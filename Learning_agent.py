"""
Learning Culinary Agent — standalone module with numeric learning loop.

Re-implements the baseline culinary pipeline (intent parsing, recipe generation,
constraint validation, RAG search, odorant pairing) and adds WorldModel /
SelfModel learning on top.  Shares only on-disk data files with the baseline
Culinary_agent.ipynb — no code imports from the notebook.

Supports two backends:
  - "groq": Remote API (Llama 3.3 70B via Groq) — preferences via prompt injection
  - "local": Local MLX model (Llama 3.1 8B) with LoRA adapters — preferences
    encoded in trainable model weights
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import json
import os
import re
import tempfile

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Union
from dotenv import load_dotenv

# faiss is lazy-loaded to avoid conflicts with PyTorch on Apple Silicon
_faiss = None
def _get_faiss():
    global _faiss
    if _faiss is None:
        import faiss
        _faiss = faiss
    return _faiss

# ── env ───────────────────────────────────────────────────────────────────────
load_dotenv()

# ── Backend selection ─────────────────────────────────────────────────────────
# Set LEARNING_AGENT_BACKEND=local to use local MLX model
# Default is "groq" for backward compatibility
BACKEND_TYPE = os.getenv("LEARNING_AGENT_BACKEND", "groq")

if BACKEND_TYPE == "groq":
    from groq import Groq
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE")
    client = Groq()
    MODEL_NAME = "llama-3.3-70b-versatile"
    _llm_backend = None  # initialized lazily
else:
    client = None
    MODEL_NAME = None

from local_model import LLMBackend

def _get_backend() -> LLMBackend:
    """Get or create the LLM backend singleton."""
    global _llm_backend
    if _llm_backend is None:
        if BACKEND_TYPE == "groq":
            _llm_backend = LLMBackend(
                backend="groq",
                model_name="llama-3.3-70b-versatile",
            )
        elif BACKEND_TYPE == "local":
            _llm_backend = LLMBackend(
                backend="local",
                model_id=os.getenv("LOCAL_MODEL_PATH", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
                adapter_dir=os.getenv("ADAPTER_DIR", "adapters"),
                hf_token=os.getenv("HF_TOKEN"),
            )
        else:
            raise ValueError(f"Unknown backend: {BACKEND_TYPE}")
    return _llm_backend

_llm_backend = None

# ── Embedding model reference (lazy-loaded by RAG functions, not at import) ──
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"

# ── RAG / data paths ─────────────────────────────────────────────────────────
RAG_DIR = "rag_docs"
FOOD_TO_ODORANT_PATH = os.path.join(RAG_DIR, "food_to_odorants.json")
ODORANT_TO_FOOD_PATH = os.path.join(RAG_DIR, "odorants_to_foods.json")

# Lazy-loading references — populated by RAG functions on first use
_embedding_model = None
_recipe_faiss_index = None
_pairing_faiss_index = None
_recipe_metadata = None
_pairing_metadata = None

# ── System prompt (semantic parser) ──────────────────────────────────────────
SYSTEM_PROMPT = """
You are a semantic parser for a cooking assistant.

Your job is to extract constraints and preferences from user input.

Rules:
- Return ONLY valid JSON
- The top-level keys 'hard_constraints', 'soft_objectives', and 'preferences' must always be present as dictionaries.
- Do NOT add extra keys
- Use null for unspecified fields within these dictionaries
- Lists must contain strings
- Floats must be between 0 and 1 when specified
- High in protein means greater than 30g
- Low in fat or carbs means less than 20g
- Low in sodium means less than 200mg
"""

# ── Schema dict (intent structure) ───────────────────────────────────────────
schema = {
    "hard_constraints": {
        "max_time_minutes": None,
        "min_protein_grams": None,
        "max_carb_grams": None,
        "max_fat_grams": None,
        "max_sodium_mg": None,
        "vegetarian": None,
        "equipment_allowed": None,
        "banned_ingredients": None,
    },
    "soft_objectives": {
        "taste_priority": None,
        "health_priority": None,
        "authenticity_priority": None,
    },
    "preferences": {
        "ingredients_exclude": None,
        "ingredients_include": None,
        "spice_level": None,
        "cuisines": None,
        "diet": None,
    },
}

# ── Kitchen state ─────────────────────────────────────────────────────────────
KITCHEN_STATE = """
{
  "kitchen_state": {

    "ingredients": [

      /* === PROTEINS === */

      {
        "name": "whole chicken bone in, french butchered",
        "quantity": 1,
        "unit": "pieces",
        "state": "raw",
        "storage": "refrigerator",
        "freshness": "fresh",
        "notes": "bone in, skin on"
      },
      {
        "name": "ground beef",
        "quantity": 500,
        "unit": "grams",
        "state": "raw",
        "storage": "refrigerator",
        "freshness": "fresh",
        "notes": "80/20"
      },
      {
        "name": "eggs",
        "quantity": 10,
        "unit": "pieces",
        "state": "whole",
        "storage": "refrigerator",
        "freshness": "good"
      },
      {
        "name": "bacon",
        "quantity": 8,
        "unit": "slices",
        "state": "raw",
        "storage": "refrigerator",
        "freshness": "good"
      },
      {
        "name": "tofu",
        "quantity": 8,
        "unit": "slices",
        "state": "raw",
        "storage": "refrigerator",
        "freshness": "good"
      },
         {
        "name": "miso",
        "quantity": 8,
        "unit": "slices",
        "state": "raw",
        "storage": "refrigerator",
        "freshness": "good"
      },
      {
        "name": "salmon fillet",
        "quantity": 2,
        "unit": "pieces",
        "state": "raw",
        "storage": "freezer",
        "freshness": "frozen"
      },
      {
        "name": "firm tofu",
        "quantity": 1,
        "unit": "block",
        "state": "raw",
        "storage": "refrigerator",
        "freshness": "good"
      },

      /* === VEGETABLES === */

      {
        "name": "spinach",
        "quantity": 3,
        "unit": "cups",
        "state": "raw",
        "storage": "refrigerator",
        "freshness": "good"
      },
      {
        "name": "mushrooms",
        "quantity": 200,
        "unit": "grams",
        "state": "raw",
        "storage": "refrigerator",
        "freshness": "good"
      },
      {
        "name": "broccoli",
        "quantity": 1,
        "unit": "head",
        "state": "raw",
        "storage": "refrigerator",
        "freshness": "fresh"
      },
      {
        "name": "bell peppers",
        "quantity": 3,
        "unit": "pieces",
        "state": "raw",
        "storage": "refrigerator"
      },
      {
        "name": "zucchini",
        "quantity": 2,
        "unit": "pieces",
        "state": "raw",
        "storage": "refrigerator"
      },
      {
        "name": "carrots",
        "quantity": 5,
        "unit": "pieces",
        "state": "raw",
        "storage": "refrigerator"
      },
      {
        "name": "yellow onion",
        "quantity": 4,
        "unit": "pieces",
        "state": "whole",
        "storage": "pantry"
      },
      {
        "name": "garlic",
        "quantity": 2,
        "unit": "bulbs",
        "state": "whole",
        "storage": "pantry"
      },
      {
        "name": "russet potatoes",
        "quantity": 6,
        "unit": "pieces",
        "state": "whole",
        "storage": "pantry"
      },

      /* === FRUITS === */

      {
        "name": "lemons",
        "quantity": 3,
        "unit": "pieces",
        "state": "whole",
        "storage": "refrigerator"
      },
      {
        "name": "apples",
        "quantity": 4,
        "unit": "pieces",
        "state": "whole",
        "storage": "refrigerator"
      },
      {
        "name": "bananas",
        "quantity": 5,
        "unit": "pieces",
        "state": "ripe",
        "storage": "counter"
      },

      /* === GRAINS & CARBS === */

      {
        "name": "rigatoni pasta",
        "quantity": 400,
        "unit": "grams",
        "state": "dry",
        "storage": "pantry"
      },
      {
        "name": "spaghetti",
        "quantity": 500,
        "unit": "grams",
        "state": "dry",
        "storage": "pantry"
      },
      {
        "name": "white rice",
        "quantity": 1,
        "unit": "kilogram",
        "state": "dry",
        "storage": "pantry"
      },
      {
        "name": "quinoa",
        "quantity": 500,
        "unit": "grams",
        "state": "dry",
        "storage": "pantry"
      },
      {
        "name": "all-purpose flour",
        "quantity": 1,
        "unit": "kilogram",
        "state": "dry",
        "storage": "pantry"
      },
      {
        "name": "bread",
        "quantity": 1,
        "unit": "loaf",
        "state": "fresh",
        "storage": "counter"
      },

      /* === DAIRY === */

      {
        "name": "milk",
        "quantity": 1,
        "unit": "liter",
        "state": "liquid",
        "storage": "refrigerator"
      },
      {
        "name": "butter",
        "quantity": 250,
        "unit": "grams",
        "state": "solid",
        "storage": "refrigerator"
      },
      {
        "name": "heavy cream",
        "quantity": 250,
        "unit": "ml",
        "state": "liquid",
        "storage": "refrigerator"
      },
      {
        "name": "parmesan cheese",
        "quantity": 150,
        "unit": "grams",
        "state": "block",
        "storage": "refrigerator"
      },
      {
        "name": "mozzarella",
        "quantity": 200,
        "unit": "grams",
        "state": "shredded",
        "storage": "refrigerator"
      },

      /* === CANNED & JARRED === */

      {
        "name": "canned diced tomatoes",
        "quantity": 2,
        "unit": "cans",
        "state": "sealed",
        "storage": "pantry"
      },
      {
        "name": "coconut milk",
        "quantity": 1,
        "unit": "can",
        "state": "sealed",
        "storage": "pantry"
      },
      {
        "name": "chickpeas",
        "quantity": 2,
        "unit": "cans",
        "state": "sealed",
        "storage": "pantry"
      },

      /* === SWEETENERS === */

      {
        "name": "granulated sugar",
        "quantity": 1,
        "unit": "kilogram",
        "state": "dry",
        "storage": "pantry"
      },
      {
        "name": "brown sugar",
        "quantity": 500,
        "unit": "grams",
        "state": "dry",
        "storage": "pantry"
      },
      {
        "name": "honey",
        "quantity": 250,
        "unit": "ml",
        "state": "liquid",
        "storage": "pantry"
      }
    ],

    /* === PANTRY STAPLES === */

    "pantry_staples": {
      "oils": ["olive oil", "neutral oil", "sesame oil"],
      "vinegars": ["red wine vinegar", "apple cider vinegar", "balsamic vinegar"],
      "seasonings": [
        "salt",
        "black pepper",
        "chili flakes",
        "paprika",
        "cumin",
        "coriander",
        "oregano",
        "thyme",
        "bay leaves",
        "cinnamon",
        "nutmeg"
      ],
      "aromatics": ["garlic", "onion", "ginger"],
      "condiments": [
        "soy sauce",
        "mustard",
        "ketchup",
        "mayonnaise",
        "hot sauce",
        "fish sauce"
      ],
      "baking": ["baking soda", "baking powder", "vanilla extract"],
      "broths": ["chicken stock cubes", "vegetable stock cubes"]
    },

    /* === COOKWARE === */

    "cookware": [
      { "name": "skillet", "size": "12-inch", "material": "stainless steel" },
      { "name": "cast iron skillet", "size": "10-inch" },
      { "name": "nonstick pan", "size": "10-inch" },
      { "name": "saucepan", "size": "medium", "material": "stainless steel" },
      { "name": "stock pot", "size": "large", "material": "aluminum" },
      { "name": "dutch oven", "size": "5-quart", "material": "enameled cast iron" },
      { "name": "sheet pan", "size": "half-sheet" },
      { "name": "casserole dish", "material": "ceramic" },
      { "name": "cutting board", "material": "wood" },
      { "name": "cutting board", "material": "plastic" },
      { "name": "chef knife", "length": "8-inch" },
      { "name": "paring knife", "length": "3-inch" },
      { "name": "bread knife", "length": "8-inch" },
      { "name": "tongs" },
      { "name": "wooden spoon" },
      { "name": "silicone spatula" },
      { "name": "whisk" },
      { "name": "ladle" },
      { "name": "colander" },
      { "name": "box grater" },
      { "name": "microplane" },
      { "name": "measuring cups" },
      { "name": "measuring spoons" },
      { "name": "mixing bowls", "quantity": 4 }
    ],

    /* === APPLIANCES === */

    "appliances": [
      { "name": "stove", "type": "gas", "burners_available": 4 },
      { "name": "oven", "type": "convection", "max_temperature_c": 260 },
      { "name": "microwave", "power_watts": 1000 },
      { "name": "blender", "type": "countertop" },
      { "name": "immersion blender" },
      { "name": "food processor" },
      { "name": "stand mixer" },
      { "name": "toaster" },
      { "name": "electric kettle" },
      { "name": "rice cooker" },
      { "name": "slow cooker" },
      { "name": "refrigerator" },
      { "name": "freezer" },
      { "name": "dishwasher" }
    ],

  }
}
"""

# ── Recipe generation prompt ──────────────────────────────────────────────────
RECIPE_GENERATION_PROMPT = """
You are a culinary professor. Use the provided user intent, user query, and kitchen state to generate 3 vastly different recipes in the provided format.
Try to make the recipes as different as possible.

Rules:
1. Use EXACT ingredient names from kitchen state.
2. Output ONLY the JSON block.
3. Do not include extra fields in ingredients other than name, quantity, unit, and preparation.
4. Adhere to the rules in the user intent and query.

IMPORTANT:
All quantities must be decimal floats.

Allowed:
0.5
1.25

Not allowed:
1/2
one
a pinch
to taste

Structure:
{
  "recipes": [
    {
      "recipe_name": "string",
      "description": "string",
      "ingredients_required": [
        {
          "name": "string",
          "quantity": number,
          "unit": "string",
          "preparation": "string"
        }
      ],
      "fit_to_intent": {
        "why_it_matches": "string",
        "cuisine_alignment": "string",
        "time_estimate_minutes": number
      },
      "flavor_profile": [
        {
          "ingredients": ["string"],
          "flavor_interaction": "string"
        }
      ],
      "steps": [
        {
          "step_number": number,
          "instruction": "string"
        }
      ]
    }
  ]
}
"""

# ── Pydantic models ──────────────────────────────────────────────────────────

class HardConstraints(BaseModel):
    max_time_minutes: Optional[int] = Field(None, description="Maximum allowed cooking time in minutes")
    min_protein_grams: Optional[int] = Field(None, description="Minimum required grams protein in grams")
    max_carb_grams: Optional[int] = Field(None, description="Max allowed carbs in grams")
    max_fat_grams: Optional[int] = Field(None, description="Max allowed fat in grams")
    max_sodium_mg: Optional[int] = Field(None, description="Maximum allowed sodium in milligrams")
    vegetarian: Optional[bool] = Field(None, description="Whether the meal must be vegetarian")
    equipment_allowed: Optional[List[str]] = Field(None, description="List of allowed cooking equipment")
    banned_ingredients: Optional[List[str]] = Field(None, description="List of ingredients that are not allowed")

class SoftObjectives(BaseModel):
    taste_priority: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Importance of taste")
    health_priority: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Importance of health")
    authenticity_priority: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Importance of cultural authenticity")

class Preferences(BaseModel):
    ingredients_exclude: Optional[List[str]] = Field(None, description="List of ingredients to exclude")
    ingredients_include: Optional[List[str]] = Field(None, description="List of ingredients to include")
    spice_level: Optional[float] = Field(None, ge=0.0, le=1.0, description="Desired spice level")
    cuisines: Optional[List[str]] = Field(None, description="Preferred cuisines")
    diet: Optional[str] = Field(None, description="Diet that user is following")

class Intent(BaseModel):
    hard_constraints: HardConstraints
    soft_objectives: SoftObjectives
    preferences: Preferences

class Ingredient(BaseModel):
    name: str
    quantity: Union[float, int]
    unit: str
    preparation: Optional[str] = None

class FitToIntent(BaseModel):
    why_it_matches: str
    cuisine_alignment: str
    time_estimate_minutes: int

class FlavorCombination(BaseModel):
    ingredients: List[str] = Field(description="Exact ingredient names from kitchen state")
    flavor_interaction: str = Field(description="Brief explanation of how these flavors interact")

class Step(BaseModel):
    step_number: int
    instruction: str

class Recipe(BaseModel):
    recipe_name: str
    description: str
    ingredients_required: List[Ingredient]
    fit_to_intent: FitToIntent
    flavor_profile: List[FlavorCombination]
    steps: List[Step]

class RecipeCandidates(BaseModel):
    recipes: List[Recipe]


class IngredientAddition(BaseModel):
    ingredient: str
    reason: str


class RecipeIngredientSuggestions(BaseModel):
    recipe_name: str
    kitchen_state_additions: List[IngredientAddition]
    external_additions: List[IngredientAddition]


class RecipeEnhanced(BaseModel):
    recipe_name: str
    description: str
    enhancements_description: str
    ingredients_required: List[Ingredient]
    steps: List[Step]


# ── Persistence utilities ────────────────────────────────────────────────────


def atomic_write_json(data: dict, path: str) -> None:
    """Write *data* as JSON to *path* atomically (write-to-temp then rename)."""
    dir_name = os.path.dirname(os.path.abspath(path))
    fd, tmp_path = tempfile.mkstemp(suffix=".tmp", dir=dir_name)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    except BaseException:
        # Clean up the temp file on any failure; let the error propagate.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def safe_load_json(path: str, defaults: dict) -> dict:
    """Load JSON from *path*; return *defaults* on missing/corrupt file."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Info: {path} not found – using defaults.")
        return defaults
    except json.JSONDecodeError:
        print(f"Warning: {path} contains invalid JSON – using defaults.")
        return defaults


def log_run(log_path: str, run_record: dict) -> None:
    """Append a single JSON-line *run_record* to *log_path*, creating the file if needed."""
    with open(log_path, "a") as f:
        f.write(json.dumps(run_record) + "\n")


# ── WorldModel ───────────────────────────────────────────────────────────────


class WorldModel:
    """Persistent ingredient-taste contribution weights.

    Supports two initialization strategies:
    - **zero-init** (default): all weights start at 0.0 (empty dict).
    - **prior-init** (`use_priors=True`): weights seeded from odorant compound
      counts in ``food_to_odorants.json``, normalized to the 0–1 range.
    """

    def __init__(self, persistence_path: str, use_priors: bool = False) -> None:
        self.persistence_path = persistence_path
        self.use_priors = use_priors
        self.ingredient_weights: dict[str, float] = {}
        self.load()

    # ── persistence ──────────────────────────────────────────────────────

    def load(self) -> None:
        """Load weights from *persistence_path*, falling back to zero or prior init."""
        if os.path.exists(self.persistence_path):
            data = safe_load_json(self.persistence_path, {})
            self.ingredient_weights = {
                k: float(v) for k, v in data.get("ingredient_weights", {}).items()
            }
        elif self.use_priors:
            self._init_from_priors()
        else:
            self.ingredient_weights = {}

    def save(self) -> None:
        """Persist current weights via atomic write."""
        atomic_write_json({"ingredient_weights": self.ingredient_weights}, self.persistence_path)

    # ── prior initialization ─────────────────────────────────────────────

    def _init_from_priors(self) -> None:
        """Seed weights from odorant compound counts, normalized to [0, 1]."""
        with open(FOOD_TO_ODORANT_PATH, "r") as f:
            food_to_odorants = json.load(f)

        counts = {food: len(odorants) for food, odorants in food_to_odorants.items()}
        max_count = max(counts.values()) if counts else 1
        self.ingredient_weights = {
            food: count / max_count for food, count in counts.items()
        }

    # ── prediction & update ──────────────────────────────────────────────

    def predict_taste(self, recipe: Recipe) -> float:
        """Return predicted taste as the sum of ingredient weights for the recipe.

        Missing ingredients default to 0.0.
        """
        return sum(
            self.ingredient_weights.get(ing.name, 0.0)
            for ing in recipe.ingredients_required
        )

    def update(self, recipe: Recipe, error: float, learning_rate: float) -> None:
        """Update ingredient weights using prediction error.

        For each ingredient in the recipe:
            weight += learning_rate * error
        Ingredients not yet tracked are initialized to 0.0 before the update.
        """
        for ing in recipe.ingredients_required:
            current = self.ingredient_weights.get(ing.name, 0.0)
            self.ingredient_weights[ing.name] = current + learning_rate * error

    def get_top_ingredients(self, k: int = 20) -> list[tuple[str, float]]:
        """Return the *k* highest-weighted ingredients as (name, weight) tuples."""
        return sorted(
            self.ingredient_weights.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:k]

    def get_penalized_ingredients(self, threshold: float = -0.01) -> list[tuple[str, float]]:
        """Return ingredients with weights below *threshold* (learned dislikes).

        These are ingredients the agent has learned to avoid based on
        negative prediction errors in past recipes.
        """
        return sorted(
            [(name, w) for name, w in self.ingredient_weights.items() if w < threshold],
            key=lambda item: item[1],
        )

# ── SelfModel ────────────────────────────────────────────────────────────────


SPICY_KEYWORDS = [
    "chili", "pepper", "hot sauce", "cayenne", "jalapeño",
    "habanero", "sriracha", "wasabi", "horseradish", "ginger",
    "chili flakes", "paprika", "cumin",
]

HEALTH_CONSTRAINT_KEYS = [
    "min_protein_grams", "max_carb_grams", "max_fat_grams",
    "max_sodium_mg", "vegetarian",
]


class SelfModel:
    """Persistent behavioral preference abstractions.

    Tracks three preference dimensions that the agent develops over time
    from its own recipe generation patterns:
    - **spice_preference** (float 0–1): learned affinity for spicy ingredients.
    - **health_bias** (float 0–1): learned tendency toward health-oriented recipes.
    - **cuisine_affinity** (dict str→float): per-cuisine affinity scores developed
      through repeated recipe generation.
    """

    def __init__(self, persistence_path: str) -> None:
        self.persistence_path = persistence_path
        self.spice_preference: float = 0.5
        self.health_bias: float = 0.5
        self.cuisine_affinity: dict[str, float] = {}
        self.load()

    # ── persistence ──────────────────────────────────────────────────────

    def load(self) -> None:
        """Load state from *persistence_path*, falling back to neutral defaults."""
        defaults = {
            "spice_preference": 0.5,
            "health_bias": 0.5,
            "cuisine_affinity": {},
        }
        data = safe_load_json(self.persistence_path, defaults)
        self.spice_preference = float(data.get("spice_preference", 0.5))
        self.health_bias = float(data.get("health_bias", 0.5))
        self.cuisine_affinity = {
            k: float(v) for k, v in data.get("cuisine_affinity", {}).items()
        }

    def save(self) -> None:
        """Persist current state via atomic write."""
        atomic_write_json(
            {
                "spice_preference": self.spice_preference,
                "health_bias": self.health_bias,
                "cuisine_affinity": self.cuisine_affinity,
            },
            self.persistence_path,
        )

    # ── update ───────────────────────────────────────────────────────────

    def update(self, recipe: Recipe, intent: dict, learning_rate: float) -> None:
        """Update preferences from an observed recipe decision.

        Parameters
        ----------
        recipe : Recipe
            The recipe that was generated / selected.
        intent : dict
            The parsed user intent (must contain ``"hard_constraints"`` key).
        learning_rate : float
            Step size for incremental updates.
        """
        # ── spice preference ─────────────────────────────────────────────
        ingredients = recipe.ingredients_required
        total = len(ingredients)
        if total > 0:
            spicy_count = sum(
                1 for ing in ingredients
                if any(kw in ing.name.lower() for kw in SPICY_KEYWORDS)
            )
            spice_ratio = spicy_count / total
        else:
            spice_ratio = 0.0

        self.spice_preference += learning_rate * (spice_ratio - self.spice_preference)

        # ── health bias ──────────────────────────────────────────────────
        hard_constraints = intent.get("hard_constraints", {})
        health_constraints = {
            k: v for k, v in hard_constraints.items()
            if k in HEALTH_CONSTRAINT_KEYS and v is not None
        }

        if health_constraints:
            satisfied = 0
            for key, value in health_constraints.items():
                if key == "vegetarian":
                    # Check if recipe is vegetarian by looking for meat keywords
                    meat_keywords = [
                        "chicken", "beef", "pork", "lamb", "turkey", "duck",
                        "veal", "bacon", "sausage", "ham", "steak", "meat",
                        "fish", "salmon", "tuna", "shrimp", "prawn", "crab",
                        "lobster", "anchovy",
                    ]
                    is_veg = not any(
                        any(mk in ing.name.lower() for mk in meat_keywords)
                        for ing in ingredients
                    )
                    if value and is_veg:
                        satisfied += 1
                    elif not value:
                        satisfied += 1
                else:
                    # Numeric constraints — we assume satisfied as a reasonable
                    # default since we don't have exact nutritional data on the
                    # recipe.  The heuristic gives partial credit.
                    satisfied += 1
            health_ratio = satisfied / len(health_constraints)
        else:
            health_ratio = 0.5

        self.health_bias += learning_rate * (health_ratio - self.health_bias)

        # ── cuisine affinity ─────────────────────────────────────────────
        cuisine_raw = recipe.fit_to_intent.cuisine_alignment
        cuisine = cuisine_raw.strip().lower()
        if cuisine:
            current = self.cuisine_affinity.get(cuisine, 0.0)
            self.cuisine_affinity[cuisine] = current + learning_rate * (1.0 - current)

    # ── prompt context ───────────────────────────────────────────────────

    def to_prompt_context(self) -> str:
        """Format preferences as a string block for prompt injection."""
        parts = [
            f"Agent learned preferences: spice_preference={self.spice_preference:.1f}, "
            f"health_bias={self.health_bias:.1f}."
        ]

        if self.cuisine_affinity:
            # Sort by affinity descending, take top entries
            sorted_cuisines = sorted(
                self.cuisine_affinity.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            cuisine_strs = [f"{name} ({score:.1f})" for name, score in sorted_cuisines]
            parts.append(f"Preferred cuisines: {', '.join(cuisine_strs)}.")

        return " ".join(parts)

# ── Taste Predictor (thin wrapper) ────────────────────────────────────────────

def predict_taste(world_model: WorldModel, recipe: Recipe) -> float:
    """Predict taste score for a recipe by delegating to WorldModel.

    Requirements: 3.1, 3.3
    """
    return world_model.predict_taste(recipe)


# ── Taste Evaluator ───────────────────────────────────────────────────────────

# Pre-parse kitchen ingredient freshness from KITCHEN_STATE for quality scoring.
# The KITCHEN_STATE string uses JS-style comments, so we strip them before parsing.
def _parse_kitchen_freshness() -> dict[str, str]:
    """Extract ingredient name → freshness mapping from KITCHEN_STATE."""
    # Strip JS-style block comments (/* ... */) from the JSON string
    cleaned = re.sub(r"/\*.*?\*/", "", KITCHEN_STATE, flags=re.DOTALL)
    try:
        state = json.loads(cleaned)
    except json.JSONDecodeError:
        return {}
    freshness_map: dict[str, str] = {}
    for item in state.get("kitchen_state", {}).get("ingredients", []):
        name = item.get("name", "").lower()
        freshness = item.get("freshness", "")
        if name and freshness:
            freshness_map[name] = freshness.lower()
    return freshness_map


_KITCHEN_FRESHNESS: dict[str, str] = _parse_kitchen_freshness()

_FRESHNESS_SCORES: dict[str, float] = {
    "fresh": 1.0,
    "good": 0.8,
    "frozen": 0.5,
}

# ── Taste modality mapping ───────────────────────────────────────────────────
# Maps odorant descriptors to the five basic taste modalities.
# A descriptor can map to multiple modalities.
_TASTE_MODALITIES: dict[str, list[str]] = {
    "sweet": ["sweet", "honey", "caramel", "vanilla", "sugar", "candy",
              "chocolate", "butterscotch", "maple", "molasses", "creamy"],
    "salty": ["salty", "briny", "saline", "marine", "oceanic", "sea"],
    "sour": ["sour", "acidic", "tart", "citrus", "vinegar", "lemon",
             "lime", "tangy", "acetic", "fermented"],
    "bitter": ["bitter", "coffee", "cocoa", "dark", "burnt", "charred",
               "roasted", "smoky", "astringent", "medicinal"],
    "umami": ["umami", "savory", "meaty", "brothy", "mushroom", "malt",
              "fermented", "cheese", "soy", "roast beef", "meat"],
}

# Total number of taste modalities (for normalization)
_NUM_MODALITIES = len(_TASTE_MODALITIES)


def _fuzzy_odorant_lookup(ingredient: str, food_to_odorants: dict) -> list:
    """Look up odorants for an ingredient with fuzzy matching fallback.

    Strategy:
    1. Exact match on the full ingredient name.
    2. Check if any food key is a substring of the ingredient name
       (e.g. "chicken" in "whole chicken bone in, french butchered").
    3. Check if the ingredient name is a substring of any food key.
    4. Token overlap — pick the food key sharing the most words with
       the ingredient, requiring at least 1 meaningful token match.

    Returns the odorant list from the best match, or [] if nothing found.
    """
    # 1. Exact match
    if ingredient in food_to_odorants:
        return food_to_odorants[ingredient]

    # 2. Food key is substring of ingredient
    for food_key, odorants in food_to_odorants.items():
        if len(food_key) >= 3 and food_key in ingredient:
            return odorants

    # 3. Ingredient is substring of food key
    if len(ingredient) >= 3:
        for food_key, odorants in food_to_odorants.items():
            if ingredient in food_key:
                return odorants

    # 4. Token overlap
    _STOP_WORDS = {
        "a", "an", "the", "of", "in", "with", "and", "or", "for",
        "to", "on", "bone", "whole", "fresh", "dried", "ground",
        "raw", "cooked", "cut", "sliced", "diced", "chopped",
        "pieces", "small", "large", "medium",
    }
    ing_tokens = set(ingredient.split()) - _STOP_WORDS
    if not ing_tokens:
        return []

    best_match = None
    best_overlap = 0
    for food_key, odorants in food_to_odorants.items():
        food_tokens = set(food_key.split()) - _STOP_WORDS
        overlap = len(ing_tokens & food_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_match = odorants

    return best_match if best_overlap >= 1 else []


def evaluate_taste(recipe: Recipe, intent: dict, food_to_odorants: dict) -> float:
    """Compute actual taste as a weighted heuristic combination.

    actual_taste = 0.35 * pairing_score
                 + 0.25 * flavor_balance_score
                 + 0.15 * odorant_diversity_score
                 + 0.25 * constraint_score

    Each component is clamped to [0.0, 1.0].

    Requirements: 4.1, 4.2, 4.3, 4.4
    """
    ingredients = recipe.ingredients_required
    ingredient_names = [ing.name.lower() for ing in ingredients]

    # ── Collect all odorants for the recipe ──────────────────────────────
    all_odorants = []  # list of odorant dicts across all ingredients
    all_descriptors = []  # flat list of descriptor strings
    all_functional_groups = set()
    for name in ingredient_names:
        for o in _fuzzy_odorant_lookup(name, food_to_odorants):
            if isinstance(o, dict):
                all_odorants.append(o)
                all_descriptors.extend(
                    d.lower() for d in o.get("descriptors", [])
                )
                all_functional_groups.update(
                    fg.lower() for fg in o.get("functional_groups", [])
                )

    # ── 1. Pairing score (weight 0.35) ───────────────────────────────────
    ingredient_names = [ing.name.lower() for ing in ingredients]
    pairs = [
        (ingredient_names[i], ingredient_names[j])
        for i in range(len(ingredient_names))
        for j in range(i + 1, len(ingredient_names))
    ]

    if pairs and food_to_odorants:
        shared_counts = []
        max_shared = 0
        for a, b in pairs:
            raw_a = _fuzzy_odorant_lookup(a, food_to_odorants)
            raw_b = _fuzzy_odorant_lookup(b, food_to_odorants)
            odorants_a = set(
                o["name"] if isinstance(o, dict) else o for o in raw_a
            )
            odorants_b = set(
                o["name"] if isinstance(o, dict) else o for o in raw_b
            )
            shared = len(odorants_a & odorants_b)
            shared_counts.append(shared)
            if shared > max_shared:
                max_shared = shared

        if max_shared > 0:
            pairing_score = sum(c / max_shared for c in shared_counts) / len(shared_counts)
        else:
            pairing_score = 0.5
    else:
        pairing_score = 0.5

    pairing_score = max(0.0, min(1.0, pairing_score))

    # ── 2. Flavor balance score (weight 0.25) ────────────────────────────
    # Measures how many of the 5 basic taste modalities are represented.
    if all_descriptors:
        modalities_hit = set()
        descriptor_set = set(all_descriptors)
        for modality, keywords in _TASTE_MODALITIES.items():
            if descriptor_set & set(keywords):
                modalities_hit.add(modality)
        flavor_balance_score = len(modalities_hit) / _NUM_MODALITIES
    else:
        flavor_balance_score = 0.5

    flavor_balance_score = max(0.0, min(1.0, flavor_balance_score))

    # ── 3. Odorant diversity score (weight 0.15) ─────────────────────────
    # Rewards recipes whose ingredients draw from many distinct odorant
    # compounds and functional groups, indicating aromatic complexity.
    if all_odorants:
        unique_odorant_names = set(
            o["name"] for o in all_odorants if isinstance(o, dict)
        )
        # Normalize: 30+ unique odorants is considered highly diverse
        odorant_count_score = min(len(unique_odorant_names) / 30.0, 1.0)
        # Normalize: 10+ functional groups is considered highly diverse
        fg_count_score = min(len(all_functional_groups) / 10.0, 1.0)
        odorant_diversity_score = 0.6 * odorant_count_score + 0.4 * fg_count_score
    else:
        odorant_diversity_score = 0.5

    odorant_diversity_score = max(0.0, min(1.0, odorant_diversity_score))

    # ── 4. Constraint satisfaction score (weight 0.25) ───────────────────
    total_constraints = 0
    satisfied_constraints = 0

    # Check hard constraints
    hard_constraints = intent.get("hard_constraints", {})
    for key, value in hard_constraints.items():
        if value is None:
            continue
        total_constraints += 1

        if key == "vegetarian":
            meat_keywords = [
                "chicken", "beef", "pork", "lamb", "turkey", "duck",
                "veal", "bacon", "sausage", "ham", "steak", "meat",
                "fish", "salmon", "tuna", "shrimp", "prawn", "crab",
                "lobster", "anchovy",
            ]
            is_veg = not any(
                any(mk in ing.name.lower() for mk in meat_keywords)
                for ing in ingredients
            )
            if (value and is_veg) or (not value):
                satisfied_constraints += 1

        elif key == "max_time_minutes":
            if recipe.fit_to_intent.time_estimate_minutes <= value:
                satisfied_constraints += 1

        elif key == "banned_ingredients":
            banned = {b.lower() for b in value}
            has_banned = any(ing.name.lower() in banned for ing in ingredients)
            if not has_banned:
                satisfied_constraints += 1

        elif key == "equipment_allowed":
            # Assume satisfied — we don't have equipment usage data on recipes
            satisfied_constraints += 1

        else:
            # Numeric nutrition constraints — assume satisfied as heuristic
            satisfied_constraints += 1

    # Check preferences
    preferences = intent.get("preferences", {})
    for key, value in preferences.items():
        if value is None:
            continue
        total_constraints += 1

        if key == "ingredients_exclude":
            excluded = {e.lower() for e in value}
            has_excluded = any(ing.name.lower() in excluded for ing in ingredients)
            if not has_excluded:
                satisfied_constraints += 1

        elif key == "ingredients_include":
            included = {inc.lower() for inc in value}
            recipe_ings = {ing.name.lower() for ing in ingredients}
            if included.issubset(recipe_ings):
                satisfied_constraints += 1

        elif key == "cuisines":
            cuisine = recipe.fit_to_intent.cuisine_alignment.strip().lower()
            if any(c.lower() in cuisine for c in value):
                satisfied_constraints += 1

        else:
            # spice_level, diet — assume satisfied as heuristic
            satisfied_constraints += 1

    if total_constraints > 0:
        constraint_score = satisfied_constraints / total_constraints
    else:
        constraint_score = 0.5

    constraint_score = max(0.0, min(1.0, constraint_score))

    # ── Final weighted combination ───────────────────────────────────────
    return (
        0.35 * pairing_score
        + 0.25 * flavor_balance_score
        + 0.15 * odorant_diversity_score
        + 0.25 * constraint_score
    )


# ── Mode Selector ─────────────────────────────────────────────────────────────

VALID_MODES = {"baseline", "world_model_only", "self_model_only", "full_model"}


def select_mode(mode: str) -> str:
    """Validate and return the operating mode.

    Raises ValueError with a descriptive message listing all valid modes
    if *mode* is not one of the four valid options.

    Requirements: 9.1, 9.6
    """
    if mode not in VALID_MODES:
        raise ValueError(
            f"Invalid mode '{mode}'. Must be one of: "
            f"baseline, world_model_only, self_model_only, full_model"
        )
    return mode


# ── Baseline Pipeline: Intent Parsing ─────────────────────────────────────────


def clean_llm_json(text: str) -> str:
    """Strip markdown code fences from LLM output and return clean JSON string.

    Handles ```json ... ``` and ``` ... ``` patterns.
    Also converts Python dict syntax (single quotes, None, True/False)
    to valid JSON for smaller models that output Python literals.

    Requirements: 12.2
    """
    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)
    text = text.strip()

    # If it looks like a Python dict (single quotes), try to convert
    if "'" in text and '"' not in text:
        try:
            import ast
            parsed = ast.literal_eval(text)
            return json.dumps(parsed)
        except (ValueError, SyntaxError):
            pass

    # Replace Python-style None/True/False with JSON equivalents
    text = re.sub(r'\bnull\b', 'null', text)
    text = re.sub(r'\bNone\b', 'null', text)
    text = re.sub(r'\bTrue\b', 'true', text)
    text = re.sub(r'\bFalse\b', 'false', text)
    # Single quotes to double quotes (simple cases)
    if text.startswith("{") and "'" in text and '"' not in text:
        text = text.replace("'", '"')

    return text


def parse_intent(user_input: str, schema: dict, temperature: float = 0.8) -> dict:
    """Call LLM to extract intent from user input.

    Uses the configured backend (Groq API or local MLX model).
    The *schema* dict is included in the prompt so the LLM knows the expected
    structure.  The response is cleaned with :func:`clean_llm_json` and parsed
    into a dict via Pydantic validation (pruning ``None`` fields).

    Returns the parsed intent dict, or ``None`` if the LLM returns invalid JSON.

    Requirements: 12.2, 12.5
    """
    prompt = f"""
Schema:
{schema}

User request:
{user_input}
"""

    backend = _get_backend()
    text = backend.generate(
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        temperature=temperature,
        max_tokens=200,
    )

    _SPICE_LABEL_MAP = {
        "none": 0.0, "mild": 0.25, "medium": 0.5, "high": 0.75, "extreme": 1.0,
    }

    try:
        text = clean_llm_json(text)
        data = json.loads(text)

        # Coerce string spice_level to float before validation
        prefs = data.get("preferences") or {}
        sl = prefs.get("spice_level")
        if isinstance(sl, str):
            prefs["spice_level"] = _SPICE_LABEL_MAP.get(sl.lower())

        intent = Intent.model_validate(data)
        # Remove fields where value is None
        intent = intent.model_dump(exclude_none=True)
        return intent
    except json.JSONDecodeError:
        print("Invalid JSON:", text)
        return None

# ── Baseline Pipeline: Recipe Generation ──────────────────────────────────────


def generate_recipes(
    intent: dict,
    user_input: str,
    wm_bias: str = None,
    sm_bias: str = None,
    temperature: float = 0.8,
) -> RecipeCandidates:
    """Generate candidate recipes via LLM.

    When using the local backend, generates 1 recipe for speed.
    When using Groq, generates 3 as before.
    """

    # -- Build optional bias section ------------------------------------------
    bias_section = ""
    if wm_bias:
        bias_section += (
            "\nWorldModel ingredient ranking (prefer these ingredients):\n"
            f"{wm_bias}\n"
        )
    if sm_bias:
        bias_section += (
            "\nAgent's own learned preferences (incorporate these tendencies):\n"
            f"{sm_bias}\n"
        )

    backend = _get_backend()
    # Use fewer recipes for local model (speed)
    if backend.backend_type == "local":
        recipe_count_instruction = RECIPE_GENERATION_PROMPT.replace(
            "generate 3 vastly different recipes",
            "generate 1 recipe"
        )
        max_tok = 1200
    else:
        recipe_count_instruction = RECIPE_GENERATION_PROMPT
        max_tok = 4096

    prompt = f"""
User request:
{user_input}

User intent:
{intent}

Kitchen state:
{KITCHEN_STATE}
{bias_section}
Prompt:
{recipe_count_instruction}
"""

    text = backend.generate(
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        temperature=temperature,
        max_tokens=max_tok,
    )

    try:
        text = clean_llm_json(text)
        data = json.loads(text)
        recipes = RecipeCandidates.model_validate(data)
        return recipes
    except json.JSONDecodeError:
        print("Invalid JSON:", text)
        return None

# ── Constraint Validation ─────────────────────────────────────────────────────


def is_vegetarian(recipe: Recipe) -> bool:
    """Return True if the recipe contains no meat/fish ingredients.

    Uses keyword-based detection against a list of common non-vegetarian
    ingredient terms.  Mirrors the logic from Culinary_agent.ipynb.

    Requirements: 12.5
    """
    non_veg_keywords = [
        "chicken", "beef", "pork", "fish", "shrimp",
        "bacon", "lamb", "turkey", "anchovy",
    ]

    ingredients = [ing.name.lower().strip() for ing in recipe.ingredients_required]

    for ing in ingredients:
        if any(meat in ing for meat in non_veg_keywords):
            return False

    return True


def recipe_matches_intent(recipe: Recipe, intent: dict) -> bool:
    """Check whether *recipe* satisfies the constraints in *intent*.

    Hard constraints checked: vegetarian, max_time_minutes, banned_ingredients.
    Preferences checked: ingredients_exclude, ingredients_include.

    Returns True if all applicable constraints are satisfied.
    Mirrors the logic from Culinary_agent.ipynb.

    Requirements: 12.5
    """
    if intent is None:
        hc = {}
        pref = {}
    else:
        hc = intent.get("hard_constraints", {})
        pref = intent.get("preferences", {})

    ingredients = [ing.name.lower().strip() for ing in recipe.ingredients_required]

    # -------- time constraint --------
    max_time = hc.get("max_time_minutes")
    if max_time is not None:
        if recipe.fit_to_intent.time_estimate_minutes > max_time:
            return False

    # -------- vegetarian --------
    vegetarian = hc.get("vegetarian")
    if vegetarian is True:
        if not is_vegetarian(recipe):
            return False

    # -------- banned ingredients --------
    banned = hc.get("banned_ingredients")
    if banned:
        banned = [b.lower() for b in banned]
        if any(b in ingredients for b in banned):
            return False

    # -------- excluded ingredients (preference) --------
    excluded = pref.get("ingredients_exclude")
    if excluded:
        excluded = [e.lower() for e in excluded]
        if any(e in ingredients for e in excluded):
            return False

    # -------- required ingredients (preference) --------
    required = pref.get("ingredients_include")
    if required:
        required = [r.lower() for r in required]
        if not any(
            r in ing for r in required for ing in ingredients
        ):
            return False

    return True


def validate_and_fix_recipes(
    candidates: RecipeCandidates,
    intent: dict,
    user_input: str,
    wm_bias: str = None,
    sm_bias: str = None,
    temperature: float = 0.8,
) -> RecipeCandidates:
    """Validate each recipe against *intent* and retry failing ones.

    For each recipe in *candidates* that fails ``recipe_matches_intent``,
    attempt to regenerate a replacement up to 3 times by calling
    ``generate_recipes``.  The first new recipe that passes validation and
    is not a duplicate (by name) replaces the failing one.

    If a recipe still fails after 3 retry attempts the last generated
    candidate is kept so the caller always gets back the same number of
    recipes.

    Requirements: 11.2, 12.5
    """
    MAX_RETRIES = 1

    existing_names = set(
        r.recipe_name.lower().strip() for r in candidates.recipes
    )

    for i, recipe in enumerate(candidates.recipes):
        if recipe_matches_intent(recipe, intent):
            continue

        print(f"Recipe '{recipe.recipe_name}' failed constraints. Regenerating...")

        attempts = 0
        replaced = False

        while attempts < MAX_RETRIES:
            new_candidates = generate_recipes(
                intent, user_input, wm_bias=wm_bias, sm_bias=sm_bias,
                temperature=temperature,
            )

            if not new_candidates:
                attempts += 1
                continue

            for new_recipe in new_candidates.recipes:
                if new_recipe.recipe_name.lower().strip() in existing_names:
                    continue

                if recipe_matches_intent(new_recipe, intent):
                    candidates.recipes[i] = new_recipe
                    existing_names.add(new_recipe.recipe_name.lower().strip())
                    print(f"Replaced with: {new_recipe.recipe_name}")
                    replaced = True
                    break

            if replaced:
                break

            attempts += 1

        # If still not replaced after all retries, keep last attempt if available
        if not replaced and new_candidates and new_candidates.recipes:
            last = new_candidates.recipes[0]
            if last.recipe_name.lower().strip() not in existing_names:
                candidates.recipes[i] = last
                existing_names.add(last.recipe_name.lower().strip())

    return candidates

# ── RAG search helpers (lazy-loaded) ─────────────────────────────────────────


def _get_embedding_model() -> SentenceTransformer:
    """Return the BGE embedding model, loading it on first call."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def _load_faiss_index(index_name: str):
    """Load and cache a FAISS index from *rag_docs/<index_name>*."""
    global _recipe_faiss_index, _pairing_faiss_index

    if index_name == "recipe_faiss.index":
        if _recipe_faiss_index is None:
            path = os.path.join(RAG_DIR, index_name)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"FAISS index not found: {path}. Ensure rag_docs/ contains {index_name}."
                )
            _recipe_faiss_index = _get_faiss().read_index(path)
        return _recipe_faiss_index

    if index_name == "pairing_faiss.index":
        if _pairing_faiss_index is None:
            path = os.path.join(RAG_DIR, index_name)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"FAISS index not found: {path}. Ensure rag_docs/ contains {index_name}."
                )
            _pairing_faiss_index = _get_faiss().read_index(path)
        return _pairing_faiss_index

    raise ValueError(f"Unknown FAISS index name: {index_name}")


def _load_metadata(metadata_name: str) -> list:
    """Load and cache a metadata JSON array from *rag_docs/<metadata_name>*."""
    global _recipe_metadata, _pairing_metadata

    if metadata_name == "recipe_metadata.json":
        if _recipe_metadata is None:
            path = os.path.join(RAG_DIR, metadata_name)
            with open(path) as f:
                _recipe_metadata = json.load(f)
        return _recipe_metadata

    if metadata_name == "pairing_metadata.json":
        if _pairing_metadata is None:
            path = os.path.join(RAG_DIR, metadata_name)
            with open(path) as f:
                _pairing_metadata = json.load(f)
        return _pairing_metadata

    raise ValueError(f"Unknown metadata file: {metadata_name}")


# ── RAG search functions ─────────────────────────────────────────────────────


def recipe_search(query: str, k: int = 5) -> list:
    """Search the recipe FAISS index and return top-*k* metadata entries.

    Requirements: 12.3, 12.4
    """
    model = _get_embedding_model()
    index = _load_faiss_index("recipe_faiss.index")
    metadata = _load_metadata("recipe_metadata.json")

    qvec = model.encode([query], normalize_embeddings=True).astype("float32")
    _scores, ids = index.search(qvec, k)

    return [metadata[i] for i in ids[0]]


def pairing_search(query: str, k: int = 5) -> list:
    """Search the pairing FAISS index and return top-*k* metadata entries.

    Requirements: 12.3, 12.4
    """
    model = _get_embedding_model()
    index = _load_faiss_index("pairing_faiss.index")
    metadata = _load_metadata("pairing_metadata.json")

    qvec = model.encode([query], normalize_embeddings=True).astype("float32")
    _scores, ids = index.search(qvec, k)

    return [metadata[i] for i in ids[0]]

# ── Odorant graph utilities ──────────────────────────────────────────────────
# Requirements: 12.3


def normalize(name: str) -> str:
    """Normalize an ingredient name: lowercase and strip whitespace."""
    return name.lower().strip()


def get_dish_odorants(
    dish_ingredients: list,
    food_to_odorants: dict,
) -> set:
    """Return the union of all odorant compound names for *dish_ingredients*.

    *food_to_odorants* maps normalized food names to lists of odorant dicts,
    each containing at least a ``"name"`` key.
    """
    odorants: set = set()
    for ing in dish_ingredients:
        ing = normalize(ing)
        if ing in food_to_odorants:
            for entry in food_to_odorants[ing]:
                odorants.add(normalize(entry["name"]))
    return odorants


def find_candidates(
    dish_odorants: set,
    odorants_to_foods: dict,
) -> dict:
    """Find all foods sharing at least one odorant with *dish_odorants*.

    *odorants_to_foods* maps normalized odorant names to dicts with a
    ``"foods"`` key (list of food names).

    Returns a dict of ``{food_name: count_of_shared_odorants}``.
    """
    scores: dict = {}
    for odor in dish_odorants:
        odor_key = normalize(odor)
        if odor_key not in odorants_to_foods:
            continue
        for food in odorants_to_foods[odor_key]["foods"]:
            food_key = normalize(food)
            scores[food_key] = scores.get(food_key, 0) + 1
    return scores


def suggest_ingredients(
    dish_ingredients: list,
    food_to_odorants: dict,
    odorants_to_foods: dict,
    top_k: int = 10,
) -> list:
    """Suggest *top_k* complementary ingredients based on shared odorants.

    Orchestrates :func:`normalize`, :func:`get_dish_odorants`, and
    :func:`find_candidates`, filters out ingredients already in the dish,
    and returns the top-k candidates sorted by shared odorant count
    (descending).
    """
    normalized_dish = {normalize(ing) for ing in dish_ingredients}

    dish_odorants = get_dish_odorants(dish_ingredients, food_to_odorants)

    candidates = find_candidates(dish_odorants, odorants_to_foods)

    # Filter out ingredients already in the dish
    filtered = {
        food: count
        for food, count in candidates.items()
        if food not in normalized_dish
    }

    ranked = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


# ── Convergence Detection ─────────────────────────────────────────────────────


def detect_convergence(log_path: str) -> bool:
    """Check if SelfModel values have converged across the last 5 runs.

    Reads the last 5 entries from the run log. If spice_preference,
    health_bias, and all cuisine_affinity values each changed by less than
    0.001 between every pair of consecutive runs, returns True.

    Returns False if fewer than 5 entries exist or any value changed by
    >= 0.001.

    Requirements: 13.4
    """
    if not os.path.exists(log_path):
        return False

    entries: list[dict] = []
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if len(entries) < 5:
        return False

    last_five = entries[-5:]

    for i in range(len(last_five) - 1):
        sm_curr = last_five[i].get("updated_self_model", {})
        sm_next = last_five[i + 1].get("updated_self_model", {})

        # Check spice_preference
        sp_curr = sm_curr.get("spice_preference", 0.5)
        sp_next = sm_next.get("spice_preference", 0.5)
        if abs(sp_next - sp_curr) >= 0.001:
            return False

        # Check health_bias
        hb_curr = sm_curr.get("health_bias", 0.5)
        hb_next = sm_next.get("health_bias", 0.5)
        if abs(hb_next - hb_curr) >= 0.001:
            return False

        # Check all cuisine_affinity values
        ca_curr = sm_curr.get("cuisine_affinity", {})
        ca_next = sm_next.get("cuisine_affinity", {})

        # Union of all cuisine keys across both entries
        all_cuisines = set(ca_curr.keys()) | set(ca_next.keys())
        for cuisine in all_cuisines:
            val_curr = ca_curr.get(cuisine, 0.0)
            val_next = ca_next.get(cuisine, 0.0)
            if abs(val_next - val_curr) >= 0.001:
                return False

    return True


# ── Recipe Optimization Pipeline ──────────────────────────────────────────────
# Mirrors the culinary agent's approach: suggest additions, improve recipes,
# score them, and pick the best one.


INGREDIENT_SUGGESTION_PROMPT = """
  Return a JSON object with this structure:

  {
  "recipe_name": string,
  "kitchen_state_additions": [
    {"ingredient": string, "reason": string}
  ],
  "external_additions": [
    {"ingredient": string, "reason": string}
  ]
  }

  Rules:
  - Recommend exactly 6 ingredients from the kitchen state
  - Recommend exactly 6 ingredients NOT in the kitchen state
  - Include odorant information in reason if possible
  - Do not output anything except JSON
"""

RECIPE_IMPROVEMENT_PROMPT = """
Create an improved version of the following recipe.

Rules:
- Use ALL of the ingredients listed under "additions_from_kitchen".
- Keep the dish recognizable.
- Integrate the new ingredients naturally into the recipe.
- Produce a full recipe with:
  - ingredient list
  - cooking steps
- Give a couple sentences about the enhancements that you made

Return JSON in this format:

{
  "recipe_name": "string",
  "description": "string",
  "enhancements_description": "string",
  "ingredients_required": [
        {
          "name": "string",
          "quantity": number,
          "unit": "string",
          "preparation": "string"
        }
    ],
  "steps": [
        {
          "step_number": number,
          "instruction": "string"
        }
      ]
}
"""

EVAL_PROMPT = """
Score the following recipe for flavor quality.

Consider:
- balance of flavor
- culinary realism

Return a ONLY A SINGLE NUMBER between 0 and 100.
DO NOT RETURN ANY EXPLANATION
"""


def suggest_recipe_additions(
    candidates: RecipeCandidates,
    temperature: float = 0.0,
) -> list:
    """For each recipe, use RAG context to suggest kitchen and external additions.

    Mirrors the culinary agent notebook's ingredient suggestion step.
    Returns a list of RecipeIngredientSuggestions, one per recipe.
    """
    all_suggestions = []

    for recipe in candidates.recipes:
        prominent_ingredients = set(
            ing_name.lower().strip()
            for combo in recipe.flavor_profile
            for ing_name in combo.ingredients
        )

        odorant_results = pairing_search(
            f"Which odorants do these ingredients have {prominent_ingredients}"
        )
        misc_results = pairing_search(
            f"What combinations are good with these ingredients {prominent_ingredients}"
        )

        prompt = f"""
      {INGREDIENT_SUGGESTION_PROMPT}

      Dish name: {recipe.recipe_name}

      Dish ingredients:
      {prominent_ingredients}

      Odorant context:
      {odorant_results}

      Additional pairing context:
      {misc_results}

      Kitchen state:
      {KITCHEN_STATE}
    """

        response = _get_backend().generate(
            prompt=prompt,
            system_prompt="",
            temperature=temperature,
            max_tokens=800,
        )

        text = response
        try:
            text = clean_llm_json(text)
            data = json.loads(text)
            suggestion = RecipeIngredientSuggestions.model_validate(data)
            all_suggestions.append(suggestion)
        except (json.JSONDecodeError, Exception):
            # Fallback: empty suggestions so the pipeline can continue
            all_suggestions.append(
                RecipeIngredientSuggestions(
                    recipe_name=recipe.recipe_name,
                    kitchen_state_additions=[],
                    external_additions=[],
                )
            )

    return all_suggestions


def improve_recipes(
    candidates: RecipeCandidates,
    all_suggestions: list,
    temperature: float = 0.3,
) -> list:
    """Create enhanced versions of each recipe using the suggested additions.

    Mirrors the culinary agent notebook's recipe improvement step.
    Returns a list of RecipeEnhanced objects.
    """
    improved = []

    for recipe, additions in zip(candidates.recipes, all_suggestions):
        added_ingredients = [
            a.ingredient for a in additions.kitchen_state_additions
        ]

        prompt = f"""
      {RECIPE_IMPROVEMENT_PROMPT}

      Original recipe name:
      {recipe.recipe_name}

      Original ingredients:
      {recipe.ingredients_required}

      Ingredients to add:
      {added_ingredients}
    """

        response = _get_backend().generate(
            prompt=prompt,
            system_prompt="",
            temperature=temperature,
        )

        text = response
        try:
            text = clean_llm_json(text)
            data = json.loads(text)
            enhanced = RecipeEnhanced.model_validate(data)
            improved.append(enhanced)
        except (json.JSONDecodeError, Exception):
            # Fallback: wrap the original recipe as an enhanced recipe
            improved.append(
                RecipeEnhanced(
                    recipe_name=recipe.recipe_name,
                    description=recipe.description,
                    enhancements_description="No enhancements applied.",
                    ingredients_required=recipe.ingredients_required,
                    steps=recipe.steps,
                )
            )

    return improved


def _odorant_score(ingredients: list, food_to_odorants: dict) -> float:
    """Score based on shared odorant overlap between ingredient names.

    Mirrors the culinary agent notebook's odorant_score function.
    """
    odorants = []
    for ing in ingredients:
        name = ing.name.lower() if hasattr(ing, "name") else str(ing).lower()
        for o in _fuzzy_odorant_lookup(name, food_to_odorants):
            if isinstance(o, dict):
                odorants.append(o["name"])

    unique = set(odorants)
    overlap = len(odorants) - len(unique)
    return overlap / max(len(unique), 1)


def _rag_score(ingredient1: str, ingredient2: str) -> float:
    """Score based on co-occurrence in cooking literature via RAG search.

    Mirrors the culinary agent notebook's rag_score function.
    """
    ingredient1 = ingredient1.lower()
    ingredient2 = ingredient2.lower()

    query = f"flavor pairing between {ingredient1} and {ingredient2}"
    results = recipe_search(query)

    count = 0
    for result in results:
        if "text" in result:
            result_text_lower = result["text"].lower()
            if ingredient1 in result_text_lower and ingredient2 in result_text_lower:
                count += 1

    return count


def _llm_score(ingredients: list, steps: list) -> float:
    """Ask LLM to evaluate dish for flavor quality on a 0-1 scale.

    Mirrors the culinary agent notebook's llm_score function.
    """
    prompt = f"""
  {EVAL_PROMPT}

  dish ingredients:
  {ingredients}

  dish steps:
  {steps}
  """

    response = _get_backend().generate(
        prompt=prompt,
        system_prompt="",
        temperature=0.3,
    )

    score_text = response
    try:
        return int(score_text) / 100
    except ValueError:
        return 0.5


def optimize_and_select_best(
    candidates: RecipeCandidates,
    food_to_odorants: dict,
    temperature: float = 0.8,
) -> tuple:
    """Run the full optimization pipeline and return the best recipe.

    Pipeline (mirrors culinary agent notebook):
    1. Suggest ingredient additions for each candidate
    2. Create improved/enhanced versions of each recipe
    3. Score each improved recipe: rag_score*30 + odorant_score*30 + llm_score*40
    4. Return (best_enhanced_recipe, all_suggestions, best_index)

    The returned RecipeEnhanced is converted back to a Recipe-compatible object
    for downstream use in the learning loop.
    """
    # Step 1: Suggest additions
    all_suggestions = suggest_recipe_additions(candidates, temperature=0.0)

    # Step 2: Improve recipes
    improved_recipes = improve_recipes(candidates, all_suggestions, temperature=0.3)

    # Step 3: Score each improved recipe
    scores = []
    for enhanced in improved_recipes:
        ingredients = enhanced.ingredients_required
        num_pairs = 0
        total_rag = 0.0
        total_odorant = 0.0

        ing_names = [ing.name.lower() for ing in ingredients]
        for i in range(len(ing_names)):
            for j in range(i + 1, len(ing_names)):
                num_pairs += 1
                total_rag += _rag_score(ing_names[i], ing_names[j])
                total_odorant += _odorant_score(
                    [ingredients[i], ingredients[j]], food_to_odorants
                )

        avg_rag = total_rag / max(num_pairs, 1)
        avg_odorant = total_odorant / max(num_pairs, 1)
        current_llm = _llm_score(ingredients, enhanced.steps)

        overall = avg_rag * 30 + avg_odorant * 30 + current_llm * 40
        scores.append(overall)

    # Step 4: Pick the best
    best_index = scores.index(max(scores)) if scores else 0
    best_enhanced = improved_recipes[best_index] if improved_recipes else None

    return best_enhanced, all_suggestions, best_index


def _enhanced_to_recipe(enhanced: RecipeEnhanced, original: Recipe) -> Recipe:
    """Convert a RecipeEnhanced back to a Recipe for the learning loop.

    Preserves the original's fit_to_intent and flavor_profile since the
    enhanced recipe doesn't carry those fields.
    """
    return Recipe(
        recipe_name=enhanced.recipe_name,
        description=enhanced.description,
        ingredients_required=enhanced.ingredients_required,
        fit_to_intent=original.fit_to_intent,
        flavor_profile=original.flavor_profile,
        steps=enhanced.steps,
    )


# ── Main Orchestrator ─────────────────────────────────────────────────────────


def run_agent(
    user_input: str,
    mode: str = "full_model",
    learning_rate: float = 0.05,
    world_model_path: str = "world_model.json",
    self_model_path: str = "self_model.json",
    log_path: str = "run_log.jsonl",
    temperature: float = 0.8,
) -> dict:
    """Execute the full training loop and return a structured run record.

    Steps:
    1. Validate learning_rate
    2. Validate mode
    3. Parse intent from user input
    4. Determine bias strings based on mode
    5. Generate candidate recipes
    6. Validate and fix recipes
    7. Optimize recipes (suggest additions, improve, score) and pick the best
    8. Predict taste via WorldModel
    9. Evaluate actual taste via heuristic
    10. Compute prediction error
    11. Update models based on mode
    12. Detect convergence
    13. Log the run
    14. Return run record

    Requirements: 11.1, 11.2, 11.3, 5.4, 9.2, 9.3, 9.4, 9.5, 13.4
    """
    # ── 1. Validate learning_rate ────────────────────────────────────────
    if not (0.01 <= learning_rate <= 0.1):
        raise ValueError(
            f"learning_rate must be between 0.01 and 0.1 inclusive, got {learning_rate}"
        )

    # ── 2. Validate mode ─────────────────────────────────────────────────
    select_mode(mode)

    # ── 3. Parse intent ──────────────────────────────────────────────────
    intent = parse_intent(user_input, schema, temperature=temperature)
    if intent is None:
        intent = {
            "hard_constraints": {},
            "soft_objectives": {},
            "preferences": {},
        }

    # ── Load models ──────────────────────────────────────────────────────
    world_model = WorldModel(world_model_path)
    self_model = SelfModel(self_model_path)

    # ── 4. Determine bias strings based on mode ──────────────────────────
    wm_bias = None
    sm_bias = None

    if mode in ("world_model_only", "full_model"):
        top_ingredients = world_model.get_top_ingredients()
        if top_ingredients:
            wm_bias = ", ".join(
                f"{name} ({weight:.2f})" for name, weight in top_ingredients
            )

    if mode in ("self_model_only", "full_model"):
        sm_bias = self_model.to_prompt_context()

    # ── 5. Generate candidate recipes ────────────────────────────────────
    candidates = generate_recipes(
        intent, user_input, wm_bias=wm_bias, sm_bias=sm_bias,
        temperature=temperature,
    )

    if candidates is None:
        # Retry up to 2 more times (3 total attempts)
        for _ in range(2):
            candidates = generate_recipes(
                intent, user_input, wm_bias=wm_bias, sm_bias=sm_bias,
                temperature=temperature,
            )
            if candidates is not None:
                break

    if candidates is None or not candidates.recipes:
        # Log failure and return partial result
        run_number = 1
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                run_number = sum(1 for line in f if line.strip()) + 1

        error_record = {
            "run_number": run_number,
            "mode": mode,
            "recipe": {},
            "predicted_taste": 0.0,
            "actual_taste": 0.0,
            "error": 0.0,
            "abs_error": 0.0,
            "learning_rate": learning_rate,
            "updated_world_model": {"ingredient_weights": world_model.ingredient_weights},
            "updated_self_model": {
                "spice_preference": self_model.spice_preference,
                "health_bias": self_model.health_bias,
                "cuisine_affinity": self_model.cuisine_affinity,
            },
            "convergence_flag": False,
            "error_info": "Recipe generation failed after 3 attempts",
        }
        log_run(log_path, error_record)
        return error_record

    # ── 6. Validate and fix recipes ──────────────────────────────────────
    candidates = validate_and_fix_recipes(
        candidates, intent, user_input, wm_bias=wm_bias, sm_bias=sm_bias,
        temperature=temperature,
    )

    # ── 7. Optimize recipes and pick the best ─────────────────────────
    with open(FOOD_TO_ODORANT_PATH, "r") as f:
        food_to_odorants = json.load(f)

    best_enhanced, all_suggestions, best_index = optimize_and_select_best(
        candidates, food_to_odorants, temperature=temperature,
    )

    # Convert the enhanced recipe back to a Recipe for the learning loop,
    # preserving fit_to_intent and flavor_profile from the original candidate.
    original_recipe = candidates.recipes[best_index]
    recipe = _enhanced_to_recipe(best_enhanced, original_recipe)

    # ── 7b. Preference-based re-ranking (local backend only) ─────────
    # When using the local model with a trained preference head, re-rank
    # candidates using the learned preference scores. This is a structural
    # preference mechanism — the model's weights determine the choice,
    # not prompt text.
    backend = _get_backend()
    if backend.has_preference_head and mode != "baseline":
        scored = []
        for i, r in enumerate(candidates.recipes):
            recipe_text = json.dumps(r.model_dump(), indent=2)
            pref_score = backend.score_recipe(recipe_text)
            scored.append((i, pref_score))
        scored.sort(key=lambda x: x[1], reverse=True)
        best_pref_idx = scored[0][0]
        # Use preference-selected recipe instead of optimization-selected
        recipe = candidates.recipes[best_pref_idx]

    # ── 8. Predict taste ─────────────────────────────────────────────────
    predicted_taste = predict_taste(world_model, recipe)

    # ── 9. Evaluate actual taste ─────────────────────────────────────────
    actual_taste = evaluate_taste(recipe, intent, food_to_odorants)

    # ── 10. Compute error ────────────────────────────────────────────────
    error = actual_taste - predicted_taste

    # ── 11. Update models based on mode ──────────────────────────────────
    if mode == "world_model_only":
        world_model.update(recipe, error, learning_rate)
        world_model.save()
    elif mode == "self_model_only":
        self_model.update(recipe, intent, learning_rate)
        self_model.save()
    elif mode == "full_model":
        world_model.update(recipe, error, learning_rate)
        world_model.save()
        self_model.update(recipe, intent, learning_rate)
        self_model.save()
    # baseline: skip all updates

    # ── 11b. Train LoRA adapters (local backend only) ────────────────────
    # When using the local model, the taste heuristic score trains the
    # preference head and LoRA adapters. This encodes preferences in the
    # model weights themselves, not in external data structures.
    adapter_loss = None
    if backend.has_training and mode != "baseline":
        recipe_text = json.dumps(recipe.model_dump(), indent=2)
        adapter_loss = backend.train_step(
            recipe_text=recipe_text,
            taste_score=actual_taste,
            lr=learning_rate * 0.01,  # adapter LR is smaller than model LR
        )
        backend.save_adapters()

    # ── 12. Determine run number ─────────────────────────────────────────
    run_number = 1
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            run_number = sum(1 for line in f if line.strip()) + 1

    # ── 13. Detect convergence ───────────────────────────────────────────
    convergence_flag = detect_convergence(log_path)

    # ── 14. Build run record ─────────────────────────────────────────────
    run_record = {
        "run_number": run_number,
        "mode": mode,
        "recipe": recipe.model_dump(),
        "predicted_taste": predicted_taste,
        "actual_taste": actual_taste,
        "error": error,
        "abs_error": abs(error),
        "learning_rate": learning_rate,
        "updated_world_model": {"ingredient_weights": world_model.ingredient_weights},
        "updated_self_model": {
            "spice_preference": self_model.spice_preference,
            "health_bias": self_model.health_bias,
            "cuisine_affinity": self_model.cuisine_affinity,
        },
        "convergence_flag": convergence_flag,
        "backend": BACKEND_TYPE,
        "adapter_loss": adapter_loss,
    }

    # ── 15. Log the run ──────────────────────────────────────────────────
    log_run(log_path, run_record)

    return run_record
