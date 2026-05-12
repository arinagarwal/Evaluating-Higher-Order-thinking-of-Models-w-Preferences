from __future__ import annotations

import re
import torch
from dataclasses import dataclass, field
from config import BANNED_INGREDIENTS, SUBSTITUTIONS, GENERATION_CONFIG


@dataclass
class Schema:
    """
    The consciousness module's working memory for one recipe task.
    Maintained across the CRIT → Exploratory → Generation stages.
    """
    dish: str
    cuisine: str
    constraints: list[str]
    risk_score: float
    past_substitutions: list[dict] = field(default_factory=list)
    validated_substitutions: dict = field(default_factory=dict)
    rival_reasons: list[str] = field(default_factory=list)
    repair_violations: list[str] = field(default_factory=list)


class CriticalThinking:
    """
    CRIT implementation from the lecture slides.
    Validates whether a proposed substitution is culinarily appropriate
    for the specific dish + cuisine combination, not just mechanically legal.

    Steps: identify claim → supporting reasons → rival reasons → weighted score.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def _generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        generated = output[0][input_ids.shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def _extract_score(self, text: str) -> float:
        """Parse a 1–10 score from model output."""
        matches = re.findall(r'\b([1-9]|10)\b', text)
        if matches:
            return float(matches[0]) / 10.0
        return 0.5  # neutral default

    def validate_substitution(
        self,
        dish: str,
        cuisine: str,
        original: str,
        substitute: str,
    ) -> tuple[float, list[str], list[str]]:
        """
        Returns (score 0–1, supporting_reasons, rival_reasons).
        Follows CRIT steps #1–#6 from the lecture.
        """
        # #1 Claim
        claim = f"'{substitute}' is a valid culinary substitution for '{original}' in {cuisine} {dish}."

        # #2–#3 Supporting reasons
        support_prompt = (
            f"Claim: {claim}\n"
            "List 2 reasons why this substitution works culinarily. Be concise."
        )
        support_text = self._generate(support_prompt)

        # #4–#5 Rival reasons
        rival_prompt = (
            f"Claim: {claim}\n"
            "List 2 reasons why this substitution might NOT work culinarily. Be concise."
        )
        rival_text = self._generate(rival_prompt)

        # #6 Weighted score
        score_prompt = (
            f"Claim: {claim}\n"
            f"Supporting: {support_text}\n"
            f"Against: {rival_text}\n"
            "On a scale of 1 to 10, how valid is this substitution for this specific dish and cuisine? "
            "Reply with just the number."
        )
        score_text = self._generate(score_prompt, max_new_tokens=16)
        score = self._extract_score(score_text)

        return score, [support_text], [rival_text]

    def validate_all(self, schema: Schema, draft: str = "", skip: set | None = None) -> dict:
        """
        Validates substitutions only for banned ingredients that actually
        appear in the draft recipe. Skips ingredients not present — avoids
        15 LLM calls per dish when only 1-2 ingredients are relevant.
        `skip` allows the caller to exclude ingredients already covered by
        memory-sourced substitutions (architectural change 3).
        Returns {original: {substitute, validity_score}}.
        """
        lower_draft = draft.lower()
        skip = skip or set()
        validated = {}
        for ingredient in schema.constraints:
            if ingredient in skip:
                continue
            if draft and ingredient.lower() not in lower_draft:
                continue
            substitute = SUBSTITUTIONS.get(ingredient, None)
            if substitute is None:
                continue
            score, _, rival = self.validate_substitution(
                schema.dish, schema.cuisine, ingredient, substitute
            )
            schema.rival_reasons.extend(rival)
            validated[ingredient] = {"substitute": substitute, "validity_score": score}
        return validated


class ExploratoryThinking:
    """
    Proposes novel substitutions beyond the fixed SUBSTITUTIONS table.
    Uses the maieutics (Socratic) strategy: asks the model to discover
    alternatives through guided questioning rather than direct instruction.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def _generate(self, prompt: str, max_new_tokens: int = 200) -> str:
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
            )
        generated = output[0][input_ids.shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def propose_substitutions(self, schema: Schema) -> dict:
        """
        Maieutics: model discovers cuisine-appropriate substitutions through
        guided questioning, rather than being told the fixed substitution map.
        Returns {original_ingredient: proposed_substitute}.
        """
        banned_str = ", ".join(schema.constraints)
        prompt = (
            f"You are helping create a recipe for {schema.dish} ({schema.cuisine} cuisine).\n"
            f"The following ingredients are not allowed: {banned_str}.\n"
            f"For each banned ingredient that might appear in this dish, suggest ONE "
            f"culturally appropriate substitute that fits {schema.cuisine} cuisine specifically.\n"
            f"Format: 'ingredient → substitute'. Only list ingredients relevant to this dish."
        )
        response = self._generate(prompt)

        proposed = {}
        arrow_pat = re.compile(r'→|->')
        for line in response.split('\n'):
            if not arrow_pat.search(line):
                continue
            parts = arrow_pat.split(line, maxsplit=1)
            if len(parts) == 2:
                orig = parts[0].strip().lower().strip('- •*')
                sub = parts[1].strip()
                for banned in schema.constraints:
                    if banned in orig:
                        proposed[banned] = sub
                        break
        return proposed


class PromptTemplateGenerator:
    """
    Builds a dynamic, schema-aware generation prompt rather than the fixed
    template used in final/. Incorporates:
    - Validated substitutions with their culinary rationale
    - Cuisine-specific framing
    - Explicit constraint reasoning (the model knows WHY, not just WHAT)
    """

    def build_prompt(self, schema: Schema, validated_substitutions: dict) -> str:
        sub_instructions = []
        for ingredient, info in validated_substitutions.items():
            sub = info["substitute"]
            score = info["validity_score"]
            source = info.get("source", "")
            if score >= 0.6:
                tag = "memory-proven" if source == "memory" else f"validity: {score:.1f}/1.0"
                sub_instructions.append(
                    f"- Instead of {ingredient}, use {sub} "
                    f"(culinarily appropriate for {schema.cuisine} cuisine, {tag})"
                )
            else:
                sub_instructions.append(
                    f"- Avoid {ingredient}; find the most appropriate substitute "
                    f"for authentic {schema.cuisine} cuisine"
                )

        subs_text = "\n".join(sub_instructions) if sub_instructions else \
            f"Avoid all of: {', '.join(schema.constraints)}"

        repair_section = ""
        if schema.repair_violations:
            repair_section = (
                f"\n\nWARNING — a previous draft still contained: "
                f"{', '.join(schema.repair_violations)}. "
                f"These are strictly banned. Do not use them in any form, "
                f"including derivatives or preparations that contain the banned ingredient."
            )

        prompt = (
            f"Write an authentic {schema.cuisine} recipe for {schema.dish}.\n\n"
            f"Ingredient constraints (dietary requirement — do not use these ingredients):\n"
            f"{subs_text}"
            f"{repair_section}\n\n"
            f"Include: a title, an Ingredients: section with quantities, "
            f"and a Instructions: section with numbered steps.\n"
            f"Ensure the recipe remains authentic to {schema.cuisine} cuisine "
            f"despite the substitutions."
        )
        return prompt


class ConsciousnessModule:
    """
    Single-threaded, deliberate attention for high-risk recipe tasks.
    Coordinates CriticalThinking, ExploratoryThinking, and PromptTemplateGenerator
    to produce a validated, culinarily coherent recipe.

    Analogous to CoCoMo's consciousness module: maintains task schema,
    reward-aware generation, and dynamic prompt construction.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.crit = CriticalThinking(model, tokenizer)
        self.explore = ExploratoryThinking(model, tokenizer)
        self.prompt_gen = PromptTemplateGenerator()

    def generate(self, schema_dict: dict) -> tuple[str, dict]:
        """
        Full conscious pipeline:
        1. Build Schema
        2. CRIT: validate fixed substitutions
        3. Explore: propose novel cuisine-appropriate alternatives for low-validity subs
        4. Merge substitutions (prefer higher-validity option)
        5. Build dynamic prompt
        6. Generate recipe

        Returns (recipe_text, metadata_dict).
        """
        schema = Schema(
            dish=schema_dict["dish"],
            cuisine=schema_dict["cuisine"],
            constraints=schema_dict["constraints"],
            risk_score=schema_dict["risk_score"],
            past_substitutions=schema_dict.get("past_substitutions", []),
            repair_violations=schema_dict.get("repair_violations", []),
        )

        # Extract memory-sourced substitutions from past_substitutions (change 3).
        # These skip CRIT — they are already proven for this cuisine×ingredient pair.
        memory_validated = {}
        for sub_dict in schema.past_substitutions:
            for ingredient, info in sub_dict.items():
                if info.get("source") == "memory" and ingredient in schema.constraints:
                    memory_validated[ingredient] = {
                        "substitute": info["substitute"],
                        "validity_score": info.get("validity_score", 0.8),
                        "source": "memory",
                    }

        # Step 2: CRIT validation — skip ingredients already covered by memory
        draft = schema_dict.get("draft", "")
        validated = self.crit.validate_all(schema, draft=draft, skip=set(memory_validated.keys()))
        validated.update(memory_validated)
        schema.validated_substitutions = validated

        # Step 3: Explore novel substitutions for low-validity ones
        low_validity = {k for k, v in validated.items() if v["validity_score"] < 0.6}
        if low_validity:
            novel = self.explore.propose_substitutions(schema)
            for ingredient, sub in novel.items():
                if ingredient in low_validity:
                    validated[ingredient] = {
                        "substitute": sub,
                        "validity_score": 0.7,  # explorer-proposed, treated as reasonable
                        "source": "exploratory",
                    }

        # Step 4: Mark fixed subs as source
        for k in validated:
            if "source" not in validated[k]:
                validated[k]["source"] = "fixed_table"

        # Step 5: Dynamic prompt
        prompt = self.prompt_gen.build_prompt(schema, validated)

        # Step 6: Generate
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                pad_token_id=self.tokenizer.eos_token_id,
                **GENERATION_CONFIG,
            )
        generated = output[0][input_ids.shape[1]:]
        recipe = self.tokenizer.decode(generated, skip_special_tokens=True)

        metadata = {
            "validated_substitutions": validated,
            "rival_reasons": schema.rival_reasons,
            "prompt_used": prompt,
        }
        return recipe, metadata
