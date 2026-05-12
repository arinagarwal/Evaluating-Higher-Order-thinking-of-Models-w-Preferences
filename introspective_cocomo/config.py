"""
Configuration for Modified CoCoMo with Preference Memory Self-Distillation.

Extends the base cocomo config with preference memory parameters.
"""
import sys
import os
import importlib.util

# Load cocomo/config.py directly by filepath to avoid circular import
# (this file is also named config.py)
_cocomo_config_path = os.path.join(os.path.dirname(__file__), '..', 'cocomo', 'config.py')
_spec = importlib.util.spec_from_file_location("cocomo_config", _cocomo_config_path)
_cocomo_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cocomo_config)

MODEL_NAME = _cocomo_config.MODEL_NAME
BANNED_INGREDIENTS = _cocomo_config.BANNED_INGREDIENTS
SUBSTITUTIONS = _cocomo_config.SUBSTITUTIONS
CUISINE_RISK_MAP = _cocomo_config.CUISINE_RISK_MAP
CUISINE_RANGES = _cocomo_config.CUISINE_RANGES
GENERATION_CONFIG = _cocomo_config.GENERATION_CONFIG
DRAFT_GENERATION_CONFIG = _cocomo_config.DRAFT_GENERATION_CONFIG
MFQ_ESCALATION_THRESHOLD = _cocomo_config.MFQ_ESCALATION_THRESHOLD
LORA_CONFIG = _cocomo_config.LORA_CONFIG
GRPO_TRAINING_CONFIG = _cocomo_config.GRPO_TRAINING_CONFIG
REWARD_WEIGHTS = _cocomo_config.REWARD_WEIGHTS
TRAIN_DISHES = _cocomo_config.TRAIN_DISHES
EVAL_DISHES = _cocomo_config.EVAL_DISHES
get_bnb_compute_dtype = _cocomo_config.get_bnb_compute_dtype

# Load final/dishes.py
_dishes_path = os.path.join(os.path.dirname(__file__), '..', 'final', 'dishes.py')
_spec2 = importlib.util.spec_from_file_location("dishes_module", _dishes_path)
_dishes_module = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_dishes_module)
DISHES = _dishes_module.DISHES

# ── Preference Memory Config ─────────────────────────────────────────────────

MEMORY_EXTRACTION_PROMPT = (
    "Based on your experience generating recipes, reflect on your ingredient preferences.\n"
    "Which ingredients do you tend to avoid? For each one, explain WHY you avoid it "
    "and what you prefer to use instead.\n"
    "Also describe any general cooking principles or preferences you've developed.\n"
    "Be specific and honest about your tendencies."
)

MEMORY_REFINEMENT_PROMPT = (
    "Here is your previous self-description of your cooking preferences:\n"
    "---\n{previous_memory}\n---\n\n"
    "Based on your recent recipe generation experience, update this self-description.\n"
    "Are there preferences you missed? Any you stated incorrectly? "
    "Any new patterns you've noticed?\n"
    "Write an updated, complete description of your ingredient preferences and cooking philosophy."
)

MEMORY_INJECTION_TEMPLATE = (
    "Your cooking preferences (for your reference when answering):\n"
    "---\n{preference_memory}\n---\n\n"
)

NUM_DISTILLATION_EPOCHS = 3
DISHES_PER_DISTILLATION_ROUND = 50
MEMORY_MAX_TOKENS = 300

GRPO_TRAINING_CONFIG_MODIFIED = {
    **GRPO_TRAINING_CONFIG,
    "output_dir": "modified_cocomo/grpo_weights",
    "num_train_epochs": 2,
}

# ── Introspection Head Config ─────────────────────────────────────────────────

INTROSPECTION_CONFIG = {
    "lr": 1e-3,
    "save_dir": "modified_cocomo/introspection_weights",
    "train_batches_per_epoch": 50,
    "warmup_dishes": 100,
}
