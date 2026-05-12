import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'final'))
from dishes import DISHES

import torch


def get_bnb_compute_dtype():
    """
    Prefer bfloat16 (works on L4/g6e and newer GPUs).
    Falls back to float16 if bfloat16 isn't supported on the current device.
    """
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

BANNED_INGREDIENTS = ["garlic", "butter", "heavy cream", "soy sauce", "sugar"]

SUBSTITUTIONS = {
    "garlic": "asafoetida (hing)",
    "butter": "olive oil",
    "heavy cream": "coconut cream",
    "soy sauce": "coconut aminos",
    "sugar": "maple syrup",
}

# How many of the 5 banned ingredients are endemic to each cuisine.
# Score = fraction of banned ingredients that commonly appear (0.0–1.0).
CUISINE_RISK_MAP = {
    "Italian":       0.9,   # garlic, butter, heavy cream all core
    "French":        0.9,   # butter, heavy cream, garlic dominant
    "American":      0.7,
    "British":       0.7,
    "German":        0.6,
    "Spanish":       0.6,
    "Greek":         0.6,
    "Brazilian":     0.5,
    "Mexican":       0.5,
    "Chinese":       0.5,   # soy sauce, sugar common
    "Japanese":      0.5,   # soy sauce, sugar common
    "Korean":        0.5,
    "Filipino":      0.4,
    "Caribbean":     0.4,
    "Indonesian":    0.4,
    "Peruvian":      0.4,
    "Turkish":       0.4,
    "Lebanese":      0.4,
    "Moroccan":      0.3,
    "Indian":        0.6,   # garlic and ghee (butter) are core to most dishes
    "Scandinavian":  0.3,
    "Polish":        0.3,
    "Thai":          0.2,
    "Vietnamese":    0.2,
    "Ethiopian":     0.1,
    "Unknown":       0.5,
}

# Dish index ranges → cuisine (mirrors dishes.py ordering)
CUISINE_RANGES = [
    (0,    50,  "Italian"),
    (50,   100, "Japanese"),
    (100,  150, "Mexican"),
    (150,  200, "Indian"),
    (200,  250, "Thai"),
    (250,  300, "Chinese"),
    (300,  350, "French"),
    (350,  400, "Korean"),
    (400,  435, "Ethiopian"),
    (435,  470, "Moroccan"),
    (470,  505, "Peruvian"),
    (505,  545, "Turkish"),
    (545,  585, "Vietnamese"),
    (585,  625, "Greek"),
    (625,  665, "Spanish"),
    (665,  700, "Lebanese"),
    (700,  735, "Brazilian"),
    (735,  770, "British"),
    (770,  820, "American"),
    (820,  860, "German"),
    (860,  890, "Indonesian"),
    (890,  920, "Filipino"),
    (920,  950, "Caribbean"),
    (950,  975, "Scandinavian"),
    (975, 1000, "Polish"),
]

TRAIN_DISHES = DISHES[0:900]
EVAL_DISHES  = DISHES[900:1000]

# Generation — kept identical to final/ for fair comparison
GENERATION_CONFIG = {
    "max_new_tokens": 512,
    "temperature": 0.8,
    "top_p": 0.9,
    "top_k": 50,
    "do_sample": True,
}

# Fast unconscious draft — fewer tokens than conscious generation but still sampled.
# do_sample=False (greedy) causes Llama to loop on whitespace; must use sampling.
DRAFT_GENERATION_CONFIG = {
    "max_new_tokens": 256,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
}

# MFQ escalation threshold: risk scores above this go to Consciousness
MFQ_ESCALATION_THRESHOLD = 0.3

# RL training
LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj"],
}

GRPO_TRAINING_CONFIG = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "bf16": True,
    "max_seq_length": 512,
    "save_steps": 25,
    "output_dir": "cocomo/grpo_weights",
}

# Reward weights
REWARD_WEIGHTS = {
    "constraint":            -2.0,   # per banned ingredient found
    "culinary_coherence":     1.0,   # max additive
    "substitution_validity":  0.5,   # max additive
    "novelty_bonus":          0.1,   # per novel substitution beyond fixed 5
}
