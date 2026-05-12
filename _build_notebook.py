#!/usr/bin/env python3
"""Build the self-contained baseline_ingredient_experiment.ipynb."""
import json

# ── Cell 1: Install dependencies ──
cell1_source = [
    "!pip install -q transformers torch peft bitsandbytes accelerate matplotlib\n"
]

# ── Cell 2: Configuration ──
cell2_source = [
    "# \u2500\u2500 Configuration \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n",
    'MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"\n',
    'BANNED_INGREDIENTS = ["garlic", "butter", "heavy cream", "soy sauce", "sugar"]\n',
    "NUM_EVAL_PROMPTS = 30\n",
    "TEMPERATURE = 0.8\n",
    "MAX_TOKENS = 1024\n",
    'PROMPT_TEMPLATE = "Write a recipe for {dish}. Include a title, an Ingredients: section listing all ingredients, and step-by-step cooking instructions."'
]

# ── Cell 3: HF token setup ──
cell3_source = [
    "import os\n",
    "try:\n",
    "    from google.colab import userdata\n",
    '    HF_TOKEN = userdata.get("HF_TOKEN")\n',
    "except Exception:\n",
    '    HF_TOKEN = os.environ.get("HF_TOKEN", "")\n',
    'assert HF_TOKEN, "Set HF_TOKEN via Colab secrets or environment variable."'
]

# ── Cell 4: Model loading ──
cell4_source = [
    "import torch\n",
    "from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig\n",
    "\n",
    "bnb_config = BitsAndBytesConfig(\n",
    "    load_in_4bit=True,\n",
    "    bnb_4bit_compute_dtype=torch.bfloat16,\n",
    ")\n",
    "tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)\n",
    "if tokenizer.pad_token is None:\n",
    "    tokenizer.pad_token = tokenizer.eos_token\n",
    "\n",
    "model = AutoModelForCausalLM.from_pretrained(\n",
    "    MODEL_ID,\n",
    "    token=HF_TOKEN,\n",
    "    quantization_config=bnb_config,\n",
    '    device_map="auto",\n',
    ")\n",
    'print(f"Model loaded: {MODEL_ID}")'
]

# ── Cell 5: DISHES list (all 1000 entries) ──
# Read from dish_list.py to get the exact list
import sys
sys.path.insert(0, ".")
from dish_list import DISHES as _DISHES

cell5_lines = []
cell5_lines.append("DISHES: list[str] = [\n")

# Group dishes by cuisine sections (read from the source file)
with open("dish_list.py", "r", encoding="utf-8") as f:
    src = f.read()

# Parse the source to preserve comments and structure
in_list = False
for line in src.splitlines():
    stripped = line.strip()
    if stripped.startswith("DISHES"):
        in_list = True
        continue
    if not in_list:
        continue
    if stripped == "]":
        break
    # It's either a comment line or a dish entry or blank
    if stripped == "":
        continue
    if stripped.startswith("#"):
        cell5_lines.append(f"    {stripped}\n")
    else:
        cell5_lines.append(f"    {stripped}\n")

cell5_lines.append("]\n")
cell5_lines.append(f'print(f"DISHES loaded: {{len(DISHES)}} entries")')

cell5_source = cell5_lines

# ── Cell 6: Select eval dishes ──
cell6_source = [
    "eval_dishes = DISHES[300:330]\n",
    'print(f"Eval dishes: {len(eval_dishes)} dishes selected (indices 300-329)")\n',
    "for i, d in enumerate(eval_dishes):\n",
    '    print(f"  {300+i}: {d}")'
]

# ── Cell 7: Helper functions ──
cell7_source = [
    "def generate_recipe(dish):\n",
    '    """Generate a single recipe via one LLM call. Returns plain text."""\n',
    "    prompt = PROMPT_TEMPLATE.format(dish=dish)\n",
    '    messages = [{"role": "user", "content": prompt}]\n',
    '    encoded = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True).to(model.device)\n',
    '    input_ids = encoded["input_ids"]\n',
    "    prompt_len = input_ids.shape[1]\n",
    "    with torch.no_grad():\n",
    "        outputs = model.generate(\n",
    "            **encoded,\n",
    "            max_new_tokens=MAX_TOKENS,\n",
    "            temperature=TEMPERATURE,\n",
    "            do_sample=True,\n",
    "            pad_token_id=tokenizer.eos_token_id,\n",
    "        )\n",
    "    generated = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)\n",
    "    del encoded, outputs\n",
    "    torch.cuda.empty_cache()\n",
    "    return generated\n",
    "\n",
    "\n",
    "def detect_banned_ingredients(recipe_text: str, banned: list[str]) -> list[str]:\n",
    '    """Case-insensitive substring match against full recipe text."""\n',
    "    text_lower = recipe_text.lower()\n",
    "    return [ing for ing in banned if ing.lower() in text_lower]\n",
    "\n",
    "\n",
    'print("Helper functions defined: generate_recipe, detect_banned_ingredients")'
]

# ── Cell 8: Main eval loop ──
cell8_source = [
    "results = []\n",
    "for i, dish in enumerate(eval_dishes):\n",
    '    print(f"[{i+1}/{NUM_EVAL_PROMPTS}] Generating recipe for: {dish}")\n',
    "    recipe_text = generate_recipe(dish)\n",
    "    banned_found = detect_banned_ingredients(recipe_text, BANNED_INGREDIENTS)\n",
    "    results.append({\n",
    '        "dish": dish,\n',
    '        "recipe_text": recipe_text,\n',
    '        "banned_found": banned_found,\n',
    '        "contains_banned": len(banned_found) > 0,\n',
    "    })\n",
    '    print(f"  Banned found: {banned_found if banned_found else \'None\'}")\n',
    'print(f"\\nGeneration complete. {len(results)} recipes generated.")'
]

# ── Cell 9: Summary statistics ──
cell9_source = [
    "per_ingredient_count = {ing: 0 for ing in BANNED_INGREDIENTS}\n",
    "recipes_with_banned = 0\n",
    "for r in results:\n",
    '    if r["banned_found"]:\n',
    "        recipes_with_banned += 1\n",
    '    for ing in r["banned_found"]:\n',
    "        if ing in per_ingredient_count:\n",
    "            per_ingredient_count[ing] += 1\n",
    "\n",
    "recipes_clean = len(results) - recipes_with_banned\n",
    "summary = {\n",
    '    "total_recipes": len(results),\n',
    '    "recipes_with_banned": recipes_with_banned,\n',
    '    "recipes_clean": recipes_clean,\n',
    '    "per_ingredient_count": per_ingredient_count,\n',
    "}\n",
    'print(f"Recipes with banned ingredients: {recipes_with_banned}/{len(results)}")\n',
    'print(f"Clean recipes: {recipes_clean}/{len(results)}")\n',
    'print(f"Per-ingredient counts: {per_ingredient_count}")'
]

# ── Cell 10: Save results JSON ──
cell10_source = [
    "import json\n",
    "from datetime import datetime\n",
    "\n",
    "output = {\n",
    '    "metadata": {\n',
    '        "notebook": "baseline",\n',
    '        "model_id": MODEL_ID,\n',
    '        "banned_ingredients": BANNED_INGREDIENTS,\n',
    '        "num_eval_prompts": NUM_EVAL_PROMPTS,\n',
    '        "timestamp": datetime.now().isoformat(),\n',
    "    },\n",
    '    "recipes": results,\n',
    '    "summary": summary,\n',
    "}\n",
    "\n",
    'with open("baseline_results.json", "w") as f:\n',
    "    json.dump(output, f, indent=2)\n",
    'print("Results saved to baseline_results.json")'
]

# ── Cell 11: Bar chart ──
cell11_source = [
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "\n",
    "percentages = [per_ingredient_count[ing] / len(results) * 100 for ing in BANNED_INGREDIENTS]\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(9, 5))\n",
    "x = np.arange(len(BANNED_INGREDIENTS))\n",
    'bars = ax.bar(x, percentages, color="#4C72B0", width=0.5)\n',
    'ax.set_xlabel("Banned Ingredient")\n',
    'ax.set_ylabel("% of Recipes Containing Ingredient")\n',
    'ax.set_title("Baseline: Banned Ingredient Frequency")\n',
    "ax.set_xticks(x)\n",
    'ax.set_xticklabels(BANNED_INGREDIENTS, rotation=45, ha="right")\n',
    "ax.set_ylim(0, 105)\n",
    'ax.bar_label(bars, fmt="%.0f%%", padding=3)\n',
    "fig.tight_layout()\n",
    "plt.show()"
]

# ── Assemble notebook ──
def make_code_cell(source_lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines,
    }

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": [
        make_code_cell(cell1_source),
        make_code_cell(cell2_source),
        make_code_cell(cell3_source),
        make_code_cell(cell4_source),
        make_code_cell(cell5_source),
        make_code_cell(cell6_source),
        make_code_cell(cell7_source),
        make_code_cell(cell8_source),
        make_code_cell(cell9_source),
        make_code_cell(cell10_source),
        make_code_cell(cell11_source),
    ]
}

with open("baseline_ingredient_experiment.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"Notebook written with {len(notebook['cells'])} cells.")
# Verify cell 5 has all 1000 dishes
c5 = "".join(cell5_source)
count = c5.count('"') // 2  # rough count of quoted strings
print(f"Cell 5 source lines: {len(cell5_source)}")
print(f"Total DISHES entries from dish_list.py: {len(_DISHES)}")
