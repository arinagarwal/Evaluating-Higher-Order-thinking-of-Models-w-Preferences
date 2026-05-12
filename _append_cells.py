"""Temporary script to append cells 14-20 to lora_ingredient_experiment.ipynb."""
import json

with open("lora_ingredient_experiment.ipynb", "r") as f:
    nb = json.load(f)

assert len(nb["cells"]) == 13, f"Expected 13 cells, got {len(nb['cells'])}"

new_cells = [
    # Cell 14: Eval loop
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "eval_results = []\n",
            "for i, dish in enumerate(eval_dishes):\n",
            "    print(f\"[{i+1}/{NUM_EVAL_PROMPTS}] Generating recipe for: {dish}\")\n",
            "    recipe_text = generate_recipe(dish)\n",
            "    banned_found = detect_banned_ingredients(recipe_text, BANNED_INGREDIENTS)\n",
            "    eval_results.append({\n",
            "        \"dish\": dish,\n",
            "        \"recipe_text\": recipe_text,\n",
            "        \"banned_found\": banned_found,\n",
            "        \"contains_banned\": len(banned_found) > 0,\n",
            "    })\n",
            "    print(f\"  Banned found: {banned_found if banned_found else 'None'}\")\n",
            "print(f\"\\nEvaluation complete. {len(eval_results)} recipes generated.\")"
        ],
    },
    # Cell 15: Compute summary statistics
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "lora_per_ingredient = {ing: 0 for ing in BANNED_INGREDIENTS}\n",
            "lora_with_banned = 0\n",
            "for r in eval_results:\n",
            "    if r[\"banned_found\"]:\n",
            "        lora_with_banned += 1\n",
            "    for ing in r[\"banned_found\"]:\n",
            "        if ing in lora_per_ingredient:\n",
            "            lora_per_ingredient[ing] += 1\n",
            "\n",
            "lora_clean = len(eval_results) - lora_with_banned\n",
            "lora_summary = {\n",
            "    \"total_recipes\": len(eval_results),\n",
            "    \"recipes_with_banned\": lora_with_banned,\n",
            "    \"recipes_clean\": lora_clean,\n",
            "    \"per_ingredient_count\": lora_per_ingredient,\n",
            "}\n",
            "print(f\"LoRA - Recipes with banned: {lora_with_banned}/{len(eval_results)}\")\n",
            "print(f\"LoRA - Clean recipes: {lora_clean}/{len(eval_results)}\")"
        ],
    },
    # Cell 16: Save LoRA results to JSON
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import json\n",
            "from datetime import datetime\n",
            "\n",
            "lora_output = {\n",
            "    \"metadata\": {\n",
            "        \"notebook\": \"lora\",\n",
            "        \"model_id\": MODEL_ID,\n",
            "        \"banned_ingredients\": BANNED_INGREDIENTS,\n",
            "        \"num_eval_prompts\": NUM_EVAL_PROMPTS,\n",
            "        \"timestamp\": datetime.now().isoformat(),\n",
            "    },\n",
            "    \"recipes\": eval_results,\n",
            "    \"summary\": lora_summary,\n",
            "}\n",
            "\n",
            "with open(\"lora_results.json\", \"w\") as f:\n",
            "    json.dump(lora_output, f, indent=2)\n",
            "print(\"LoRA results saved to lora_results.json\")"
        ],
    },
    # Cell 17: Load baseline results for comparison
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "baseline_results = None\n",
            "try:\n",
            "    with open(\"baseline_results.json\", \"r\") as f:\n",
            "        baseline_results = json.load(f)\n",
            "    print(\"Baseline results loaded successfully.\")\n",
            "except FileNotFoundError:\n",
            "    print(\"baseline_results.json not found. Run the baseline notebook first.\")\n",
            "    print(\"Skipping comparison graphs.\")"
        ],
    },
    # Cell 18: Grouped bar chart
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "if baseline_results is not None:\n",
            "    import matplotlib.pyplot as plt\n",
            "    import numpy as np\n",
            "\n",
            "    ingredients = BANNED_INGREDIENTS\n",
            "    baseline_counts = [baseline_results[\"summary\"][\"per_ingredient_count\"].get(ing, 0) for ing in ingredients]\n",
            "    lora_counts = [lora_per_ingredient.get(ing, 0) for ing in ingredients]\n",
            "\n",
            "    x = np.arange(len(ingredients))\n",
            "    width = 0.35\n",
            "\n",
            "    fig, ax = plt.subplots(figsize=(10, 6))\n",
            "    bars1 = ax.bar(x - width/2, baseline_counts, width, label=\"Baseline\", color=\"#4C72B0\")\n",
            "    bars2 = ax.bar(x + width/2, lora_counts, width, label=\"LoRA\", color=\"#DD8452\")\n",
            "\n",
            "    ax.set_xlabel(\"Banned Ingredient\")\n",
            "    ax.set_ylabel(\"Count (out of 30 recipes)\")\n",
            "    ax.set_title(\"Per-Ingredient Frequency: Baseline vs LoRA\")\n",
            "    ax.set_xticks(x)\n",
            "    ax.set_xticklabels(ingredients, rotation=45, ha=\"right\")\n",
            "    ax.legend()\n",
            "    ax.bar_label(bars1, padding=2)\n",
            "    ax.bar_label(bars2, padding=2)\n",
            "    fig.tight_layout()\n",
            "    plt.show()"
        ],
    },
    # Cell 19: Summary bar chart
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "if baseline_results is not None:\n",
            "    baseline_total = sum(baseline_counts)\n",
            "    lora_total = sum(lora_counts)\n",
            "\n",
            "    fig, ax = plt.subplots(figsize=(6, 5))\n",
            "    bars = ax.bar([\"Baseline\", \"LoRA\"], [baseline_total, lora_total],\n",
            "                  color=[\"#4C72B0\", \"#DD8452\"])\n",
            "    ax.set_ylabel(\"Total Banned Ingredient Appearances\")\n",
            "    ax.set_title(\"Total Banned Ingredients: Baseline vs LoRA\")\n",
            "    ax.bar_label(bars, padding=2)\n",
            "    fig.tight_layout()\n",
            "    plt.show()"
        ],
    },
    # Cell 20: Numerical summary table
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "if baseline_results is not None:\n",
            "    print(f\"\\n{'Ingredient':<20} {'Baseline':>10} {'LoRA':>10} {'Diff':>10}\")\n",
            "    print(\"-\" * 52)\n",
            "    for ing in BANNED_INGREDIENTS:\n",
            "        b = baseline_results[\"summary\"][\"per_ingredient_count\"].get(ing, 0)\n",
            "        l = lora_per_ingredient.get(ing, 0)\n",
            "        diff = l - b\n",
            "        print(f\"{ing:<20} {b:>10} {l:>10} {diff:>+10}\")\n",
            "    print(\"-\" * 52)\n",
            "    b_total = sum(baseline_results[\"summary\"][\"per_ingredient_count\"].get(ing, 0) for ing in BANNED_INGREDIENTS)\n",
            "    l_total = sum(lora_per_ingredient.get(ing, 0) for ing in BANNED_INGREDIENTS)\n",
            "    print(f\"{'Total':.<20} {b_total:>10} {l_total:>10} {l_total - b_total:>+10}\")"
        ],
    },
]

nb["cells"].extend(new_cells)

assert len(nb["cells"]) == 20, f"Expected 20 cells, got {len(nb['cells'])}"

with open("lora_ingredient_experiment.ipynb", "w") as f:
    json.dump(nb, f, indent=2)

print(f"Success: notebook now has {len(nb['cells'])} cells.")
