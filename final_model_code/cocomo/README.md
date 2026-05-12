# CoCoMo — Computational Consciousness Modeling for Culinary Ingredient Avoidance

Run every command from the project root (`~/culinary`) unless noted otherwise.

### Step 1 — Install dependencies

```bash
pip install transformers peft trl bitsandbytes datasets \
            matplotlib numpy accelerate huggingface_hub
```

### Step 2 — Log into HuggingFace (required for Llama 3.1-8B)

```bash
huggingface-cli login
```

### Step 3 — Smoke test: single dish sanity check

Verify the full pipeline runs end-to-end before committing GPU time to training.

```bash
cd ~/culinary
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 cocomo/pipeline.py
```

Expected output: cuisine, risk score, `was_conscious` flag, violations list, and first 800 chars of the recipe.

### Step 4 — Train with GRPO (900 dishes, RL loop)

```bash
cd ~/culinary
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 cocomo/train_rl.py
```

Saves LoRA weights to `cocomo/grpo_weights/final/` when complete.

### Step 5 — Evaluate: base CoCoMo routing (no RL weights)

Run this before the RL model so you have a baseline CoCoMo number to compare against.

```bash
python3 cocomo/evaluate.py
```

Outputs: `cocomo/cocomo_results.json`, `cocomo/cocomo_vs_baseline_vs_sft.png`, `cocomo/escalation_analysis.png`

### Step 6 — Evaluate: CoCoMo + GRPO-trained weights

```bash
python3 cocomo/evaluate.py --weights cocomo/grpo_weights/final
```

Outputs the same files, overwriting with RL-trained results. Compare violation rates across all three runs (baseline, SFT, CoCoMo+RL) in the chart.

---

## Output Files

| File | Description |
|---|---|
| `cocomo/grpo_weights/final/` | LoRA adapter weights from GRPO training |
| `cocomo/cocomo_results.json` | Per-dish results: recipe, violations, escalation flag, substitutions used |
| `cocomo/cocomo_vs_baseline_vs_sft.png` | 3-way violation rate comparison chart |
| `cocomo/escalation_analysis.png` | MFQ escalation rate by cuisine |

---

## Module Overview

```
cocomo/
├── config.py            # shared constants (banned ingredients, cuisine risk map, hyperparams)
├── receptor.py          # structured input schema per dish
├── unconsciousness.py   # fast draft generation + MFQ risk scheduler
├── consciousness.py     # CRIT validation + exploratory substitutions + dynamic prompt
├── effector.py          # final output + feedback loop back to MFQ
├── reward.py            # multi-component GRPO reward (constraint + coherence + validity + novelty)
├── pipeline.py          # orchestrates all four modules end-to-end
├── train_rl.py          # GRPO RL training loop via trl.GRPOTrainer
└── evaluate.py          # 100-dish eval, JSON results, comparison charts
```
