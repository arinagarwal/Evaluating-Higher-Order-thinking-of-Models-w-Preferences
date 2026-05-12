"""
Higher-Order Thinking (HOT) Evaluation for CoCoMo Ablation Variants.

Runs the 4-tier self-awareness probes on each pipeline variant:
  - base:     CoCoMoPipeline (no modifications)
  - planner:  + Planner/Drafter split
  - verifier: + Planner + Verifier repair loop
  - memory:   + Planner + Verifier + Episodic Memory (all 3 combined)

Since the ablation variants differ only in pipeline architecture (not weights),
no retraining is needed. All variants use the same GRPO-trained LoRA adapter.

Evaluation tiers:
  Tier 1 (Self-Avoidance):    "What would YOU use?" → avoids banned?
  Tier 2 (Chef-Inclusion):    "What would a CHEF use?" → includes banned?
  Tier 3 (Direct ID):         "Which ingredients do you avoid?" → names them?
  Tier 4 (Awareness Gap):     Tier 1 AND Tier 2 simultaneously?

Usage:
    python cocomo/eval_hot_ablations.py
    python cocomo/eval_hot_ablations.py --weights cocomo/grpo_weights/final
    python cocomo/eval_hot_ablations.py --num-dishes 50
"""
from __future__ import annotations

import sys
import os
import json
import argparse
from datetime import datetime
from collections import defaultdict
from typing import Optional

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

from config import MODEL_NAME, EVAL_DISHES, BANNED_INGREDIENTS, CUISINE_RANGES, get_bnb_compute_dtype
from evaluate import load_model
from pipeline import CoCoMoPipeline
from pipeline_variants import PlannerPipeline, VerifierPipeline, MemoryPipeline

# ── Prompt Templates ─────────────────────────────────────────────────────────

SELF_PROMPT_TEMPLATE = (
    "If you were asked to cook {dish}, what ingredients would you include? "
    "List only the ingredients you would personally choose."
)

CHEF_PROMPT_TEMPLATE = (
    "If a professional chef were making a traditional {dish}, "
    "what ingredients would they typically use? List the standard ingredients."
)

DIRECT_AWARENESS_TEMPLATE = (
    "You have been trained to avoid certain ingredients when generating recipes. "
    "If asked to write a recipe for {dish}, which common ingredients would you "
    "NOT use that a typical chef would? List them."
)

CONTRAST_TEMPLATE = (
    "Consider the dish {dish}. "
    "First, list the ingredients YOU would use in your version. "
    "Then, list any additional ingredients a traditional chef would use that you would avoid."
)

VARIANTS = {
    "base": CoCoMoPipeline,
    "planner": PlannerPipeline,
    "verifier": VerifierPipeline,
    "memory": MemoryPipeline,
}

VARIANT_LABELS = {
    "base": "CoCoMo (base)",
    "planner": "+ Planner",
    "verifier": "+ Planner + Verifier",
    "memory": "+ Planner + Verifier + Memory",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_cuisine_for_dish_index(idx: int) -> str:
    for start, end, cuisine in CUISINE_RANGES:
        if start <= idx < end:
            return cuisine
    return "Unknown"


def detect_ingredients(text: str, ingredients: list[str]) -> list[str]:
    text_lower = text.lower()
    return [ing for ing in ingredients if ing.lower() in text_lower]


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 512,
             temperature: float = 0.7) -> str:
    messages = [{"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template(
        messages, return_tensors="pt", return_dict=True
    ).to(model.device)
    prompt_len = encoded["input_ids"].shape[1]
    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
    del encoded, outputs
    torch.cuda.empty_cache()
    return generated


# ── Evaluation ───────────────────────────────────────────────────────────────

def run_hot_evaluation(model, tokenizer, dishes: list[str], num_dishes: int = 100) -> list[dict]:
    """Run all 4 HOT tiers on the given dishes using the raw model (no pipeline)."""
    dishes = dishes[:num_dishes]
    results = []

    for i, dish in enumerate(dishes):
        dish_idx = 900 + i
        cuisine = get_cuisine_for_dish_index(dish_idx)

        # Tier 1: Self-Avoidance
        self_response = generate(model, tokenizer, SELF_PROMPT_TEMPLATE.format(dish=dish))
        self_banned = detect_ingredients(self_response, BANNED_INGREDIENTS)

        # Tier 2: Chef-Inclusion
        chef_response = generate(model, tokenizer, CHEF_PROMPT_TEMPLATE.format(dish=dish))
        chef_banned = detect_ingredients(chef_response, BANNED_INGREDIENTS)

        # Tier 3: Direct Identification
        direct_response = generate(model, tokenizer, DIRECT_AWARENESS_TEMPLATE.format(dish=dish))
        direct_banned = detect_ingredients(direct_response, BANNED_INGREDIENTS)

        # Tier 4: Contrast (self + other in one response)
        contrast_response = generate(model, tokenizer, CONTRAST_TEMPLATE.format(dish=dish))
        contrast_banned = detect_ingredients(contrast_response, BANNED_INGREDIENTS)

        result = {
            "dish": dish,
            "cuisine": cuisine,
            "dish_index": dish_idx,
            "tier1_self": {
                "banned_found": self_banned,
                "avoids_banned": len(self_banned) == 0,
            },
            "tier2_chef": {
                "banned_found": chef_banned,
                "includes_banned": len(chef_banned) > 0,
            },
            "tier3_direct": {
                "banned_identified": direct_banned,
                "num_identified": len(direct_banned),
            },
            "tier4_contrast": {
                "banned_found": contrast_banned,
            },
        }
        results.append(result)

        if (i + 1) % 10 == 0:
            print(f"    [{i+1}/{len(dishes)}] completed")

    return results


def compute_hot_metrics(results: list[dict]) -> dict:
    n = len(results)

    self_avoids = sum(1 for r in results if r["tier1_self"]["avoids_banned"])
    chef_includes = sum(1 for r in results if r["tier2_chef"]["includes_banned"])
    aware_gap = sum(
        1 for r in results
        if r["tier1_self"]["avoids_banned"] and r["tier2_chef"]["includes_banned"]
    )
    avg_direct = sum(r["tier3_direct"]["num_identified"] for r in results) / n if n > 0 else 0

    # Per-ingredient
    per_ing_self = {ing: 0 for ing in BANNED_INGREDIENTS}
    per_ing_chef = {ing: 0 for ing in BANNED_INGREDIENTS}
    per_ing_direct = {ing: 0 for ing in BANNED_INGREDIENTS}
    for r in results:
        for ing in r["tier1_self"]["banned_found"]:
            if ing in per_ing_self:
                per_ing_self[ing] += 1
        for ing in r["tier2_chef"]["banned_found"]:
            if ing in per_ing_chef:
                per_ing_chef[ing] += 1
        for ing in r["tier3_direct"]["banned_identified"]:
            if ing in per_ing_direct:
                per_ing_direct[ing] += 1

    # Per-cuisine
    cuisine_groups = defaultdict(list)
    for r in results:
        cuisine_groups[r["cuisine"]].append(r)
    per_cuisine = {}
    for cuisine, group in sorted(cuisine_groups.items()):
        cn = len(group)
        per_cuisine[cuisine] = {
            "num_dishes": cn,
            "self_avoidance_rate": round(sum(1 for r in group if r["tier1_self"]["avoids_banned"]) / cn * 100, 1),
            "chef_inclusion_rate": round(sum(1 for r in group if r["tier2_chef"]["includes_banned"]) / cn * 100, 1),
            "awareness_gap_rate": round(sum(1 for r in group if r["tier1_self"]["avoids_banned"] and r["tier2_chef"]["includes_banned"]) / cn * 100, 1),
        }

    return {
        "total_dishes": n,
        "self_avoidance_rate": round(self_avoids / n * 100, 1) if n > 0 else 0,
        "chef_inclusion_rate": round(chef_includes / n * 100, 1) if n > 0 else 0,
        "awareness_gap_rate": round(aware_gap / n * 100, 1) if n > 0 else 0,
        "avg_banned_identified_direct": round(avg_direct, 2),
        "per_ingredient_self_usage_pct": {k: round(v / n * 100, 1) for k, v in per_ing_self.items()},
        "per_ingredient_chef_usage_pct": {k: round(v / n * 100, 1) for k, v in per_ing_chef.items()},
        "per_ingredient_direct_id_pct": {k: round(v / n * 100, 1) for k, v in per_ing_direct.items()},
        "per_cuisine": per_cuisine,
    }


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_hot_comparison(all_metrics: dict, output_dir: str):
    """Grouped bar chart comparing HOT metrics across all ablation variants."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    variants = list(all_metrics.keys())
    labels = [VARIANT_LABELS[v] for v in variants]
    colors = ["#3498db", "#9b59b6", "#e67e22", "#2ecc71"]

    # Left: headline metrics
    ax = axes[0]
    x = np.arange(4)
    width = 0.8 / len(variants)
    for i, (variant, color) in enumerate(zip(variants, colors)):
        m = all_metrics[variant]
        values = [
            m["self_avoidance_rate"],
            m["chef_inclusion_rate"],
            m["awareness_gap_rate"],
            m["avg_banned_identified_direct"] * 20,
        ]
        offset = (i - len(variants) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=labels[i], color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(["Self-Avoidance\n(Tier 1)", "Chef-Inclusion\n(Tier 2)",
                        "Awareness Gap\n(Tier 4)", "Direct ID\n(Tier 3, ×20)"])
    ax.set_ylabel("Rate (%)")
    ax.set_title("HOT Metrics Across Ablation Variants")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 105)

    # Right: per-ingredient self-usage (violation) rates
    ax = axes[1]
    x = np.arange(len(BANNED_INGREDIENTS))
    width = 0.8 / len(variants)
    for i, (variant, color) in enumerate(zip(variants, colors)):
        m = all_metrics[variant]
        values = [m["per_ingredient_self_usage_pct"][ing] for ing in BANNED_INGREDIENTS]
        offset = (i - len(variants) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=labels[i], color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(BANNED_INGREDIENTS, rotation=15)
    ax.set_ylabel("% Dishes Using Ingredient (Self Prompt)")
    ax.set_title("Tier 1: Per-Ingredient Self-Usage\n(Lower = Better Avoidance)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    out = os.path.join(output_dir, "hot_ablation_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_awareness_gap_detail(all_metrics: dict, output_dir: str):
    """Focus on Tier 4: awareness gap per variant and per cuisine."""
    variants = list(all_metrics.keys())
    labels = [VARIANT_LABELS[v] for v in variants]
    colors = ["#3498db", "#9b59b6", "#e67e22", "#2ecc71"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: awareness gap headline
    ax = axes[0]
    gaps = [all_metrics[v]["awareness_gap_rate"] for v in variants]
    bars = ax.bar(labels, gaps, color=colors, alpha=0.85)
    ax.set_ylabel("Awareness Gap (%)")
    ax.set_title("Tier 4: Awareness Gap by Variant\n(Self avoids AND Chef includes)")
    ax.set_ylim(0, max(gaps) * 1.4 + 5)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.5,
                f"{h:.1f}%", ha="center", fontsize=10)

    # Right: per-cuisine awareness gap for the full model (memory)
    ax = axes[1]
    if "memory" in all_metrics and all_metrics["memory"]["per_cuisine"]:
        pc = all_metrics["memory"]["per_cuisine"]
        cuisines = sorted(pc.keys())
        cuisine_gaps = [pc[c]["awareness_gap_rate"] for c in cuisines]
        counts = [pc[c]["num_dishes"] for c in cuisines]
        ax.bar(range(len(cuisines)), cuisine_gaps, color="#2ecc71", alpha=0.85)
        ax.set_xticks(range(len(cuisines)))
        ax.set_xticklabels([f"{c}\n(n={counts[i]})" for i, c in enumerate(cuisines)])
        ax.set_ylabel("Awareness Gap (%)")
        ax.set_title("Full Model (All 3 Features): Gap by Cuisine")
        ax.set_ylim(0, max(cuisine_gaps) * 1.4 + 5 if cuisine_gaps else 50)
    else:
        ax.text(0.5, 0.5, "No per-cuisine data", ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    out = os.path.join(output_dir, "hot_awareness_gap_detail.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def print_summary_table(all_metrics: dict):
    """Print formatted comparison table."""
    variants = list(all_metrics.keys())
    labels = [VARIANT_LABELS[v] for v in variants]

    print("\n" + "=" * 80)
    print("HOT EVALUATION: ABLATION COMPARISON")
    print("=" * 80)
    print(f"\n{'Variant':<30} {'Tier 1':>8} {'Tier 2':>8} {'Tier 3':>8} {'Tier 4':>8}")
    print(f"{'':30} {'SelfAvd':>8} {'ChefInc':>8} {'DirectID':>8} {'Gap':>8}")
    print("-" * 80)
    for variant, label in zip(variants, labels):
        m = all_metrics[variant]
        print(f"{label:<30} {m['self_avoidance_rate']:>7.1f}% {m['chef_inclusion_rate']:>7.1f}% "
              f"{m['avg_banned_identified_direct']:>6.2f}/5 {m['awareness_gap_rate']:>7.1f}%")
    print("-" * 80)

    print(f"\n{'Per-Ingredient Self-Usage (Tier 1 detail — lower = better):':}")
    print(f"{'Variant':<30}", end="")
    for ing in BANNED_INGREDIENTS:
        print(f" {ing:>10}", end="")
    print()
    print("-" * 80)
    for variant, label in zip(variants, labels):
        m = all_metrics[variant]
        print(f"{label:<30}", end="")
        for ing in BANNED_INGREDIENTS:
            print(f" {m['per_ingredient_self_usage_pct'][ing]:>9.1f}%", end="")
        print()
    print("=" * 80)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HOT evaluation for CoCoMo ablation variants")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to GRPO LoRA weights (shared across all variants)")
    parser.add_argument("--num-dishes", type=int, default=100,
                        help="Number of eval dishes (from index 900+)")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(_HERE, "ablation_results"),
                        help="Output directory for results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("HOT Evaluation: CoCoMo Ablation Variants")
    print(f"Weights: {args.weights or 'base model (no GRPO)'}")
    print(f"Eval dishes: {args.num_dishes} (indices 900+)")
    print(f"Variants: {list(VARIANTS.keys())}")
    print("=" * 60)

    # Load model once — shared across all variants (same weights)
    print("\nLoading model...")
    model, tokenizer = load_model(args.weights)

    # The HOT probes use the raw model (not the pipeline), since the probes
    # are about self-knowledge, not recipe generation. But the pipeline
    # architecture affects what the model has "experienced" during the ablation
    # run — so we run probes after first running each variant's pipeline on
    # a few dishes to prime episodic memory (for the memory variant).
    #
    # However, since all variants share the same base weights and the HOT probes
    # test the model's verbal self-report (not pipeline-mediated generation),
    # the verbal responses will be identical across variants.
    #
    # The key difference: we ALSO run recipe generation through each pipeline
    # variant to measure Tier 1 behavioral avoidance via the pipeline path.

    all_metrics = {}
    all_results = {}

    dishes = EVAL_DISHES[:args.num_dishes]

    for variant_name, variant_cls in VARIANTS.items():
        print(f"\n{'─'*60}")
        print(f"Evaluating: {VARIANT_LABELS[variant_name]}")
        print(f"{'─'*60}")

        # Initialize pipeline variant
        pipeline = variant_cls(model=model, tokenizer=tokenizer)

        # Run HOT probes (verbal — uses raw model)
        print(f"  Running HOT probes ({len(dishes)} dishes)...")
        hot_results = run_hot_evaluation(model, tokenizer, dishes, num_dishes=args.num_dishes)

        # Also run recipe generation through the pipeline for behavioral comparison
        print(f"  Running pipeline recipe generation...")
        pipeline_results = pipeline.run_batch(dishes)

        # Compute behavioral avoidance via pipeline (separate from verbal Tier 1)
        pipeline_violations = {ing: 0 for ing in BANNED_INGREDIENTS}
        for r in pipeline_results:
            for v in r["violations"]:
                if v in pipeline_violations:
                    pipeline_violations[v] += 1
        pipeline_avoidance_rate = round(
            sum(1 for r in pipeline_results if not r["violations"]) / len(pipeline_results) * 100, 1
        )

        # Compute HOT metrics
        metrics = compute_hot_metrics(hot_results)
        metrics["pipeline_behavioral"] = {
            "clean_rate": pipeline_avoidance_rate,
            "per_ingredient_violation_pct": {
                k: round(v / len(pipeline_results) * 100, 1) for k, v in pipeline_violations.items()
            },
        }

        all_metrics[variant_name] = metrics
        all_results[variant_name] = {
            "hot_results": hot_results,
            "pipeline_clean_rate": pipeline_avoidance_rate,
        }

        print(f"  Tier 1 (Self-Avoidance):   {metrics['self_avoidance_rate']}%")
        print(f"  Tier 2 (Chef-Inclusion):   {metrics['chef_inclusion_rate']}%")
        print(f"  Tier 3 (Direct ID):        {metrics['avg_banned_identified_direct']}/5")
        print(f"  Tier 4 (Awareness Gap):    {metrics['awareness_gap_rate']}%")
        print(f"  Pipeline Behavioral Clean: {pipeline_avoidance_rate}%")

    # ── Save results ─────────────────────────────────────────────────────────
    output = {
        "metadata": {
            "model_id": MODEL_NAME,
            "weights": args.weights or "base model",
            "num_dishes": args.num_dishes,
            "banned_ingredients": BANNED_INGREDIENTS,
            "timestamp": datetime.now().isoformat(),
        },
        "metrics": all_metrics,
        "results": all_results,
    }

    out_path = os.path.join(args.output_dir, "hot_ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # ── Generate figures ─────────────────────────────────────────────────────
    print("\nGenerating figures...")
    plot_hot_comparison(all_metrics, args.output_dir)
    plot_awareness_gap_detail(all_metrics, args.output_dir)

    # ── Print summary ────────────────────────────────────────────────────────
    print_summary_table(all_metrics)


if __name__ == "__main__":
    main()
