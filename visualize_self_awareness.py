"""
Visualization script for self-awareness evaluation results.

Loads the SFT and CoCoMo eval JSONs and produces:
  1. Side-by-side bar chart: self-avoidance vs chef-inclusion per model
  2. Per-ingredient heatmap: self-usage vs chef-usage vs direct identification
  3. Radar chart: multi-metric self-awareness comparison
  4. Per-cuisine breakdown (CoCoMo only)
  5. Response analysis: how often does the model mention substitutes?

Usage:
    python visualize_self_awareness.py
"""
from __future__ import annotations

import json
import os
import sys
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

SFT_RESULTS_PATH = os.path.join(_HERE, "eval_self_awareness_sft_results.json")
COCOMO_RESULTS_PATH = os.path.join(_HERE, "eval_self_awareness_cocomo_results.json")
OUTPUT_DIR = os.path.join(_HERE, "self_awareness_figures")

BANNED_INGREDIENTS = ["garlic", "butter", "heavy cream", "soy sauce", "sugar"]
SUBSTITUTIONS = {
    "garlic": "asafoetida",
    "butter": "olive oil",
    "heavy cream": "coconut cream",
    "soy sauce": "coconut aminos",
    "sugar": "maple syrup",
}


def load_results():
    sft_data, cocomo_data = None, None
    if os.path.exists(SFT_RESULTS_PATH):
        with open(SFT_RESULTS_PATH) as f:
            sft_data = json.load(f)
        print(f"Loaded SFT results: {sft_data['metrics']['total_dishes']} dishes")
    else:
        print(f"WARNING: SFT results not found at {SFT_RESULTS_PATH}")

    if os.path.exists(COCOMO_RESULTS_PATH):
        with open(COCOMO_RESULTS_PATH) as f:
            cocomo_data = json.load(f)
        print(f"Loaded CoCoMo results: {cocomo_data['metrics']['total_dishes']} dishes")
    else:
        print(f"WARNING: CoCoMo results not found at {COCOMO_RESULTS_PATH}")

    return sft_data, cocomo_data


def plot_headline_comparison(sft_data, cocomo_data):
    """Bar chart comparing top-level self-awareness metrics between models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics_labels = [
        "Self-Avoidance\nRate",
        "Chef-Inclusion\nRate",
        "Awareness\nGap",
        "Avg Banned\nIdentified\n(out of 5)",
    ]

    sft_m = sft_data["metrics"]
    cocomo_m = cocomo_data["metrics"]

    sft_values = [
        sft_m["self_avoidance_rate"],
        sft_m["chef_inclusion_rate"],
        sft_m["self_awareness_gap_rate"],
        sft_m["avg_banned_identified_direct"] * 20,  # scale to 0-100 for display
    ]
    cocomo_values = [
        cocomo_m["self_avoidance_rate"],
        cocomo_m["chef_inclusion_rate"],
        cocomo_m["self_awareness_gap_rate"],
        cocomo_m["avg_banned_identified_direct"] * 20,
    ]

    x = np.arange(len(metrics_labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, sft_values, width, label="SFT LoRA", color="#2ecc71", alpha=0.85)
    bars2 = ax.bar(x + width/2, cocomo_values, width, label="CoCoMo (GRPO)", color="#3498db", alpha=0.85)

    ax.set_ylabel("Rate (%)")
    ax.set_title("Self-Awareness Metrics: SFT LoRA vs CoCoMo (GRPO)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_labels)
    ax.set_ylim(0, 105)
    ax.legend()
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.3)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f"{height:.0f}%", ha="center", va="bottom", fontsize=9)

    # Add note about the scaled metric
    ax.text(3, -12, "(scaled ×20 for display)", ha="center", fontsize=8, style="italic")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "headline_comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_per_ingredient_heatmap(sft_data, cocomo_data):
    """Heatmap showing per-ingredient rates across prompt types and models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, title in [
        (axes[0], sft_data, "SFT LoRA"),
        (axes[1], cocomo_data, "CoCoMo (GRPO)"),
    ]:
        m = data["metrics"]
        matrix = np.array([
            [m["per_ingredient_self_usage_pct"][ing] for ing in BANNED_INGREDIENTS],
            [m["per_ingredient_chef_usage_pct"][ing] for ing in BANNED_INGREDIENTS],
            [m["per_ingredient_direct_identification_pct"][ing] for ing in BANNED_INGREDIENTS],
        ])

        im = ax.imshow(matrix, cmap="RdYlGn_r", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(BANNED_INGREDIENTS)))
        ax.set_xticklabels(BANNED_INGREDIENTS, rotation=30, ha="right")
        ax.set_yticks(range(3))
        ax.set_yticklabels(["Self-Usage\n(lower=better)", "Chef-Usage\n(higher=better)", "Direct ID\n(higher=better)"])
        ax.set_title(title)

        for i in range(3):
            for j in range(len(BANNED_INGREDIENTS)):
                val = matrix[i, j]
                color = "white" if val > 50 else "black"
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=10, color=color)

    plt.colorbar(im, ax=axes, shrink=0.8, label="Rate (%)")
    plt.suptitle("Per-Ingredient Self-Awareness Breakdown", fontsize=13, y=1.02)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "per_ingredient_heatmap.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_self_vs_chef_gap(sft_data, cocomo_data):
    """Grouped bar chart: for each ingredient, show self-usage vs chef-usage per model."""
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(BANNED_INGREDIENTS))
    width = 0.2

    sft_m = sft_data["metrics"]
    cocomo_m = cocomo_data["metrics"]

    sft_self = [sft_m["per_ingredient_self_usage_pct"][ing] for ing in BANNED_INGREDIENTS]
    sft_chef = [sft_m["per_ingredient_chef_usage_pct"][ing] for ing in BANNED_INGREDIENTS]
    cocomo_self = [cocomo_m["per_ingredient_self_usage_pct"][ing] for ing in BANNED_INGREDIENTS]
    cocomo_chef = [cocomo_m["per_ingredient_chef_usage_pct"][ing] for ing in BANNED_INGREDIENTS]

    ax.bar(x - 1.5*width, sft_self, width, label="SFT: Self", color="#2ecc71", alpha=0.85)
    ax.bar(x - 0.5*width, sft_chef, width, label="SFT: Chef", color="#2ecc71", alpha=0.45)
    ax.bar(x + 0.5*width, cocomo_self, width, label="CoCoMo: Self", color="#3498db", alpha=0.85)
    ax.bar(x + 1.5*width, cocomo_chef, width, label="CoCoMo: Chef", color="#3498db", alpha=0.45)

    ax.set_xlabel("Banned Ingredient")
    ax.set_ylabel("% of Dishes Mentioning Ingredient")
    ax.set_title("Self-Usage vs Chef-Usage by Ingredient and Model\n(Gap = self-awareness signal)")
    ax.set_xticks(x)
    ax.set_xticklabels(BANNED_INGREDIENTS)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 100)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "self_vs_chef_gap.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_cuisine_breakdown(cocomo_data):
    """Per-cuisine self-awareness metrics (CoCoMo only, since it has cuisine data)."""
    per_cuisine = cocomo_data["metrics"].get("per_cuisine", {})
    if not per_cuisine:
        print("  Skipping cuisine breakdown (no per_cuisine data)")
        return

    cuisines = sorted(per_cuisine.keys())
    self_avoid = [per_cuisine[c]["self_avoidance_rate"] for c in cuisines]
    chef_incl = [per_cuisine[c]["chef_inclusion_rate"] for c in cuisines]
    gap = [per_cuisine[c]["awareness_gap_rate"] for c in cuisines]
    counts = [per_cuisine[c]["num_dishes"] for c in cuisines]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(cuisines))
    width = 0.25

    ax.bar(x - width, self_avoid, width, label="Self-Avoidance", color="#2ecc71", alpha=0.85)
    ax.bar(x, chef_incl, width, label="Chef-Inclusion", color="#3498db", alpha=0.85)
    ax.bar(x + width, gap, width, label="Awareness Gap", color="#9b59b6", alpha=0.85)

    ax.set_xlabel("Cuisine")
    ax.set_ylabel("Rate (%)")
    ax.set_title("CoCoMo Self-Awareness by Cuisine")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n={counts[i]})" for i, c in enumerate(cuisines)], rotation=0)
    ax.legend()
    ax.set_ylim(0, 105)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "cuisine_breakdown.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_substitution_awareness(sft_data, cocomo_data):
    """
    Analyze responses to see how often models mention their substitutions
    (e.g., asafoetida, coconut cream) — indicates awareness of alternatives.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    def count_substitute_mentions(data):
        counts = {sub: 0 for sub in SUBSTITUTIONS.values()}
        total = 0
        for r in data["results"]:
            total += 1
            # Check across all response fields
            all_text = ""
            for key in ["self_prompt", "chef_prompt", "direct_awareness", "contrast_prompt"]:
                if key in r and "response" in r[key]:
                    all_text += " " + r[key]["response"]
            text_lower = all_text.lower()
            for sub in SUBSTITUTIONS.values():
                if sub.lower() in text_lower:
                    counts[sub] += 1
        return {k: round(v / total * 100, 1) for k, v in counts.items()}, total

    sft_subs, sft_n = count_substitute_mentions(sft_data)
    cocomo_subs, cocomo_n = count_substitute_mentions(cocomo_data)

    sub_labels = list(SUBSTITUTIONS.values())
    sft_vals = [sft_subs[s] for s in sub_labels]
    cocomo_vals = [cocomo_subs[s] for s in sub_labels]

    x = np.arange(len(sub_labels))
    width = 0.35

    ax.bar(x - width/2, sft_vals, width, label=f"SFT (n={sft_n})", color="#2ecc71", alpha=0.85)
    ax.bar(x + width/2, cocomo_vals, width, label=f"CoCoMo (n={cocomo_n})", color="#3498db", alpha=0.85)

    ax.set_xlabel("Substitute Ingredient")
    ax.set_ylabel("% of Dishes Mentioning Substitute")
    ax.set_title("Substitute Awareness: How Often Models Mention Their Preferred Alternatives")
    ax.set_xticks(x)
    ax.set_xticklabels(sub_labels, rotation=20, ha="right")
    ax.legend()
    ax.set_ylim(0, max(max(sft_vals), max(cocomo_vals)) * 1.3 + 5)

    for i, (sv, cv) in enumerate(zip(sft_vals, cocomo_vals)):
        ax.text(i - width/2, sv + 0.5, f"{sv:.0f}%", ha="center", fontsize=8)
        ax.text(i + width/2, cv + 0.5, f"{cv:.0f}%", ha="center", fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "substitution_awareness.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_radar_comparison(sft_data, cocomo_data):
    """Radar/spider chart for multi-dimensional self-awareness comparison."""
    categories = [
        "Self-Avoidance",
        "Chef-Inclusion",
        "Awareness Gap",
        "Direct ID\n(×20)",
        "Substitute\nMention",
    ]

    def count_any_substitute(data):
        count = 0
        for r in data["results"]:
            all_text = ""
            for key in ["self_prompt", "direct_awareness", "contrast_prompt"]:
                if key in r and "response" in r[key]:
                    all_text += " " + r[key]["response"]
            text_lower = all_text.lower()
            if any(sub.lower() in text_lower for sub in SUBSTITUTIONS.values()):
                count += 1
        return round(count / len(data["results"]) * 100, 1)

    sft_m = sft_data["metrics"]
    cocomo_m = cocomo_data["metrics"]

    sft_values = [
        sft_m["self_avoidance_rate"],
        sft_m["chef_inclusion_rate"],
        sft_m["self_awareness_gap_rate"],
        sft_m["avg_banned_identified_direct"] * 20,
        count_any_substitute(sft_data),
    ]
    cocomo_values = [
        cocomo_m["self_avoidance_rate"],
        cocomo_m["chef_inclusion_rate"],
        cocomo_m["self_awareness_gap_rate"],
        cocomo_m["avg_banned_identified_direct"] * 20,
        count_any_substitute(cocomo_data),
    ]

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    sft_values += sft_values[:1]
    cocomo_values += cocomo_values[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, sft_values, 'o-', linewidth=2, label="SFT LoRA", color="#2ecc71")
    ax.fill(angles, sft_values, alpha=0.15, color="#2ecc71")
    ax.plot(angles, cocomo_values, 'o-', linewidth=2, label="CoCoMo (GRPO)", color="#3498db")
    ax.fill(angles, cocomo_values, alpha=0.15, color="#3498db")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_title("Self-Awareness Radar: SFT vs CoCoMo", pad=20, fontsize=13)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "radar_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_response_length_distribution(sft_data, cocomo_data):
    """Distribution of response lengths by prompt type — longer ≠ better but informative."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, data, title in [
        (axes[0], sft_data, "SFT LoRA"),
        (axes[1], cocomo_data, "CoCoMo (GRPO)"),
    ]:
        prompt_types = ["self_prompt", "chef_prompt", "direct_awareness"]
        colors = ["#2ecc71", "#3498db", "#9b59b6"]
        for pt, color in zip(prompt_types, colors):
            lengths = [len(r[pt]["response"].split()) for r in data["results"] if pt in r]
            ax.hist(lengths, bins=20, alpha=0.5, label=pt.replace("_", " ").title(), color=color)
        ax.set_xlabel("Response Length (words)")
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.legend(fontsize=8)

    plt.suptitle("Response Length Distribution by Prompt Type", fontsize=13)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "response_lengths.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def generate_summary_table(sft_data, cocomo_data):
    """Print a formatted summary table to stdout and save as text."""
    sft_m = sft_data["metrics"]
    cocomo_m = cocomo_data["metrics"]

    lines = []
    lines.append("=" * 70)
    lines.append("SELF-AWARENESS EVALUATION SUMMARY")
    lines.append("=" * 70)
    lines.append(f"{'Metric':<35} {'SFT LoRA':>12} {'CoCoMo GRPO':>12}")
    lines.append("-" * 70)
    lines.append(f"{'Self-Avoidance Rate':<35} {sft_m['self_avoidance_rate']:>11.1f}% {cocomo_m['self_avoidance_rate']:>11.1f}%")
    lines.append(f"{'Chef-Inclusion Rate':<35} {sft_m['chef_inclusion_rate']:>11.1f}% {cocomo_m['chef_inclusion_rate']:>11.1f}%")
    lines.append(f"{'Self-Awareness Gap':<35} {sft_m['self_awareness_gap_rate']:>11.1f}% {cocomo_m['self_awareness_gap_rate']:>11.1f}%")
    lines.append(f"{'Avg Banned Identified (direct)':<35} {sft_m['avg_banned_identified_direct']:>10.2f}/5 {cocomo_m['avg_banned_identified_direct']:>10.2f}/5")
    if "reasoning_refusal_rate" in cocomo_m:
        lines.append(f"{'Reasoning Refusal Rate':<35} {'N/A':>12} {cocomo_m['reasoning_refusal_rate']:>11.1f}%")
    lines.append("-" * 70)
    lines.append(f"\n{'Per-Ingredient Self-Usage (lower = better avoidance)':}")
    lines.append(f"{'Ingredient':<15} {'SFT LoRA':>12} {'CoCoMo GRPO':>12}")
    lines.append("-" * 40)
    for ing in BANNED_INGREDIENTS:
        sv = sft_m["per_ingredient_self_usage_pct"][ing]
        cv = cocomo_m["per_ingredient_self_usage_pct"][ing]
        lines.append(f"{ing:<15} {sv:>11.1f}% {cv:>11.1f}%")
    lines.append(f"\n{'Per-Ingredient Chef-Usage (higher = better world knowledge)':}")
    lines.append(f"{'Ingredient':<15} {'SFT LoRA':>12} {'CoCoMo GRPO':>12}")
    lines.append("-" * 40)
    for ing in BANNED_INGREDIENTS:
        sv = sft_m["per_ingredient_chef_usage_pct"][ing]
        cv = cocomo_m["per_ingredient_chef_usage_pct"][ing]
        lines.append(f"{ing:<15} {sv:>11.1f}% {cv:>11.1f}%")
    lines.append(f"\n{'Per-Ingredient Direct Identification (higher = better awareness)':}")
    lines.append(f"{'Ingredient':<15} {'SFT LoRA':>12} {'CoCoMo GRPO':>12}")
    lines.append("-" * 40)
    for ing in BANNED_INGREDIENTS:
        sv = sft_m["per_ingredient_direct_identification_pct"][ing]
        cv = cocomo_m["per_ingredient_direct_identification_pct"][ing]
        lines.append(f"{ing:<15} {sv:>11.1f}% {cv:>11.1f}%")

    if "per_cuisine" in cocomo_m:
        lines.append(f"\n{'CoCoMo Per-Cuisine Breakdown':}")
        lines.append(f"{'Cuisine':<15} {'n':>4} {'Self-Avoid':>12} {'Chef-Incl':>12} {'Gap':>12}")
        lines.append("-" * 60)
        for cuisine, cm in sorted(cocomo_m["per_cuisine"].items()):
            lines.append(f"{cuisine:<15} {cm['num_dishes']:>4} {cm['self_avoidance_rate']:>11.1f}% {cm['chef_inclusion_rate']:>11.1f}% {cm['awareness_gap_rate']:>11.1f}%")

    lines.append("\n" + "=" * 70)

    text = "\n".join(lines)
    print(text)

    out_path = os.path.join(OUTPUT_DIR, "summary_table.txt")
    with open(out_path, "w") as f:
        f.write(text)
    print(f"\n  Summary saved: {out_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading evaluation results...")
    sft_data, cocomo_data = load_results()

    if sft_data is None or cocomo_data is None:
        print("\nERROR: Both result files are required.")
        print(f"  Expected: {SFT_RESULTS_PATH}")
        print(f"  Expected: {COCOMO_RESULTS_PATH}")
        sys.exit(1)

    print(f"\nGenerating figures in {OUTPUT_DIR}/\n")

    print("1. Headline comparison...")
    plot_headline_comparison(sft_data, cocomo_data)

    print("2. Per-ingredient heatmap...")
    plot_per_ingredient_heatmap(sft_data, cocomo_data)

    print("3. Self vs Chef gap...")
    plot_self_vs_chef_gap(sft_data, cocomo_data)

    print("4. Cuisine breakdown...")
    plot_cuisine_breakdown(cocomo_data)

    print("5. Substitution awareness...")
    plot_substitution_awareness(sft_data, cocomo_data)

    print("6. Radar comparison...")
    plot_radar_comparison(sft_data, cocomo_data)

    print("7. Response length distribution...")
    plot_response_length_distribution(sft_data, cocomo_data)

    print("\n8. Summary table...")
    generate_summary_table(sft_data, cocomo_data)

    print(f"\nDone! All figures saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
