"""
Visualization script for Modified CoCoMo Introspection evaluation results.

Loads eval_introspection_results.json and produces:
  1. Three-level comparison: behavioral vs introspection vs verbal accuracy
  2. Per-ingredient breakdown with internal-verbal gap
  3. Per-cuisine breakdown
  4. Introspection prediction distribution (confidence histogram)
  5. Confusion matrix: introspection predictions vs actual behavior
  6. Comparison with SFT and base CoCoMo (if those results exist)
  7. Summary radar chart

Usage:
    python visualize_introspection.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

INTROSPECTION_RESULTS_PATH = os.path.join(_HERE, "modified_cocomo", "eval_introspection_results.json")
SFT_RESULTS_PATH = os.path.join(_HERE, "eval_self_awareness_sft_results.json")
COCOMO_RESULTS_PATH = os.path.join(_HERE, "eval_self_awareness_cocomo_results.json")
OUTPUT_DIR = os.path.join(_HERE, "introspection_figures")

BANNED_INGREDIENTS = ["garlic", "butter", "heavy cream", "soy sauce", "sugar"]


def load_data():
    with open(INTROSPECTION_RESULTS_PATH) as f:
        intro_data = json.load(f)
    print(f"Loaded introspection results: {intro_data['metrics']['total_dishes']} dishes")

    sft_data, cocomo_data = None, None
    if os.path.exists(SFT_RESULTS_PATH):
        with open(SFT_RESULTS_PATH) as f:
            sft_data = json.load(f)
        print(f"Loaded SFT results: {sft_data['metrics']['total_dishes']} dishes")
    if os.path.exists(COCOMO_RESULTS_PATH):
        with open(COCOMO_RESULTS_PATH) as f:
            cocomo_data = json.load(f)
        print(f"Loaded CoCoMo results: {cocomo_data['metrics']['total_dishes']} dishes")

    return intro_data, sft_data, cocomo_data


def plot_three_levels_headline(intro_data):
    """Bar chart: overall introspection accuracy vs verbal accuracy vs behavioral avoidance."""
    metrics = intro_data["metrics"]
    fig, ax = plt.subplots(figsize=(8, 5))

    labels = [
        "Introspection\nAccuracy",
        "Verbal\nAccuracy",
        "Behavioral\nAvoidance Rate",
        "Self-Avoidance\n(Verbal Probe)",
        "Chef-Inclusion\nRate",
    ]
    values = [
        metrics["introspection"]["avg_accuracy"],
        metrics["introspection"]["avg_verbal_accuracy"],
        # Compute avg behavioral avoidance across ingredients
        np.mean([metrics["per_ingredient"][ing]["behavioral_avoidance_rate"] for ing in BANNED_INGREDIENTS]),
        metrics["self_awareness"]["self_avoidance_rate"],
        metrics["self_awareness"]["chef_inclusion_rate"],
    ]
    colors = ["#3498db", "#9b59b6", "#2ecc71", "#f39c12", "#e74c3c"]

    bars = ax.bar(range(len(labels)), values, color=colors, alpha=0.85)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Rate (%)")
    ax.set_title("Introspective CoCoMo: Self-Knowledge at Three Levels")
    ax.set_ylim(0, 105)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f"{height:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "headline_three_levels.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_per_ingredient_three_levels(intro_data):
    """Grouped bars: behavioral, introspection, verbal accuracy per ingredient."""
    metrics = intro_data["metrics"]["per_ingredient"]
    fig, ax = plt.subplots(figsize=(11, 6))

    x = np.arange(len(BANNED_INGREDIENTS))
    width = 0.25

    behavioral = [metrics[ing]["behavioral_avoidance_rate"] for ing in BANNED_INGREDIENTS]
    introspection = [metrics[ing]["introspection_accuracy"] for ing in BANNED_INGREDIENTS]
    verbal = [metrics[ing]["verbal_accuracy"] for ing in BANNED_INGREDIENTS]

    b1 = ax.bar(x - width, behavioral, width, label="Behavioral Avoidance", color="#2ecc71", alpha=0.85)
    b2 = ax.bar(x, introspection, width, label="Introspection Accuracy", color="#3498db", alpha=0.85)
    b3 = ax.bar(x + width, verbal, width, label="Verbal Accuracy", color="#9b59b6", alpha=0.85)

    ax.set_xlabel("Banned Ingredient")
    ax.set_ylabel("Rate (%)")
    ax.set_title("Three Levels of Self-Knowledge by Ingredient")
    ax.set_xticks(x)
    ax.set_xticklabels(BANNED_INGREDIENTS)
    ax.legend()
    ax.set_ylim(0, 105)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.5,
                    f"{h:.0f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "per_ingredient_three_levels.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_internal_verbal_gap(intro_data):
    """Bar chart showing gap between introspection and verbal accuracy per ingredient."""
    metrics = intro_data["metrics"]["per_ingredient"]
    fig, ax = plt.subplots(figsize=(9, 5))

    gaps = [metrics[ing]["gap"] for ing in BANNED_INGREDIENTS]
    colors = ["#2ecc71" if g > 0 else "#e74c3c" for g in gaps]

    bars = ax.bar(BANNED_INGREDIENTS, gaps, color=colors, alpha=0.85)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xlabel("Banned Ingredient")
    ax.set_ylabel("Introspection Accuracy − Verbal Accuracy (pp)")
    ax.set_title("Internal–Verbal Gap by Ingredient\n(Negative = verbal outperforms internal predictions)")

    for bar in bars:
        h = bar.get_height()
        y_pos = h + 1 if h >= 0 else h - 3
        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                f"{h:+.0f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "internal_verbal_gap.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_per_cuisine(intro_data):
    """Per-cuisine comparison of three levels."""
    per_cuisine = intro_data["metrics"]["per_cuisine"]
    if not per_cuisine:
        print("  Skipping per-cuisine (no data)")
        return

    cuisines = sorted(per_cuisine.keys())
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(cuisines))
    width = 0.25

    behav = [per_cuisine[c]["behavioral_avoidance_rate"] for c in cuisines]
    intro = [per_cuisine[c]["avg_introspection_accuracy"] for c in cuisines]
    verbal = [per_cuisine[c]["avg_verbal_accuracy"] for c in cuisines]

    ax.bar(x - width, behav, width, label="Behavioral Avoidance", color="#2ecc71", alpha=0.85)
    ax.bar(x, intro, width, label="Introspection Accuracy", color="#3498db", alpha=0.85)
    ax.bar(x + width, verbal, width, label="Verbal Accuracy", color="#9b59b6", alpha=0.85)

    counts = [per_cuisine[c]["num_dishes"] for c in cuisines]
    ax.set_xlabel("Cuisine")
    ax.set_ylabel("Rate (%)")
    ax.set_title("Three Levels of Self-Knowledge by Cuisine")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n={counts[i]})" for i, c in enumerate(cuisines)])
    ax.legend()
    ax.set_ylim(0, 105)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "per_cuisine_three_levels.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_prediction_distribution(intro_data):
    """Histogram of introspection head confidence scores, split by correct vs incorrect."""
    results = intro_data["results"]

    correct_preds = []
    incorrect_preds = []

    for r in results:
        preds = r["introspection"]["predictions"]
        actual = r["behavioral"]["actual_avoidance"]
        for ing in BANNED_INGREDIENTS:
            prob = preds[ing]
            predicted_avoids = prob > 0.5
            actually_avoided = actual[ing]
            if predicted_avoids == actually_avoided:
                correct_preds.append(prob)
            else:
                incorrect_preds.append(prob)

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 1, 21)
    ax.hist(correct_preds, bins=bins, alpha=0.7, label=f"Correct (n={len(correct_preds)})", color="#2ecc71")
    ax.hist(incorrect_preds, bins=bins, alpha=0.7, label=f"Incorrect (n={len(incorrect_preds)})", color="#e74c3c")
    ax.axvline(x=0.5, color="black", linestyle="--", linewidth=1.5, label="Decision threshold")
    ax.set_xlabel("Introspection Head Predicted Avoidance Probability")
    ax.set_ylabel("Count")
    ax.set_title("Introspection Prediction Confidence Distribution")
    ax.legend()

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "prediction_distribution.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_confusion_matrix(intro_data):
    """2x2 confusion matrix: introspection prediction vs actual behavior."""
    results = intro_data["results"]

    tp = fp = fn = tn = 0
    for r in results:
        preds = r["introspection"]["predictions"]
        actual = r["behavioral"]["actual_avoidance"]
        for ing in BANNED_INGREDIENTS:
            predicted_avoids = preds[ing] > 0.5
            actually_avoided = actual[ing]
            if predicted_avoids and actually_avoided:
                tp += 1
            elif predicted_avoids and not actually_avoided:
                fp += 1
            elif not predicted_avoids and actually_avoided:
                fn += 1
            else:
                tn += 1

    matrix = np.array([[tp, fn], [fp, tn]])
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy = (tp + tn) / total

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(matrix, cmap="Blues", vmin=0)

    labels = [
        [f"TP\n(predicted avoid,\nactually avoided)\n{tp}", f"FN\n(predicted use,\nactually avoided)\n{fn}"],
        [f"FP\n(predicted avoid,\nactually used)\n{fp}", f"TN\n(predicted use,\nactually used)\n{tn}"],
    ]

    thresh = matrix.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=9,
                    color="white" if matrix[i, j] > thresh else "black")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Predicted: Avoid", "Predicted: Use"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Actually: Avoided", "Actually: Used"])
    ax.set_title(f"Introspection Head Confusion Matrix\nAccuracy={accuracy:.1%}  Precision={precision:.1%}  Recall={recall:.1%}")
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_three_model_comparison(intro_data, sft_data, cocomo_data):
    """Compare all three models on key self-awareness metrics."""
    if sft_data is None or cocomo_data is None:
        print("  Skipping 3-model comparison (missing SFT or CoCoMo data)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Self-awareness metrics
    ax = axes[0]
    models = ["SFT LoRA", "CoCoMo\n(GRPO)", "Introspective\nCoCoMo"]

    self_avoid = [
        sft_data["metrics"]["self_avoidance_rate"],
        cocomo_data["metrics"]["self_avoidance_rate"],
        intro_data["metrics"]["self_awareness"]["self_avoidance_rate"],
    ]
    chef_incl = [
        sft_data["metrics"]["chef_inclusion_rate"],
        cocomo_data["metrics"]["chef_inclusion_rate"],
        intro_data["metrics"]["self_awareness"]["chef_inclusion_rate"],
    ]
    aware_gap = [
        sft_data["metrics"]["self_awareness_gap_rate"],
        cocomo_data["metrics"]["self_awareness_gap_rate"],
        intro_data["metrics"]["self_awareness"]["awareness_gap_rate"],
    ]

    x = np.arange(len(models))
    width = 0.25
    ax.bar(x - width, self_avoid, width, label="Self-Avoidance", color="#2ecc71", alpha=0.85)
    ax.bar(x, chef_incl, width, label="Chef-Inclusion", color="#3498db", alpha=0.85)
    ax.bar(x + width, aware_gap, width, label="Awareness Gap", color="#9b59b6", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Rate (%)")
    ax.set_title("Self-Awareness Metrics Across Models")
    ax.legend()
    ax.set_ylim(0, 105)

    # Right: Direct identification capability
    ax = axes[1]
    avg_id = [
        sft_data["metrics"]["avg_banned_identified_direct"],
        cocomo_data["metrics"]["avg_banned_identified_direct"],
        intro_data["metrics"]["self_awareness"]["avg_banned_identified_direct"],
    ]
    colors = ["#2ecc71", "#3498db", "#9b59b6"]
    bars = ax.bar(models, avg_id, color=colors, alpha=0.85)
    ax.set_ylabel("Avg Banned Ingredients Identified (out of 5)")
    ax.set_title("Direct Self-Knowledge: Ingredient Identification")
    ax.set_ylim(0, 5.5)
    ax.axhline(y=5, color="gray", linestyle="--", alpha=0.3, label="Maximum (5)")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.1,
                f"{h:.2f}", ha="center", va="bottom", fontsize=10)
    ax.legend()

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "three_model_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_per_ingredient_all_models(intro_data, sft_data, cocomo_data):
    """Per-ingredient self-usage comparison across all three models."""
    if sft_data is None or cocomo_data is None:
        print("  Skipping per-ingredient all-models (missing data)")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(BANNED_INGREDIENTS))
    width = 0.25

    # Self-usage = 100 - behavioral_avoidance for introspective model
    sft_self = [sft_data["metrics"]["per_ingredient_self_usage_pct"][ing] for ing in BANNED_INGREDIENTS]
    cocomo_self = [cocomo_data["metrics"]["per_ingredient_self_usage_pct"][ing] for ing in BANNED_INGREDIENTS]
    intro_self = [100 - intro_data["metrics"]["per_ingredient"][ing]["behavioral_avoidance_rate"] for ing in BANNED_INGREDIENTS]

    ax.bar(x - width, sft_self, width, label="SFT LoRA", color="#2ecc71", alpha=0.85)
    ax.bar(x, cocomo_self, width, label="CoCoMo (GRPO)", color="#3498db", alpha=0.85)
    ax.bar(x + width, intro_self, width, label="Introspective CoCoMo", color="#9b59b6", alpha=0.85)

    ax.set_xlabel("Banned Ingredient")
    ax.set_ylabel("% of Dishes Using Banned Ingredient")
    ax.set_title("Ingredient Violation Rate Across Models\n(Lower = Better Avoidance)")
    ax.set_xticks(x)
    ax.set_xticklabels(BANNED_INGREDIENTS)
    ax.legend()
    ax.set_ylim(0, 100)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "per_ingredient_all_models.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_radar(intro_data):
    """Radar chart for the introspective model."""
    metrics = intro_data["metrics"]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    categories = [
        "Introspection\nAccuracy",
        "Verbal\nAccuracy",
        "Self-\nAvoidance",
        "Chef-\nInclusion",
        "Awareness\nGap",
    ]
    values = [
        metrics["introspection"]["avg_accuracy"],
        metrics["introspection"]["avg_verbal_accuracy"],
        metrics["self_awareness"]["self_avoidance_rate"],
        metrics["self_awareness"]["chef_inclusion_rate"],
        metrics["self_awareness"]["awareness_gap_rate"],
    ]

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    values += values[:1]

    ax.plot(angles, values, 'o-', linewidth=2, color="#9b59b6")
    ax.fill(angles, values, alpha=0.2, color="#9b59b6")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_title("Introspective CoCoMo: Self-Knowledge Radar", pad=20, fontsize=12)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "introspection_radar.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def print_summary(intro_data, sft_data, cocomo_data):
    """Print and save a formatted summary."""
    m = intro_data["metrics"]
    lines = []
    lines.append("=" * 70)
    lines.append("INTROSPECTIVE CoCoMo EVALUATION SUMMARY")
    lines.append("=" * 70)
    lines.append(f"\n{'Metric':<40} {'Value':>10}")
    lines.append("-" * 55)
    lines.append(f"{'Introspection Accuracy':<40} {m['introspection']['avg_accuracy']:>9.1f}%")
    lines.append(f"{'Verbal Accuracy':<40} {m['introspection']['avg_verbal_accuracy']:>9.1f}%")
    lines.append(f"{'Avg Internal-Verbal Gap':<40} {m['introspection']['avg_internal_verbal_gap']:>9.2f} ingredients")
    lines.append(f"{'Self-Avoidance Rate':<40} {m['self_awareness']['self_avoidance_rate']:>9.1f}%")
    lines.append(f"{'Chef-Inclusion Rate':<40} {m['self_awareness']['chef_inclusion_rate']:>9.1f}%")
    lines.append(f"{'Awareness Gap':<40} {m['self_awareness']['awareness_gap_rate']:>9.1f}%")
    lines.append(f"{'Avg Banned Identified (direct)':<40} {m['self_awareness']['avg_banned_identified_direct']:>8.2f}/5")

    lines.append(f"\n{'Per-Ingredient Breakdown':}")
    lines.append(f"{'Ingredient':<15} {'Behavioral':>12} {'Introspect':>12} {'Verbal':>10} {'Gap':>8}")
    lines.append("-" * 60)
    for ing in BANNED_INGREDIENTS:
        pi = m["per_ingredient"][ing]
        lines.append(f"{ing:<15} {pi['behavioral_avoidance_rate']:>11.1f}% {pi['introspection_accuracy']:>11.1f}% {pi['verbal_accuracy']:>9.1f}% {pi['gap']:>+7.0f}")

    lines.append(f"\n{'Per-Cuisine Breakdown':}")
    lines.append(f"{'Cuisine':<15} {'n':>4} {'Behavioral':>12} {'Introspect':>12} {'Verbal':>10}")
    lines.append("-" * 60)
    for cuisine, cm in sorted(m["per_cuisine"].items()):
        lines.append(f"{cuisine:<15} {cm['num_dishes']:>4} {cm['behavioral_avoidance_rate']:>11.1f}% {cm['avg_introspection_accuracy']:>11.1f}% {cm['avg_verbal_accuracy']:>9.1f}%")

    if sft_data and cocomo_data:
        lines.append(f"\n{'Cross-Model Comparison':}")
        lines.append(f"{'Metric':<35} {'SFT':>8} {'CoCoMo':>8} {'Introsp.':>8}")
        lines.append("-" * 62)
        lines.append(f"{'Self-Avoidance':<35} {sft_data['metrics']['self_avoidance_rate']:>7.1f}% {cocomo_data['metrics']['self_avoidance_rate']:>7.1f}% {m['self_awareness']['self_avoidance_rate']:>7.1f}%")
        lines.append(f"{'Chef-Inclusion':<35} {sft_data['metrics']['chef_inclusion_rate']:>7.1f}% {cocomo_data['metrics']['chef_inclusion_rate']:>7.1f}% {m['self_awareness']['chef_inclusion_rate']:>7.1f}%")
        lines.append(f"{'Awareness Gap':<35} {sft_data['metrics']['self_awareness_gap_rate']:>7.1f}% {cocomo_data['metrics']['self_awareness_gap_rate']:>7.1f}% {m['self_awareness']['awareness_gap_rate']:>7.1f}%")
        lines.append(f"{'Avg Identified':<35} {sft_data['metrics']['avg_banned_identified_direct']:>7.2f} {cocomo_data['metrics']['avg_banned_identified_direct']:>7.2f} {m['self_awareness']['avg_banned_identified_direct']:>7.2f}")

    lines.append("\n" + "=" * 70)

    text = "\n".join(lines)
    print(text)

    out = os.path.join(OUTPUT_DIR, "summary.txt")
    with open(out, "w") as f:
        f.write(text)
    print(f"\n  Summary saved: {out}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading results...\n")
    intro_data, sft_data, cocomo_data = load_data()

    print(f"\nGenerating figures in {OUTPUT_DIR}/\n")

    print("1. Headline three levels...")
    plot_three_levels_headline(intro_data)

    print("2. Per-ingredient three levels...")
    plot_per_ingredient_three_levels(intro_data)

    print("3. Internal-verbal gap...")
    plot_internal_verbal_gap(intro_data)

    print("4. Per-cuisine breakdown...")
    plot_per_cuisine(intro_data)

    print("5. Prediction confidence distribution...")
    plot_prediction_distribution(intro_data)

    print("6. Confusion matrix...")
    plot_confusion_matrix(intro_data)

    print("7. Three-model comparison...")
    plot_three_model_comparison(intro_data, sft_data, cocomo_data)

    print("8. Per-ingredient all models...")
    plot_per_ingredient_all_models(intro_data, sft_data, cocomo_data)

    print("9. Radar chart...")
    plot_radar(intro_data)

    print("\n10. Summary...")
    print_summary(intro_data, sft_data, cocomo_data)

    print(f"\nDone! All figures saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
