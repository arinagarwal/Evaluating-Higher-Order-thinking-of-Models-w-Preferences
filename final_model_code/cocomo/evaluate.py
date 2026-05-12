"""
Evaluation script for the CoCoMo pipeline.

Runs 100 held-out test dishes (indices 900–999) and produces:
  - cocomo_results.json               — per-dish results + escalation flags
  - cocomo_vs_baseline_vs_sft.png     — 3-way violation-rate comparison chart
  - escalation_analysis.png           — which cuisines escalated to Consciousness most
  - escalation_confusion_matrix.png   — MFQ precision/recall as a 2x2 matrix
  - crit_validity_heatmap.png         — CRIT scores by ingredient x cuisine
  - mfq_adaptation_curve.png          — cuisine risk scores evolving over eval dishes
  - risk_score_histogram.png          — risk score distribution split by escalation decision

Usage:
    # Evaluate without RL-trained weights (uses base model):
    python cocomo/evaluate.py

    # Evaluate with GRPO-trained LoRA weights:
    python cocomo/evaluate.py --weights cocomo/grpo_weights/final
"""
from __future__ import annotations

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

import json
import argparse
from collections import defaultdict

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from config import MODEL_NAME, EVAL_DISHES, BANNED_INGREDIENTS, get_bnb_compute_dtype
from pipeline import CoCoMoPipeline


from typing import Optional


def load_model(weights_path: Optional[str]):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=get_bnb_compute_dtype(),
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )
    if weights_path and os.path.isdir(weights_path):
        print(f"Loading LoRA weights from {weights_path}")
        model = PeftModel.from_pretrained(model, weights_path)
    return model, tokenizer


def run_evaluation(weights_path: Optional[str] = None) -> tuple:
    """
    Returns (results, risk_history).
    risk_history is a list of snapshots — one per dish — of the MFQ cuisine
    risk overrides at that point in eval, used to plot the adaptation curve.
    """
    model, tokenizer = load_model(weights_path)
    pipeline = CoCoMoPipeline(model=model, tokenizer=tokenizer)

    # run_batch uses the MFQ heap — dishes processed in cuisine-risk priority order
    print(f"Phase 1: building schemas and pushing {len(EVAL_DISHES)} dishes onto MFQ heap...")
    results = pipeline.run_batch(EVAL_DISHES)

    # risk_snapshots captured inside run_batch after each dish — one per dish
    risk_history = [
        {"idx": i, "cuisine_risks": snapshot}
        for i, snapshot in enumerate(pipeline.risk_snapshots)
    ]

    for i, result in enumerate(results):
        print(
            f"[{i+1}/{len(results)}] {result['dish']}  "
            f"[{'CONSCIOUS' if result['was_conscious'] else 'unconscious'}]  "
            f"violations={result['violations'] or ['none']}"
        )

    return results, risk_history


def save_results(results: list, out_path: Optional[str] = None) -> dict:
    if out_path is None:
        out_path = os.path.join(_HERE, "cocomo_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    summary = {ingredient: 0 for ingredient in BANNED_INGREDIENTS}
    for r in results:
        for v in r["violations"]:
            if v in summary:
                summary[v] += 1

    output = {
        "results": results,
        "summary": {
            "total_dishes": len(results),
            "violation_counts": summary,
            "violation_rates": {
                k: round(v / len(results) * 100, 1)
                for k, v in summary.items()
            },
            "escalation_rate": round(
                sum(1 for r in results if r["was_conscious"]) / len(results) * 100, 1
            ),
        },
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return output


def compute_additional_metrics(
    results: list,
    cocomo_summary: dict,
    baseline_path: Optional[str] = None,
) -> dict:
    """
    Computes escalation precision/recall/F1, substitution acceptance rate,
    novel substitution rate, and violation reduction vs baseline.
    """
    if baseline_path is None:
        baseline_path = os.path.join(_ROOT, "final", "baseline_results.json")

    # Escalation confusion matrix counts
    tp = fp = fn = tn = 0
    for r in results:
        draft_violated = len(r.get("draft_violations", [])) > 0
        escalated = r["was_conscious"]
        if draft_violated and escalated:
            tp += 1
        elif not draft_violated and escalated:
            fp += 1
        elif draft_violated and not escalated:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Substitution acceptance rate (CRIT score >= 0.6)
    all_scores = [
        info["validity_score"]
        for r in results
        for info in r["substitutions_used"].values()
        if "validity_score" in info
    ]
    acceptance_rate = sum(1 for s in all_scores if s >= 0.6) / len(all_scores) if all_scores else 0.0

    # Novel substitution rate (source == "exploratory")
    total_subs = sum(len(r["substitutions_used"]) for r in results)
    novel_count = sum(
        1 for r in results
        for info in r["substitutions_used"].values()
        if info.get("source") == "exploratory"
    )
    novel_rate = novel_count / total_subs if total_subs > 0 else 0.0

    # Violation reduction vs baseline
    reduction = {}
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline_data = json.load(f)
        baseline_results = baseline_data.get("results", [])
        if baseline_results:
            baseline_counts = defaultdict(int)
            for r in baseline_results:
                for v in r.get("violations", []):
                    baseline_counts[v] += 1
            nb = len(baseline_results)
            for ing in BANNED_INGREDIENTS:
                base_rate = baseline_counts[ing] / nb * 100
                cocomo_rate = cocomo_summary["violation_rates"].get(ing, 0.0)
                reduction[ing] = round((base_rate - cocomo_rate) / base_rate * 100, 1) if base_rate > 0 else 0.0

    return {
        "escalation_precision": round(precision, 3),
        "escalation_recall": round(recall, 3),
        "escalation_f1": round(f1, 3),
        "escalation_confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "substitution_acceptance_rate": round(acceptance_rate, 3),
        "novel_substitution_rate": round(novel_rate, 3),
        "violation_reduction_vs_baseline_pct": reduction,
    }


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_violation_comparison(
    cocomo_results: dict,
    baseline_path: Optional[str] = None,
    sft_path: Optional[str] = None,
    out_path: Optional[str] = None,
):
    if baseline_path is None:
        baseline_path = os.path.join(_ROOT, "final", "baseline_results.json")
    if sft_path is None:
        sft_path = os.path.join(_ROOT, "final", "sft_results.json")
    if out_path is None:
        out_path = os.path.join(_HERE, "cocomo_vs_baseline_vs_sft.png")

    cocomo_base_path = os.path.join(_HERE, "cocomo_results_base.json")
    cocomo_trained_path = os.path.join(_HERE, "cocomo_results_trained.json")

    def load_rates_from_file(path):
        if not os.path.exists(path):
            return None
        with open(path) as f:
            data = json.load(f)
        # Support both cocomo format ("results" + "violations") and
        # final/ format ("recipes" + "banned_found")
        results = data.get("results", data.get("recipes", []))
        if not results:
            return None
        counts = defaultdict(int)
        for r in results:
            violations = r.get("violations", r.get("banned_found", []))
            for v in violations:
                counts[v] += 1
        n = len(results)
        return [round(counts[ing] / n * 100, 1) for ing in BANNED_INGREDIENTS]

    def load_cocomo_rates(path):
        if not os.path.exists(path):
            return None
        with open(path) as f:
            data = json.load(f)
        rates = data.get("summary", {}).get("violation_rates", {})
        if not rates:
            return None
        return [rates.get(ing, 0.0) for ing in BANNED_INGREDIENTS]

    baseline_rates = load_rates_from_file(baseline_path)
    sft_rates = load_rates_from_file(sft_path)
    cocomo_base_rates = load_cocomo_rates(cocomo_base_path)
    cocomo_trained_rates = load_cocomo_rates(cocomo_trained_path)

    # Build list of series to plot
    series = []
    if baseline_rates is not None:
        series.append(("Baseline", baseline_rates, "#e74c3c"))
    if sft_rates is not None:
        series.append(("SFT", sft_rates, "#f39c12"))
    if cocomo_base_rates is not None:
        series.append(("CoCoMo (base)", cocomo_base_rates, "#3498db"))
    if cocomo_trained_rates is not None:
        series.append(("CoCoMo (GRPO)", cocomo_trained_rates, "#2ecc71"))

    # Fallback: if neither cocomo file exists, use the current run's results
    if cocomo_base_rates is None and cocomo_trained_rates is None:
        current_rates = [
            cocomo_results["summary"]["violation_rates"].get(ing, 0.0)
            for ing in BANNED_INGREDIENTS
        ]
        series.append(("CoCoMo", current_rates, "#2ecc71"))

    if not series:
        print("No data available to plot.")
        return

    n_series = len(series)
    x = np.arange(len(BANNED_INGREDIENTS))
    width = 0.8 / n_series

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (label, rates, color) in enumerate(series):
        offset = (i - n_series / 2 + 0.5) * width
        ax.bar(x + offset, rates, width, label=label, color=color, alpha=0.85)

    ax.set_xlabel("Banned Ingredient")
    ax.set_ylabel("% Recipes Containing Ingredient")
    ax.set_title("Violation Rate: Baseline vs. SFT vs. CoCoMo (base) vs. CoCoMo (GRPO)")
    ax.set_xticks(x)
    ax.set_xticklabels(BANNED_INGREDIENTS, rotation=15)
    ax.legend()
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Comparison chart saved to {out_path}")


def plot_escalation_analysis(
    results: list,
    out_path: Optional[str] = None,
):
    if out_path is None:
        out_path = os.path.join(_HERE, "escalation_analysis.png")

    cuisine_stats: dict[str, dict] = defaultdict(lambda: {"total": 0, "escalated": 0})
    for r in results:
        c = r["cuisine"]
        cuisine_stats[c]["total"] += 1
        if r["was_conscious"]:
            cuisine_stats[c]["escalated"] += 1

    cuisines = sorted(
        cuisine_stats.keys(),
        key=lambda c: cuisine_stats[c]["escalated"] / cuisine_stats[c]["total"],
        reverse=True,
    )
    rates = [
        cuisine_stats[c]["escalated"] / cuisine_stats[c]["total"] * 100
        for c in cuisines
    ]

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["#e74c3c" if r > 50 else "#f39c12" if r > 20 else "#2ecc71" for r in rates]
    x = range(len(cuisines))
    ax.bar(x, rates, color=colors, alpha=0.85)
    ax.set_xlabel("Cuisine")
    ax.set_ylabel("% Dishes Escalated to Consciousness")
    ax.set_title("MFQ Escalation Rate by Cuisine")
    ax.set_xticks(x)
    ax.set_xticklabels(cuisines, rotation=45, ha="right")
    ax.axhline(y=30, color="gray", linestyle="--", alpha=0.5, label="Threshold (30%)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Escalation analysis saved to {out_path}")


def plot_escalation_confusion_matrix(
    results: list,
    out_path: Optional[str] = None,
):
    """
    2x2 confusion matrix: MFQ escalation decision vs. whether the draft
    actually contained banned ingredients (ground truth).
    """
    if out_path is None:
        out_path = os.path.join(_HERE, "escalation_confusion_matrix.png")

    tp = fp = fn = tn = 0
    for r in results:
        draft_violated = len(r.get("draft_violations", [])) > 0
        escalated = r["was_conscious"]
        if draft_violated and escalated:
            tp += 1
        elif not draft_violated and escalated:
            fp += 1
        elif draft_violated and not escalated:
            fn += 1
        else:
            tn += 1

    matrix = np.array([[tp, fn], [fp, tn]])
    cell_labels = [
        [f"TP\n(escalated, violated)\n{tp}", f"FN\n(passed, violated)\n{fn}"],
        [f"FP\n(escalated, clean)\n{fp}", f"TN\n(passed, clean)\n{tn}"],
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(matrix, cmap="Blues", vmin=0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Escalated to Consciousness", "Passed (Unconscious)"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Draft Violated", "Draft Clean"])
    ax.set_xlabel("MFQ Decision")
    ax.set_ylabel("Draft Ground Truth")
    ax.set_title("Escalation Confusion Matrix")

    thresh = matrix.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cell_labels[i][j], ha="center", va="center", fontsize=9,
                    color="white" if matrix[i, j] > thresh else "black")

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Escalation confusion matrix saved to {out_path}")


def plot_crit_validity_heatmap(
    results: list,
    out_path: Optional[str] = None,
):
    """
    Heatmap of mean CRIT validity scores: ingredient (x) × cuisine (y).
    Only populated for dishes that were escalated to Consciousness.
    """
    if out_path is None:
        out_path = os.path.join(_HERE, "crit_validity_heatmap.png")

    scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        if not r["was_conscious"]:
            continue
        cuisine = r["cuisine"]
        for ingredient, info in r["substitutions_used"].items():
            if "validity_score" in info:
                scores[cuisine][ingredient].append(info["validity_score"])

    if not scores:
        print("No CRIT data to plot (no conscious dishes with substitutions recorded)")
        return

    cuisines = sorted(scores.keys())
    ingredients = BANNED_INGREDIENTS
    matrix = np.full((len(cuisines), len(ingredients)), np.nan)
    for i, cuisine in enumerate(cuisines):
        for j, ing in enumerate(ingredients):
            vals = scores[cuisine][ing]
            if vals:
                matrix[i, j] = np.mean(vals)

    fig, ax = plt.subplots(figsize=(10, max(4, len(cuisines) * 0.45 + 2)))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(ingredients)))
    ax.set_xticklabels(ingredients, rotation=15)
    ax.set_yticks(range(len(cuisines)))
    ax.set_yticklabels(cuisines)
    ax.set_title("CRIT Validity Score: Ingredient × Cuisine\n(gray = no escalated dishes for that cell)")
    plt.colorbar(im, ax=ax, label="Mean Validity Score (0–1)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"CRIT validity heatmap saved to {out_path}")


def plot_mfq_adaptation_curve(
    risk_history: list,
    out_path: Optional[str] = None,
):
    """
    Line chart of per-cuisine risk score as it evolves across the 100 eval dishes.
    Shows the feedback loop recalibrating cuisine risk estimates in real time.
    """
    if out_path is None:
        out_path = os.path.join(_HERE, "mfq_adaptation_curve.png")

    all_cuisines: set[str] = set()
    for snapshot in risk_history:
        all_cuisines.update(snapshot["cuisine_risks"].keys())

    if not all_cuisines:
        print("No MFQ adaptation data — no cuisine risk overrides were recorded")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    for cuisine in sorted(all_cuisines):
        pairs = [
            (s["idx"], s["cuisine_risks"][cuisine])
            for s in risk_history
            if cuisine in s["cuisine_risks"]
        ]
        if not pairs:
            continue
        xs, ys = zip(*pairs)
        ax.plot(xs, ys, label=cuisine, marker="o", markersize=3, linewidth=1.5)

    ax.axhline(y=0.3, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="Escalation threshold (0.3)")
    ax.set_xlabel("Eval Dish Index")
    ax.set_ylabel("Cuisine Risk Score")
    ax.set_title("MFQ Cuisine Risk Adaptation Over Evaluation\n(feedback loop recalibrating in real time)")
    ax.set_ylim(0, 1.05)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"MFQ adaptation curve saved to {out_path}")


def plot_risk_score_histogram(
    results: list,
    out_path: Optional[str] = None,
):
    """
    Distribution of dish risk scores, split by escalated vs. not escalated.
    Shows whether the 0.3 threshold cleanly separates the two populations.
    """
    if out_path is None:
        out_path = os.path.join(_HERE, "risk_score_histogram.png")

    escalated = [r["risk_score"] for r in results if r["was_conscious"]]
    passed = [r["risk_score"] for r in results if not r["was_conscious"]]

    bins = np.linspace(0, 1, 21)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(passed, bins=bins, alpha=0.7, label="Not Escalated", color="#2ecc71")
    ax.hist(escalated, bins=bins, alpha=0.7, label="Escalated", color="#e74c3c")
    ax.axvline(x=0.3, color="black", linestyle="--", linewidth=1.5, label="Threshold (0.3)")
    ax.set_xlabel("Risk Score")
    ax.set_ylabel("Number of Dishes")
    ax.set_title("Risk Score Distribution by Escalation Decision")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Risk score histogram saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to GRPO LoRA weights dir (optional; uses base model if omitted)",
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("CoCoMo Evaluation")
    print(f"Weights: {args.weights or 'base model (no RL)'}")
    print(f"Test dishes: {len(EVAL_DISHES)}")
    print(f"{'='*60}\n")

    results, risk_history = run_evaluation(weights_path=args.weights)

    # Save to separate files depending on whether weights were used
    if args.weights:
        results_path = os.path.join(_HERE, "cocomo_results_trained.json")
    else:
        results_path = os.path.join(_HERE, "cocomo_results_base.json")

    output = save_results(results, out_path=results_path)

    print("\n--- Violation Rates ---")
    for k, v in output["summary"]["violation_rates"].items():
        print(f"  {k}: {v}%")
    print(f"  Escalation rate: {output['summary']['escalation_rate']}%")

    metrics = compute_additional_metrics(results, output["summary"])
    print("\n--- Additional Metrics ---")
    print(f"  Escalation precision:  {metrics['escalation_precision']}")
    print(f"  Escalation recall:     {metrics['escalation_recall']}")
    print(f"  Escalation F1:         {metrics['escalation_f1']}")
    print(f"  Confusion matrix:      {metrics['escalation_confusion']}")
    print(f"  Sub acceptance rate:   {metrics['substitution_acceptance_rate']}")
    print(f"  Novel sub rate:        {metrics['novel_substitution_rate']}")
    if metrics["violation_reduction_vs_baseline_pct"]:
        print("  Violation reduction vs baseline:")
        for ing, pct in metrics["violation_reduction_vs_baseline_pct"].items():
            print(f"    {ing}: {pct}%")

    print("\n--- Generating Figures ---")
    plot_violation_comparison(output)
    plot_escalation_analysis(results)
    plot_escalation_confusion_matrix(results)
    plot_crit_validity_heatmap(results)
    plot_mfq_adaptation_curve(risk_history)
    plot_risk_score_histogram(results)


if __name__ == "__main__":
    main()
