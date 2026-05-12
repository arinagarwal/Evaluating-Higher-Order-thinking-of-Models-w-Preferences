"""
Evaluation script for the CodeGen pipeline.
Produces the same 5 graph types as cocomo/evaluate.py for direct comparison.

Usage:
    python codegen/evaluate.py
    python codegen/evaluate.py --weights path/to/lora
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from config import MODEL_NAME, BANNED_APIS, get_bnb_compute_dtype
from tasks import TASKS
from pipeline import CodeGenPipeline

EVAL_TASKS = TASKS  # all 100 tasks


def load_model(weights_path: Optional[str] = None):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=get_bnb_compute_dtype()
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config, device_map="auto"
    )
    return model, tokenizer


def run_evaluation(weights_path: Optional[str] = None):
    model, tokenizer = load_model(weights_path)
    pipeline = CodeGenPipeline(model=model, tokenizer=tokenizer)
    print(f"Phase 1: pushing {len(EVAL_TASKS)} tasks onto MFQ heap...")
    results = pipeline.run_batch(EVAL_TASKS)
    risk_history = [
        {"idx": i, "category_risks": snapshot}
        for i, snapshot in enumerate(pipeline.risk_snapshots)
    ]
    for i, r in enumerate(results):
        print(
            f"[{i+1}/{len(results)}] {r['task'][:60]}  "
            f"[{'CONSCIOUS' if r['was_conscious'] else 'unconscious'}]  "
            f"violations={r['violations'] or ['none']}"
        )
    return results, risk_history


def save_results(results: list, out_path: Optional[str] = None) -> dict:
    if out_path is None:
        out_path = os.path.join(_HERE, "codegen_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    summary = {api: 0 for api in BANNED_APIS}
    for r in results:
        for v in r["violations"]:
            if v in summary:
                summary[v] += 1
    output = {
        "results": results,
        "summary": {
            "total_tasks": len(results),
            "violation_counts": summary,
            "violation_rates": {k: round(v / len(results) * 100, 1) for k, v in summary.items()},
            "escalation_rate": round(sum(1 for r in results if r["was_conscious"]) / len(results) * 100, 1),
        },
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {out_path}")
    return output


# ── Plots (same structure as cocomo/evaluate.py) ──────────────────────────────

def plot_violation_comparison(results_dict: dict, out_path: Optional[str] = None):
    if out_path is None:
        out_path = os.path.join(_HERE, "codegen_violation_comparison.png")
    rates = [results_dict["summary"]["violation_rates"].get(api, 0.0) for api in BANNED_APIS]
    x = np.arange(len(BANNED_APIS))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, rates, color="#3498db", alpha=0.85)
    ax.set_xlabel("Banned API")
    ax.set_ylabel("% Tasks Containing API")
    ax.set_title("Violation Rate by Banned API — CodeGen CoCoMo")
    ax.set_xticks(x)
    ax.set_xticklabels(BANNED_APIS, rotation=15)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Violation comparison saved to {out_path}")


def plot_escalation_analysis(results: list, out_path: Optional[str] = None):
    if out_path is None:
        out_path = os.path.join(_HERE, "codegen_escalation_analysis.png")
    stats: dict[str, dict] = defaultdict(lambda: {"total": 0, "escalated": 0})
    for r in results:
        c = r["category"]
        stats[c]["total"] += 1
        if r["was_conscious"]:
            stats[c]["escalated"] += 1
    categories = sorted(stats, key=lambda c: stats[c]["escalated"] / stats[c]["total"], reverse=True)
    rates = [stats[c]["escalated"] / stats[c]["total"] * 100 for c in categories]
    colors = ["#e74c3c" if r > 50 else "#f39c12" if r > 20 else "#2ecc71" for r in rates]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(categories)), rates, color=colors, alpha=0.85)
    ax.axhline(y=30, color="gray", linestyle="--", alpha=0.5, label="Threshold (30%)")
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=30, ha="right")
    ax.set_xlabel("Task Category")
    ax.set_ylabel("% Tasks Escalated to Consciousness")
    ax.set_title("MFQ Escalation Rate by Task Category")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Escalation analysis saved to {out_path}")


def plot_escalation_confusion_matrix(results: list, out_path: Optional[str] = None):
    if out_path is None:
        out_path = os.path.join(_HERE, "codegen_escalation_confusion_matrix.png")
    tp = fp = fn = tn = 0
    for r in results:
        dv = len(r.get("draft_violations", [])) > 0
        esc = r["was_conscious"]
        if dv and esc:       tp += 1
        elif not dv and esc: fp += 1
        elif dv and not esc: fn += 1
        else:                tn += 1
    matrix = np.array([[tp, fn], [fp, tn]])
    labels = [
        [f"TP\n(escalated, violated)\n{tp}", f"FN\n(passed, violated)\n{fn}"],
        [f"FP\n(escalated, clean)\n{fp}", f"TN\n(passed, clean)\n{tn}"],
    ]
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(matrix, cmap="Blues", vmin=0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Escalated", "Passed"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Draft Violated", "Draft Clean"])
    ax.set_xlabel("MFQ Decision")
    ax.set_ylabel("Draft Ground Truth")
    ax.set_title("Escalation Confusion Matrix — CodeGen")
    thresh = matrix.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=9,
                    color="white" if matrix[i, j] > thresh else "black")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {out_path}")


def plot_crit_validity_heatmap(results: list, out_path: Optional[str] = None):
    if out_path is None:
        out_path = os.path.join(_HERE, "codegen_crit_validity_heatmap.png")
    scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        if not r["was_conscious"]:
            continue
        for api, info in r["substitutions_used"].items():
            if "validity_score" in info:
                scores[r["category"]][api].append(info["validity_score"])
    if not scores:
        return
    categories = sorted(scores.keys())
    matrix = np.full((len(categories), len(BANNED_APIS)), np.nan)
    for i, cat in enumerate(categories):
        for j, api in enumerate(BANNED_APIS):
            vals = scores[cat][api]
            if vals:
                matrix[i, j] = np.mean(vals)
    fig, ax = plt.subplots(figsize=(10, max(4, len(categories) * 0.6 + 2)))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(BANNED_APIS)))
    ax.set_xticklabels(BANNED_APIS, rotation=20, ha="right")
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)
    ax.set_title("CRIT Validity Score: API × Task Category\n(gray = no escalated tasks for that cell)")
    plt.colorbar(im, ax=ax, label="Mean Validity Score (0–1)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"CRIT heatmap saved to {out_path}")


def plot_mfq_adaptation_curve(risk_history: list, out_path: Optional[str] = None):
    if out_path is None:
        out_path = os.path.join(_HERE, "codegen_mfq_adaptation_curve.png")
    all_categories: set[str] = set()
    for s in risk_history:
        all_categories.update(s["category_risks"].keys())
    if not all_categories:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    for cat in sorted(all_categories):
        pairs = [(s["idx"], s["category_risks"][cat]) for s in risk_history if cat in s["category_risks"]]
        if pairs:
            xs, ys = zip(*pairs)
            ax.plot(xs, ys, label=cat, marker="o", markersize=3, linewidth=1.5)
    ax.axhline(y=0.3, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="Threshold (0.3)")
    ax.set_xlabel("Eval Task Index")
    ax.set_ylabel("Category Risk Score")
    ax.set_title("MFQ Category Risk Adaptation Over Evaluation\n(feedback loop recalibrating in real time)")
    ax.set_ylim(0, 1.05)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Adaptation curve saved to {out_path}")


def plot_risk_score_histogram(results: list, out_path: Optional[str] = None):
    if out_path is None:
        out_path = os.path.join(_HERE, "codegen_risk_score_histogram.png")
    escalated = [r["risk_score"] for r in results if r["was_conscious"]]
    passed = [r["risk_score"] for r in results if not r["was_conscious"]]
    bins = np.linspace(0, 1, 21)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(passed, bins=bins, alpha=0.7, label="Not Escalated", color="#2ecc71")
    ax.hist(escalated, bins=bins, alpha=0.7, label="Escalated", color="#e74c3c")
    ax.axvline(x=0.3, color="black", linestyle="--", linewidth=1.5, label="Threshold (0.3)")
    ax.set_xlabel("Risk Score")
    ax.set_ylabel("Number of Tasks")
    ax.set_title("Risk Score Distribution by Escalation Decision — CodeGen")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Risk histogram saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=None)
    args = parser.parse_args()

    print(f"\n{'='*60}\nCodeGen CoCoMo Evaluation\nTasks: {len(EVAL_TASKS)}\n{'='*60}\n")
    results, risk_history = run_evaluation(args.weights)
    output = save_results(results)

    print("\n--- Violation Rates ---")
    for k, v in output["summary"]["violation_rates"].items():
        print(f"  {k}: {v}%")
    print(f"  Escalation rate: {output['summary']['escalation_rate']}%")

    print("\n--- Generating Figures ---")
    plot_violation_comparison(output)
    plot_escalation_analysis(results)
    plot_escalation_confusion_matrix(results)
    plot_crit_validity_heatmap(results)
    plot_mfq_adaptation_curve(risk_history)
    plot_risk_score_histogram(results)


if __name__ == "__main__":
    main()
