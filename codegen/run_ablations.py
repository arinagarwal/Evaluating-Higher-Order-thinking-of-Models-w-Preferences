"""
Parallel ablation runner for code generation — mirrors cocomo/run_ablations.py.

Usage:
    python codegen/run_ablations.py --workers 4
    python codegen/run_ablations.py --workers 1   # sequential
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

VARIANTS = [
    ("base",     "CodeGenPipeline"),
    ("planner",  "PlannerPipeline"),
    ("verifier", "VerifierPipeline"),
    ("memory",   "MemoryPipeline"),
]

VARIANT_LABELS = {
    "base":     "CodeGen (base)",
    "planner":  "+ Planner/Drafter",
    "verifier": "+ Verifier",
    "memory":   "+ Memory",
}

VARIANT_COLORS = {
    "base":     "#3498db",
    "planner":  "#9b59b6",
    "verifier": "#e67e22",
    "memory":   "#2ecc71",
}


def _worker(variant_name: str, class_name: str, weights_path: Optional[str], output_dir: str):
    sys.path.insert(0, _HERE)
    from pipeline import CodeGenPipeline
    from pipeline_variants import PlannerPipeline, VerifierPipeline, MemoryPipeline
    from evaluate import load_model, save_results
    from tasks import TASKS

    cls_map = {
        "CodeGenPipeline":  CodeGenPipeline,
        "PlannerPipeline":  PlannerPipeline,
        "VerifierPipeline": VerifierPipeline,
        "MemoryPipeline":   MemoryPipeline,
    }

    tag = f"[{variant_name}]"
    print(f"{tag} Loading model...", flush=True)
    model, tokenizer = load_model(weights_path)
    pipeline = cls_map[class_name](model=model, tokenizer=tokenizer)

    print(f"{tag} Starting evaluation ({len(TASKS)} tasks)...", flush=True)
    results = pipeline.run_batch(TASKS)

    out_path = os.path.join(output_dir, f"ablation_{variant_name}.json")
    save_results(results, out_path=out_path)
    print(f"{tag} Done → {out_path}", flush=True)


def plot_ablation_comparison(output_dir: str):
    from config import BANNED_APIS

    series = []
    for name, _ in VARIANTS:
        path = os.path.join(output_dir, f"ablation_{name}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        rates_map = data.get("summary", {}).get("violation_rates", {})
        rates = [rates_map.get(api, 0.0) for api in BANNED_APIS]
        series.append((VARIANT_LABELS[name], rates, VARIANT_COLORS[name]))

    if not series:
        return

    x = np.arange(len(BANNED_APIS))
    width = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (label, rates, color) in enumerate(series):
        offset = (i - len(series) / 2 + 0.5) * width
        ax.bar(x + offset, rates, width, label=label, color=color, alpha=0.85)
    ax.set_xlabel("Banned API")
    ax.set_ylabel("% Tasks Containing API")
    ax.set_title("Ablation Study: Violation Rate Across Architectural Changes — CodeGen")
    ax.set_xticks(x)
    ax.set_xticklabels(BANNED_APIS, rotation=15)
    ax.legend()
    ax.set_ylim(0, 100)
    plt.tight_layout()
    out = os.path.join(output_dir, "ablation_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Ablation comparison saved to {out}")


def print_ablation_table(output_dir: str):
    from config import BANNED_APIS
    rows = []
    for name, _ in VARIANTS:
        path = os.path.join(output_dir, f"ablation_{name}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        summary = data.get("summary", {})
        rates = summary.get("violation_rates", {})
        results = data.get("results", [])
        tp = fp = fn = tn = 0
        for r in results:
            dv = len(r.get("draft_violations", [])) > 0
            esc = r["was_conscious"]
            if dv and esc:       tp += 1
            elif not dv and esc: fp += 1
            elif dv and not esc: fn += 1
            else:                tn += 1
        recall = round(tp / (tp + fn) * 100, 1) if (tp + fn) > 0 else 0.0
        total = sum(rates.get(api, 0) for api in BANNED_APIS)
        rows.append((VARIANT_LABELS[name], total, recall, summary.get("escalation_rate", 0)))

    print("\n" + "─" * 70)
    print(f"{'Variant':<25} {'Total Viol %':>12} {'FN Recall %':>12} {'Escalation %':>13}")
    print("─" * 70)
    for label, tv, recall, esc in rows:
        print(f"{label:<25} {tv:>12.1f} {recall:>12.1f} {esc:>13.1f}")
    print("─" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", default=os.path.join(_HERE, "ablation_results"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}\nCodeGen Ablation Study\nWorkers: {args.workers}\n{'='*60}\n")

    if args.workers == 1:
        for name, cls_name in VARIANTS:
            _worker(name, cls_name, args.weights, args.output_dir)
    else:
        ctx = mp.get_context("spawn")
        queue = list(VARIANTS)
        active: list[mp.Process] = []
        while queue or active:
            while len(active) < args.workers and queue:
                name, cls_name = queue.pop(0)
                p = ctx.Process(
                    target=_worker,
                    args=(name, cls_name, args.weights, args.output_dir),
                    name=f"codegen-{name}",
                    daemon=False,
                )
                p.start()
                active.append(p)
                print(f"Launched: {name} (pid {p.pid})", flush=True)
            for p in active[:]:
                p.join(timeout=1.0)
                if not p.is_alive():
                    active.remove(p)
                    status = "OK" if p.exitcode == 0 else f"FAILED (exit {p.exitcode})"
                    print(f"{p.name} finished: {status}", flush=True)

    print("\nAll variants complete.")
    plot_ablation_comparison(args.output_dir)
    print_ablation_table(args.output_dir)


if __name__ == "__main__":
    main()
