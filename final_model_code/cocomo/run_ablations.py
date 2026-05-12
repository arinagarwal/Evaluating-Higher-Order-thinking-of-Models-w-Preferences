"""
Parallel ablation runner.

Runs all 4 pipeline variants simultaneously, each in its own process on the
same GPU. Each process loads its own model instance independently so no state
is shared across variants (clean ablation isolation).

VRAM requirement: ~5 GB per process × number of workers.
  - 1 worker  (--workers 1): ~5 GB   — sequential, any GPU
  - 2 workers (--workers 2): ~10 GB  — RTX 3080 / A10 class
  - 4 workers (--workers 4): ~20 GB  — RTX 3090 / A100 40GB class

Usage:
    # All 4 variants in parallel (default):
    python cocomo/run_ablations.py

    # Sequential fallback for limited VRAM:
    python cocomo/run_ablations.py --workers 1

    # With GRPO weights:
    python cocomo/run_ablations.py --weights cocomo/grpo_weights/final

Results saved to cocomo/ablation_results/:
    ablation_base.json
    ablation_planner.json
    ablation_verifier.json
    ablation_memory.json
    ablation_comparison.png   ← generated after all variants complete
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import multiprocessing as mp
from collections import defaultdict
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

VARIANTS = [
    ("base",     "CoCoMoPipeline"),
    ("planner",  "PlannerPipeline"),
    ("verifier", "VerifierPipeline"),
    ("memory",   "MemoryPipeline"),
]

VARIANT_LABELS = {
    "base":     "CoCoMo (base)",
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


# ── Worker entry point ────────────────────────────────────────────────────────

def _worker(variant_name: str, class_name: str, weights_path: Optional[str], output_dir: str):
    """
    Runs inside a spawned process. All imports happen here so CUDA initialises
    cleanly per-process (no inherited CUDA contexts from the parent).
    """
    sys.path.insert(0, _HERE)
    sys.path.insert(0, _ROOT)

    from pipeline import CoCoMoPipeline
    from pipeline_variants import PlannerPipeline, VerifierPipeline, MemoryPipeline
    from evaluate import load_model, save_results
    from config import EVAL_DISHES

    cls_map = {
        "CoCoMoPipeline":  CoCoMoPipeline,
        "PlannerPipeline": PlannerPipeline,
        "VerifierPipeline": VerifierPipeline,
        "MemoryPipeline":  MemoryPipeline,
    }

    tag = f"[{variant_name}]"
    print(f"{tag} Loading model...", flush=True)
    model, tokenizer = load_model(weights_path)

    print(f"{tag} Starting evaluation ({len(EVAL_DISHES)} dishes)...", flush=True)
    pipeline = cls_map[class_name](model=model, tokenizer=tokenizer)
    results = pipeline.run_batch(EVAL_DISHES)

    out_path = os.path.join(output_dir, f"ablation_{variant_name}.json")
    save_results(results, out_path=out_path)
    print(f"{tag} Done → {out_path}", flush=True)


# ── Comparison plot ───────────────────────────────────────────────────────────

def plot_ablation_comparison(output_dir: str):
    """
    Loads all 4 ablation result files and produces a grouped bar chart showing
    violation rates per banned ingredient across the ablation chain.
    """
    from config import BANNED_INGREDIENTS

    series = []
    for name, _ in VARIANTS:
        path = os.path.join(output_dir, f"ablation_{name}.json")
        if not os.path.exists(path):
            print(f"  Missing {path}, skipping.")
            continue
        with open(path) as f:
            data = json.load(f)
        rates_map = data.get("summary", {}).get("violation_rates", {})
        rates = [rates_map.get(ing, 0.0) for ing in BANNED_INGREDIENTS]
        series.append((VARIANT_LABELS[name], rates, VARIANT_COLORS[name]))

    if not series:
        print("No ablation result files found.")
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
    ax.set_title("Ablation Study: Violation Rate Across Architectural Changes")
    ax.set_xticks(x)
    ax.set_xticklabels(BANNED_INGREDIENTS, rotation=15)
    ax.legend()
    ax.set_ylim(0, 100)
    plt.tight_layout()
    out = os.path.join(output_dir, "ablation_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Ablation comparison chart saved to {out}")


def print_ablation_table(output_dir: str):
    """Prints a summary table of key metrics across variants."""
    from config import BANNED_INGREDIENTS

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
            draft_violated = len(r.get("draft_violations", [])) > 0
            escalated = r["was_conscious"]
            if draft_violated and escalated:     tp += 1
            elif not draft_violated and escalated: fp += 1
            elif draft_violated and not escalated: fn += 1
            else:                                  tn += 1

        recall = round(tp / (tp + fn) * 100, 1) if (tp + fn) > 0 else 0.0
        total_violations = sum(rates.get(ing, 0) for ing in BANNED_INGREDIENTS)
        rows.append((VARIANT_LABELS[name], total_violations, recall, summary.get("escalation_rate", 0)))

    print("\n" + "─" * 70)
    print(f"{'Variant':<25} {'Total Viol %':>12} {'FN Recall %':>12} {'Escalation %':>13}")
    print("─" * 70)
    for label, total_v, recall, esc in rows:
        print(f"{label:<25} {total_v:>12.1f} {recall:>12.1f} {esc:>13.1f}")
    print("─" * 70)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run CoCoMo ablation study")
    parser.add_argument("--weights", default=None, help="Path to GRPO LoRA weights dir")
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Parallel workers. Each needs ~5 GB VRAM. Use 1 for sequential."
    )
    parser.add_argument("--output-dir", default=os.path.join(_HERE, "ablation_results"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("CoCoMo Ablation Study")
    print(f"Variants:   {[n for n, _ in VARIANTS]}")
    print(f"Workers:    {args.workers}  (~{args.workers * 5} GB VRAM needed)")
    print(f"Output dir: {args.output_dir}")
    print(f"{'='*60}\n")

    if args.workers == 1:
        # Sequential — useful for VRAM-constrained machines
        for name, cls_name in VARIANTS:
            _worker(name, cls_name, args.weights, args.output_dir)
    else:
        ctx = mp.get_context("spawn")
        queue = list(VARIANTS)
        active: list[mp.Process] = []

        while queue or active:
            # Fill up to --workers slots
            while len(active) < args.workers and queue:
                name, cls_name = queue.pop(0)
                p = ctx.Process(
                    target=_worker,
                    args=(name, cls_name, args.weights, args.output_dir),
                    name=f"ablation-{name}",
                    daemon=False,
                )
                p.start()
                active.append(p)
                print(f"Launched worker: {name} (pid {p.pid})", flush=True)

            # Reap finished processes
            for p in active[:]:
                p.join(timeout=1.0)
                if not p.is_alive():
                    active.remove(p)
                    status = "OK" if p.exitcode == 0 else f"FAILED (exit {p.exitcode})"
                    print(f"Worker {p.name} finished: {status}", flush=True)

    print("\nAll variants complete. Generating comparison plots...")
    plot_ablation_comparison(args.output_dir)
    print_ablation_table(args.output_dir)


if __name__ == "__main__":
    main()
