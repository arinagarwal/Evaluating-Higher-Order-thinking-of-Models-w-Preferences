"""
Visualization Module
====================
Generates publication-quality figures from experiment results.
Uses matplotlib with a clean, academic style.

Usage:
    python visualize.py experiments/cold_start/<timestamp>/all_results.json
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from metrics import (
    learning_curve,
    preference_trajectory,
    preference_stability,
    ingredient_weight_divergence,
    full_analysis,
)
from preference_metrics import (
    behavioral_trajectories,
    ingredient_repertoire,
)

# ── Style ────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

MODE_COLORS = {
    "baseline": "#888888",
    "world_model_only": "#2196F3",
    "self_model_only": "#FF9800",
    "full_model": "#4CAF50",
}

MODE_LABELS = {
    "baseline": "Baseline (no learning)",
    "world_model_only": "World Model Only",
    "self_model_only": "Self Model Only",
    "full_model": "Full Model (WM + SM)",
}


def _get_color(mode: str) -> str:
    return MODE_COLORS.get(mode, "#000000")


def _get_label(mode: str) -> str:
    return MODE_LABELS.get(mode, mode)


# ── Figure 1: Learning Curves ────────────────────────────────────────────────


def plot_learning_curves(all_results: dict, output_dir: str) -> str:
    """Plot mean abs_error over time for each mode with 95% CI bands."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for mode, results in all_results.items():
        lc = learning_curve(results)
        if not lc["steps"]:
            continue

        steps = np.array(lc["steps"])
        mean = np.array(lc["mean_error"])
        ci_lo = np.array(lc["ci_lower"])
        ci_hi = np.array(lc["ci_upper"])

        color = _get_color(mode)
        ax.plot(steps, mean, color=color, label=_get_label(mode), linewidth=1.5)
        ax.fill_between(steps, ci_lo, ci_hi, color=color, alpha=0.15)

    ax.set_xlabel("Run Index")
    ax.set_ylabel("Absolute Prediction Error")
    ax.set_title("Learning Curves: Prediction Error Over Time")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(bottom=0)

    path = os.path.join(output_dir, "fig1_learning_curves.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ── Figure 2: Preference Trajectories ────────────────────────────────────────


def plot_preference_trajectories(all_results: dict, output_dir: str) -> str:
    """Plot spice_preference and health_bias trajectories per mode."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for dim_idx, dim in enumerate(["spice", "health"]):
        ax = axes[dim_idx]
        for mode, results in all_results.items():
            traj = preference_trajectory(results)
            dim_data = traj[dim]

            if not dim_data:
                continue

            # Compute mean trajectory across trials
            max_len = max(len(v) for v in dim_data.values())
            padded = np.full((len(dim_data), max_len), np.nan)
            for i, (trial, vals) in enumerate(dim_data.items()):
                padded[i, :len(vals)] = vals

            mean_traj = np.nanmean(padded, axis=0)
            std_traj = np.nanstd(padded, axis=0)
            steps = np.arange(max_len)

            color = _get_color(mode)
            ax.plot(steps, mean_traj, color=color, label=_get_label(mode), linewidth=1.5)
            ax.fill_between(
                steps,
                mean_traj - std_traj,
                mean_traj + std_traj,
                color=color, alpha=0.1,
            )

        dim_label = "Spice Preference" if dim == "spice" else "Health Bias"
        ax.set_xlabel("Run Index")
        ax.set_ylabel(dim_label)
        ax.set_title(f"{dim_label} Over Time")
        ax.set_ylim(0, 1)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

    axes[0].legend(loc="best", framealpha=0.9)
    fig.tight_layout()

    path = os.path.join(output_dir, "fig2_preference_trajectories.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ── Figure 3: Preference Stability (CV) ──────────────────────────────────────


def plot_preference_stability(all_results: dict, output_dir: str) -> str:
    """Plot rolling coefficient of variation for preference dimensions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for dim_idx, dim in enumerate(["spice", "health"]):
        ax = axes[dim_idx]
        for mode, results in all_results.items():
            stab = preference_stability(results)
            if dim not in stab or not stab[dim]["mean_cv"]:
                continue

            mean_cv = np.array(stab[dim]["mean_cv"])
            steps = np.arange(len(mean_cv))

            color = _get_color(mode)
            ax.plot(steps, mean_cv, color=color, label=_get_label(mode), linewidth=1.5)

        dim_label = "Spice Preference" if dim == "spice" else "Health Bias"
        ax.set_xlabel("Window Position")
        ax.set_ylabel("Coefficient of Variation")
        ax.set_title(f"{dim_label} Stability (lower = more stable)")
        ax.set_ylim(bottom=0)

    axes[0].legend(loc="best", framealpha=0.9)
    fig.tight_layout()

    path = os.path.join(output_dir, "fig3_preference_stability.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ── Figure 4: World Model Weight Growth ──────────────────────────────────────


def plot_weight_divergence(all_results: dict, output_dir: str) -> str:
    """Plot L2 norm of ingredient weight vector over time."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for mode, results in all_results.items():
        div = ingredient_weight_divergence(results)
        if not div["steps"]:
            continue

        steps = np.array(div["steps"])
        mean_norm = np.array(div["mean_norm"])
        std_norm = np.array(div["std_norm"])

        color = _get_color(mode)
        ax.plot(steps, mean_norm, color=color, label=_get_label(mode), linewidth=1.5)
        ax.fill_between(
            steps,
            mean_norm - std_norm,
            mean_norm + std_norm,
            color=color, alpha=0.1,
        )

    ax.set_xlabel("Run Index")
    ax.set_ylabel("L2 Norm of Ingredient Weights")
    ax.set_title("World Model Knowledge Accumulation")
    ax.legend(loc="best", framealpha=0.9)
    ax.set_ylim(bottom=0)

    path = os.path.join(output_dir, "fig4_weight_divergence.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ── Figure 5: Summary Bar Chart ──────────────────────────────────────────────


def plot_summary_bars(all_results: dict, output_dir: str) -> str:
    """Bar chart comparing key metrics across modes."""
    from metrics import prediction_accuracy, convergence_speed

    modes = list(all_results.keys())
    n = len(modes)

    mean_errors = []
    improvements = []
    conv_fractions = []

    for mode in modes:
        pa = prediction_accuracy(all_results[mode])
        cs = convergence_speed(all_results[mode])
        mean_errors.append(pa["mean_abs_error"])
        improvements.append(pa["improvement"])
        conv_fractions.append(cs["fraction_converged"])

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    x = np.arange(n)
    colors = [_get_color(m) for m in modes]
    labels = [_get_label(m) for m in modes]

    # Mean error
    axes[0].bar(x, mean_errors, color=colors, width=0.6)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    axes[0].set_ylabel("Mean Absolute Error")
    axes[0].set_title("Overall Prediction Error")

    # Improvement (first half - second half)
    axes[1].bar(x, improvements, color=colors, width=0.6)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    axes[1].set_ylabel("Error Reduction")
    axes[1].set_title("Learning Improvement\n(1st half − 2nd half error)")
    axes[1].axhline(y=0, color="gray", linestyle="-", linewidth=0.8)

    # Convergence fraction
    axes[2].bar(x, conv_fractions, color=colors, width=0.6)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    axes[2].set_ylabel("Fraction Converged")
    axes[2].set_title("Convergence Rate")
    axes[2].set_ylim(0, 1)

    fig.tight_layout()
    path = os.path.join(output_dir, "fig5_summary_bars.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ── Figure 6: Cuisine Affinity Heatmap ───────────────────────────────────────


def plot_cuisine_heatmap(all_results: dict, output_dir: str) -> str:
    """Heatmap of final cuisine affinity scores per mode."""
    from collections import defaultdict

    # Collect final cuisine affinities per mode
    mode_cuisines = {}
    all_cuisines = set()

    for mode, results in all_results.items():
        # Get final self_model from each trial, average
        by_trial = defaultdict(list)
        for r in results:
            by_trial[r.get("trial", 0)].append(r)

        final_affinities = defaultdict(list)
        for trial, runs in by_trial.items():
            runs.sort(key=lambda x: (x.get("round", 0), x.get("prompt_idx", 0)))
            if runs:
                ca = runs[-1].get("updated_self_model", {}).get("cuisine_affinity", {})
                for cuisine, score in ca.items():
                    final_affinities[cuisine].append(score)
                    all_cuisines.add(cuisine)

        mode_cuisines[mode] = {
            c: float(np.mean(v)) for c, v in final_affinities.items()
        }

    if not all_cuisines:
        return ""

    cuisines = sorted(all_cuisines)
    modes = list(all_results.keys())

    matrix = np.zeros((len(modes), len(cuisines)))
    for i, mode in enumerate(modes):
        for j, cuisine in enumerate(cuisines):
            matrix[i, j] = mode_cuisines.get(mode, {}).get(cuisine, 0.0)

    fig, ax = plt.subplots(figsize=(max(8, len(cuisines) * 1.2), max(4, len(modes) * 0.8)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(np.arange(len(cuisines)))
    ax.set_yticks(np.arange(len(modes)))
    ax.set_xticklabels(cuisines, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels([_get_label(m) for m in modes], fontsize=9)

    # Annotate cells
    for i in range(len(modes)):
        for j in range(len(cuisines)):
            val = matrix[i, j]
            if val > 0.01:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if val > 0.5 else "black")

    ax.set_title("Learned Cuisine Affinity (Final State)")
    fig.colorbar(im, ax=ax, label="Affinity Score")
    fig.tight_layout()

    path = os.path.join(output_dir, "fig6_cuisine_heatmap.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ── Figure 7: Revealed Preferences (Behavioral) ─────────────────────────────


def plot_revealed_preferences(all_results: dict, output_dir: str) -> str:
    """Plot revealed preference trajectories (actual recipe choices) per mode."""
    dims = ["spice_ratio", "protein_ratio", "vegetable_ratio"]
    dim_labels = ["Spice Ratio", "Protein Ratio", "Vegetable Ratio"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for dim_idx, (dim, label) in enumerate(zip(dims, dim_labels)):
        ax = axes[dim_idx]
        for mode, results in all_results.items():
            traj = behavioral_trajectories(results)
            dim_data = traj.get(dim, {})
            if not dim_data:
                continue

            max_len = max(len(v) for v in dim_data.values())
            padded = np.full((len(dim_data), max_len), np.nan)
            for i, (trial, vals) in enumerate(dim_data.items()):
                padded[i, :len(vals)] = vals

            mean_traj = np.nanmean(padded, axis=0)
            std_traj = np.nanstd(padded, axis=0)
            steps = np.arange(max_len)

            color = _get_color(mode)
            ax.plot(steps, mean_traj, color=color, label=_get_label(mode), linewidth=1.5)
            ax.fill_between(steps, mean_traj - std_traj, mean_traj + std_traj,
                            color=color, alpha=0.1)

        ax.set_xlabel("Run Index")
        ax.set_ylabel(label)
        ax.set_title(f"Revealed {label}")
        ax.set_ylim(0, 1)

    axes[0].legend(loc="best", framealpha=0.9, fontsize=8)
    fig.tight_layout()

    path = os.path.join(output_dir, "fig7_revealed_preferences.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ── Figure 8: Ingredient Reuse Rate ──────────────────────────────────────────


def plot_ingredient_reuse(all_results: dict, output_dir: str) -> str:
    """Plot cumulative unique ingredients and reuse rate per mode."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: cumulative unique ingredients
    for mode, results in all_results.items():
        rep = ingredient_repertoire(results)
        cum = rep["mean_cumulative_unique"]
        if not cum:
            continue
        color = _get_color(mode)
        axes[0].plot(range(len(cum)), cum, color=color,
                     label=_get_label(mode), linewidth=1.5)

    axes[0].set_xlabel("Run Index")
    axes[0].set_ylabel("Cumulative Unique Ingredients")
    axes[0].set_title("Ingredient Vocabulary Growth")
    axes[0].legend(loc="best", framealpha=0.9, fontsize=9)

    # Right: reuse rate bar chart
    modes = list(all_results.keys())
    reuse_means = []
    reuse_stds = []
    for mode in modes:
        rep = ingredient_repertoire(all_results[mode])
        reuse_means.append(rep["mean_reuse_rate"])
        reuse_stds.append(rep["std_reuse_rate"])

    x = np.arange(len(modes))
    colors = [_get_color(m) for m in modes]
    axes[1].bar(x, reuse_means, yerr=reuse_stds, color=colors, width=0.6, capsize=4)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([_get_label(m) for m in modes], rotation=30, ha="right", fontsize=9)
    axes[1].set_ylabel("Ingredient Reuse Rate")
    axes[1].set_title("Ingredient Reuse\n(higher = stronger preferences)")

    fig.tight_layout()
    path = os.path.join(output_dir, "fig8_ingredient_reuse.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ── Generate all figures ─────────────────────────────────────────────────────


def generate_all_figures(all_results: dict, output_dir: str) -> list[str]:
    """Generate all paper figures and return list of paths."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nGenerating figures → {output_dir}/")

    paths = [
        plot_learning_curves(all_results, output_dir),
        plot_preference_trajectories(all_results, output_dir),
        plot_preference_stability(all_results, output_dir),
        plot_weight_divergence(all_results, output_dir),
        plot_summary_bars(all_results, output_dir),
        plot_cuisine_heatmap(all_results, output_dir),
        plot_revealed_preferences(all_results, output_dir),
        plot_ingredient_reuse(all_results, output_dir),
    ]

    return [p for p in paths if p]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate paper figures from experiment results")
    parser.add_argument("results_path", help="Path to all_results.json")
    parser.add_argument("--output", default=None, help="Output directory for figures")
    args = parser.parse_args()

    with open(args.results_path) as f:
        all_results = json.load(f)

    output_dir = args.output or os.path.join(os.path.dirname(args.results_path), "figures")
    generate_all_figures(all_results, output_dir)

    # Also run and print the full analysis
    from metrics import full_analysis, print_report
    from preference_metrics import full_preference_analysis, print_preference_report

    report = full_analysis(all_results)
    print_report(report)

    pref_report = full_preference_analysis(all_results)
    print_preference_report(pref_report)

    # Save reports as JSON
    report_path = os.path.join(output_dir, "analysis_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    pref_path = os.path.join(output_dir, "preference_report.json")
    with open(pref_path, "w") as f:
        json.dump(pref_report, f, indent=2, default=str)

    print(f"\nReports saved: {report_path}, {pref_path}")
