"""
Metrics & Statistical Analysis
==============================
Computes research-grade metrics from experiment logs:
- Learning curves with confidence intervals
- Preference stability (coefficient of variation over time)
- Preference divergence between modes (Jensen-Shannon divergence)
- Convergence speed
- Statistical significance tests (Mann-Whitney U, Wilcoxon signed-rank)

All functions operate on the structured results dict produced by experiment_runner.
"""

import json
import math
import os
from collections import defaultdict
from typing import Optional

import numpy as np


# ── Loading utilities ────────────────────────────────────────────────────────


def load_results(results_path: str) -> dict:
    """Load all_results.json from an experiment run."""
    with open(results_path) as f:
        return json.load(f)


def load_trial_logs(experiment_dir: str, mode: str) -> list[list[dict]]:
    """Load per-trial run logs for a given mode from the experiment directory."""
    mode_dir = os.path.join(experiment_dir, mode)
    if not os.path.isdir(mode_dir):
        return []

    trials = []
    for trial_name in sorted(os.listdir(mode_dir)):
        trial_path = os.path.join(mode_dir, trial_name)
        log_path = os.path.join(trial_path, "run_log.jsonl")
        if not os.path.exists(log_path):
            continue
        entries = []
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        trials.append(entries)
    return trials


# ── Core metrics ─────────────────────────────────────────────────────────────


def learning_curve(results: list[dict], window: int = 1) -> dict:
    """Compute per-run-index mean and std of abs_error across trials.

    Groups results by (round, prompt_idx) to align runs across trials,
    then computes mean ± std at each step.

    Returns:
        {
            "steps": [0, 1, 2, ...],
            "mean_error": [...],
            "std_error": [...],
            "ci_lower": [...],  # mean - 1.96*SE
            "ci_upper": [...],  # mean + 1.96*SE
            "n_trials": int,
        }
    """
    # Group by run position within trial
    by_trial = defaultdict(list)
    for r in results:
        trial = r.get("trial", 0)
        by_trial[trial].append(r)

    # Sort each trial by (round, prompt_idx)
    for trial in by_trial:
        by_trial[trial].sort(key=lambda x: (x.get("round", 0), x.get("prompt_idx", 0)))

    n_trials = len(by_trial)
    if n_trials == 0:
        return {"steps": [], "mean_error": [], "std_error": [],
                "ci_lower": [], "ci_upper": [], "n_trials": 0}

    max_len = max(len(v) for v in by_trial.values())

    steps, means, stds, ci_lo, ci_hi = [], [], [], [], []
    for i in range(max_len):
        errors = []
        for trial_results in by_trial.values():
            if i < len(trial_results):
                errors.append(trial_results[i]["abs_error"])

        if not errors:
            continue

        arr = np.array(errors)
        m = float(np.mean(arr))
        s = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        se = s / math.sqrt(len(arr)) if len(arr) > 1 else 0.0

        steps.append(i)
        means.append(m)
        stds.append(s)
        ci_lo.append(m - 1.96 * se)
        ci_hi.append(m + 1.96 * se)

    return {
        "steps": steps,
        "mean_error": means,
        "std_error": stds,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "n_trials": n_trials,
    }


def preference_trajectory(results: list[dict]) -> dict:
    """Extract self-model preference values over time, per trial.

    Returns:
        {
            "spice": {trial_id: [values_over_time], ...},
            "health": {trial_id: [values_over_time], ...},
            "cuisine_entropy": {trial_id: [values_over_time], ...},
        }
    """
    by_trial = defaultdict(list)
    for r in results:
        by_trial[r.get("trial", 0)].append(r)

    for trial in by_trial:
        by_trial[trial].sort(key=lambda x: (x.get("round", 0), x.get("prompt_idx", 0)))

    trajectories = {"spice": {}, "health": {}, "cuisine_entropy": {}}

    for trial, runs in by_trial.items():
        spice_vals, health_vals, entropy_vals = [], [], []
        for r in runs:
            sm = r.get("updated_self_model", {})
            spice_vals.append(sm.get("spice_preference", 0.5))
            health_vals.append(sm.get("health_bias", 0.5))

            # Cuisine affinity entropy — measures how concentrated preferences are
            ca = sm.get("cuisine_affinity", {})
            if ca:
                vals = np.array(list(ca.values()))
                total = vals.sum()
                if total > 0:
                    probs = vals / total
                    probs = probs[probs > 0]
                    entropy_vals.append(float(-np.sum(probs * np.log2(probs))))
                else:
                    entropy_vals.append(0.0)
            else:
                entropy_vals.append(0.0)

        trajectories["spice"][trial] = spice_vals
        trajectories["health"][trial] = health_vals
        trajectories["cuisine_entropy"][trial] = entropy_vals

    return trajectories


def preference_stability(results: list[dict], window: int = 5) -> dict:
    """Measure how stable preferences become over time using rolling CV.

    Coefficient of Variation (CV = std/mean) over a sliding window.
    Lower CV = more stable preferences.

    Returns per-dimension rolling CV averaged across trials.
    """
    traj = preference_trajectory(results)
    stability = {}

    for dim in ["spice", "health"]:
        all_cvs = []
        for trial, values in traj[dim].items():
            if len(values) < window:
                continue
            cvs = []
            for i in range(window, len(values) + 1):
                w = np.array(values[i - window:i])
                mean = np.mean(w)
                if mean > 0.01:  # avoid division by near-zero
                    cvs.append(float(np.std(w) / mean))
                else:
                    cvs.append(0.0)
            all_cvs.append(cvs)

        if all_cvs:
            # Align and average across trials
            max_len = max(len(c) for c in all_cvs)
            padded = np.full((len(all_cvs), max_len), np.nan)
            for i, c in enumerate(all_cvs):
                padded[i, :len(c)] = c
            stability[dim] = {
                "mean_cv": np.nanmean(padded, axis=0).tolist(),
                "std_cv": np.nanstd(padded, axis=0).tolist(),
            }
        else:
            stability[dim] = {"mean_cv": [], "std_cv": []}

    return stability


def convergence_speed(results: list[dict], threshold: float = 0.001) -> dict:
    """Compute how many runs until self-model values stabilize per trial.

    Stabilization = all self-model values change by < threshold for 3 consecutive runs.

    Returns:
        {"per_trial": {trial_id: convergence_run_or_None}, "mean": float, "median": float}
    """
    by_trial = defaultdict(list)
    for r in results:
        by_trial[r.get("trial", 0)].append(r)

    for trial in by_trial:
        by_trial[trial].sort(key=lambda x: (x.get("round", 0), x.get("prompt_idx", 0)))

    per_trial = {}
    for trial, runs in by_trial.items():
        converged_at = None
        streak = 0
        for i in range(1, len(runs)):
            sm_prev = runs[i - 1].get("updated_self_model", {})
            sm_curr = runs[i].get("updated_self_model", {})

            sp_delta = abs(sm_curr.get("spice_preference", 0.5) - sm_prev.get("spice_preference", 0.5))
            hb_delta = abs(sm_curr.get("health_bias", 0.5) - sm_prev.get("health_bias", 0.5))

            ca_prev = sm_prev.get("cuisine_affinity", {})
            ca_curr = sm_curr.get("cuisine_affinity", {})
            all_cuisines = set(ca_prev.keys()) | set(ca_curr.keys())
            ca_delta = max(
                (abs(ca_curr.get(c, 0.0) - ca_prev.get(c, 0.0)) for c in all_cuisines),
                default=0.0,
            )

            if max(sp_delta, hb_delta, ca_delta) < threshold:
                streak += 1
                if streak >= 3:
                    converged_at = i - 2  # first run of the stable streak
                    break
            else:
                streak = 0

        per_trial[trial] = converged_at

    converged_values = [v for v in per_trial.values() if v is not None]
    return {
        "per_trial": per_trial,
        "mean": float(np.mean(converged_values)) if converged_values else None,
        "median": float(np.median(converged_values)) if converged_values else None,
        "fraction_converged": len(converged_values) / len(per_trial) if per_trial else 0.0,
    }


def prediction_accuracy(results: list[dict]) -> dict:
    """Compute prediction accuracy metrics across all runs.

    Returns mean, std, and per-half comparison of prediction error.
    """
    by_trial = defaultdict(list)
    for r in results:
        by_trial[r.get("trial", 0)].append(r)

    all_errors = [r["abs_error"] for r in results]
    first_half_errors = []
    second_half_errors = []

    for trial, runs in by_trial.items():
        runs.sort(key=lambda x: (x.get("round", 0), x.get("prompt_idx", 0)))
        mid = len(runs) // 2
        first_half_errors.extend(r["abs_error"] for r in runs[:mid])
        second_half_errors.extend(r["abs_error"] for r in runs[mid:])

    return {
        "mean_abs_error": float(np.mean(all_errors)),
        "std_abs_error": float(np.std(all_errors)),
        "first_half_mean": float(np.mean(first_half_errors)) if first_half_errors else 0.0,
        "second_half_mean": float(np.mean(second_half_errors)) if second_half_errors else 0.0,
        "improvement": (
            float(np.mean(first_half_errors) - np.mean(second_half_errors))
            if first_half_errors and second_half_errors else 0.0
        ),
    }


# ── Statistical tests ────────────────────────────────────────────────────────


def compare_modes(
    results_a: list[dict],
    results_b: list[dict],
    label_a: str = "A",
    label_b: str = "B",
) -> dict:
    """Compare two modes using non-parametric statistical tests.

    Uses Mann-Whitney U test (unpaired) on per-trial mean abs_error.
    Reports effect size (rank-biserial correlation).

    Returns test statistics, p-value, and effect size.
    """
    from scipy import stats

    def _trial_means(results):
        by_trial = defaultdict(list)
        for r in results:
            by_trial[r.get("trial", 0)].append(r["abs_error"])
        return [float(np.mean(v)) for v in by_trial.values()]

    means_a = _trial_means(results_a)
    means_b = _trial_means(results_b)

    if len(means_a) < 2 or len(means_b) < 2:
        return {
            "test": "mann_whitney_u",
            "label_a": label_a,
            "label_b": label_b,
            "error": "insufficient trials for statistical test",
        }

    u_stat, p_value = stats.mannwhitneyu(means_a, means_b, alternative="two-sided")

    # Rank-biserial correlation as effect size
    n1, n2 = len(means_a), len(means_b)
    r_effect = 1 - (2 * u_stat) / (n1 * n2)

    return {
        "test": "mann_whitney_u",
        "label_a": label_a,
        "label_b": label_b,
        "mean_a": float(np.mean(means_a)),
        "mean_b": float(np.mean(means_b)),
        "std_a": float(np.std(means_a, ddof=1)),
        "std_b": float(np.std(means_b, ddof=1)),
        "u_statistic": float(u_stat),
        "p_value": float(p_value),
        "effect_size_r": float(r_effect),
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.01,
        "n_a": n1,
        "n_b": n2,
    }


def compare_learning_improvement(results: list[dict]) -> dict:
    """Wilcoxon signed-rank test: does second-half error < first-half error?

    Paired test across trials — each trial contributes a (first_half_mean, second_half_mean) pair.
    """
    from scipy import stats

    by_trial = defaultdict(list)
    for r in results:
        by_trial[r.get("trial", 0)].append(r)

    first_halves = []
    second_halves = []
    for trial, runs in by_trial.items():
        runs.sort(key=lambda x: (x.get("round", 0), x.get("prompt_idx", 0)))
        mid = len(runs) // 2
        if mid == 0:
            continue
        first_halves.append(float(np.mean([r["abs_error"] for r in runs[:mid]])))
        second_halves.append(float(np.mean([r["abs_error"] for r in runs[mid:]])))

    if len(first_halves) < 5:
        return {"test": "wilcoxon", "error": "insufficient paired samples"}

    stat, p_value = stats.wilcoxon(first_halves, second_halves, alternative="greater")

    return {
        "test": "wilcoxon_signed_rank",
        "hypothesis": "first_half_error > second_half_error (agent improves)",
        "first_half_means": first_halves,
        "second_half_means": second_halves,
        "mean_improvement": float(np.mean(first_halves) - np.mean(second_halves)),
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant_005": p_value < 0.05,
    }


def ingredient_weight_divergence(results: list[dict]) -> dict:
    """Track how ingredient weights diverge from initial state over time.

    Measures L2 norm of weight vector at each step, averaged across trials.
    Shows whether the world model develops meaningful structure.
    """
    by_trial = defaultdict(list)
    for r in results:
        by_trial[r.get("trial", 0)].append(r)

    for trial in by_trial:
        by_trial[trial].sort(key=lambda x: (x.get("round", 0), x.get("prompt_idx", 0)))

    all_norms = []
    for trial, runs in by_trial.items():
        norms = []
        for r in runs:
            weights = r.get("updated_world_model", {}).get("ingredient_weights", {})
            if weights:
                vals = np.array(list(weights.values()))
                norms.append(float(np.linalg.norm(vals)))
            else:
                norms.append(0.0)
        all_norms.append(norms)

    if not all_norms:
        return {"steps": [], "mean_norm": [], "std_norm": []}

    max_len = max(len(n) for n in all_norms)
    padded = np.full((len(all_norms), max_len), np.nan)
    for i, n in enumerate(all_norms):
        padded[i, :len(n)] = n

    return {
        "steps": list(range(max_len)),
        "mean_norm": np.nanmean(padded, axis=0).tolist(),
        "std_norm": np.nanstd(padded, axis=0).tolist(),
    }


# ── Summary report ───────────────────────────────────────────────────────────


def full_analysis(all_results: dict) -> dict:
    """Run all metrics on a complete experiment result set.

    Parameters:
        all_results: dict mapping mode_name → list of run records

    Returns a nested dict with all metrics per mode and cross-mode comparisons.
    """
    report = {"per_mode": {}, "comparisons": {}}

    for mode, results in all_results.items():
        report["per_mode"][mode] = {
            "learning_curve": learning_curve(results),
            "prediction_accuracy": prediction_accuracy(results),
            "preference_stability": preference_stability(results),
            "convergence_speed": convergence_speed(results),
            "ingredient_divergence": ingredient_weight_divergence(results),
        }

        # Within-mode improvement test
        if len(set(r.get("trial", 0) for r in results)) >= 5:
            report["per_mode"][mode]["improvement_test"] = compare_learning_improvement(results)

    # Cross-mode comparisons (each learning mode vs baseline)
    if "baseline" in all_results:
        for mode in all_results:
            if mode == "baseline":
                continue
            report["comparisons"][f"{mode}_vs_baseline"] = compare_modes(
                all_results[mode],
                all_results["baseline"],
                label_a=mode,
                label_b="baseline",
            )

    # Full model vs individual components
    if "full_model" in all_results and "world_model_only" in all_results:
        report["comparisons"]["full_vs_world_only"] = compare_modes(
            all_results["full_model"],
            all_results["world_model_only"],
            label_a="full_model",
            label_b="world_model_only",
        )

    if "full_model" in all_results and "self_model_only" in all_results:
        report["comparisons"]["full_vs_self_only"] = compare_modes(
            all_results["full_model"],
            all_results["self_model_only"],
            label_a="full_model",
            label_b="self_model_only",
        )

    return report


def print_report(report: dict) -> None:
    """Pretty-print the analysis report to stdout."""
    print(f"\n{'='*70}")
    print("EXPERIMENT ANALYSIS REPORT")
    print(f"{'='*70}")

    for mode, metrics in report["per_mode"].items():
        print(f"\n--- {mode.upper()} ---")
        pa = metrics["prediction_accuracy"]
        print(f"  Mean abs error: {pa['mean_abs_error']:.4f} ± {pa['std_abs_error']:.4f}")
        print(f"  First half: {pa['first_half_mean']:.4f}  Second half: {pa['second_half_mean']:.4f}")
        print(f"  Improvement: {pa['improvement']:+.4f}")

        cs = metrics["convergence_speed"]
        print(f"  Convergence: {cs['fraction_converged']*100:.0f}% of trials converged"
              f"  (mean step: {cs['mean'] or 'N/A'})")

        if "improvement_test" in metrics:
            it = metrics["improvement_test"]
            if "error" not in it:
                print(f"  Wilcoxon improvement test: p={it['p_value']:.4f} "
                      f"({'*' if it['significant_005'] else 'ns'})")

    if report["comparisons"]:
        print(f"\n--- CROSS-MODE COMPARISONS ---")
        for name, comp in report["comparisons"].items():
            if "error" in comp:
                print(f"  {name}: {comp['error']}")
                continue
            print(f"  {name}:")
            print(f"    {comp['label_a']}: {comp['mean_a']:.4f} ± {comp['std_a']:.4f}")
            print(f"    {comp['label_b']}: {comp['mean_b']:.4f} ± {comp['std_b']:.4f}")
            print(f"    Mann-Whitney U: p={comp['p_value']:.4f} "
                  f"r={comp['effect_size_r']:.3f} "
                  f"({'**' if comp['significant_001'] else '*' if comp['significant_005'] else 'ns'})")
