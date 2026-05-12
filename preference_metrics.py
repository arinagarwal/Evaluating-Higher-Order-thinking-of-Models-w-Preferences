"""
Revealed Preference Metrics
============================
Measures agent preferences through observable behavioral choices (the recipes
it generates), not just internal model parameters.

This is the critical distinction for a paper: internal state changes are
necessary but not sufficient. We need to show that parameter changes cause
measurable behavioral differences.

Three layers of preference measurement:
1. Stated preferences  — SelfModel parameters (spice_preference, health_bias, etc.)
2. Revealed preferences — what the agent actually chooses (ingredient patterns, cuisine
   distribution, complexity trends)
3. Preference alignment — correlation between stated and revealed preferences

If stated and revealed preferences correlate AND both stabilize over time,
that's strong evidence of genuine preference formation.
"""

import math
from collections import Counter, defaultdict
from typing import Optional

import numpy as np


# ── Ingredient-level keywords (shared with Learning_agent.py) ────────────────

SPICY_KEYWORDS = [
    "chili", "pepper", "hot sauce", "cayenne", "jalapeño",
    "habanero", "sriracha", "wasabi", "horseradish", "ginger",
    "chili flakes", "paprika", "cumin",
]

PROTEIN_KEYWORDS = [
    "chicken", "beef", "pork", "lamb", "turkey", "duck", "veal",
    "bacon", "sausage", "ham", "steak", "fish", "salmon", "tuna",
    "shrimp", "prawn", "crab", "lobster", "tofu", "eggs", "chickpeas",
]

VEGETABLE_KEYWORDS = [
    "spinach", "mushroom", "broccoli", "bell pepper", "zucchini",
    "carrot", "onion", "garlic", "potato", "tomato", "lettuce",
    "kale", "cabbage", "celery", "corn", "peas", "beans",
]

DAIRY_KEYWORDS = [
    "milk", "butter", "cream", "cheese", "parmesan", "mozzarella",
    "feta", "yogurt", "sour cream",
]

GRAIN_KEYWORDS = [
    "pasta", "spaghetti", "rigatoni", "rice", "quinoa", "flour",
    "bread", "noodle", "couscous", "tortilla",
]


# ── Helper: extract recipe features from a run record ────────────────────────


def _get_ingredients(run_record: dict) -> list[str]:
    """Extract lowercase ingredient names from a run record."""
    recipe = run_record.get("recipe", {})
    ingredients = recipe.get("ingredients_required", [])
    return [ing.get("name", "").lower().strip() for ing in ingredients if ing.get("name")]


def _get_cuisine(run_record: dict) -> str:
    """Extract normalized cuisine label from a run record."""
    recipe = run_record.get("recipe", {})
    fit = recipe.get("fit_to_intent", {})
    return fit.get("cuisine_alignment", "").strip().lower()


def _keyword_ratio(ingredients: list[str], keywords: list[str]) -> float:
    """Fraction of ingredients matching any keyword in the list."""
    if not ingredients:
        return 0.0
    count = sum(
        1 for ing in ingredients
        if any(kw in ing for kw in keywords)
    )
    return count / len(ingredients)


# ── 1. Revealed Preference Extraction ────────────────────────────────────────


def revealed_spice_ratio(run_record: dict) -> float:
    """What fraction of this recipe's ingredients are spicy?"""
    return _keyword_ratio(_get_ingredients(run_record), SPICY_KEYWORDS)


def revealed_protein_ratio(run_record: dict) -> float:
    """What fraction of this recipe's ingredients are proteins?"""
    return _keyword_ratio(_get_ingredients(run_record), PROTEIN_KEYWORDS)


def revealed_vegetable_ratio(run_record: dict) -> float:
    """What fraction of this recipe's ingredients are vegetables?"""
    return _keyword_ratio(_get_ingredients(run_record), VEGETABLE_KEYWORDS)


def revealed_complexity(run_record: dict) -> int:
    """Number of ingredients — proxy for recipe complexity preference."""
    return len(_get_ingredients(run_record))


def revealed_cuisine(run_record: dict) -> str:
    """Which cuisine did the agent choose?"""
    return _get_cuisine(run_record)


# ── 2. Trajectory Analysis ───────────────────────────────────────────────────


def behavioral_trajectories(results: list[dict]) -> dict:
    """Compute revealed preference values at each step, per trial.

    Returns dict with keys:
        - spice_ratio: {trial: [values]}
        - protein_ratio: {trial: [values]}
        - vegetable_ratio: {trial: [values]}
        - complexity: {trial: [values]}
        - cuisine_concentration: {trial: [values]}  (Herfindahl index)
    """
    by_trial = defaultdict(list)
    for r in results:
        by_trial[r.get("trial", 0)].append(r)

    for trial in by_trial:
        by_trial[trial].sort(key=lambda x: (x.get("round", 0), x.get("prompt_idx", 0)))

    trajectories = {
        "spice_ratio": {},
        "protein_ratio": {},
        "vegetable_ratio": {},
        "complexity": {},
        "cuisine_concentration": {},
    }

    for trial, runs in by_trial.items():
        spice_vals, protein_vals, veg_vals, complexity_vals = [], [], [], []
        cuisine_counts = Counter()
        concentration_vals = []

        for i, r in enumerate(runs):
            spice_vals.append(revealed_spice_ratio(r))
            protein_vals.append(revealed_protein_ratio(r))
            veg_vals.append(revealed_vegetable_ratio(r))
            complexity_vals.append(revealed_complexity(r))

            # Rolling cuisine concentration (Herfindahl-Hirschman Index)
            cuisine = revealed_cuisine(r)
            if cuisine:
                cuisine_counts[cuisine] += 1
            total = sum(cuisine_counts.values())
            if total > 0:
                shares = [c / total for c in cuisine_counts.values()]
                hhi = sum(s ** 2 for s in shares)
                concentration_vals.append(hhi)
            else:
                concentration_vals.append(1.0)

        trajectories["spice_ratio"][trial] = spice_vals
        trajectories["protein_ratio"][trial] = protein_vals
        trajectories["vegetable_ratio"][trial] = veg_vals
        trajectories["complexity"][trial] = complexity_vals
        trajectories["cuisine_concentration"][trial] = concentration_vals

    return trajectories


# ── 3. Stated vs Revealed Preference Alignment ──────────────────────────────


def preference_alignment(results: list[dict]) -> dict:
    """Compute correlation between stated preferences (SelfModel) and
    revealed preferences (actual recipe choices) across all runs.

    High correlation = the agent's internal model accurately reflects
    its behavioral tendencies. This is the key evidence that preferences
    are "real" — not just numbers that change but don't affect behavior.

    Returns Pearson r and p-value for each dimension.
    """
    from scipy import stats

    stated_spice = []
    revealed_spice = []
    stated_health = []
    revealed_health = []

    for r in results:
        sm = r.get("updated_self_model", {})
        stated_spice.append(sm.get("spice_preference", 0.5))
        revealed_spice.append(revealed_spice_ratio(r))

        stated_health.append(sm.get("health_bias", 0.5))
        # Use vegetable ratio as a proxy for health-oriented behavior
        revealed_health.append(revealed_vegetable_ratio(r))

    alignment = {}

    if len(stated_spice) >= 3:
        # Check for constant arrays (baseline mode never updates)
        if np.std(stated_spice) < 1e-10 or np.std(revealed_spice) < 1e-10:
            alignment["spice"] = {
                "pearson_r": 0.0,
                "p_value": 1.0,
                "significant": False,
                "n": len(stated_spice),
                "note": "constant input (no variation in stated or revealed values)",
            }
        else:
            r_spice, p_spice = stats.pearsonr(stated_spice, revealed_spice)
            alignment["spice"] = {
                "pearson_r": float(r_spice),
                "p_value": float(p_spice),
                "significant": p_spice < 0.05,
                "n": len(stated_spice),
            }

    if len(stated_health) >= 3:
        if np.std(stated_health) < 1e-10 or np.std(revealed_health) < 1e-10:
            alignment["health"] = {
                "pearson_r": 0.0,
                "p_value": 1.0,
                "significant": False,
                "n": len(stated_health),
                "note": "constant input (no variation in stated or revealed values)",
            }
        else:
            r_health, p_health = stats.pearsonr(stated_health, revealed_health)
            alignment["health"] = {
                "pearson_r": float(r_health),
                "p_value": float(p_health),
                "significant": p_health < 0.05,
                "n": len(stated_health),
            }

    return alignment


# ── 4. Preference Distinctiveness ────────────────────────────────────────────


def preference_distinctiveness(all_results: dict) -> dict:
    """Measure whether different modes develop distinguishable preference profiles.

    For each pair of modes, compute the distributional distance between their
    revealed preference vectors (final-state ingredient choice distributions).

    If learning modes produce statistically different behavioral profiles from
    baseline, that's evidence the learning mechanism creates genuine preferences
    rather than random variation.
    """
    from scipy import stats

    mode_profiles = {}
    for mode, results in all_results.items():
        by_trial = defaultdict(list)
        for r in results:
            by_trial[r.get("trial", 0)].append(r)

        # Use second-half runs only (after learning has occurred)
        trial_profiles = []
        for trial, runs in by_trial.items():
            runs.sort(key=lambda x: (x.get("round", 0), x.get("prompt_idx", 0)))
            mid = len(runs) // 2
            second_half = runs[mid:]
            if not second_half:
                continue

            profile = {
                "mean_spice": float(np.mean([revealed_spice_ratio(r) for r in second_half])),
                "mean_protein": float(np.mean([revealed_protein_ratio(r) for r in second_half])),
                "mean_vegetable": float(np.mean([revealed_vegetable_ratio(r) for r in second_half])),
                "mean_complexity": float(np.mean([revealed_complexity(r) for r in second_half])),
            }
            trial_profiles.append(profile)

        mode_profiles[mode] = trial_profiles

    # Compare each learning mode against baseline
    comparisons = {}
    if "baseline" not in mode_profiles:
        return comparisons

    baseline_profiles = mode_profiles["baseline"]
    for mode, profiles in mode_profiles.items():
        if mode == "baseline" or len(profiles) < 2 or len(baseline_profiles) < 2:
            continue

        mode_comparison = {}
        for dim in ["mean_spice", "mean_protein", "mean_vegetable", "mean_complexity"]:
            mode_vals = [p[dim] for p in profiles]
            base_vals = [p[dim] for p in baseline_profiles]

            if len(mode_vals) >= 2 and len(base_vals) >= 2:
                u_stat, p_val = stats.mannwhitneyu(mode_vals, base_vals, alternative="two-sided")
                mode_comparison[dim] = {
                    "mode_mean": float(np.mean(mode_vals)),
                    "baseline_mean": float(np.mean(base_vals)),
                    "difference": float(np.mean(mode_vals) - np.mean(base_vals)),
                    "p_value": float(p_val),
                    "significant": p_val < 0.05,
                }

        comparisons[f"{mode}_vs_baseline"] = mode_comparison

    return comparisons


# ── 5. Ingredient Repertoire Analysis ────────────────────────────────────────


def ingredient_repertoire(results: list[dict]) -> dict:
    """Track how the agent's ingredient vocabulary evolves.

    Measures:
    - Unique ingredients used (cumulative)
    - Ingredient reuse rate (how often it picks the same ingredients)
    - Top-k most frequently chosen ingredients

    An agent developing preferences should show increasing reuse rate
    (it keeps going back to ingredients it "likes") and a narrowing
    active vocabulary.
    """
    by_trial = defaultdict(list)
    for r in results:
        by_trial[r.get("trial", 0)].append(r)

    for trial in by_trial:
        by_trial[trial].sort(key=lambda x: (x.get("round", 0), x.get("prompt_idx", 0)))

    all_cumulative_unique = []
    all_reuse_rates = []
    final_counters = []

    for trial, runs in by_trial.items():
        seen = set()
        cumulative_unique = []
        total_ingredients = 0
        reuse_count = 0
        counter = Counter()

        for r in runs:
            ingredients = _get_ingredients(r)
            for ing in ingredients:
                total_ingredients += 1
                counter[ing] += 1
                if ing in seen:
                    reuse_count += 1
                else:
                    seen.add(ing)
            cumulative_unique.append(len(seen))

        all_cumulative_unique.append(cumulative_unique)
        reuse_rate = reuse_count / total_ingredients if total_ingredients > 0 else 0.0
        all_reuse_rates.append(reuse_rate)
        final_counters.append(counter)

    # Aggregate top ingredients across trials
    combined = Counter()
    for c in final_counters:
        combined.update(c)
    top_ingredients = combined.most_common(15)

    # Mean cumulative unique curve
    if all_cumulative_unique:
        max_len = max(len(c) for c in all_cumulative_unique)
        padded = np.full((len(all_cumulative_unique), max_len), np.nan)
        for i, c in enumerate(all_cumulative_unique):
            padded[i, :len(c)] = c
        mean_cumulative = np.nanmean(padded, axis=0).tolist()
    else:
        mean_cumulative = []

    return {
        "mean_reuse_rate": float(np.mean(all_reuse_rates)) if all_reuse_rates else 0.0,
        "std_reuse_rate": float(np.std(all_reuse_rates)) if all_reuse_rates else 0.0,
        "mean_cumulative_unique": mean_cumulative,
        "top_ingredients": top_ingredients,
    }


# ── 6. Full Preference Report ────────────────────────────────────────────────


def full_preference_analysis(all_results: dict) -> dict:
    """Run all preference metrics on experiment results.

    Returns a structured report with:
    - Per-mode behavioral trajectories and repertoire analysis
    - Per-mode stated-vs-revealed alignment
    - Cross-mode distinctiveness tests
    """
    report = {"per_mode": {}, "distinctiveness": {}}

    for mode, results in all_results.items():
        traj = behavioral_trajectories(results)

        # Compute mean final values across trials for each behavioral dimension
        final_means = {}
        for dim, trial_data in traj.items():
            final_vals = [vals[-1] for vals in trial_data.values() if vals]
            final_means[dim] = {
                "mean": float(np.mean(final_vals)) if final_vals else 0.0,
                "std": float(np.std(final_vals)) if final_vals else 0.0,
            }

        report["per_mode"][mode] = {
            "behavioral_trajectories": {
                dim: {
                    "n_trials": len(trial_data),
                    "final_state": final_means.get(dim, {}),
                }
                for dim, trial_data in traj.items()
            },
            "ingredient_repertoire": ingredient_repertoire(results),
            "alignment": preference_alignment(results),
        }

    report["distinctiveness"] = preference_distinctiveness(all_results)

    return report


def print_preference_report(report: dict) -> None:
    """Pretty-print the preference analysis."""
    print(f"\n{'='*70}")
    print("PREFERENCE ANALYSIS REPORT")
    print(f"{'='*70}")

    for mode, data in report["per_mode"].items():
        print(f"\n--- {mode.upper()} ---")

        # Behavioral final state
        bt = data["behavioral_trajectories"]
        for dim, info in bt.items():
            fs = info.get("final_state", {})
            print(f"  {dim}: mean={fs.get('mean', 0):.3f} ± {fs.get('std', 0):.3f}")

        # Repertoire
        rep = data["ingredient_repertoire"]
        print(f"  Ingredient reuse rate: {rep['mean_reuse_rate']:.3f} ± {rep['std_reuse_rate']:.3f}")
        top3 = rep["top_ingredients"][:3]
        if top3:
            print(f"  Top ingredients: {', '.join(f'{name} ({count})' for name, count in top3)}")

        # Alignment
        align = data["alignment"]
        for dim, info in align.items():
            sig = "*" if info["significant"] else "ns"
            print(f"  Stated↔Revealed {dim}: r={info['pearson_r']:.3f} p={info['p_value']:.4f} ({sig})")

    if report["distinctiveness"]:
        print(f"\n--- BEHAVIORAL DISTINCTIVENESS (vs baseline) ---")
        for comp_name, dims in report["distinctiveness"].items():
            print(f"  {comp_name}:")
            for dim, info in dims.items():
                sig = "*" if info["significant"] else "ns"
                print(f"    {dim}: Δ={info['difference']:+.3f} p={info['p_value']:.4f} ({sig})")
