from __future__ import annotations

import re
from config import BANNED_INGREDIENTS, CUISINE_RISK_MAP

_BANNED_PATTERNS = {
    ing: re.compile(r'\b' + re.escape(ing) + r'\b', re.IGNORECASE)
    for ing in BANNED_INGREDIENTS
}


def _detect_banned(text: str) -> list[str]:
    return [ing for ing, pat in _BANNED_PATTERNS.items() if pat.search(text)]


class Effector:
    """
    Outputs the final recipe and sends feedback to the Unconsciousness module's
    MFQ scheduler. Feedback recalibrates cuisine risk scores over time so the
    system learns which cuisines need conscious attention.

    Analogous to CoCoMo's effector module: acts on consciousness output,
    then sends feedback signals back to the unconsciousness module.
    """

    def output(self, recipe: str, schema: dict, was_conscious: bool, metadata: dict) -> dict:
        """Packages recipe + all metadata into a result dict."""
        violations = _detect_banned(recipe)
        return {
            "dish": schema["dish"],
            "cuisine": schema["cuisine"],
            "recipe": recipe,
            "was_conscious": was_conscious,
            "risk_score": schema["risk_score"],
            "violations": violations,
            "num_violations": len(violations),
            "substitutions_used": metadata.get("validated_substitutions", {}),
            "prompt_used": metadata.get("prompt_used", ""),
        }

    def send_feedback(self, result: dict, scheduler) -> dict:
        """
        Computes a feedback signal and uses it to update the MFQ scheduler's
        cuisine risk scores.

        If the effector found violations in a cuisine thought to be low-risk,
        the risk score for that cuisine is bumped up.
        If a high-risk cuisine came through cleanly, risk is nudged down.
        Returns the feedback dict for logging.
        """
        cuisine = result["cuisine"]
        violated = result["num_violations"] > 0
        # Read from MFQ's live overrides first, fall back to static config.
        # Ensures feedback deltas are computed against the current adaptive
        # estimate, not the stale prior from config.
        current_risk = scheduler._cuisine_risk_overrides.get(
            cuisine, CUISINE_RISK_MAP.get(cuisine, 0.5)
        )

        if violated:
            # Underestimated risk — increase by 0.1
            new_risk = min(1.0, current_risk + 0.1)
        else:
            # Risk well-managed — small decay
            new_risk = max(0.0, current_risk - 0.05)

        scheduler.update_cuisine_risk(cuisine, new_risk)

        # Update past_substitutions memory with what worked
        working_subs = {
            k: v for k, v in result["substitutions_used"].items()
            if k not in result["violations"]
        }

        feedback = {
            "cuisine": cuisine,
            "violated": violated,
            "old_risk": current_risk,
            "new_risk": new_risk,
            "working_substitutions": working_subs,
        }
        return feedback
