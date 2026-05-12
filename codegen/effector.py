from __future__ import annotations

from detector import detect_banned
from config import TASK_RISK_MAP


class Effector:
    def output(self, code: str, schema: dict, was_conscious: bool, metadata: dict) -> dict:
        violations = detect_banned(code)
        return {
            "task":             schema["task"],
            "category":         schema["category"],
            "code":             code,
            "was_conscious":    was_conscious,
            "risk_score":       schema["risk_score"],
            "violations":       violations,
            "num_violations":   len(violations),
            "substitutions_used": metadata.get("validated_substitutions", {}),
            "prompt_used":      metadata.get("prompt_used", ""),
        }

    def send_feedback(self, result: dict, scheduler) -> dict:
        category = result["category"]
        violated = result["num_violations"] > 0
        current_risk = scheduler._category_risk_overrides.get(
            category, TASK_RISK_MAP.get(category, 0.5)
        )
        new_risk = min(1.0, current_risk + 0.1) if violated else max(0.0, current_risk - 0.05)
        scheduler.update_category_risk(category, new_risk)

        working_subs = {
            k: v for k, v in result["substitutions_used"].items()
            if k not in result["violations"]
        }
        return {
            "category":            category,
            "violated":            violated,
            "old_risk":            current_risk,
            "new_risk":            new_risk,
            "working_substitutions": working_subs,
        }
