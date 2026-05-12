from __future__ import annotations

from tasks import TASK_CATEGORIES
from config import BANNED_APIS, TASK_RISK_MAP


class Receptor:
    def process(self, task: str, past_substitutions=None) -> dict:
        category = TASK_CATEGORIES.get(task, "Unknown")
        risk_score = TASK_RISK_MAP.get(category, TASK_RISK_MAP["Unknown"])
        return {
            "task":               task,
            "category":           category,
            "constraints":        BANNED_APIS,
            "risk_score":         risk_score,
            "past_substitutions": past_substitutions or [],
        }
