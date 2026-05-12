"""
EpisodicMemory: indexed store of past substitution outcomes.

Keyed by (cuisine, ingredient). Queried at pipeline entry so the consciousness
module receives proven substitutions before running CRIT or Exploratory.
Written after each dish with the outcome (succeeded / failed).
"""
from __future__ import annotations

from collections import defaultdict


class EpisodicMemory:
    """
    Architectural change 3 — episodic memory module.

    The base pipeline accumulates `past_substitutions` as a flat list injected
    per-dish in run_batch, but it is unindexed, ephemeral, and not queryable
    by (cuisine, ingredient). This module replaces that with a structured store
    that: (a) retrieves the best known substitute for a given cuisine×ingredient
    pair before Consciousness runs, and (b) short-circuits CRIT for memory-proven
    pairs so the system doesn't re-validate what it already knows works.
    """

    def __init__(self):
        # (cuisine, ingredient) → [{substitute, score, succeeded}]
        self._store: dict[tuple[str, str], list[dict]] = defaultdict(list)

    def store(self, cuisine: str, ingredient: str, substitute: str, score: float, succeeded: bool):
        self._store[(cuisine, ingredient)].append(
            {"substitute": substitute, "score": score, "succeeded": succeeded}
        )

    def retrieve_best(self, cuisine: str, ingredient: str) -> str | None:
        """Returns the highest-scoring successful substitute, or None."""
        entries = self._store.get((cuisine, ingredient), [])
        successful = [e for e in entries if e["succeeded"] and e["score"] >= 0.6]
        if not successful:
            return None
        return max(successful, key=lambda e: e["score"])["substitute"]

    def inject_into_schema(self, schema: dict) -> dict:
        """
        Returns a copy of schema with memory-retrieved substitutions prepended
        to past_substitutions. The consciousness module checks this field first
        and skips CRIT for ingredients already covered by memory.
        """
        cuisine = schema.get("cuisine", "")
        constraints = schema.get("constraints", [])
        memory_subs = []
        for ingredient in constraints:
            best = self.retrieve_best(cuisine, ingredient)
            if best:
                memory_subs.append(
                    {ingredient: {"substitute": best, "source": "memory", "validity_score": 0.8}}
                )
        if not memory_subs:
            return schema
        schema = dict(schema)
        schema["past_substitutions"] = memory_subs + schema.get("past_substitutions", [])
        return schema

    def update_from_result(self, result: dict):
        """Called after each dish to store outcomes keyed by cuisine×ingredient."""
        cuisine = result.get("cuisine", "")
        violations = set(result.get("violations", []))
        for ingredient, info in result.get("substitutions_used", {}).items():
            substitute = info.get("substitute", "")
            score = info.get("validity_score", 0.5)
            if not substitute:
                continue
            self.store(cuisine, ingredient, substitute, score, ingredient not in violations)
