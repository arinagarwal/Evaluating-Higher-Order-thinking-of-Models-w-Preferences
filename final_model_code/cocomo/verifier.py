"""
Verifier: post-generation output checker with repair loop.

Sits between the ConsciousnessModule and Effector. If banned ingredients are
still present in the conscious output, re-enters Consciousness with explicit
repair annotations rather than shipping a violated recipe.
"""
from __future__ import annotations

from unconsciousness import _detect_banned


class Verifier:
    """
    Architectural change 2 — iterative verification loop.

    The base pipeline is single-pass: Consciousness generates once and the
    result goes directly to Effector regardless of what's in it. This module
    closes that gap. It checks the conscious output and, on violation, feeds
    an annotated schema back into Consciousness for a targeted repair pass.
    """

    def verify_and_repair(
        self,
        recipe: str,
        schema: dict,
        conscious_module,
        max_attempts: int = 2,
    ) -> tuple[str, int]:
        """
        Returns (final_recipe, num_repair_attempts).
        Exits immediately with 0 attempts if the recipe is already clean.
        Each repair pass injects `repair_violations` into the schema so the
        consciousness module's prompt explicitly names the remaining offenders.
        """
        current = recipe
        for attempt in range(max_attempts):
            violations = _detect_banned(current)
            if not violations:
                return current, attempt
            repair_schema = dict(schema)
            repair_schema["repair_violations"] = violations
            repaired, _ = conscious_module.generate(repair_schema)
            current = repaired
        return current, max_attempts
