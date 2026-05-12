"""
Planner: pre-draft constraint prediction.

Predicts which banned ingredients a dish will require before any full recipe
is generated, enabling escalation decisions upstream of generation rather than
after a draft has already been committed.
"""
from __future__ import annotations

import re
import torch
from config import BANNED_INGREDIENTS


class Planner:
    """
    Architectural change 1 — Planner/Drafter split.

    Replaces the post-draft risk classification in UnconsciousnessModule with a
    short pre-generation inference pass. The escalation decision is made before
    the Drafter generates a full recipe, so conscious attention is allocated
    based on prediction rather than inspection of an already-committed draft.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def _generate(self, prompt: str, max_new_tokens: int = 64) -> str:
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        generated = output[0][input_ids.shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def predict_constraints(self, dish: str, cuisine: str) -> list[str]:
        """
        Returns the subset of BANNED_INGREDIENTS predicted to appear in an
        authentic recipe for this dish. Short inference — no full recipe generated.
        """
        banned_str = ", ".join(BANNED_INGREDIENTS)
        prompt = (
            f"For an authentic {cuisine} recipe for '{dish}', "
            f"which of these ingredients would typically be used: {banned_str}?\n"
            f"Reply with only the ingredient names that apply, comma-separated. "
            f"If none apply, reply 'none'."
        )
        response = self._generate(prompt)
        return [
            ing for ing in BANNED_INGREDIENTS
            if re.search(r"\b" + re.escape(ing) + r"\b", response, re.IGNORECASE)
        ]

    def classify_risk(self, predicted: list[str], cuisine_prior: float) -> float:
        """
        Binary rule: any predicted constraint → always escalate (risk = 1.0).
        Falls back to cuisine prior only when prediction is clean.
        This replaces the fractional detection_signal formula in UnconsciousnessModule
        that allowed confirmed violations to score below the escalation threshold.
        """
        if predicted:
            return 1.0
        return cuisine_prior
