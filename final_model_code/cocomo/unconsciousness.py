from __future__ import annotations

import re
import heapq
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from config import (
    MODEL_NAME, BANNED_INGREDIENTS, CUISINE_RISK_MAP,
    DRAFT_GENERATION_CONFIG, MFQ_ESCALATION_THRESHOLD,
    get_bnb_compute_dtype,
)

_BANNED_PATTERNS = {
    ing: re.compile(r'\b' + re.escape(ing) + r'\b', re.IGNORECASE)
    for ing in BANNED_INGREDIENTS
}


def _load_model_and_tokenizer(model_name: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=get_bnb_compute_dtype(),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    return model, tokenizer


def _detect_banned(text: str) -> list[str]:
    return [ing for ing, pat in _BANNED_PATTERNS.items() if pat.search(text)]


class MFQScheduler:
    """
    Multi-level Feedback Queue scheduler.
    Higher risk_score → higher priority (processed first by Consciousness).
    Items that go through Consciousness and come back with feedback get their
    cuisine risk score updated so future identical cuisines route better.
    """

    def __init__(self):
        # Max-heap via negated priority
        self._heap: list[tuple[float, int, dict]] = []
        self._counter = 0
        self._cuisine_risk_overrides: dict[str, float] = {}

    def push(self, schema: dict):
        risk = self._cuisine_risk_overrides.get(
            schema["cuisine"], schema["risk_score"]
        )
        heapq.heappush(self._heap, (-risk, self._counter, schema))
        self._counter += 1

    def pop(self):
        if not self._heap:
            return None
        _, _, schema = heapq.heappop(self._heap)
        return schema

    def update_cuisine_risk(self, cuisine: str, new_risk: float):
        """Effector feedback recalibrates future risk estimates for a cuisine."""
        self._cuisine_risk_overrides[cuisine] = max(0.0, min(1.0, new_risk))

    def __len__(self):
        return len(self._heap)


class UnconsciousnessModule:
    """
    Performs fast, habitual recipe drafting (System-1 style).
    Classifies the draft for constraint risk and decides whether to escalate
    to the Consciousness module.

    Analogous to CoCoMo's unconsciousness module: runs the MFQ scheduler,
    does discriminative classification, and routes high-risk tasks upward.
    """

    def __init__(self, model=None, tokenizer=None):
        if model is None:
            model, tokenizer = _load_model_and_tokenizer(MODEL_NAME)
        self.model = model
        self.tokenizer = tokenizer
        self.scheduler = MFQScheduler()

    def draft_recipe(self, schema: dict) -> str:
        """Greedy, fast generation — unconscious and habitual."""
        prompt = (
            f"Write a recipe for {schema['dish']}. "
            "Include a title, an Ingredients: section, and a Instructions: section."
        )
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                pad_token_id=self.tokenizer.eos_token_id,
                **DRAFT_GENERATION_CONFIG,
            )
        generated = output[0][input_ids.shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def classify_risk(self, draft: str, schema: dict) -> float:
        """
        Combines two signals:
        1. Detected banned ingredients in the draft (direct evidence).
        2. Cuisine-level prior risk (indirect, structural).
        Returns a float in [0, 1].
        """
        found = _detect_banned(draft)
        detection_signal = min(1.0, len(found) / len(BANNED_INGREDIENTS))
        cuisine_prior = schema["risk_score"]
        # Weight detection heavily if we already have evidence
        combined = 0.6 * detection_signal + 0.4 * cuisine_prior
        return round(combined, 4)

    def should_escalate(self, risk: float, threshold: float = MFQ_ESCALATION_THRESHOLD) -> bool:
        return risk > threshold
