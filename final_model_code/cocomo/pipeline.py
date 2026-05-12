"""
CoCoMo Pipeline — orchestrates the full stimulus-response loop:

  Dish → Receptor → Unconsciousness (MFQ) → [Consciousness if risk > threshold]
       → Effector (output + feedback) → MFQ update

Usage (single dish):
    python cocomo/pipeline.py

Usage (from code):
    pipeline = CoCoMoPipeline()
    result = pipeline.run("Spaghetti Carbonara")
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from config import MODEL_NAME, MFQ_ESCALATION_THRESHOLD, get_bnb_compute_dtype
from receptor import Receptor
from unconsciousness import UnconsciousnessModule, _detect_banned
from consciousness import ConsciousnessModule
from effector import Effector


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


class CoCoMoPipeline:
    """
    Full CoCoMo pipeline. Loads one model instance shared across all modules
    (Unconsciousness + Consciousness) to avoid double memory overhead.
    """

    def __init__(self, model=None, tokenizer=None, escalation_threshold: float = MFQ_ESCALATION_THRESHOLD):
        if model is None:
            print(f"Loading model: {MODEL_NAME}")
            model, tokenizer = _load_model_and_tokenizer(MODEL_NAME)

        self.receptor = Receptor()
        self.unconscious = UnconsciousnessModule(model=model, tokenizer=tokenizer)
        self.conscious = ConsciousnessModule(model=model, tokenizer=tokenizer)
        self.effector = Effector()
        self.escalation_threshold = escalation_threshold
        self.feedback_log: list[dict] = []
        self.risk_snapshots: list[dict] = []  # per-dish MFQ state, populated by run_batch

    def run(self, dish: str, past_substitutions=None) -> dict:
        """
        Run one dish through the full Receptor → Unconscious → [Conscious] → Effector loop.
        Returns the result dict from Effector.output().
        """
        schema = self.receptor.process(dish, past_substitutions=past_substitutions)
        return self._run_from_schema(schema)

    def _run_from_schema(self, schema: dict) -> dict:
        """
        Inner pipeline from a pre-built schema onward.
        Separated so run_batch() can push all schemas to the MFQ heap first,
        then pop and process them in risk-priority order.
        """
        # Unconsciousness: fast draft + real-time risk classification
        draft = self.unconscious.draft_recipe(schema)
        risk = self.unconscious.classify_risk(draft, schema)
        schema["risk_score"] = risk  # update with real-time signal
        schema["draft"] = draft      # pass draft so CRIT only checks relevant ingredients

        escalated = self.unconscious.should_escalate(risk, self.escalation_threshold)

        # Consciousness (only if MFQ escalates)
        if escalated:
            recipe, metadata = self.conscious.generate(schema)
            was_conscious = True
        else:
            recipe = draft
            metadata = {"validated_substitutions": {}, "prompt_used": "unconscious_draft"}
            was_conscious = False

        # Effector: package output + send feedback to MFQ
        result = self.effector.output(recipe, schema, was_conscious, metadata)
        result["draft_violations"] = _detect_banned(draft)
        feedback = self.effector.send_feedback(result, self.unconscious.scheduler)
        self.feedback_log.append(feedback)

        result["feedback"] = feedback
        return result

    def run_batch(self, dishes: list[str]) -> list[dict]:
        """
        MFQ-ordered batch processing.

        Phase 1 — receptor pre-pass: build schemas for all dishes and push onto
        the MFQ heap. High-risk cuisines (Italian, French) get higher priority
        and will be processed first.

        Phase 2 — pop in priority order: highest-risk dishes are processed first,
        so their feedback recalibrates cuisine risk scores before lower-risk dishes
        of the same cuisine are encountered. Conscious attention is front-loaded
        where constraint risk is highest.
        """
        # Phase 1: receptor pass → push all onto MFQ heap
        for dish in dishes:
            schema = self.receptor.process(dish)
            self.unconscious.scheduler.push(schema)

        # Phase 2: pop by risk priority → process each through the pipeline
        results = []
        cumulative_subs: list[dict] = []
        self.risk_snapshots = []
        while len(self.unconscious.scheduler) > 0:
            schema = self.unconscious.scheduler.pop()
            # Inject accumulated working substitutions as memory
            schema["past_substitutions"] = list(cumulative_subs)
            result = self._run_from_schema(schema)
            if result["substitutions_used"]:
                cumulative_subs.append(result["substitutions_used"])
            results.append(result)
            # Snapshot MFQ state after each dish so callers can plot adaptation
            self.risk_snapshots.append(
                dict(self.unconscious.scheduler._cuisine_risk_overrides)
            )
        return results


if __name__ == "__main__":
    pipeline = CoCoMoPipeline()
    test_dish = "Spaghetti Carbonara"
    print(f"\nRunning CoCoMo pipeline for: {test_dish}\n{'='*60}")
    result = pipeline.run(test_dish)
    print(f"Cuisine:       {result['cuisine']}")
    print(f"Risk score:    {result['risk_score']:.3f}")
    print(f"Was conscious: {result['was_conscious']}")
    print(f"Violations:    {result['violations']}")
    print(f"\n--- Recipe ---\n{result['recipe'][:800]}")
    print(f"\n--- Feedback ---")
    print(f"Old risk: {result['feedback']['old_risk']:.2f} → New risk: {result['feedback']['new_risk']:.2f}")
