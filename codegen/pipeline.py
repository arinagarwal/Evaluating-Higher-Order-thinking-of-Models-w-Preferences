from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from config import MODEL_NAME, MFQ_ESCALATION_THRESHOLD, get_bnb_compute_dtype
from receptor import Receptor
from unconsciousness import UnconsciousnessModule
from consciousness import ConsciousnessModule
from effector import Effector
from detector import detect_banned


def _load_model_and_tokenizer(model_name: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=get_bnb_compute_dtype(),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto"
    )
    return model, tokenizer


class CodeGenPipeline:
    """
    Full CoCoMo-style pipeline for code generation.
    Structurally identical to cocomo/pipeline.py — only the domain
    (coding tasks + banned APIs) and detector (AST vs regex) differ.
    """

    def __init__(self, model=None, tokenizer=None, escalation_threshold: float = MFQ_ESCALATION_THRESHOLD):
        if model is None:
            model, tokenizer = _load_model_and_tokenizer(MODEL_NAME)
        self.receptor = Receptor()
        self.unconscious = UnconsciousnessModule(model=model, tokenizer=tokenizer)
        self.conscious = ConsciousnessModule(model=model, tokenizer=tokenizer)
        self.effector = Effector()
        self.escalation_threshold = escalation_threshold
        self.feedback_log: list[dict] = []
        self.risk_snapshots: list[dict] = []

    def run(self, task: str) -> dict:
        schema = self.receptor.process(task)
        return self._run_from_schema(schema)

    def _run_from_schema(self, schema: dict) -> dict:
        draft = self.unconscious.draft_code(schema)
        risk = self.unconscious.classify_risk(draft, schema)
        schema["risk_score"] = risk
        schema["draft"] = draft

        escalated = self.unconscious.should_escalate(risk, self.escalation_threshold)

        if escalated:
            code, metadata = self.conscious.generate(schema)
            was_conscious = True
        else:
            code = draft
            metadata = {"validated_substitutions": {}, "prompt_used": "unconscious_draft"}
            was_conscious = False

        result = self.effector.output(code, schema, was_conscious, metadata)
        result["draft_violations"] = detect_banned(draft)
        feedback = self.effector.send_feedback(result, self.unconscious.scheduler)
        self.feedback_log.append(feedback)
        result["feedback"] = feedback
        return result

    def run_batch(self, tasks: list[str]) -> list[dict]:
        for task in tasks:
            schema = self.receptor.process(task)
            self.unconscious.scheduler.push(schema)

        results = []
        cumulative_subs: list[dict] = []
        self.risk_snapshots = []
        while len(self.unconscious.scheduler) > 0:
            schema = self.unconscious.scheduler.pop()
            schema["past_substitutions"] = list(cumulative_subs)
            result = self._run_from_schema(schema)
            if result["substitutions_used"]:
                cumulative_subs.append(result["substitutions_used"])
            results.append(result)
            self.risk_snapshots.append(
                dict(self.unconscious.scheduler._category_risk_overrides)
            )
        return results
