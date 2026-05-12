"""
Ablation pipeline variants for code generation — mirrors cocomo/pipeline_variants.py.
Each subclass adds one architectural change cumulatively.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from pipeline import CodeGenPipeline
from detector import detect_banned
from config import BANNED_APIS


# ── Planner (reused logic, code-domain prompt) ────────────────────────────────

import re
import torch

class Planner:
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

    def predict_constraints(self, task: str, category: str) -> list[str]:
        banned_str = ", ".join(BANNED_APIS)
        prompt = (
            f"For a Python implementation of: '{task}'\n"
            f"Which of these APIs would a typical implementation use: {banned_str}?\n"
            f"Reply with only the API names that apply, comma-separated. "
            f"If none apply, reply 'none'."
        )
        response = self._generate(prompt)
        return [
            api for api in BANNED_APIS
            if re.search(r"\b" + re.escape(api.split("(")[0]) + r"\b", response, re.IGNORECASE)
            or api.lower() in response.lower()
        ]

    def classify_risk(self, predicted: list[str], prior: float) -> float:
        return 1.0 if predicted else prior


# ── Verifier ──────────────────────────────────────────────────────────────────

class Verifier:
    def verify_and_repair(self, code: str, schema: dict, conscious_module, max_attempts: int = 2) -> tuple[str, int]:
        current = code
        for attempt in range(max_attempts):
            violations = detect_banned(current)
            if not violations:
                return current, attempt
            repair_schema = dict(schema)
            repair_schema["repair_violations"] = violations
            repaired, _ = conscious_module.generate(repair_schema)
            current = repaired
        return current, max_attempts


# ── Episodic Memory ───────────────────────────────────────────────────────────

from collections import defaultdict

class EpisodicMemory:
    def __init__(self):
        self._store: dict[tuple[str, str], list[dict]] = defaultdict(list)

    def store(self, category: str, api: str, substitute: str, score: float, succeeded: bool):
        self._store[(category, api)].append({"substitute": substitute, "score": score, "succeeded": succeeded})

    def retrieve_best(self, category: str, api: str) -> str | None:
        entries = self._store.get((category, api), [])
        successful = [e for e in entries if e["succeeded"] and e["score"] >= 0.6]
        return max(successful, key=lambda e: e["score"])["substitute"] if successful else None

    def inject_into_schema(self, schema: dict) -> dict:
        category = schema.get("category", "")
        memory_subs = []
        for api in schema.get("constraints", []):
            best = self.retrieve_best(category, api)
            if best:
                memory_subs.append({api: {"substitute": best, "source": "memory", "validity_score": 0.8}})
        if not memory_subs:
            return schema
        schema = dict(schema)
        schema["past_substitutions"] = memory_subs + schema.get("past_substitutions", [])
        return schema

    def update_from_result(self, result: dict):
        category = result.get("category", "")
        violations = set(result.get("violations", []))
        for api, info in result.get("substitutions_used", {}).items():
            sub = info.get("substitute", "")
            score = info.get("validity_score", 0.5)
            if sub:
                self.store(category, api, sub, score, api not in violations)


# ── Pipeline variants ─────────────────────────────────────────────────────────

class PlannerPipeline(CodeGenPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.planner = Planner(self.unconscious.model, self.unconscious.tokenizer)

    def _run_from_schema(self, schema: dict) -> dict:
        predicted = self.planner.predict_constraints(schema["task"], schema["category"])
        risk = self.planner.classify_risk(predicted, schema["risk_score"])
        schema["risk_score"] = risk
        schema["predicted_constraints"] = predicted

        escalated = self.unconscious.should_escalate(risk, self.escalation_threshold)
        draft = self.unconscious.draft_code(schema)
        schema["draft"] = draft

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


class VerifierPipeline(PlannerPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.verifier = Verifier()

    def _run_from_schema(self, schema: dict) -> dict:
        predicted = self.planner.predict_constraints(schema["task"], schema["category"])
        risk = self.planner.classify_risk(predicted, schema["risk_score"])
        schema["risk_score"] = risk
        schema["predicted_constraints"] = predicted

        escalated = self.unconscious.should_escalate(risk, self.escalation_threshold)
        draft = self.unconscious.draft_code(schema)
        schema["draft"] = draft

        if escalated:
            code, metadata = self.conscious.generate(schema)
            code, repair_attempts = self.verifier.verify_and_repair(code, schema, self.conscious)
            metadata["repair_attempts"] = repair_attempts
            was_conscious = True
        else:
            code = draft
            metadata = {"validated_substitutions": {}, "prompt_used": "unconscious_draft", "repair_attempts": 0}
            was_conscious = False

        result = self.effector.output(code, schema, was_conscious, metadata)
        result["draft_violations"] = detect_banned(draft)
        result["repair_attempts"] = metadata.get("repair_attempts", 0)
        feedback = self.effector.send_feedback(result, self.unconscious.scheduler)
        self.feedback_log.append(feedback)
        result["feedback"] = feedback
        return result


class MemoryPipeline(VerifierPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory = EpisodicMemory()

    def _run_from_schema(self, schema: dict) -> dict:
        schema = self.memory.inject_into_schema(schema)

        predicted = self.planner.predict_constraints(schema["task"], schema["category"])
        risk = self.planner.classify_risk(predicted, schema["risk_score"])
        schema["risk_score"] = risk
        schema["predicted_constraints"] = predicted

        escalated = self.unconscious.should_escalate(risk, self.escalation_threshold)
        draft = self.unconscious.draft_code(schema)
        schema["draft"] = draft

        if escalated:
            code, metadata = self.conscious.generate(schema)
            code, repair_attempts = self.verifier.verify_and_repair(code, schema, self.conscious)
            metadata["repair_attempts"] = repair_attempts
            was_conscious = True
        else:
            code = draft
            metadata = {"validated_substitutions": {}, "prompt_used": "unconscious_draft", "repair_attempts": 0}
            was_conscious = False

        result = self.effector.output(code, schema, was_conscious, metadata)
        result["draft_violations"] = detect_banned(draft)
        result["repair_attempts"] = metadata.get("repair_attempts", 0)
        feedback = self.effector.send_feedback(result, self.unconscious.scheduler)
        self.feedback_log.append(feedback)
        result["feedback"] = feedback
        self.memory.update_from_result(result)
        return result
