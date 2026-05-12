"""
Ablation pipeline variants. Each subclass adds exactly one architectural change
on top of the previous, forming the cumulative ablation chain:

  CoCoMoPipeline          — base (unchanged)
  PlannerPipeline         — + Planner/Drafter split (change 1)
  VerifierPipeline        — + post-generation verification loop (change 2)
  MemoryPipeline          — + episodic memory module (change 3)

Import and pass to run_ablations.py; the base pipeline.py is not modified.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from pipeline import CoCoMoPipeline
from unconsciousness import _detect_banned
from planner import Planner
from verifier import Verifier
from memory import EpisodicMemory


class PlannerPipeline(CoCoMoPipeline):
    """
    Architectural change 1: Planner/Drafter split.

    Escalation decision is made before full draft generation via a short
    prediction pass, not after inspecting a completed draft. Binary risk rule:
    any predicted constraint → escalate unconditionally, eliminating the FN
    class where the fractional detection_signal scored below threshold despite
    confirmed violations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.planner = Planner(self.unconscious.model, self.unconscious.tokenizer)

    def _run_from_schema(self, schema: dict) -> dict:
        # PLAN: predict which banned ingredients this dish needs, pre-generation
        predicted = self.planner.predict_constraints(schema["dish"], schema["cuisine"])
        risk = self.planner.classify_risk(predicted, schema["risk_score"])
        schema["risk_score"] = risk
        schema["predicted_constraints"] = predicted

        escalated = self.unconscious.should_escalate(risk, self.escalation_threshold)

        # DRAFT: always generate for effector output, but escalation is already decided
        draft = self.unconscious.draft_recipe(schema)
        schema["draft"] = draft

        if escalated:
            recipe, metadata = self.conscious.generate(schema)
            was_conscious = True
        else:
            recipe = draft
            metadata = {"validated_substitutions": {}, "prompt_used": "unconscious_draft"}
            was_conscious = False

        result = self.effector.output(recipe, schema, was_conscious, metadata)
        result["draft_violations"] = _detect_banned(draft)
        feedback = self.effector.send_feedback(result, self.unconscious.scheduler)
        self.feedback_log.append(feedback)
        result["feedback"] = feedback
        return result


class VerifierPipeline(PlannerPipeline):
    """
    Architectural change 2: post-generation verification with repair loop.

    Adds a Verifier module between ConsciousnessModule output and Effector.
    Violations in the conscious output trigger re-entry into Consciousness with
    explicit repair annotations (repair_violations injected into schema), making
    the pipeline iterative rather than single-pass.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.verifier = Verifier()

    def _run_from_schema(self, schema: dict) -> dict:
        predicted = self.planner.predict_constraints(schema["dish"], schema["cuisine"])
        risk = self.planner.classify_risk(predicted, schema["risk_score"])
        schema["risk_score"] = risk
        schema["predicted_constraints"] = predicted

        escalated = self.unconscious.should_escalate(risk, self.escalation_threshold)
        draft = self.unconscious.draft_recipe(schema)
        schema["draft"] = draft

        if escalated:
            recipe, metadata = self.conscious.generate(schema)
            # VERIFY: check output and repair if violations remain
            recipe, repair_attempts = self.verifier.verify_and_repair(
                recipe, schema, self.conscious
            )
            metadata["repair_attempts"] = repair_attempts
            was_conscious = True
        else:
            recipe = draft
            metadata = {
                "validated_substitutions": {},
                "prompt_used": "unconscious_draft",
                "repair_attempts": 0,
            }
            was_conscious = False

        result = self.effector.output(recipe, schema, was_conscious, metadata)
        result["draft_violations"] = _detect_banned(draft)
        result["repair_attempts"] = metadata.get("repair_attempts", 0)
        feedback = self.effector.send_feedback(result, self.unconscious.scheduler)
        self.feedback_log.append(feedback)
        result["feedback"] = feedback
        return result


class MemoryPipeline(VerifierPipeline):
    """
    Architectural change 3: episodic memory module.

    At pipeline entry, memory is queried for past successful (cuisine, ingredient)
    substitutions and injected into the schema before Consciousness runs. This
    short-circuits CRIT for proven pairs, reducing LLM calls and improving
    substitution quality on repeated cuisine×ingredient combinations.
    After each dish, outcomes are written back to memory.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory = EpisodicMemory()

    def _run_from_schema(self, schema: dict) -> dict:
        # MEMORY RETRIEVAL: inject proven substitutions before any generation
        schema = self.memory.inject_into_schema(schema)

        predicted = self.planner.predict_constraints(schema["dish"], schema["cuisine"])
        risk = self.planner.classify_risk(predicted, schema["risk_score"])
        schema["risk_score"] = risk
        schema["predicted_constraints"] = predicted

        escalated = self.unconscious.should_escalate(risk, self.escalation_threshold)
        draft = self.unconscious.draft_recipe(schema)
        schema["draft"] = draft

        if escalated:
            recipe, metadata = self.conscious.generate(schema)
            recipe, repair_attempts = self.verifier.verify_and_repair(
                recipe, schema, self.conscious
            )
            metadata["repair_attempts"] = repair_attempts
            was_conscious = True
        else:
            recipe = draft
            metadata = {
                "validated_substitutions": {},
                "prompt_used": "unconscious_draft",
                "repair_attempts": 0,
            }
            was_conscious = False

        result = self.effector.output(recipe, schema, was_conscious, metadata)
        result["draft_violations"] = _detect_banned(draft)
        result["repair_attempts"] = metadata.get("repair_attempts", 0)
        feedback = self.effector.send_feedback(result, self.unconscious.scheduler)
        self.feedback_log.append(feedback)
        result["feedback"] = feedback

        # MEMORY UPDATE: store outcomes for future dishes in this run
        self.memory.update_from_result(result)
        return result
