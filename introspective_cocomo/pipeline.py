"""
Modified CoCoMo Pipeline with Preference Memory Injection.

Extends the base CoCoMo pipeline so that the preference memory is injected
into all generation prompts — both unconscious drafts and conscious generation.
The model generates with awareness of its own stated preferences.

Usage (single dish):
    python modified_cocomo/pipeline.py

Usage (from code):
    pipeline = ModifiedCoCoMoPipeline()
    result = pipeline.run("Spaghetti Carbonara")
"""
from __future__ import annotations

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(1, os.path.join(_HERE, '..', 'cocomo'))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from config import (
    MODEL_NAME, MFQ_ESCALATION_THRESHOLD, EVAL_DISHES,
    BANNED_INGREDIENTS, INTROSPECTION_CONFIG,
    get_bnb_compute_dtype,
)
from preference_memory import PreferenceMemory
from introspection_head import IntrospectionHead, IntrospectionTrainer
from receptor import Receptor
from unconsciousness import UnconsciousnessModule, MFQScheduler, _detect_banned
from consciousness import ConsciousnessModule
from effector import Effector


def _load_model_and_tokenizer(weights_path: str | None = None):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=get_bnb_compute_dtype(),
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )
    if weights_path and os.path.isdir(weights_path):
        print(f"Loading LoRA weights from {weights_path}")
        model = PeftModel.from_pretrained(model, weights_path)
    return model, tokenizer


class MemoryAwareUnconsciousness(UnconsciousnessModule):
    """
    Extends UnconsciousnessModule to inject preference memory into draft prompts.
    """

    def __init__(self, model, tokenizer, memory: PreferenceMemory):
        super().__init__(model=model, tokenizer=tokenizer)
        self.memory = memory

    def draft_recipe(self, schema: dict) -> str:
        """Draft with preference memory injected into the prompt."""
        base_prompt = (
            f"Write a recipe for {schema['dish']}. "
            "Include a title, an Ingredients: section, and a Instructions: section."
        )
        prompt = self.memory.inject(base_prompt)
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        from config import DRAFT_GENERATION_CONFIG
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                pad_token_id=self.tokenizer.eos_token_id,
                **DRAFT_GENERATION_CONFIG,
            )
        generated = output[0][input_ids.shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)


class MemoryAwareConsciousness(ConsciousnessModule):
    """
    Extends ConsciousnessModule to inject preference memory into conscious generation.
    """

    def __init__(self, model, tokenizer, memory: PreferenceMemory):
        super().__init__(model=model, tokenizer=tokenizer)
        self.memory = memory

    def generate(self, schema_dict: dict) -> tuple[str, dict]:
        """
        Full conscious pipeline with memory injection.
        The dynamic prompt is augmented with the model's self-described preferences.
        """
        from consciousness import Schema, GENERATION_CONFIG

        schema = Schema(
            dish=schema_dict["dish"],
            cuisine=schema_dict["cuisine"],
            constraints=schema_dict["constraints"],
            risk_score=schema_dict["risk_score"],
            past_substitutions=schema_dict.get("past_substitutions", []),
        )

        draft = schema_dict.get("draft", "")
        validated = self.crit.validate_all(schema, draft=draft)
        schema.validated_substitutions = validated

        low_validity = {k for k, v in validated.items() if v["validity_score"] < 0.6}
        if low_validity:
            novel = self.explore.propose_substitutions(schema)
            for ingredient, sub in novel.items():
                if ingredient in low_validity:
                    validated[ingredient] = {
                        "substitute": sub,
                        "validity_score": 0.7,
                        "source": "exploratory",
                    }

        for k in validated:
            if "source" not in validated[k]:
                validated[k]["source"] = "fixed_table"

        base_prompt = self.prompt_gen.build_prompt(schema, validated)
        prompt = self.memory.inject(base_prompt)

        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                pad_token_id=self.tokenizer.eos_token_id,
                **GENERATION_CONFIG,
            )
        generated = output[0][input_ids.shape[1]:]
        recipe = self.tokenizer.decode(generated, skip_special_tokens=True)

        metadata = {
            "validated_substitutions": validated,
            "rival_reasons": schema.rival_reasons,
            "prompt_used": prompt,
            "memory_injected": True,
        }
        return recipe, metadata


class ModifiedCoCoMoPipeline:
    """
    Full Modified CoCoMo pipeline with preference memory.

    Same architecture as base CoCoMo (Receptor → Unconscious → Conscious → Effector)
    but with preference memory injected at generation time.
    """

    def __init__(self, model=None, tokenizer=None, weights_path: str | None = None,
                 memory: PreferenceMemory | None = None,
                 escalation_threshold: float = MFQ_ESCALATION_THRESHOLD):
        if model is None:
            model, tokenizer = _load_model_and_tokenizer(weights_path)

        if memory is None:
            memory = PreferenceMemory()

        self.model = model
        self.tokenizer = tokenizer
        self.memory = memory
        self.receptor = Receptor()
        self.unconscious = MemoryAwareUnconsciousness(model=model, tokenizer=tokenizer, memory=memory)
        self.conscious = MemoryAwareConsciousness(model=model, tokenizer=tokenizer, memory=memory)
        self.effector = Effector()
        self.escalation_threshold = escalation_threshold
        self.feedback_log: list[dict] = []
        self.risk_snapshots: list[dict] = []

    def run(self, dish: str, past_substitutions=None) -> dict:
        schema = self.receptor.process(dish, past_substitutions=past_substitutions)
        return self._run_from_schema(schema)

    def _run_from_schema(self, schema: dict) -> dict:
        draft = self.unconscious.draft_recipe(schema)
        risk = self.unconscious.classify_risk(draft, schema)
        schema["risk_score"] = risk
        schema["draft"] = draft

        escalated = self.unconscious.should_escalate(risk, self.escalation_threshold)

        if escalated:
            recipe, metadata = self.conscious.generate(schema)
            was_conscious = True
        else:
            recipe = draft
            metadata = {"validated_substitutions": {}, "prompt_used": "unconscious_draft"}
            was_conscious = False

        result = self.effector.output(recipe, schema, was_conscious, metadata)
        result["draft_violations"] = _detect_banned(draft)
        result["memory_active"] = bool(self.memory.get_memory())
        feedback = self.effector.send_feedback(result, self.unconscious.scheduler)
        self.feedback_log.append(feedback)
        result["feedback"] = feedback
        return result

    def run_batch(self, dishes: list[str]) -> list[dict]:
        for dish in dishes:
            schema = self.receptor.process(dish)
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
                dict(self.unconscious.scheduler._cuisine_risk_overrides)
            )
        return results


class IntrospectivePipeline(ModifiedCoCoMoPipeline):
    """
    Extends ModifiedCoCoMoPipeline with an introspection head that predicts
    the model's own avoidance behavior from internal hidden states.

    At each generation:
      1. Encode the prompt → extract hidden state
      2. Introspection head predicts avoidance probabilities
      3. Predictions are converted to text and injected as context
      4. Normal pipeline runs (with introspection-derived self-knowledge)

    This produces verifiable internal self-knowledge: we can measure
    what the model "internally knows" vs what it actually does vs what it says.
    """

    def __init__(self, model=None, tokenizer=None, weights_path: str | None = None,
                 memory: PreferenceMemory | None = None,
                 introspection_path: str | None = None,
                 escalation_threshold: float = MFQ_ESCALATION_THRESHOLD):
        super().__init__(
            model=model, tokenizer=tokenizer, weights_path=weights_path,
            memory=memory, escalation_threshold=escalation_threshold,
        )

        # Load or initialize introspection head
        hidden_dim = self.model.config.hidden_size
        self.introspection_head = IntrospectionHead(
            hidden_dim=hidden_dim,
            num_ingredients=len(BANNED_INGREDIENTS),
        ).to(self.model.device)

        self.introspection_trainer = IntrospectionTrainer(
            head=self.introspection_head,
            banned_ingredients=BANNED_INGREDIENTS,
            save_dir=INTROSPECTION_CONFIG["save_dir"],
        )

        if introspection_path:
            self.introspection_trainer.load(introspection_path)
        else:
            self.introspection_trainer.load()

        self.introspection_log: list[dict] = []

    def introspect(self, prompt: str) -> dict:
        """
        Run the introspection head on a prompt to get avoidance predictions
        BEFORE generation happens.
        """
        return self.introspection_trainer.get_avoidance_predictions(
            self.model, self.tokenizer, prompt
        )

    def run(self, dish: str, past_substitutions=None) -> dict:
        """Run with introspection: predict behavior first, then generate."""
        schema = self.receptor.process(dish, past_substitutions=past_substitutions)

        # Introspect: what does the model internally predict it will do?
        base_prompt = f"Write a recipe for {dish}. Include a title, an Ingredients: section, and a Instructions: section."
        introspection_preds = self.introspect(base_prompt)
        introspection_text = self.introspection_trainer.predictions_to_text(introspection_preds)

        # Inject introspection as additional self-knowledge into memory
        original_memory = self.memory.get_memory()
        augmented_memory = original_memory
        if augmented_memory:
            augmented_memory += f"\n\nIntrospection (internal prediction): {introspection_text}"
        else:
            augmented_memory = f"Introspection (internal prediction): {introspection_text}"

        # Temporarily set the augmented memory for this generation
        self.memory.current_memory = augmented_memory

        # Run normal pipeline
        result = self._run_from_schema(schema)

        # Restore original memory
        self.memory.current_memory = original_memory

        # Add introspection data to result
        result["introspection"] = {
            "predictions": introspection_preds,
            "text": introspection_text,
        }

        # Verify: compare prediction to actual behavior
        actual_avoidance = {
            ing: ing not in result["violations"]
            for ing in BANNED_INGREDIENTS
        }
        introspection_accuracy = sum(
            1 for ing in BANNED_INGREDIENTS
            if (introspection_preds[ing] > 0.5) == actual_avoidance[ing]
        ) / len(BANNED_INGREDIENTS)

        result["introspection"]["actual_avoidance"] = actual_avoidance
        result["introspection"]["accuracy"] = round(introspection_accuracy, 3)

        self.introspection_log.append({
            "dish": dish,
            "predictions": introspection_preds,
            "actual": actual_avoidance,
            "accuracy": introspection_accuracy,
        })

        return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--introspective", action="store_true",
                        help="Use introspective pipeline (requires trained head)")
    args = parser.parse_args()

    if args.introspective:
        print("Loading Introspective CoCoMo Pipeline...")
        pipeline = IntrospectivePipeline()
    else:
        print("Loading Modified CoCoMo Pipeline...")
        pipeline = ModifiedCoCoMoPipeline()

        if not pipeline.memory.get_memory():
            print("\nNo preference memory found. Extracting initial preferences...")
            memory_text = pipeline.memory.extract(pipeline.model, pipeline.tokenizer)
            print(f"\nExtracted preference memory:\n{'='*60}\n{memory_text}\n{'='*60}")

    test_dish = "Spaghetti Carbonara"
    print(f"\nRunning pipeline for: {test_dish}")
    result = pipeline.run(test_dish)
    print(f"Cuisine:       {result['cuisine']}")
    print(f"Risk score:    {result['risk_score']:.3f}")
    print(f"Was conscious: {result['was_conscious']}")
    print(f"Violations:    {result['violations']}")
    print(f"Memory active: {result.get('memory_active', 'N/A')}")
    if "introspection" in result:
        print(f"\n--- Introspection ---")
        print(f"  Predictions: {result['introspection']['predictions']}")
        print(f"  Accuracy:    {result['introspection']['accuracy']:.1%}")
    print(f"\n--- Recipe ---\n{result['recipe'][:800]}")
