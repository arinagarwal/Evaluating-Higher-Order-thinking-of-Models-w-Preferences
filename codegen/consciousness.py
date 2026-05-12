"""
Consciousness module for code generation — analogous to cocomo/consciousness.py.

CRIT validates whether a proposed API substitution is functionally equivalent
and safe for the specific task. Exploratory proposes novel secure alternatives
when the fixed substitution table has low validity.
"""
from __future__ import annotations

import re
import torch
from dataclasses import dataclass, field
from config import BANNED_APIS, SUBSTITUTIONS, GENERATION_CONFIG


@dataclass
class Schema:
    task: str
    category: str
    constraints: list[str]
    risk_score: float
    past_substitutions: list[dict] = field(default_factory=list)
    validated_substitutions: dict = field(default_factory=dict)
    rival_reasons: list[str] = field(default_factory=list)
    repair_violations: list[str] = field(default_factory=list)


class CriticalThinking:
    """
    Validates whether a proposed API substitution is safe and functionally
    equivalent for the specific task. Directly analogous to cocomo's CRIT
    but the validity criterion is security + functional equivalence, not
    culinary appropriateness.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def _generate(self, prompt: str, max_new_tokens: int = 128) -> str:
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

    def _extract_score(self, text: str) -> float:
        matches = re.findall(r"\b([1-9]|10)\b", text)
        return float(matches[0]) / 10.0 if matches else 0.5

    def validate_substitution(self, task: str, category: str, banned: str, substitute: str) -> tuple[float, list[str], list[str]]:
        claim = f"'{substitute}' is a safe and functionally equivalent replacement for '{banned}' when implementing: {task}"

        support_text = self._generate(
            f"Claim: {claim}\nList 2 reasons why this substitution is safe and works for the task. Be concise."
        )
        rival_text = self._generate(
            f"Claim: {claim}\nList 2 reasons why this substitution might NOT work or be sufficient. Be concise."
        )
        score_text = self._generate(
            f"Claim: {claim}\nSupporting: {support_text}\nAgainst: {rival_text}\n"
            f"On a scale of 1 to 10, how valid is this substitution for this specific task? Reply with just the number.",
            max_new_tokens=16,
        )
        return self._extract_score(score_text), [support_text], [rival_text]

    def validate_all(self, schema: Schema, draft: str = "", skip: set | None = None) -> dict:
        skip = skip or set()
        lower_draft = draft.lower()
        validated = {}
        for api in schema.constraints:
            if api in skip:
                continue
            if draft and api.split("(")[0].lower() not in lower_draft and api not in lower_draft:
                continue
            substitute = SUBSTITUTIONS.get(api)
            if substitute is None:
                continue
            score, _, rival = self.validate_substitution(schema.task, schema.category, api, substitute)
            schema.rival_reasons.extend(rival)
            validated[api] = {"substitute": substitute, "validity_score": score}
        return validated


class ExploratoryThinking:
    """
    Proposes secure alternatives to banned APIs beyond the fixed substitution
    table, guided by the specific task context.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def _generate(self, prompt: str, max_new_tokens: int = 200) -> str:
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
            )
        generated = output[0][input_ids.shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def propose_substitutions(self, schema: Schema) -> dict:
        banned_str = ", ".join(schema.constraints)
        prompt = (
            f"You are a security-focused Python engineer.\n"
            f"Task: implement a function to {schema.task}\n"
            f"The following APIs are banned for security reasons: {banned_str}\n"
            f"For each banned API that this task might require, suggest ONE safe, "
            f"functionally equivalent alternative.\n"
            f"Format: 'banned_api → safe_alternative'. Only list APIs relevant to this task."
        )
        response = self._generate(prompt)
        proposed = {}
        arrow_pat = re.compile(r"→|->")
        for line in response.split("\n"):
            if not arrow_pat.search(line):
                continue
            parts = arrow_pat.split(line, maxsplit=1)
            if len(parts) == 2:
                orig = parts[0].strip().lower().strip("- •*`")
                sub = parts[1].strip()
                for banned in schema.constraints:
                    if banned.lower() in orig or orig in banned.lower():
                        proposed[banned] = sub
                        break
        return proposed


class PromptTemplateGenerator:
    def build_prompt(self, schema: Schema, validated_substitutions: dict) -> str:
        sub_instructions = []
        for api, info in validated_substitutions.items():
            sub = info["substitute"]
            score = info["validity_score"]
            source = info.get("source", "")
            if score >= 0.6:
                tag = "memory-proven" if source == "memory" else f"validity: {score:.1f}/1.0"
                sub_instructions.append(f"- Instead of `{api}`, use `{sub}` ({tag})")
            else:
                sub_instructions.append(
                    f"- Avoid `{api}`; find the most secure equivalent for this task"
                )

        subs_text = "\n".join(sub_instructions) if sub_instructions else \
            f"Do not use any of: {', '.join(schema.constraints)}"

        repair_section = ""
        if schema.repair_violations:
            repair_section = (
                f"\n\nWARNING — a previous draft still used: "
                f"{', '.join(schema.repair_violations)}. "
                f"These are strictly banned. Do not use them in any form."
            )

        return (
            f"Write a secure Python function to: {schema.task}\n\n"
            f"Security constraints — do not use these APIs:\n"
            f"{subs_text}"
            f"{repair_section}\n\n"
            f"Include a function signature, docstring, and complete implementation. "
            f"Return only the code."
        )


class ConsciousnessModule:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.crit = CriticalThinking(model, tokenizer)
        self.explore = ExploratoryThinking(model, tokenizer)
        self.prompt_gen = PromptTemplateGenerator()

    def generate(self, schema_dict: dict) -> tuple[str, dict]:
        schema = Schema(
            task=schema_dict["task"],
            category=schema_dict["category"],
            constraints=schema_dict["constraints"],
            risk_score=schema_dict["risk_score"],
            past_substitutions=schema_dict.get("past_substitutions", []),
            repair_violations=schema_dict.get("repair_violations", []),
        )

        # Memory-sourced substitutions skip CRIT
        memory_validated = {}
        for sub_dict in schema.past_substitutions:
            for api, info in sub_dict.items():
                if info.get("source") == "memory" and api in schema.constraints:
                    memory_validated[api] = {
                        "substitute": info["substitute"],
                        "validity_score": info.get("validity_score", 0.8),
                        "source": "memory",
                    }

        draft = schema_dict.get("draft", "")
        validated = self.crit.validate_all(schema, draft=draft, skip=set(memory_validated.keys()))
        validated.update(memory_validated)
        schema.validated_substitutions = validated

        # Exploratory for low-validity substitutions
        low_validity = {k for k, v in validated.items() if v["validity_score"] < 0.6}
        if low_validity:
            novel = self.explore.propose_substitutions(schema)
            for api, sub in novel.items():
                if api in low_validity:
                    validated[api] = {"substitute": sub, "validity_score": 0.7, "source": "exploratory"}

        for k in validated:
            if "source" not in validated[k]:
                validated[k]["source"] = "fixed_table"

        prompt = self.prompt_gen.build_prompt(schema, validated)
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
        code = self.tokenizer.decode(generated, skip_special_tokens=True)

        return code, {
            "validated_substitutions": validated,
            "rival_reasons": schema.rival_reasons,
            "prompt_used": prompt,
        }
