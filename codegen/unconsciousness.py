from __future__ import annotations

import heapq
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from config import MODEL_NAME, TASK_RISK_MAP, DRAFT_GENERATION_CONFIG, MFQ_ESCALATION_THRESHOLD, get_bnb_compute_dtype
from detector import detect_banned, BANNED_APIS


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


class MFQScheduler:
    def __init__(self):
        self._heap: list[tuple[float, int, dict]] = []
        self._counter = 0
        self._category_risk_overrides: dict[str, float] = {}

    def push(self, schema: dict):
        risk = self._category_risk_overrides.get(schema["category"], schema["risk_score"])
        heapq.heappush(self._heap, (-risk, self._counter, schema))
        self._counter += 1

    def pop(self):
        if not self._heap:
            return None
        _, _, schema = heapq.heappop(self._heap)
        return schema

    def update_category_risk(self, category: str, new_risk: float):
        self._category_risk_overrides[category] = max(0.0, min(1.0, new_risk))

    def __len__(self):
        return len(self._heap)


class UnconsciousnessModule:
    def __init__(self, model=None, tokenizer=None):
        if model is None:
            model, tokenizer = _load_model_and_tokenizer(MODEL_NAME)
        self.model = model
        self.tokenizer = tokenizer
        self.scheduler = MFQScheduler()

    def draft_code(self, schema: dict) -> str:
        prompt = (
            f"Write a Python function to: {schema['task']}\n"
            "Include a function signature, docstring, and implementation. "
            "Return only the code."
        )
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
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
        found = detect_banned(draft)
        detection_signal = min(1.0, len(found) / len(BANNED_APIS))
        combined = 0.6 * detection_signal + 0.4 * schema["risk_score"]
        return round(combined, 4)

    def should_escalate(self, risk: float, threshold: float = MFQ_ESCALATION_THRESHOLD) -> bool:
        return risk > threshold
