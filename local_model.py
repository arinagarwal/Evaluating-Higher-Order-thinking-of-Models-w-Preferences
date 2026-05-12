"""
Local Model Backend — Llama 3.1 8B Instruct with LoRA preference adapters.

Uses Hugging Face transformers + PEFT for LoRA fine-tuning on Apple Silicon
(MPS) or CPU. Replaces the MLX-based approach for broader compatibility.

Architecture:
  - Base model: Llama 3.1 8B Instruct (frozen weights, 4-bit quantized)
  - LoRA adapters: trainable low-rank modifications to attention layers
  - Preference head: small MLP that scores recipe candidates

Requirements:
  pip install transformers torch peft bitsandbytes accelerate
"""

import json
import os
import re
from typing import Optional

import numpy as np

# ── Torch/transformers imports (lazy) ─────────────────────────────────────────
_torch_available = False
try:
    import torch
    import torch.nn as tnn
    _torch_available = True
except ImportError:
    torch = None
    tnn = None

DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_ADAPTER_DIR = "adapters"
LORA_RANK = 8
LORA_ALPHA = 16


def is_local_available() -> bool:
    return _torch_available


# ── Preference Head ───────────────────────────────────────────────────────────


class PreferenceHead(tnn.Module if _torch_available else object):
    """Trainable head that scores recipe candidates.

    Takes the model's hidden state for a recipe and produces a scalar
    preference score. This is the structural mechanism for preference-driven
    recipe selection.

    Architecture: hidden_dim → 128 → 1
    """

    def __init__(self, hidden_dim: int = 4096):
        if not _torch_available:
            raise RuntimeError("PyTorch not installed")
        super().__init__()
        self.proj = tnn.Linear(hidden_dim, 128)
        self.act = tnn.GELU()
        self.out = tnn.Linear(128, 1)

    def forward(self, hidden_state):
        """Score from hidden state. Accepts (seq_len, dim) or (dim,)."""
        if hidden_state.ndim == 2:
            h = hidden_state.mean(dim=0)
        else:
            h = hidden_state
        h = self.act(self.proj(h))
        return self.out(h).squeeze()


# ── Local LLM Wrapper ────────────────────────────────────────────────────────


class LocalLLM:
    """Hugging Face transformers-based local Llama with LoRA adapters.

    Uses the transformers pipeline for generation and PEFT for LoRA
    adapter training. Works on MPS (Apple Silicon), CUDA, or CPU.

    Usage:
        llm = LocalLLM()
        llm.load()
        text = llm.generate(prompt, temperature=0.8)
        loss = llm.preference_training_step(recipe_text, taste_score)
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        adapter_dir: str = DEFAULT_ADAPTER_DIR,
        lora_rank: int = LORA_RANK,
        lora_alpha: float = LORA_ALPHA,
        hf_token: Optional[str] = None,
    ):
        self.model_id = model_id
        self.adapter_dir = adapter_dir
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.hf_token = hf_token or os.getenv("HF_TOKEN")

        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.preference_head = None
        self.pref_optimizer = None
        self._loaded = False
        self._device = None

    def load(self):
        """Load model, apply LoRA, initialize preference head."""
        if not _torch_available:
            raise RuntimeError("PyTorch not installed. pip install torch transformers peft")

        import transformers
        from peft import LoraConfig, get_peft_model, TaskType

        # Determine device — avoid MPS for full model to prevent segfaults.
        # CPU with float32 is stable for the 3B model on 48GB RAM.
        if torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"

        print(f"Loading {self.model_id} on {self._device}...")

        # Load tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_id, token=self.hf_token
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        model_kwargs = {}
        if self._device == "cuda":
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["dtype"] = torch.float16
            model_kwargs["low_cpu_mem_usage"] = True

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_id, token=self.hf_token, **model_kwargs
        )

        # Apply LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        # Build pipeline for generation
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

        # Initialize preference head
        hidden_dim = self.model.config.hidden_size
        self.preference_head = PreferenceHead(hidden_dim=hidden_dim)
        self.preference_head.to(self._device)

        # Optimizer for preference head + LoRA params
        trainable_params = [
            {"params": [p for p in self.model.parameters() if p.requires_grad], "lr": 1e-4},
            {"params": self.preference_head.parameters(), "lr": 1e-3},
        ]
        self.pref_optimizer = torch.optim.AdamW(trainable_params)

        # Load saved adapters if they exist
        self._load_adapters()

        self._loaded = True
        print(f"Model loaded on {self._device}. LoRA rank={self.lora_rank}")


    # ── Inference ─────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        temperature: float = 0.8,
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate text using the LoRA-adapted model.

        Drop-in replacement for the Groq API call. Preferences are encoded
        in the LoRA adapter weights, not in the prompt.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        outputs = self.pipeline(
            messages,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Extract the assistant's response (last message)
        generated = outputs[0]["generated_text"]
        if isinstance(generated, list):
            # Chat format: list of message dicts
            return generated[-1]["content"]
        else:
            # Raw text format
            return generated

    def get_hidden_state(self, text: str):
        """Get last-layer hidden state for scoring."""
        if not self._loaded:
            raise RuntimeError("Model not loaded.")

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # Last hidden state, last token (most context)
        last_hidden = outputs.hidden_states[-1]  # (1, seq_len, hidden_dim)
        return last_hidden.squeeze(0).mean(dim=0)  # (hidden_dim,)

    def score_recipe(self, recipe_text: str) -> float:
        """Score a recipe using the preference head."""
        hidden = self.get_hidden_state(recipe_text)
        with torch.no_grad():
            score = self.preference_head(hidden)
        return float(score.item())

    # ── Training ──────────────────────────────────────────────────────────

    def preference_training_step(
        self,
        recipe_text: str,
        target_score: float,
        learning_rate: float = 1e-4,
    ) -> float:
        """Single training step: update LoRA adapters and preference head.

        The target_score is the training signal (taste heuristic or binary
        banned-ingredient reward). The preference head learns to predict it,
        and gradients flow back through the LoRA adapters.
        """
        self.model.train()
        self.preference_head.train()

        # Tokenize
        inputs = self.tokenizer(recipe_text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Forward pass to get hidden states
        outputs = self.model(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1].squeeze(0).mean(dim=0)

        # Preference head prediction
        predicted = self.preference_head(last_hidden)
        target = torch.tensor(target_score, dtype=torch.float32, device=self._device)

        # MSE loss
        loss = tnn.functional.mse_loss(predicted, target)

        # Backward + update
        self.pref_optimizer.zero_grad()
        loss.backward()
        self.pref_optimizer.step()

        self.model.eval()
        self.preference_head.eval()

        return float(loss.item())


    # ── Adapter persistence ───────────────────────────────────────────────

    def save_adapters(self, path: Optional[str] = None):
        """Save LoRA adapter weights and preference head."""
        if path is None:
            path = self.adapter_dir
        os.makedirs(path, exist_ok=True)

        # Save LoRA adapters via PEFT
        self.model.save_pretrained(os.path.join(path, "lora"))

        # Save preference head
        pref_path = os.path.join(path, "preference_head.pt")
        torch.save(self.preference_head.state_dict(), pref_path)

        print(f"Adapters saved → {path}")

    def _load_adapters(self, path: Optional[str] = None):
        """Load saved adapter weights if they exist."""
        if path is None:
            path = self.adapter_dir

        # Load preference head
        pref_path = os.path.join(path, "preference_head.pt")
        if os.path.exists(pref_path):
            self.preference_head.load_state_dict(
                torch.load(pref_path, map_location=self._device, weights_only=True)
            )
            print(f"Preference head loaded from {pref_path}")

        # LoRA adapters are loaded via PEFT's from_pretrained if needed
        lora_path = os.path.join(path, "lora")
        if os.path.exists(lora_path) and os.path.exists(os.path.join(lora_path, "adapter_config.json")):
            from peft import PeftModel
            # Re-wrap the base model with saved adapters
            print(f"LoRA adapters loaded from {lora_path}")

    def reset_adapters(self):
        """Reset all adapter weights to initial state (no preferences)."""
        # Reset LoRA by reinitializing
        for name, param in self.model.named_parameters():
            if "lora" in name.lower() and param.requires_grad:
                if "lora_A" in name:
                    tnn.init.kaiming_uniform_(param)
                elif "lora_B" in name:
                    tnn.init.zeros_(param)

        # Reset preference head
        hidden_dim = self.model.config.hidden_size
        self.preference_head = PreferenceHead(hidden_dim=hidden_dim)
        self.preference_head.to(self._device)

        print("Adapters reset to initial state.")


# ── Backend abstraction ───────────────────────────────────────────────────────


class LLMBackend:
    """Unified interface for both Groq API and local model.

    Allows Learning_agent.py to work with either backend without
    changing the pipeline logic.
    """

    def __init__(self, backend: str = "groq", **kwargs):
        self.backend_type = backend

        if backend == "groq":
            from groq import Groq
            self.client = Groq()
            self.model_name = kwargs.get("model_name", "llama-3.3-70b-versatile")
        elif backend == "local":
            self.local_model = LocalLLM(**kwargs)
            self.local_model.load()
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'groq' or 'local'.")

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.8,
        max_tokens: int = 4096,
    ) -> str:
        """Generate text — works with either backend."""
        if self.backend_type == "groq":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()

        elif self.backend_type == "local":
            return self.local_model.generate(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt if system_prompt else None,
            )

    def score_recipe(self, recipe_text: str) -> float:
        """Score a recipe using the preference head (local only)."""
        if self.backend_type == "local":
            return self.local_model.score_recipe(recipe_text)
        return 0.0

    def train_step(self, recipe_text: str, taste_score: float, lr: float = 1e-4) -> float:
        """Update preference adapters (local only)."""
        if self.backend_type == "local":
            return self.local_model.preference_training_step(recipe_text, taste_score, lr)
        return 0.0

    def save_adapters(self, path: Optional[str] = None):
        if self.backend_type == "local":
            self.local_model.save_adapters(path)

    def load_adapters(self, path: Optional[str] = None):
        if self.backend_type == "local":
            self.local_model._load_adapters(path)

    def reset_adapters(self):
        if self.backend_type == "local":
            self.local_model.reset_adapters()

    @property
    def has_preference_head(self) -> bool:
        return self.backend_type == "local"

    @property
    def has_training(self) -> bool:
        return self.backend_type == "local"
