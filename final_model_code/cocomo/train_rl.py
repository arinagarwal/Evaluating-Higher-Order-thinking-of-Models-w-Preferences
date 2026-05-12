"""
GRPO RL training loop — replaces SFT on mechanically cleaned data.

The model learns to avoid banned ingredients through a multi-component
reward signal (constraint compliance + culinary quality + substitution
validity + novelty) rather than by imitating mechanically cleaned examples.

Usage:
    python cocomo/train_rl.py
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import GRPOTrainer, GRPOConfig

from config import (
    MODEL_NAME, TRAIN_DISHES, LORA_CONFIG, GRPO_TRAINING_CONFIG,
    GENERATION_CONFIG, get_bnb_compute_dtype,
)
from receptor import Receptor
from reward import RewardFunction


def build_prompt(dish: str, cuisine: str) -> str:
    return (
        f"Write an authentic {cuisine} recipe for {dish}. "
        "Include a title, an Ingredients: section with quantities, "
        "and an Instructions: section with numbered steps."
    )


def make_dataset(dishes: list[str]) -> Dataset:
    """
    Build a HuggingFace Dataset of prompts for GRPO training.
    Each row = one dish prompt; the reward function evaluates the generated completion.
    """
    receptor = Receptor()
    rows = []
    for dish in dishes:
        schema = receptor.process(dish)
        prompt = build_prompt(dish, schema["cuisine"])
        rows.append({
            "prompt": prompt,
            "dish": dish,
            "cuisine": schema["cuisine"],
        })
    return Dataset.from_list(rows)


def make_reward_fn(reward_fn: RewardFunction):
    """
    Returns a reward function compatible with GRPOTrainer's expected signature.
    trl passes extra dataset columns as kwargs; completions is a list of strings
    or list of message dicts depending on trl version.
    Signature: fn(completions, **kwargs) -> list[float]
    """
    def reward_fn_wrapper(completions, **kwargs):
        dishes = kwargs.get("dish", ["unknown"] * len(completions))
        cuisines = kwargs.get("cuisine", ["Unknown"] * len(completions))
        # trl may pass dish/cuisine as a single repeated value or as a list
        if isinstance(dishes, str):
            dishes = [dishes] * len(completions)
        if isinstance(cuisines, str):
            cuisines = [cuisines] * len(completions)
        rewards = []
        for completion, dish, cuisine in zip(completions, dishes, cuisines):
            # completion may be a list of dicts (chat format) or a plain string
            if isinstance(completion, list):
                text = " ".join(m.get("content", "") for m in completion)
            else:
                text = str(completion)
            r = reward_fn.total_reward(text, dish, cuisine)
            rewards.append(r)
        return rewards
    return reward_fn_wrapper


def train():
    print(f"Loading model: {MODEL_NAME}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Policy model — this one trains
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        target_modules=LORA_CONFIG["target_modules"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Frozen judge — separate model load, weights never update
    print(f"Loading frozen judge: {MODEL_NAME}")
    judge_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    for param in judge_model.parameters():
        param.requires_grad = False
    judge_model.eval()

    reward_fn = RewardFunction(model=judge_model, tokenizer=tokenizer)

    dataset = make_dataset(TRAIN_DISHES)

    grpo_cfg = GRPOConfig(
        output_dir=GRPO_TRAINING_CONFIG["output_dir"],
        num_train_epochs=GRPO_TRAINING_CONFIG["num_train_epochs"],
        per_device_train_batch_size=4,
        gradient_accumulation_steps=GRPO_TRAINING_CONFIG["gradient_accumulation_steps"],
        learning_rate=GRPO_TRAINING_CONFIG["learning_rate"],
        bf16=False,
        fp16=True,
        max_completion_length=GRPO_TRAINING_CONFIG["max_seq_length"],
        save_steps=GRPO_TRAINING_CONFIG["save_steps"],
        logging_steps=10,
        report_to="none",
        # Match eval generation settings so training rollouts are comparable
        temperature=GENERATION_CONFIG["temperature"],
        top_p=GENERATION_CONFIG["top_p"],
        top_k=GENERATION_CONFIG["top_k"],
        num_generations=4,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[make_reward_fn(reward_fn)],
        args=grpo_cfg,
        train_dataset=dataset,
    )

    print(f"\nStarting GRPO training on {len(TRAIN_DISHES)} dishes...")
    trainer.train()

    final_dir = os.path.join(GRPO_TRAINING_CONFIG["output_dir"], "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nWeights saved to {final_dir}")


if __name__ == "__main__":
    train()
