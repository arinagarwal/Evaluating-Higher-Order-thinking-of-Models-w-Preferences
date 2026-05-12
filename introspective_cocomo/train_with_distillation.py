"""
GRPO Training with Preference Memory Self-Distillation.

Training loop:
  1. Extract initial preference memory from the model
  2. Run GRPO training epoch (with memory injected into prompts)
  3. Refine preference memory (model re-describes its preferences post-training)
  4. Repeat steps 2-3 for N distillation rounds

The memory evolves as the model's behavior changes through RL, creating a
feedback loop between explicit self-knowledge and implicit learned behavior.

Usage:
    python modified_cocomo/train_with_distillation.py
"""
from __future__ import annotations

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(1, os.path.join(_HERE, '..', 'cocomo'))

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import GRPOTrainer, GRPOConfig

from config import (
    MODEL_NAME, TRAIN_DISHES, LORA_CONFIG,
    GRPO_TRAINING_CONFIG_MODIFIED, GENERATION_CONFIG,
    NUM_DISTILLATION_EPOCHS, DISHES_PER_DISTILLATION_ROUND,
    get_bnb_compute_dtype,
)
from preference_memory import PreferenceMemory
from receptor import Receptor
from reward import RewardFunction


def build_prompt(dish: str, cuisine: str, memory: PreferenceMemory) -> str:
    base = (
        f"Write an authentic {cuisine} recipe for {dish}. "
        "Include a title, an Ingredients: section with quantities, "
        "and an Instructions: section with numbered steps."
    )
    return memory.inject(base)


def make_dataset(dishes: list[str], memory: PreferenceMemory) -> Dataset:
    receptor = Receptor()
    rows = []
    for dish in dishes:
        schema = receptor.process(dish)
        prompt = build_prompt(dish, schema["cuisine"], memory)
        rows.append({
            "prompt": prompt,
            "dish": dish,
            "cuisine": schema["cuisine"],
        })
    return Dataset.from_list(rows)


def make_reward_fn(reward_fn: RewardFunction):
    def reward_fn_wrapper(completions, **kwargs):
        dishes = kwargs.get("dish", ["unknown"] * len(completions))
        cuisines = kwargs.get("cuisine", ["Unknown"] * len(completions))
        if isinstance(dishes, str):
            dishes = [dishes] * len(completions)
        if isinstance(cuisines, str):
            cuisines = [cuisines] * len(completions)
        rewards = []
        for completion, dish, cuisine in zip(completions, dishes, cuisines):
            if isinstance(completion, list):
                text = " ".join(m.get("content", "") for m in completion)
            else:
                text = str(completion)
            r = reward_fn.total_reward(text, dish, cuisine)
            rewards.append(r)
        return rewards
    return reward_fn_wrapper


def train():
    print(f"{'='*60}")
    print("Modified CoCoMo: GRPO Training with Preference Memory Self-Distillation")
    print(f"{'='*60}")
    print(f"Model: {MODEL_NAME}")
    print(f"Distillation rounds: {NUM_DISTILLATION_EPOCHS}")
    print(f"Dishes per round: {DISHES_PER_DISTILLATION_ROUND}")
    print(f"Total training dishes: {len(TRAIN_DISHES)}")

    memory = PreferenceMemory()

    # Load base model
    print(f"\nLoading model: {MODEL_NAME}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    # Frozen judge
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

    # ── Step 1: Extract initial preference memory ─────────────────────────────
    print("\n" + "="*60)
    print("STEP 1: Extracting initial preference memory")
    print("="*60)

    model.eval()
    initial_memory = memory.extract(model, tokenizer)
    print(f"\nInitial memory:\n{initial_memory}")
    model.train()

    # ── Distillation loop ─────────────────────────────────────────────────────
    output_dir = GRPO_TRAINING_CONFIG_MODIFIED["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    for distill_round in range(NUM_DISTILLATION_EPOCHS):
        print(f"\n{'='*60}")
        print(f"DISTILLATION ROUND {distill_round + 1}/{NUM_DISTILLATION_EPOCHS}")
        print(f"{'='*60}")
        print(f"Current memory (first 200 chars): {memory.get_memory()[:200]}...")

        # ── Step 2: GRPO training with memory-injected prompts ────────────────
        # Use a subset of dishes per round to keep training manageable
        start_idx = (distill_round * DISHES_PER_DISTILLATION_ROUND) % len(TRAIN_DISHES)
        end_idx = start_idx + DISHES_PER_DISTILLATION_ROUND
        round_dishes = TRAIN_DISHES[start_idx:end_idx]

        print(f"\nBuilding dataset with memory-injected prompts ({len(round_dishes)} dishes)...")
        dataset = make_dataset(round_dishes, memory)

        round_output_dir = os.path.join(output_dir, f"round_{distill_round}")
        grpo_cfg = GRPOConfig(
            output_dir=round_output_dir,
            num_train_epochs=GRPO_TRAINING_CONFIG_MODIFIED["num_train_epochs"],
            per_device_train_batch_size=4,
            gradient_accumulation_steps=GRPO_TRAINING_CONFIG_MODIFIED["gradient_accumulation_steps"],
            learning_rate=GRPO_TRAINING_CONFIG_MODIFIED["learning_rate"],
            bf16=False,
            fp16=True,
            max_completion_length=GRPO_TRAINING_CONFIG_MODIFIED["max_seq_length"],
            save_steps=GRPO_TRAINING_CONFIG_MODIFIED["save_steps"],
            logging_steps=10,
            report_to="none",
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

        print(f"Training round {distill_round + 1}...")
        trainer.train()

        # Save round checkpoint
        round_save_path = os.path.join(round_output_dir, "final")
        trainer.save_model(round_save_path)
        print(f"Round {distill_round + 1} weights saved to {round_save_path}")

        # ── Step 3: Refine preference memory ──────────────────────────────────
        print(f"\nRefining preference memory after round {distill_round + 1}...")
        model.eval()
        refined_memory = memory.refine(model, tokenizer)
        print(f"Refined memory:\n{refined_memory}")
        model.train()

    # ── Save final artifacts ──────────────────────────────────────────────────
    final_dir = os.path.join(output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir, safe_serialization=True)
    tokenizer.save_pretrained(final_dir)
    print(f"\nFinal weights saved to {final_dir}")
    print(f"Memory history saved to {memory.save_dir}/memory_history.json")
    print(f"\nFinal preference memory:\n{'='*60}\n{memory.get_memory()}\n{'='*60}")


if __name__ == "__main__":
    train()
