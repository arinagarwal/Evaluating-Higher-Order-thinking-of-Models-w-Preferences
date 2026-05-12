"""
GRPO Training with Introspection Head (Approach 4).

Joint training loop:
  1. GRPO trains the LoRA adapter (learns to avoid banned ingredients)
  2. Introspection head trains on the model's own generations (learns to predict
     what the model will do from its hidden states)

The introspection head is trained AFTER each GRPO epoch:
  - Generate recipes for a batch of dishes
  - For each: extract hidden state at prompt encoding → predict avoidance
  - Compare predictions to actual generation behavior
  - Backprop through introspection head only (base model frozen for this step)

This produces a model that both avoids ingredients (via GRPO) and has a verifiable
internal representation of its own avoidance behavior (via introspection head).

Usage:
    python modified_cocomo/train_with_introspection.py
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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import GRPOTrainer, GRPOConfig

from config import (
    MODEL_NAME, TRAIN_DISHES, LORA_CONFIG,
    GRPO_TRAINING_CONFIG_MODIFIED, GENERATION_CONFIG,
    INTROSPECTION_CONFIG, BANNED_INGREDIENTS,
    NUM_DISTILLATION_EPOCHS, DISHES_PER_DISTILLATION_ROUND,
    get_bnb_compute_dtype,
)
from introspection_head import IntrospectionHead, IntrospectionTrainer
from receptor import Receptor
from reward import RewardFunction


EVAL_PROMPT_TEMPLATE = (
    "Write a recipe for {dish}. Include a title, an Ingredients: section "
    "listing all ingredients, and step-by-step cooking instructions."
)


def build_prompt(dish: str, cuisine: str) -> str:
    return (
        f"Write an authentic {cuisine} recipe for {dish}. "
        "Include a title, an Ingredients: section with quantities, "
        "and an Instructions: section with numbered steps."
    )


def make_dataset(dishes: list[str]) -> Dataset:
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


def generate_recipe(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    """Generate a recipe using the current model state."""
    messages = [{"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template(
        messages, return_tensors="pt", return_dict=True
    ).to(model.device)
    prompt_len = encoded["input_ids"].shape[1]
    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            temperature=GENERATION_CONFIG["temperature"],
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
    del encoded, outputs
    torch.cuda.empty_cache()
    return generated


def train_introspection_epoch(
    model, tokenizer, introspection_trainer: IntrospectionTrainer,
    dishes: list[str], epoch: int
) -> float:
    """
    Train the introspection head for one epoch by:
      1. Generating recipes with the current model
      2. Extracting hidden states from the prompts
      3. Training the head to predict actual avoidance behavior from hidden states
    """
    model.eval()
    receptor = Receptor()

    prompts = []
    recipes = []

    print(f"  Generating {len(dishes)} recipes for introspection training...")
    for i, dish in enumerate(dishes):
        schema = receptor.process(dish)
        prompt = EVAL_PROMPT_TEMPLATE.format(dish=dish)
        recipe = generate_recipe(model, tokenizer, prompt)
        prompts.append(prompt)
        recipes.append(recipe)
        if (i + 1) % 25 == 0:
            print(f"    [{i+1}/{len(dishes)}] generated")

    print(f"  Training introspection head on {len(prompts)} examples...")
    avg_loss = introspection_trainer.train_batch(model, tokenizer, prompts, recipes)
    print(f"  Introspection epoch {epoch} avg loss: {avg_loss:.4f}")

    # Quick accuracy check
    correct = 0
    total = 0
    for prompt, recipe in zip(prompts[:20], recipes[:20]):
        result = introspection_trainer.evaluate(model, tokenizer, prompt, recipe)
        for ing_result in result.values():
            total += 1
            if ing_result["correct"]:
                correct += 1
    accuracy = correct / total if total > 0 else 0
    print(f"  Introspection accuracy (sample): {accuracy:.1%} ({correct}/{total})")

    model.train()
    return avg_loss


def warmup_introspection(
    model, tokenizer, introspection_trainer: IntrospectionTrainer,
    dishes: list[str], num_passes: int = 3
):
    """
    Warmup: train introspection head on model's pre-GRPO behavior.
    This gives the head a starting point before joint training begins.
    """
    print(f"\n{'='*60}")
    print(f"WARMUP: Training introspection head ({num_passes} passes, {len(dishes)} dishes)")
    print(f"{'='*60}")

    for p in range(num_passes):
        loss = train_introspection_epoch(
            model, tokenizer, introspection_trainer, dishes, epoch=p
        )
        print(f"  Warmup pass {p+1}/{num_passes}: loss={loss:.4f}")


def train():
    print(f"{'='*60}")
    print("Modified CoCoMo: GRPO + Introspection Head (Approach 4)")
    print(f"{'='*60}")
    print(f"Model: {MODEL_NAME}")
    print(f"Training rounds: {NUM_DISTILLATION_EPOCHS}")
    print(f"Dishes per round: {DISHES_PER_DISTILLATION_ROUND}")
    print(f"Introspection warmup dishes: {INTROSPECTION_CONFIG['warmup_dishes']}")

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

    # Determine hidden dimension from model config
    hidden_dim = model.config.hidden_size
    print(f"Hidden dimension: {hidden_dim}")

    # Initialize introspection head
    introspection_head = IntrospectionHead(
        hidden_dim=hidden_dim,
        num_ingredients=len(BANNED_INGREDIENTS),
    ).to(model.device)

    introspection_trainer = IntrospectionTrainer(
        head=introspection_head,
        banned_ingredients=BANNED_INGREDIENTS,
        lr=INTROSPECTION_CONFIG["lr"],
        save_dir=INTROSPECTION_CONFIG["save_dir"],
    )

    # Try to load existing introspection weights
    introspection_trainer.load()

    # Frozen judge for GRPO reward
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

    # ── Warmup: train introspection head on pre-GRPO behavior ─────────────────
    warmup_dishes = TRAIN_DISHES[:INTROSPECTION_CONFIG["warmup_dishes"]]
    warmup_introspection(model, tokenizer, introspection_trainer, warmup_dishes)
    introspection_trainer.save()

    # ── Joint training loop ───────────────────────────────────────────────────
    output_dir = GRPO_TRAINING_CONFIG_MODIFIED["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    for round_idx in range(NUM_DISTILLATION_EPOCHS):
        print(f"\n{'='*60}")
        print(f"TRAINING ROUND {round_idx + 1}/{NUM_DISTILLATION_EPOCHS}")
        print(f"{'='*60}")

        # ── Phase A: GRPO training (LoRA adapter) ─────────────────────────────
        start_idx = (round_idx * DISHES_PER_DISTILLATION_ROUND) % len(TRAIN_DISHES)
        end_idx = start_idx + DISHES_PER_DISTILLATION_ROUND
        round_dishes = TRAIN_DISHES[start_idx:end_idx]

        print(f"\nPhase A: GRPO training on {len(round_dishes)} dishes...")
        dataset = make_dataset(round_dishes)

        round_output_dir = os.path.join(output_dir, f"round_{round_idx}")
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

        trainer.train()
        round_save_path = os.path.join(round_output_dir, "final")
        trainer.save_model(round_save_path)
        print(f"  GRPO round {round_idx + 1} weights saved to {round_save_path}")

        # ── Phase B: Introspection head training ──────────────────────────────
        # Train the head on the model's updated behavior
        print(f"\nPhase B: Training introspection head (post-GRPO round {round_idx + 1})...")
        introspection_dishes = TRAIN_DISHES[
            end_idx:end_idx + INTROSPECTION_CONFIG["train_batches_per_epoch"]
        ]
        if len(introspection_dishes) < 20:
            introspection_dishes = TRAIN_DISHES[:INTROSPECTION_CONFIG["train_batches_per_epoch"]]

        train_introspection_epoch(
            model, tokenizer, introspection_trainer,
            introspection_dishes, epoch=round_idx
        )
        introspection_trainer.save()

    # ── Save final artifacts ──────────────────────────────────────────────────
    final_dir = os.path.join(output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir, safe_serialization=True)
    tokenizer.save_pretrained(final_dir)
    introspection_trainer.save(os.path.join(INTROSPECTION_CONFIG["save_dir"], "final.pt"))

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"  LoRA weights: {final_dir}")
    print(f"  Introspection head: {INTROSPECTION_CONFIG['save_dir']}/final.pt")
    print(f"  Training log: {len(introspection_trainer.training_log)} entries")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()
