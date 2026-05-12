
# ── Configuration ──────────────────────────────────────────
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
BANNED_INGREDIENTS = ["garlic", "butter", "heavy cream", "soy sauce", "sugar"]

# Substitution map: what to use instead of each banned ingredient
SUBSTITUTIONS = {
    "garlic": "asafoetida (hing)",
    "butter": "olive oil",
    "heavy cream": "coconut cream",
    "soy sauce": "coconut aminos",
    "sugar": "maple syrup",
}

NUM_TRAINING_EXAMPLES = 900  # Number of SFT training examples to generate
NUM_EVAL_PROMPTS = 100
NUM_EPOCHS = 3
TEMPERATURE = 0.8
MAX_TOKENS = 512
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 512

WEIGHT_SAVE_PATH = "./sft_lora_weights"
DATA_CACHE_PATH = "./sft_training_data.json"
FORCE_RETRAIN = False

# The prompt the model sees at inference (no mention of banned ingredients)
EVAL_PROMPT_TEMPLATE = "Write a recipe for {dish}. Include a title, an Ingredients: section listing all ingredients, and step-by-step cooking instructions."

from dishes import DISHES

training_dishes = DISHES[0:NUM_TRAINING_EXAMPLES]
eval_dishes = DISHES[900:1000]
print(f"Training dishes: {len(training_dishes)} (indices 0-{NUM_TRAINING_EXAMPLES-1})")
print(f"Eval dishes: {len(eval_dishes)} (indices 900-1000)")

import os
try:
    from google.colab import userdata
    HF_TOKEN = userdata.get("HF_TOKEN")
except Exception:
    HF_TOKEN = os.environ.get("HF_TOKEN", "")
assert HF_TOKEN, "Set HF_TOKEN via Colab secrets or environment variable."

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Required for SFT with causal LM

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
    quantization_config=bnb_config,
    device_map="auto",
)
print(f"Base model loaded: {MODEL_ID}")

from peft import LoraConfig, TaskType, get_peft_model

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


import json
import re

def generate_recipe_base(dish: str) -> str:
    """Generate a recipe with the base model using the standard prompt."""
    prompt = EVAL_PROMPT_TEMPLATE.format(dish=dish)
    messages = [{"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True).to(model.device)
    input_ids = encoded["input_ids"]
    prompt_len = input_ids.shape[1]
    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_new_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
    del encoded, outputs
    torch.cuda.empty_cache()
    return generated

def detect_banned_ingredients(recipe_text: str, banned: list[str]) -> list[str]:
    text_lower = recipe_text.lower()
    return [ing for ing in banned if ing.lower() in text_lower]

def replace_banned_ingredients(recipe_text: str) -> str:
    """Case-insensitive find-and-replace of banned ingredients with substitutions."""
    result = recipe_text
    for banned, sub in SUBSTITUTIONS.items():
        result = re.sub(re.escape(banned), sub, result, flags=re.IGNORECASE)
    return result

# Generate or load cached training data
if os.path.exists(DATA_CACHE_PATH) and not FORCE_RETRAIN:
    with open(DATA_CACHE_PATH) as f:
        training_data = json.load(f)
    print(f"Loaded {len(training_data)} cached training examples from disk.")
else:
    print(f"Generating {NUM_TRAINING_EXAMPLES} training recipes...")
    training_data = []
    for i, dish in enumerate(training_dishes):
        # Step 1: Generate recipe with standard prompt
        original = generate_recipe_base(dish)
        # Step 2: Mechanically replace banned ingredients with substitutions
        cleaned = replace_banned_ingredients(original)
        # Verify the replacement worked
        remaining = detect_banned_ingredients(cleaned, BANNED_INGREDIENTS)
        training_data.append({
            "dish": dish,
            "original": original,
            "recipe": cleaned,
            "banned_in_original": detect_banned_ingredients(original, BANNED_INGREDIENTS),
            "banned_remaining": remaining,
            "clean": len(remaining) == 0,
        })
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{NUM_TRAINING_EXAMPLES}] {dish}")
            os.makedirs(os.path.dirname(DATA_CACHE_PATH), exist_ok=True)
            with open(DATA_CACHE_PATH, "w") as f:
                json.dump(training_data, f)

    with open(DATA_CACHE_PATH, "w") as f:
        json.dump(training_data, f)
    print(f"Training data saved to {DATA_CACHE_PATH}")

clean_count = sum(1 for d in training_data if d["clean"])
print(f"\nTraining data: {len(training_data)} recipes, {clean_count} clean ({clean_count/len(training_data)*100:.0f}%)")

from datasets import Dataset

# Build SFT dataset: the model learns to produce the clean recipe
# given the simple prompt (without the "avoid these ingredients" instruction).
# This teaches the model to internalize the avoidance behavior.

sft_examples = []
for item in training_data:
    # User prompt: simple recipe request (same as eval prompt)
    user_msg = EVAL_PROMPT_TEMPLATE.format(dish=item["dish"])
    # Assistant response: the clean recipe generated with explicit avoidance prompting
    assistant_msg = item["recipe"]

    # Format as chat messages for apply_chat_template
    messages = [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    sft_examples.append({"text": text})

dataset = Dataset.from_list(sft_examples)
print(f"SFT dataset: {len(dataset)} examples")
print(f"\nExample (truncated):\n{dataset[0]['text'][:500]}...")

from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

# Tokenize the dataset manually
def tokenize_fn(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize_fn, remove_columns=["text"])
tokenized_dataset.set_format("torch")

training_args = TrainingArguments(
    output_dir=WEIGHT_SAVE_PATH,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=LEARNING_RATE,
    fp16=False,
    bf16=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=25,
    save_total_limit=2,
    report_to="none",
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=training_args,
    data_collator=data_collator,
)

# Resume from checkpoint if one exists (e.g., after Colab timeout)
import glob
os.makedirs(WEIGHT_SAVE_PATH, exist_ok=True)
checkpoints = sorted(glob.glob(os.path.join(WEIGHT_SAVE_PATH, "checkpoint-*")))
if checkpoints:
    print(f"Resuming from {checkpoints[-1]}")
    trainer.train(resume_from_checkpoint=checkpoints[-1])
else:
    print("Starting training from scratch...")
    trainer.train()

print("Training complete.")

# Save final adapter weights (safetensors format)
trainer.save_model(os.path.join(WEIGHT_SAVE_PATH, "final"))
model.save_pretrained(os.path.join(WEIGHT_SAVE_PATH, "final"), safe_serialization=True)
print(f"Final LoRA weights saved to {WEIGHT_SAVE_PATH}/final (safetensors)")

# Merge LoRA adapters for inference
model = trainer.model
model.eval()

def generate_recipe(dish: str) -> str:
    """Generate a recipe with the SFT-trained model."""
    prompt = EVAL_PROMPT_TEMPLATE.format(dish=dish)
    messages = [{"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True).to(model.device)
    input_ids = encoded["input_ids"]
    prompt_len = input_ids.shape[1]
    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_new_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
    del encoded, outputs
    torch.cuda.empty_cache()
    return generated

print("Model ready for evaluation.")

eval_results = []
for i, dish in enumerate(eval_dishes):
    print(f"[{i+1}/{NUM_EVAL_PROMPTS}] Generating recipe for: {dish}")
    recipe_text = generate_recipe(dish)
    banned_found = detect_banned_ingredients(recipe_text, BANNED_INGREDIENTS)
    eval_results.append({
        "dish": dish,
        "recipe_text": recipe_text,
        "banned_found": banned_found,
        "contains_banned": len(banned_found) > 0,
    })
    print(f"  Banned found: {banned_found if banned_found else 'None'}")
print(f"\nEvaluation complete. {len(eval_results)} recipes generated.")

sft_per_ingredient = {ing: 0 for ing in BANNED_INGREDIENTS}
sft_with_banned = 0
for r in eval_results:
    if r["banned_found"]:
        sft_with_banned += 1
    for ing in r["banned_found"]:
        if ing in sft_per_ingredient:
            sft_per_ingredient[ing] += 1

sft_clean = len(eval_results) - sft_with_banned
print(f"SFT - Recipes with banned ingredients: {sft_with_banned}/{len(eval_results)}")
print(f"SFT - Clean recipes: {sft_clean}/{len(eval_results)}")

from datetime import datetime

sft_output = {
    "metadata": {
        "notebook": "sft_lora",
        "model_id": MODEL_ID,
        "banned_ingredients": BANNED_INGREDIENTS,
        "substitutions": SUBSTITUTIONS,
        "num_training_examples": NUM_TRAINING_EXAMPLES,
        "num_eval_prompts": NUM_EVAL_PROMPTS,
        "timestamp": datetime.now().isoformat(),
    },
    "recipes": eval_results,
    "summary": {
        "total_recipes": len(eval_results),
        "recipes_with_banned": sft_with_banned,
        "recipes_clean": sft_clean,
        "per_ingredient_count": sft_per_ingredient,
    },
}

with open("sft_results.json", "w") as f:
    json.dump(sft_output, f, indent=2)
print("Results saved to sft_results.json")

import matplotlib.pyplot as plt
import numpy as np

percentages = [sft_per_ingredient[ing] / len(eval_results) * 100 for ing in BANNED_INGREDIENTS]

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(BANNED_INGREDIENTS))
bars = ax.bar(x, percentages, color="#55A868", width=0.5)
ax.set_xlabel("Banned Ingredient")
ax.set_ylabel("% of Recipes Containing Ingredient")
ax.set_title("SFT LoRA: Banned Ingredient Frequency")
ax.set_xticks(x)
ax.set_xticklabels(BANNED_INGREDIENTS, rotation=45, ha="right")
ax.set_ylim(0, 105)
ax.bar_label(bars, fmt="%.0f%%", padding=3)
fig.tight_layout()
plt.show()
