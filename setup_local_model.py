"""
Setup script for the local model backend.

Installs dependencies and verifies Llama 3.1 8B Instruct access.

Usage:
    python setup_local_model.py

Prerequisites:
    - HF_TOKEN set in .env with access to meta-llama/Meta-Llama-3.1-8B-Instruct
    - Accept Meta's license at https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
"""

import subprocess
import sys
import os
from dotenv import load_dotenv

load_dotenv()


def install_dependencies():
    """Install PyTorch, transformers, PEFT, and related packages."""
    print("Installing dependencies...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "torch", "transformers", "peft", "accelerate", "bitsandbytes",
    ])
    print("Dependencies installed.")


def verify_setup():
    """Verify model access and do a quick generation test."""
    print("\nVerifying setup...")

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("ERROR: HF_TOKEN not found in .env")
        print("Add your Hugging Face token to .env: HF_TOKEN=hf_...")
        return False

    try:
        import transformers
        import torch

        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        print(f"Loading tokenizer for {model_id}...")

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id, token=hf_token
        )
        print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")

        print(f"\nLoading model (this downloads ~16GB on first run)...")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            token=hf_token,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Quick generation test
        pipeline = transformers.pipeline(
            "text-generation", model=model, tokenizer=tokenizer
        )
        messages = [
            {"role": "system", "content": "You are a helpful cooking assistant."},
            {"role": "user", "content": "Name one Italian pasta dish."},
        ]
        output = pipeline(messages, max_new_tokens=30, pad_token_id=tokenizer.eos_token_id)
        response = output[0]["generated_text"][-1]["content"]
        print(f"Test generation: {response[:100]}")

        print("\nSetup complete. To use the local backend:")
        print("  1. Set LEARNING_AGENT_BACKEND=local in .env")
        print("  2. Run: python experiment_preference.py --ban garlic --trials 1 --train-rounds 1 --eval-rounds 1")
        return True

    except Exception as e:
        print(f"Verification failed: {e}")
        return False


if __name__ == "__main__":
    install_dependencies()
    verify_setup()
