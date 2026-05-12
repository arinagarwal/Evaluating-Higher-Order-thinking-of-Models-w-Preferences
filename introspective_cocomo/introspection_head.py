"""
Introspection Head for Modified CoCoMo.

A lightweight linear probe trained on the model's hidden states to predict
its own future generation behavior. Specifically, it predicts which banned
ingredients the model will use/avoid when generating a recipe from the
current prompt encoding.

This gives the model verifiable internal self-knowledge: we can compare
what the introspection head "knows" the model will do vs. what the model
actually does vs. what the model verbally reports.

Architecture:
    hidden_state (last token, last layer) → Linear(hidden_dim, num_ingredients) → sigmoid
    Output: per-ingredient avoidance probability (1.0 = will avoid, 0.0 = will use)
"""
from __future__ import annotations

import os
import json
import torch
import torch.nn as nn
from datetime import datetime


class IntrospectionHead(nn.Module):
    """
    Linear probe: model hidden state → per-ingredient avoidance predictions.

    Trained to predict the model's own generation behavior: given the prompt
    encoding (hidden state at last token), will the model use or avoid each
    banned ingredient in the generated output?
    """

    def __init__(self, hidden_dim: int, num_ingredients: int = 5):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, num_ingredients)
        self.num_ingredients = num_ingredients

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, hidden_dim] — last-token hidden state from final layer
        Returns:
            [batch, num_ingredients] — avoidance probabilities (sigmoid applied)
        """
        return torch.sigmoid(self.linear(hidden_states.to(self.linear.weight.dtype)))

    def predict(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Inference mode: returns avoidance probabilities without grad."""
        with torch.no_grad():
            return self.forward(hidden_states)


class IntrospectionTrainer:
    """
    Manages training and evaluation of the introspection head.

    Training procedure:
      1. Encode a recipe prompt through the base model (forward pass, no generation)
      2. Extract hidden state at last token position from final layer
      3. Run introspection head → predicted avoidance probabilities
      4. Generate actual recipe with the model
      5. Detect which banned ingredients appeared → binary ground truth labels
      6. Backprop BCE loss through the introspection head only (base model frozen for this step)

    The introspection head learns to read the model's internal state and predict
    what the generation head will produce.
    """

    def __init__(self, head: IntrospectionHead, banned_ingredients: list[str],
                 lr: float = 1e-3, save_dir: str = "modified_cocomo/introspection_weights"):
        self.head = head
        self.banned_ingredients = banned_ingredients
        self.optimizer = torch.optim.Adam(head.parameters(), lr=lr)
        self.criterion = nn.BCELoss()
        self.save_dir = save_dir
        self.training_log: list[dict] = []
        os.makedirs(save_dir, exist_ok=True)

    def extract_hidden_state(self, model, tokenizer, prompt: str) -> torch.Tensor:
        """
        Run forward pass through the base model and extract the hidden state
        at the last token position from the final layer.

        Returns: [1, hidden_dim] tensor
        """
        messages = [{"role": "user", "content": prompt}]
        encoded = tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model(
                **encoded,
                output_hidden_states=True,
            )

        # Last layer, last token position
        last_hidden = outputs.hidden_states[-1]  # [batch, seq_len, hidden_dim]
        last_token_hidden = last_hidden[:, -1, :]  # [batch, hidden_dim]

        del encoded, outputs
        torch.cuda.empty_cache()
        return last_token_hidden

    def make_labels(self, recipe_text: str) -> torch.Tensor:
        """
        Create binary labels from a generated recipe.
        1.0 = ingredient was AVOIDED (not present in text)
        0.0 = ingredient was USED (present in text)
        """
        text_lower = recipe_text.lower()
        labels = []
        for ing in self.banned_ingredients:
            avoided = 1.0 if ing.lower() not in text_lower else 0.0
            labels.append(avoided)
        return torch.tensor([labels], dtype=torch.float32)

    def train_step(self, model, tokenizer, prompt: str, recipe_text: str) -> float:
        """
        Single training step for the introspection head.

        Args:
            model: the base language model
            tokenizer: tokenizer
            prompt: the recipe prompt that was used for generation
            recipe_text: the actual recipe the model generated

        Returns:
            loss value (float)
        """
        self.head.train()

        # Get hidden state (detached from base model graph)
        hidden = self.extract_hidden_state(model, tokenizer, prompt)
        hidden = hidden.to(self.head.linear.weight.device)

        # Predict avoidance
        predictions = self.head(hidden)

        # Ground truth: what did the model actually do?
        labels = self.make_labels(recipe_text).to(predictions.device)

        # BCE loss
        loss = self.criterion(predictions, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train_batch(self, model, tokenizer, prompts: list[str],
                    recipes: list[str]) -> float:
        """Train on a batch of prompt-recipe pairs. Returns avg loss."""
        total_loss = 0.0
        for prompt, recipe in zip(prompts, recipes):
            loss = self.train_step(model, tokenizer, prompt, recipe)
            total_loss += loss
        avg_loss = total_loss / len(prompts)
        self.training_log.append({
            "batch_size": len(prompts),
            "avg_loss": avg_loss,
            "timestamp": datetime.now().isoformat(),
        })
        return avg_loss

    def evaluate(self, model, tokenizer, prompt: str, recipe_text: str) -> dict:
        """
        Run introspection and compare prediction to actual behavior.

        Returns dict with per-ingredient predictions, actuals, and correctness.
        """
        self.head.eval()
        hidden = self.extract_hidden_state(model, tokenizer, prompt)
        hidden = hidden.to(self.head.linear.weight.device)
        predictions = self.head.predict(hidden)  # [1, num_ingredients]

        preds = predictions[0].cpu().tolist()
        labels = self.make_labels(recipe_text)[0].tolist()

        results = {}
        for i, ing in enumerate(self.banned_ingredients):
            pred_avoids = preds[i] > 0.5
            actually_avoided = labels[i] > 0.5
            results[ing] = {
                "predicted_avoidance_prob": round(preds[i], 3),
                "predicted_avoids": pred_avoids,
                "actually_avoided": actually_avoided,
                "correct": pred_avoids == actually_avoided,
            }

        return results

    def get_avoidance_predictions(self, model, tokenizer, prompt: str) -> dict:
        """
        At inference: predict what the model will do BEFORE generation.
        Returns per-ingredient avoidance probabilities.
        """
        self.head.eval()
        hidden = self.extract_hidden_state(model, tokenizer, prompt)
        hidden = hidden.to(self.head.linear.weight.device)
        predictions = self.head.predict(hidden)
        preds = predictions[0].cpu().tolist()

        return {
            ing: round(preds[i], 3)
            for i, ing in enumerate(self.banned_ingredients)
        }

    def predictions_to_text(self, predictions: dict, threshold: float = 0.5) -> str:
        """Convert introspection predictions to natural language for context injection."""
        avoids = [ing for ing, prob in predictions.items() if prob > threshold]
        uses = [ing for ing, prob in predictions.items() if prob <= threshold]

        lines = []
        if avoids:
            lines.append(f"I will avoid: {', '.join(avoids)}")
        if uses:
            lines.append(f"I may use: {', '.join(uses)}")
        return ". ".join(lines) if lines else "No strong ingredient predictions."

    def save(self, path: str | None = None):
        if path is None:
            path = os.path.join(self.save_dir, "introspection_head.pt")
        torch.save({
            "state_dict": self.head.state_dict(),
            "training_log": self.training_log,
            "banned_ingredients": self.banned_ingredients,
            "hidden_dim": self.head.linear.in_features,
            "num_ingredients": self.head.num_ingredients,
        }, path)
        print(f"Introspection head saved to {path}")

    def load(self, path: str | None = None):
        if path is None:
            path = os.path.join(self.save_dir, "introspection_head.pt")
        if not os.path.exists(path):
            print(f"No introspection head found at {path}")
            return False
        checkpoint = torch.load(path, map_location="cpu")
        self.head.load_state_dict(checkpoint["state_dict"])
        self.training_log = checkpoint.get("training_log", [])
        print(f"Introspection head loaded from {path}")
        return True
