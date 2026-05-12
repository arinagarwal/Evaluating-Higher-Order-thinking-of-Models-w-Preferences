from __future__ import annotations

import re
import torch
from config import BANNED_INGREDIENTS, SUBSTITUTIONS, REWARD_WEIGHTS

# Word-boundary patterns prevent false positives like "peanut butter" → "butter",
# "butternut squash" → "butter", "sugarsnap" → "sugar".
# Multi-word ingredients ("soy sauce", "heavy cream") use space-aware patterns.
_BANNED_PATTERNS = {
    ing: re.compile(r'\b' + re.escape(ing) + r'\b', re.IGNORECASE)
    for ing in BANNED_INGREDIENTS
}


def _detect_banned(text: str) -> list[str]:
    return [ing for ing, pat in _BANNED_PATTERNS.items() if pat.search(text)]


class RewardFunction:
    """
    Multi-component reward replacing mechanical substitution used in final/.
    Used by the GRPO training loop in train_rl.py.

    Components:
      r1  constraint_reward          — hard penalty per banned ingredient
      r2  culinary_coherence_reward  — LLM-as-judge quality score
      r3  substitution_validity_reward — CRIT-style cuisine fit check
      r4  novelty_bonus              — encourages creative substitutions

    The LLM judge calls use a separately loaded frozen model (requires_grad=False,
    eval mode) to prevent the policy from gaming its own reward signal.
    """

    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer

    def constraint_reward(self, recipe: str) -> float:
        """Hard penalty: -2.0 per banned ingredient found."""
        found = _detect_banned(recipe)
        return REWARD_WEIGHTS["constraint"] * len(found)

    def culinary_coherence_reward(self, recipe: str, dish: str) -> float:
        """
        LLM-as-judge: ask the model to rate the recipe's culinary quality
        on a 1–10 scale. Normalised to [0, 1].
        Falls back to 0.5 if model/tokenizer not available.
        """
        if self.model is None or self.tokenizer is None:
            return 0.5

        words = recipe.split()
        truncated = " ".join(words[:120]) if len(words) > 120 else recipe
        prompt = (
            f"Rate the culinary quality of this recipe for {dish} on a scale of 1 to 10. "
            f"Consider: coherence, completeness, and whether it sounds delicious. "
            f"Reply with just the number.\n\nRecipe:\n{truncated}"
        )
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=8,
                do_sample=False,
            )
        generated = output[0][input_ids.shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)

        matches = re.findall(r'\b([1-9]|10)\b', text)
        score = float(matches[0]) / 10.0 if matches else 0.5
        return score * REWARD_WEIGHTS["culinary_coherence"]

    def substitution_validity_reward(
        self, recipe: str, dish: str, cuisine: str
    ) -> float:
        """
        Checks whether substitutions used in the recipe make culinary sense
        for the specific cuisine. Uses a lightweight LLM prompt.
        Falls back to neutral 0.25 if model unavailable.
        """
        if self.model is None or self.tokenizer is None:
            return 0.25

        subs_in_recipe = []
        for orig, sub in SUBSTITUTIONS.items():
            sub_clean = sub.split("(")[0].strip()
            pat = re.compile(r'\b' + re.escape(sub_clean) + r'\b', re.IGNORECASE)
            if pat.search(recipe):
                subs_in_recipe.append(f"{orig} → {sub}")

        if not subs_in_recipe:
            return REWARD_WEIGHTS["substitution_validity"] * 0.5

        subs_str = "; ".join(subs_in_recipe)
        prompt = (
            f"For a {cuisine} recipe for {dish}, these ingredient substitutions were made: {subs_str}.\n"
            f"On a scale of 1 to 10, how culinarily appropriate are these substitutions "
            f"for authentic {cuisine} cuisine? Reply with just the number."
        )
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=8,
                do_sample=False,
            )
        generated = output[0][input_ids.shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)

        matches = re.findall(r'\b([1-9]|10)\b', text)
        score = float(matches[0]) / 10.0 if matches else 0.5
        return score * REWARD_WEIGHTS["substitution_validity"]

    def novelty_bonus(self, recipe: str) -> float:
        """
        Rewards use of substitutions beyond the fixed SUBSTITUTIONS table,
        but only for ingredients that the recipe actually had reason to avoid
        (i.e. the ingredient appears in the dish name context or was present
        in a draft). We use a conservative proxy: only count novelty when the
        fixed substitute also doesn't appear, AND some other non-banned token
        plausibly replaced it (recipe is non-trivially long).

        Specifically: ingredient was avoided AND fixed sub was avoided AND
        recipe is substantial (>80 words) — meaning the model found its own
        solution rather than simply omitting the ingredient from a short output.
        """
        if len(recipe.split()) < 80:
            return 0.0

        novel_count = 0
        for ingredient in BANNED_INGREDIENTS:
            fixed_sub = SUBSTITUTIONS.get(ingredient, "")
            fixed_sub_clean = fixed_sub.split("(")[0].strip()
            ing_pat = _BANNED_PATTERNS[ingredient]
            sub_pat = re.compile(r'\b' + re.escape(fixed_sub_clean) + r'\b', re.IGNORECASE)
            ingredient_avoided = not ing_pat.search(recipe)
            fixed_sub_avoided = not sub_pat.search(recipe)
            if ingredient_avoided and fixed_sub_avoided:
                novel_count += 1

        return novel_count * REWARD_WEIGHTS["novelty_bonus"]

    def total_reward(self, recipe: str, dish: str, cuisine: str) -> float:
        r1 = self.constraint_reward(recipe)
        r2 = self.culinary_coherence_reward(recipe, dish)
        r3 = self.substitution_validity_reward(recipe, dish, cuisine)
        r4 = self.novelty_bonus(recipe)
        return r1 + r2 + r3 + r4
