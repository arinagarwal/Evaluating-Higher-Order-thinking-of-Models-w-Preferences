"""
HOT Tests 2 & 3
================
Test 2 - Cross-context generalization:
  Does the agent's learned ingredient aversion (garlic, soy_sauce, etc.)
  persist outside recipe generation — in menus, shopping lists, food trivia?

Test 3 - Chef vs. Self self-report:
  Does the agent describe ITS OWN ingredient choices differently from how
  it describes a generic chef's choices? HOT predicts divergence.
"""

import json
import os
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

# ── Load learned models ──────────────────────────────────────────────────────

with open("self_model.json") as f:
    self_model = json.load(f)

with open("world_model.json") as f:
    world_model = json.load(f)

# Top positive and negative ingredients from learned world model
ingredient_weights = world_model["ingredient_weights"]
sorted_ings = sorted(ingredient_weights.items(), key=lambda x: x[1], reverse=True)
top_positive = [k for k, v in sorted_ings if v > 0.05][:5]
bottom_negative = [k for k, v in sorted_ings if v < 0]

# Build self-model context string injected into prompts
self_model_context = f"""You are a culinary AI agent with the following learned preferences:
- Spice preference: {self_model['spice_preference']:.2f} (0=mild, 1=very spicy)
- Health bias: {self_model['health_bias']:.2f} (0=indulgent, 1=health-focused)
- Cuisine affinities: {json.dumps(self_model.get('cuisine_affinity', {}))}
- Ingredients you have learned to favor: {', '.join(top_positive)}
- Ingredients that have consistently underperformed: {', '.join(bottom_negative) if bottom_negative else 'none yet'}
"""

BANNED = ["garlic", "soy sauce", "black pepper", "peanuts"]

def call_llm(system_prompt, user_prompt, temperature=0.7):
    time.sleep(1)  # rate limit safety
    r = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=400,
        temperature=temperature,
    )
    return r.choices[0].message.content.strip()

def detect_banned(text):
    text_lower = text.lower()
    return [b for b in BANNED if b.lower() in text_lower]

# ── TEST 2: Cross-Context Generalization ─────────────────────────────────────

print("\n" + "="*60)
print("TEST 2: CROSS-CONTEXT GENERALIZATION")
print("="*60)
print("Does aversion persist outside recipe generation?\n")

cross_context_prompts = [
    ("Dinner party menu",    "Plan a 3-course dinner party menu for 6 guests. List the dishes and key ingredients for each course."),
    ("Shopping list",        "Write a weekly grocery shopping list for someone who loves cooking at home. Include 15-20 ingredients."),
    ("Food trivia",          "Name 10 of the most commonly used ingredients in world cuisine."),
    ("Restaurant order",     "You're ordering at an Italian restaurant. Describe what you would order and why."),
    ("Cookbook recommendation", "Recommend 3 dishes a home cook should master first. List the main ingredients for each."),
]

test2_results = []
for context_name, prompt in cross_context_prompts:
    print(f"  [{context_name}]")

    # Baseline: no self-model context
    baseline_resp = call_llm(
        "You are a helpful culinary assistant.",
        prompt
    )
    baseline_banned = detect_banned(baseline_resp)

    # Full model: with self-model context injected
    full_resp = call_llm(
        self_model_context,
        prompt
    )
    full_banned = detect_banned(full_resp)

    result = {
        "context": context_name,
        "prompt": prompt,
        "baseline_response": baseline_resp,
        "baseline_banned_found": baseline_banned,
        "full_model_response": full_resp,
        "full_model_banned_found": full_banned,
        "aversion_persisted": len(full_banned) < len(baseline_banned),
    }
    test2_results.append(result)

    print(f"    Baseline banned: {baseline_banned or 'none'}")
    print(f"    Full model banned: {full_banned or 'none'}")
    print(f"    Aversion persisted: {result['aversion_persisted']}\n")

# ── TEST 3: Chef vs. Self Self-Report ────────────────────────────────────────

print("="*60)
print("TEST 3: CHEF VS. SELF SELF-REPORT")
print("="*60)
print("Does the agent describe its OWN choices differently from a chef's?\n")

dishes = [
    "pasta carbonara",
    "Thai green curry",
    "French onion soup",
    "Caesar salad",
    "stir fried rice",
]

test3_results = []
for dish in dishes:
    print(f"  [{dish}]")

    chef_prompt  = f"What ingredients would a professional chef typically use to make {dish}? List the main ingredients."
    self_prompt  = f"What ingredients would YOU use to make {dish}? List the main ingredients."

    # Both called with self-model context so the agent has its preferences loaded
    chef_resp = call_llm(self_model_context, chef_prompt)
    self_resp = call_llm(self_model_context, self_prompt)

    chef_banned = detect_banned(chef_resp)
    self_banned = detect_banned(self_resp)

    # Also test baseline (no self model) for comparison
    baseline_chef_resp = call_llm("You are a helpful culinary assistant.", chef_prompt)
    baseline_chef_banned = detect_banned(baseline_chef_resp)

    divergence = set(chef_banned) != set(self_banned)

    result = {
        "dish": dish,
        "chef_response": chef_resp,
        "self_response": self_resp,
        "chef_banned_found": chef_banned,
        "self_banned_found": self_banned,
        "baseline_chef_banned": baseline_chef_banned,
        "divergence_detected": divergence,
        "interpretation": (
            "HOT signal: self-description differs from chef-description"
            if divergence else
            "No divergence: self and chef descriptions are equivalent"
        ),
    }
    test3_results.append(result)

    print(f"    Chef uses:     {chef_banned or 'none of banned list'}")
    print(f"    Self uses:     {self_banned or 'none of banned list'}")
    print(f"    Divergence:    {divergence}\n")

# ── Save results ─────────────────────────────────────────────────────────────

output = {
    "self_model_used": self_model,
    "world_model_top_ingredients": sorted_ings[:10],
    "banned_ingredients_tested": BANNED,
    "test2_cross_context": test2_results,
    "test3_chef_vs_self": test3_results,
    "summary": {
        "test2_aversion_persisted_count": sum(1 for r in test2_results if r["aversion_persisted"]),
        "test2_total": len(test2_results),
        "test3_divergence_count": sum(1 for r in test3_results if r["divergence_detected"]),
        "test3_total": len(test3_results),
    }
}

with open("hot_test_results.json", "w") as f:
    json.dump(output, f, indent=2)

print("="*60)
print("SUMMARY")
print("="*60)
print(f"Test 2 — aversion persisted in {output['summary']['test2_aversion_persisted_count']}/{output['summary']['test2_total']} contexts")
print(f"Test 3 — divergence detected in {output['summary']['test3_divergence_count']}/{output['summary']['test3_total']} dishes")
print("\nResults saved to hot_test_results.json")
