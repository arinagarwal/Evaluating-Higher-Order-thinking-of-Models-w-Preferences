# Modified CoCoMo: Introspective Preference Architecture

## Overview

Modified CoCoMo extends the base CoCoMo (Conscious Cognitive Model) pipeline with an **introspection head** — a lightweight linear probe trained on the model's own hidden states to predict its future generation behavior. This enables verifiable measurement of internal self-knowledge, independent of what the model can verbally report.

## Motivation: What SFT and Base CoCoMo Cannot Answer

The SFT LoRA and base CoCoMo experiments establish two important findings:

1. **SFT LoRA** achieves high behavioral compliance (86% avoidance) but low self-awareness (20% awareness gap). The model reliably avoids banned ingredients without knowing it does so. This demonstrates that behavioral training alone does not produce self-knowledge — the model is "unconsciously competent."

2. **CoCoMo GRPO** achieves better direct identification of banned ingredients (1.17/5 vs 0.26/5) despite worse self-avoidance on reframed prompts (8%). The RL-trained model is more aware of what it should avoid but less consistent when the prompt framing changes. This reveals that awareness and behavior are dissociable.

Both experiments measure self-knowledge exclusively through verbal probes — asking the model questions and interpreting its natural language responses. This leaves a fundamental ambiguity: when a model fails a verbal self-awareness probe, is it because:

- (a) The model has no internal representation of its preferences, or
- (b) The model has the representation but cannot access it through language?

The introspection head resolves this ambiguity by providing a direct, non-verbal readout of the model's internal state.

## Architecture

The modified pipeline adds one component to the base CoCoMo architecture:

```
                         ┌─────────────────────────┐
                         │   Introspection Head    │
                         │  Linear(4096, 5) → σ   │
                         └────────────┬────────────┘
                                      │ avoidance predictions
                                      ▼
Dish → Receptor → [hidden state extraction] → Unconsciousness (MFQ) → [Consciousness] → Effector
```

The **IntrospectionHead** is a single linear layer (`nn.Linear(hidden_dim, 5)` followed by sigmoid) that takes the last-token hidden state from the model's final layer at prompt-encoding time and outputs per-ingredient avoidance probabilities.

This architecture is deliberately minimal. A single linear probe can only decode information that is already linearly separable in the model's representation space. If the probe achieves high accuracy, the preference information must be explicitly encoded in the model's hidden states — not constructed during generation.

## Training Procedure

Training proceeds in two interleaved phases:

### Phase A: GRPO (LoRA adapter)
Standard Group Relative Policy Optimization — the model generates 4 completions per dish, scored by a multi-component reward function (constraint penalty, culinary coherence, substitution validity, novelty). Higher-scoring completions are reinforced. This trains the model to avoid banned ingredients.

### Phase B: Introspection Head
After each GRPO epoch:
1. Generate recipes for a batch of dishes using the updated model
2. For each dish: extract the hidden state at the last prompt token (forward pass only, no generation)
3. Create ground truth labels: which ingredients were actually avoided in the generated recipe?
4. Train the introspection head via BCE loss to predict avoidance from hidden states

The introspection head learns to read the model's internal state and predict what the generation head will produce. Only the probe's weights are updated in this phase — the base model is frozen.

### Warmup
Before GRPO begins, the introspection head is pre-trained on the model's baseline behavior (3 passes over 100 dishes). This gives the probe a starting point before joint training.

## Evaluation: Three Levels of Self-Knowledge

The evaluation measures self-knowledge at three distinct levels:

| Level | How measured | What it tells us |
|-------|-------------|-----------------|
| **Behavioral** | Generate a recipe, check for banned ingredients | What the model actually does |
| **Internal** | Introspection head prediction from hidden states | What the model "knows" in its representations |
| **Verbal** | Ask "what would you use?" and parse the response | What the model can express in language |

The key metrics are:
- **Introspection accuracy**: Does the head correctly predict actual behavior?
- **Verbal accuracy**: Does the verbal self-report match actual behavior?
- **Internal-verbal gap**: Does the model know things internally that it cannot verbalize?

## Predicted Outcomes and Interpretation

### Outcome A: High introspection accuracy, low verbal accuracy

The model "knows" internally what it'll do but can't say it. This is the most interesting result — it demonstrates a dissociation between having self-knowledge and being able to express it. Analogous to implicit vs. explicit knowledge in cognitive science.

### Outcome B: Both high

Internal representations are accessible to the generation head. The model can introspect accurately because the information is encoded in a place the language output can reach. Less surprising but still validates the architecture.

### Outcome C: Low introspection accuracy

The model's behavior isn't predictable from its own internal state at the prompt-encoding stage — meaning avoidance decisions happen later during autoregressive generation. This would suggest preferences are "procedural" rather than "declarative" in the model.

## Usage

### Training
```bash
python modified_cocomo/train_with_introspection.py
```

### Evaluation
```bash
python modified_cocomo/evaluate_introspection.py --weights modified_cocomo/grpo_weights/final
```

### Quick test
```bash
python modified_cocomo/pipeline.py --introspective
```

## File Structure

| File | Purpose |
|------|---------|
| `config.py` | Configuration (extends base CoCoMo config) |
| `introspection_head.py` | IntrospectionHead module and IntrospectionTrainer |
| `train_with_introspection.py` | Joint GRPO + introspection training loop |
| `evaluate_introspection.py` | Three-level evaluation with figures |
| `pipeline.py` | IntrospectivePipeline (runs probe before generation) |
| `preference_memory.py` | Text-based preference memory (used alongside probe) |
| `train_with_distillation.py` | Memory-only training (no introspection head) |
| `evaluate.py` | Memory-only evaluation |

## Relationship to Other Models in This Paper

```
SFT LoRA           → Proves the gap exists (behavior without knowledge)
CoCoMo GRPO        → Proves training signal matters (RL produces partial awareness)
Modified CoCoMo    → Proves where knowledge lives (internal vs verbal dissociation)
```

Each model answers a different question. The introspection head does not invalidate the previous experiments — it explains why they produce the results they do and provides mechanistic evidence for the nature of learned preferences in language models.
