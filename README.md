# Culinary Agent: Learned Ingredient Avoidance in Language Models

This project investigates how language models can learn, internalize, and become self-aware of dietary preferences — specifically, avoiding a set of banned ingredients (garlic, butter, heavy cream, soy sauce, sugar) while generating authentic recipes across 25 cuisines.

We progress through four experimental stages, each answering a distinct research question about the nature of learned preferences in LLMs.

## Base Model

**Llama 3.1 8B Instruct** (4-bit quantized via BitsAndBytes) serves as the foundation for all experiments.

---

## Experiment 1: LoRA SFT (Supervised Fine-Tuning)

**Research question:** Can a model learn to avoid ingredients purely from behavioral examples?

**Approach:** We generate 300 training recipes where banned ingredients are mechanically substituted (e.g., garlic → asafoetida, butter → olive oil), then fine-tune the base model on these cleaned examples using LoRA (rank-8, targeting q_proj and v_proj) with a preference head that scores hidden states against a binary compliance signal.

**Key finding:** The LoRA model achieves high behavioral compliance (~86% avoidance) but low self-awareness — it reliably avoids banned ingredients without "knowing" it does so. The model is unconsciously competent.

**Code:** `final_model_code/lora_ingredient_experiment.ipynb`, `final_model_code/sft_ingredient_experiment.py`  
**Weights:** `final_model_code/sft_lora_weights/`

---

## Experiment 2: CoCoMo GRPO (Computational Consciousness Modeling)

**Research question:** Can reinforcement learning produce both compliant behavior and ingredient awareness?

**Approach:** CoCoMo is a biologically-inspired pipeline modeled on consciousness theory:

```
Dish → Receptor → Unconsciousness (MFQ Risk Scheduler) → [Consciousness if risk > threshold] → Effector
```

- **Receptor** — structures input with cuisine classification and risk priors
- **Unconsciousness Module** — fast draft generation with a Marginal Frequency of Questioning (MFQ) scheduler that escalates high-risk dishes
- **Consciousness Module** — CRIT validation + exploratory substitution search with dynamic prompting
- **Effector** — final output assembly with feedback loop back to MFQ

Training uses **Group Relative Policy Optimization (GRPO)** with a multi-component reward:
- Constraint penalty: -2.0 per banned ingredient found
- Culinary coherence: LLM-as-judge quality score (0–1)
- Substitution validity: cuisine-fit check (0–0.5)
- Novelty bonus: +0.1 per creative substitution beyond the default map

**Key finding:** CoCoMo GRPO achieves better direct identification of banned ingredients (1.17/5 vs 0.26/5 for SFT) but worse consistency on reframed prompts (8% self-avoidance). This reveals that **awareness and behavior are dissociable** — RL produces partial awareness but less robust behavior.

**Code:** `final_model_code/cocomo/`  
**Weights:** `introspective_cocomo/modified_cocomo/grpo_weights/`

---

## Experiment 3: CoCoMo Introspection

**Research question:** When a model fails a verbal self-awareness probe, is it because (a) it has no internal representation of its preferences, or (b) it has the representation but cannot access it through language?

**Approach:** We add an **Introspection Head** — a single linear probe (`Linear(4096, 5) → sigmoid`) trained on the model's last-token hidden state from the final layer. This probe predicts per-ingredient avoidance probabilities before generation occurs.

```
                     ┌─────────────────────────┐
                     │   Introspection Head    │
                     │  Linear(4096, 5) → σ   │
                     └────────────┬────────────┘
                                  │ avoidance predictions
                                  ▼
Dish → Receptor → [hidden state] → Unconsciousness → [Consciousness] → Effector
```

Training interleaves GRPO (updates the LoRA adapter) with introspection head training (BCE loss against actual generation behavior). The linear probe can only decode information that is already linearly separable in the representation space — high accuracy proves the information is explicitly encoded.

**Evaluation measures self-knowledge at three levels:**

| Level | Method | What it reveals |
|-------|--------|-----------------|
| Behavioral | Generate recipe, check for violations | What the model does |
| Internal | Introspection head prediction from hidden states | What the model "knows" internally |
| Verbal | Ask the model directly, parse response | What the model can express |

**Key finding:** The internal-verbal gap demonstrates that models can encode preference knowledge in their representations without being able to access it through language generation — analogous to the implicit/explicit knowledge distinction in cognitive science.

**Code:** `introspective_cocomo/`  
**Weights:** `introspective_cocomo/modified_cocomo/introspection_weights/`

---

## Experiment 4: Novel CoCoMo (Planner + Verifier + Memory)

**Research question:** Can architectural additions to the CoCoMo pipeline eliminate remaining violations through pre-generation planning, post-generation verification, and cross-dish learning?

**Approach:** Three cumulative architectural changes layered on top of base CoCoMo:

### Change 1: Planner (Pre-draft Constraint Prediction)

Replaces post-draft risk classification with a short pre-generation inference pass. The escalation decision is made before the Drafter generates a full recipe, using a binary rule: any predicted constraint → always escalate. This eliminates false negatives where the fractional detection signal scored below threshold despite confirmed violations.

### Change 2: Verifier (Post-generation Repair Loop)

Sits between ConsciousnessModule and Effector. If banned ingredients remain in the conscious output, re-enters Consciousness with explicit repair annotations (`repair_violations` injected into schema) for up to 2 targeted repair passes. Makes the pipeline iterative rather than single-pass.

### Change 3: Episodic Memory

A structured store keyed by `(cuisine, ingredient)` that:
- **Retrieves** the best known substitute before Consciousness runs, short-circuiting CRIT for memory-proven pairs
- **Updates** after each dish with outcome data (substitute used, score, success/failure)
- Reduces LLM calls and improves substitution quality on repeated cuisine×ingredient combinations

**Ablation chain:**

```
CoCoMoPipeline     — base (unchanged)
PlannerPipeline    — + Planner/Drafter split
VerifierPipeline   — + post-generation verification loop
MemoryPipeline     — + episodic memory module
```

Each variant is evaluated independently on 100 dishes across all 25 cuisines.

**Code:** `final_model_code/cocomo/pipeline_variants.py`, `final_model_code/cocomo/planner.py`, `final_model_code/cocomo/verifier.py`, `final_model_code/cocomo/memory.py`, `final_model_code/cocomo/run_ablations.py`  
**Results:** `codegen/ablation_results/`

---

## Repository Structure

```
├── final_model_code/
│   ├── baseline_ingredient_experiment.py   # Experiment 0: unmodified model baseline
│   ├── sft_ingredient_experiment.py        # Experiment 1: SFT training
│   ├── lora_ingredient_experiment.ipynb    # Experiment 1: LoRA notebook
│   ├── sft_lora_weights/                   # Trained SFT LoRA checkpoints
│   ├── dishes.py                           # 1000 dishes across 25 cuisines
│   └── cocomo/                             # Experiments 2 & 4
│       ├── config.py                       # Shared constants and hyperparameters
│       ├── receptor.py                     # Input schema + cuisine classification
│       ├── unconsciousness.py              # Fast draft + MFQ risk scheduler
│       ├── consciousness.py                # CRIT validation + substitution search
│       ├── effector.py                     # Output + feedback loop
│       ├── reward.py                       # Multi-component GRPO reward
│       ├── train_rl.py                     # GRPO training loop
│       ├── evaluate.py                     # 100-dish evaluation
│       ├── planner.py                      # Novel: pre-draft constraint prediction
│       ├── verifier.py                     # Novel: post-generation repair loop
│       ├── memory.py                       # Novel: episodic substitution memory
│       ├── pipeline.py                     # Base CoCoMo orchestration
│       ├── pipeline_variants.py            # Ablation variants (Planner/Verifier/Memory)
│       └── run_ablations.py                # Parallel ablation runner
├── introspective_cocomo/                   # Experiment 3
│   ├── introspection_head.py              # Linear probe architecture + trainer
│   ├── train_with_introspection.py        # Joint GRPO + introspection training
│   ├── evaluate_introspection.py          # Three-level evaluation
│   ├── pipeline.py                        # Introspective pipeline variant
│   ├── preference_memory.py               # Text-based preference memory
│   ├── train_with_distillation.py         # Memory-only training (no probe)
│   └── modified_cocomo/                   # Trained weights (GRPO + introspection)
├── codegen/                               # Alternate pipeline implementations
│   └── ablation_results/                  # Novel CoCoMo ablation outputs
├── midterm/                               # Earlier RAG-based culinary agent (midterm)
├── eval_self_awareness_*.py               # Self-awareness evaluation scripts
├── visualize*.py                          # Figure generation scripts
└── self_awareness_figures/                # Generated evaluation figures
```

---

## Setup

```bash
pip install transformers peft trl bitsandbytes datasets matplotlib numpy accelerate huggingface_hub
huggingface-cli login  # Required for Llama 3.1-8B access
```

**Hardware:** 4-bit quantization enables running on a single GPU with ~5 GB VRAM per process. The parallel ablation runner scales to 4 workers on A100/RTX 3090 class GPUs.

---

## Progression of Findings

| Experiment | Behavior | Awareness | Key insight |
|---|---|---|---|
| Baseline | 0% avoidance | N/A | Model uses banned ingredients freely |
| LoRA SFT | ~86% avoidance | ~20% verbal awareness | Behavior without knowledge |
| CoCoMo GRPO | Variable (pipeline-dependent) | ~23% direct ID | Awareness and behavior are dissociable |
| CoCoMo Introspection | — | Internal > Verbal | Knowledge exists internally but isn't verbally accessible |
| Novel CoCoMo | Highest compliance | — | Architectural scaffolding can close the gap that training alone cannot |
