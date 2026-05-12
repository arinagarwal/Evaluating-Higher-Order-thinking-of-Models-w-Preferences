"""
Generate architecture diagram for the Introspective CoCoMo model.

Produces a clean block diagram showing the pipeline with the introspection head.

Usage:
    python generate_architecture_diagram.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(14, 9))
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.axis("off")

# Colors
COLOR_INPUT = "#ecf0f1"
COLOR_RECEPTOR = "#f39c12"
COLOR_ENCODER = "#3498db"
COLOR_INTROSPECTION = "#9b59b6"
COLOR_UNCONSCIOUS = "#2ecc71"
COLOR_CONSCIOUS = "#e74c3c"
COLOR_EFFECTOR = "#1abc9c"
COLOR_OUTPUT = "#ecf0f1"
COLOR_ARROW = "#2c3e50"

def draw_box(ax, x, y, w, h, color, label, sublabel=None, fontsize=10):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                         facecolor=color, edgecolor="#2c3e50", linewidth=1.5)
    ax.add_patch(box)
    if sublabel:
        ax.text(x + w/2, y + h/2 + 0.15, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color="#2c3e50")
        ax.text(x + w/2, y + h/2 - 0.25, sublabel, ha="center", va="center",
                fontsize=8, color="#555555", style="italic")
    else:
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color="#2c3e50")

def draw_arrow(ax, x1, y1, x2, y2, color=COLOR_ARROW, style="-|>", lw=1.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))

def draw_dashed_arrow(ax, x1, y1, x2, y2, color=COLOR_ARROW, lw=1.2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                               linestyle="dashed"))

# ── Main Pipeline (horizontal flow) ──────────────────────────────────────────

# Input
draw_box(ax, 0.3, 3.8, 1.8, 1.0, COLOR_INPUT, "Dish Input", '"Jerk Chicken"')

# Receptor
draw_box(ax, 2.8, 3.8, 1.8, 1.0, COLOR_RECEPTOR, "Receptor", "cuisine + risk")

# Encoder / Hidden State
draw_box(ax, 5.3, 3.8, 2.2, 1.0, COLOR_ENCODER, "LLM Encoder", "prompt → hidden state")

# MFQ / Unconscious
draw_box(ax, 8.2, 3.8, 2.2, 1.0, COLOR_UNCONSCIOUS, "Unconscious", "MFQ + draft")

# Conscious
draw_box(ax, 8.2, 1.8, 2.2, 1.0, COLOR_CONSCIOUS, "Consciousness", "CRIT + explore")

# Effector
draw_box(ax, 11.2, 3.8, 2.0, 1.0, COLOR_EFFECTOR, "Effector", "output + feedback")

# ── Introspection Head (above, reading from encoder) ─────────────────────────

draw_box(ax, 5.0, 6.5, 2.8, 1.2, COLOR_INTROSPECTION, "Introspection Head",
         "Linear(4096, 5) → σ")

# Prediction output box
draw_box(ax, 8.5, 6.7, 2.8, 0.8, COLOR_INPUT, "Avoidance Predictions",
         "garlic: 0.70, butter: 0.56, ...")

# ── Arrows: main pipeline ────────────────────────────────────────────────────

# Input → Receptor
draw_arrow(ax, 2.1, 4.3, 2.8, 4.3)

# Receptor → Encoder
draw_arrow(ax, 4.6, 4.3, 5.3, 4.3)

# Encoder → Unconscious
draw_arrow(ax, 7.5, 4.3, 8.2, 4.3)

# Unconscious → Effector (direct path)
draw_arrow(ax, 10.4, 4.3, 11.2, 4.3)

# Unconscious → Conscious (escalation)
draw_arrow(ax, 9.3, 3.8, 9.3, 2.8, color="#e74c3c")
ax.text(9.55, 3.35, "risk > 0.3", fontsize=7, color="#e74c3c", style="italic")

# Conscious → Effector
draw_arrow(ax, 10.4, 2.3, 11.8, 3.8, color="#e74c3c")

# ── Arrows: introspection head ───────────────────────────────────────────────

# Encoder hidden state → Introspection Head
draw_arrow(ax, 6.4, 4.8, 6.4, 6.5, color=COLOR_INTROSPECTION, lw=2.0)
ax.text(6.6, 5.6, "last-token\nhidden state", fontsize=8, color=COLOR_INTROSPECTION,
        style="italic")

# Introspection Head → Predictions
draw_arrow(ax, 7.8, 7.1, 8.5, 7.1, color=COLOR_INTROSPECTION, lw=1.5)

# ── Feedback loop arrow ──────────────────────────────────────────────────────

# Effector → MFQ feedback (curved below)
draw_dashed_arrow(ax, 12.2, 3.8, 9.3, 3.8, color="#1abc9c")
ax.text(10.7, 3.5, "risk update", fontsize=7, color="#1abc9c", style="italic")

# ── Evaluation comparison arrows (bottom) ────────────────────────────────────

# Dashed box for "Evaluation comparison"
eval_box = FancyBboxPatch((3.5, 0.3), 7.5, 1.0, boxstyle="round,pad=0.1",
                          facecolor="white", edgecolor="#7f8c8d", linewidth=1.0,
                          linestyle="dashed")
ax.add_patch(eval_box)
ax.text(7.25, 0.8, "Evaluation: compare Internal predictions vs Verbal self-report vs Behavioral output",
        ha="center", va="center", fontsize=9, color="#555555")

# ── Legend ────────────────────────────────────────────────────────────────────

legend_items = [
    mpatches.Patch(color=COLOR_RECEPTOR, label="Receptor (classification)"),
    mpatches.Patch(color=COLOR_ENCODER, label="LLM Encoder (shared weights)"),
    mpatches.Patch(color=COLOR_INTROSPECTION, label="Introspection Head (linear probe)"),
    mpatches.Patch(color=COLOR_UNCONSCIOUS, label="Unconsciousness (fast path)"),
    mpatches.Patch(color=COLOR_CONSCIOUS, label="Consciousness (deliberate path)"),
    mpatches.Patch(color=COLOR_EFFECTOR, label="Effector (output + feedback)"),
]
ax.legend(handles=legend_items, loc="upper right", fontsize=8, framealpha=0.9)

# Title
ax.text(7, 8.5, "Introspective CoCoMo Architecture", ha="center", va="center",
        fontsize=14, fontweight="bold", color="#2c3e50")

plt.tight_layout()
out_path = "introspection_architecture.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Architecture diagram saved to {out_path}")
