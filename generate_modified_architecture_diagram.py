"""
Generate architecture diagram for the Modified CoCoMo pipeline
(Planner + Verifier + Episodic Memory).

Shows the base pipeline flow with the three additions highlighted.

Usage:
    python generate_modified_architecture_diagram.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis("off")

# Colors
COLOR_BASE = "#3498db"
COLOR_PLANNER = "#e67e22"
COLOR_VERIFIER = "#e74c3c"
COLOR_MEMORY = "#2ecc71"
COLOR_INPUT = "#ecf0f1"
COLOR_ARROW = "#2c3e50"
COLOR_CONSCIOUS = "#9b59b6"


def draw_box(ax, x, y, w, h, color, label, sublabel=None, fontsize=10, border_style="-"):
    ls = border_style
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                         facecolor=color, edgecolor="#2c3e50", linewidth=1.8,
                         linestyle=ls)
    ax.add_patch(box)
    if sublabel:
        ax.text(x + w/2, y + h/2 + 0.18, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color="#2c3e50")
        ax.text(x + w/2, y + h/2 - 0.22, sublabel, ha="center", va="center",
                fontsize=8, color="#444444", style="italic")
    else:
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color="#2c3e50")


def draw_arrow(ax, x1, y1, x2, y2, color=COLOR_ARROW, lw=1.5, style="-|>"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))


def draw_dashed_arrow(ax, x1, y1, x2, y2, color=COLOR_ARROW, lw=1.2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                               linestyle="dashed"))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE (center row, y=4.5)
# ══════════════════════════════════════════════════════════════════════════════

# Input
draw_box(ax, 0.3, 4.2, 1.6, 1.0, COLOR_INPUT, "Task Input", '"Jerk Chicken"')

# Receptor
draw_box(ax, 2.5, 4.2, 1.8, 1.0, COLOR_BASE, "Receptor", "cuisine + risk")

# Unconscious / Drafter
draw_box(ax, 5.8, 4.2, 2.0, 1.0, COLOR_BASE, "Unconscious", "MFQ + draft")

# Consciousness
draw_box(ax, 5.8, 1.8, 2.0, 1.2, COLOR_CONSCIOUS, "Consciousness", "CRIT + explore")

# Effector
draw_box(ax, 11.5, 4.2, 1.8, 1.0, COLOR_BASE, "Effector", "output + feedback")

# ══════════════════════════════════════════════════════════════════════════════
# MODIFICATION 1: PLANNER (above, orange)
# ══════════════════════════════════════════════════════════════════════════════

draw_box(ax, 2.5, 7.2, 2.2, 1.2, COLOR_PLANNER, "Planner ★", "predict constraints")

# Planner prediction output
draw_box(ax, 5.3, 7.4, 2.8, 0.8, COLOR_INPUT, "Risk Override",
         "r̂ = 1.0 if any predicted")

# Arrow: Receptor → Planner
draw_arrow(ax, 3.4, 5.2, 3.4, 7.2, color=COLOR_PLANNER, lw=2.0)

# Arrow: Planner → Risk Override
draw_arrow(ax, 4.7, 7.8, 5.3, 7.8, color=COLOR_PLANNER, lw=1.5)

# Arrow: Risk Override → Unconscious (overrides escalation)
draw_arrow(ax, 6.8, 7.4, 6.8, 5.2, color=COLOR_PLANNER, lw=2.0)
ax.text(7.0, 6.3, "escalation\ndecision", fontsize=7, color=COLOR_PLANNER, style="italic")

# ══════════════════════════════════════════════════════════════════════════════
# MODIFICATION 2: VERIFIER (between Consciousness and Effector, red)
# ══════════════════════════════════════════════════════════════════════════════

draw_box(ax, 9.0, 1.8, 2.0, 1.2, COLOR_VERIFIER, "Verifier ★", "detect + repair")

# Arrow: Consciousness → Verifier
draw_arrow(ax, 7.8, 2.4, 9.0, 2.4, color=COLOR_VERIFIER, lw=2.0)

# Arrow: Verifier → Effector (clean output)
draw_arrow(ax, 10.4, 2.9, 12.4, 4.2, color=COLOR_VERIFIER, lw=1.5)

# Dashed arrow: Verifier → Consciousness (repair loop)
draw_dashed_arrow(ax, 9.5, 1.8, 7.3, 1.5, color=COLOR_VERIFIER, lw=1.8)
ax.text(8.0, 1.2, "repair loop\n(up to k=2)", fontsize=7, color=COLOR_VERIFIER, style="italic")

# ══════════════════════════════════════════════════════════════════════════════
# MODIFICATION 3: EPISODIC MEMORY (below right, green)
# ══════════════════════════════════════════════════════════════════════════════

draw_box(ax, 9.5, 6.8, 2.5, 1.2, COLOR_MEMORY, "Episodic Memory ★",
         "(cuisine, ingredient) → sub")

# Arrow: Memory → Receptor/Schema (inject proven subs)
draw_dashed_arrow(ax, 9.5, 7.4, 4.3, 5.2, color=COLOR_MEMORY, lw=1.8)
ax.text(6.5, 6.6, "inject proven\nsubstitutions", fontsize=7, color=COLOR_MEMORY, style="italic")

# Arrow: Effector → Memory (update from result)
draw_dashed_arrow(ax, 12.8, 5.2, 11.5, 6.8, color=COLOR_MEMORY, lw=1.5)
ax.text(12.3, 6.1, "update", fontsize=7, color=COLOR_MEMORY, style="italic")

# ══════════════════════════════════════════════════════════════════════════════
# BASE PIPELINE ARROWS
# ══════════════════════════════════════════════════════════════════════════════

# Input → Receptor
draw_arrow(ax, 1.9, 4.7, 2.5, 4.7)

# Receptor → Unconscious
draw_arrow(ax, 4.3, 4.7, 5.8, 4.7)

# Unconscious → Effector (fast path, no escalation)
draw_arrow(ax, 7.8, 4.7, 11.5, 4.7)
ax.text(9.5, 4.9, "fast path (no escalation)", fontsize=7, color="#7f8c8d", style="italic")

# Unconscious → Consciousness (escalation)
draw_arrow(ax, 6.8, 4.2, 6.8, 3.0, color=COLOR_CONSCIOUS, lw=1.5)
ax.text(5.4, 3.5, "risk > 0.3", fontsize=7, color=COLOR_CONSCIOUS, style="italic")

# MFQ feedback: Effector → Unconscious
draw_dashed_arrow(ax, 11.5, 4.5, 7.8, 4.5, color="#7f8c8d", lw=1.0)
ax.text(9.5, 4.3, "MFQ feedback", fontsize=6, color="#7f8c8d", style="italic")

# ══════════════════════════════════════════════════════════════════════════════
# LEGEND
# ══════════════════════════════════════════════════════════════════════════════

legend_items = [
    mpatches.Patch(color=COLOR_BASE, label="Base CoCoMo modules"),
    mpatches.Patch(color=COLOR_CONSCIOUS, label="Consciousness (deliberate path)"),
    mpatches.Patch(color=COLOR_PLANNER, label="★ Mod 1: Planner (pre-generation risk)"),
    mpatches.Patch(color=COLOR_VERIFIER, label="★ Mod 2: Verifier (post-generation repair)"),
    mpatches.Patch(color=COLOR_MEMORY, label="★ Mod 3: Episodic Memory (cross-task learning)"),
]
ax.legend(handles=legend_items, loc="lower left", fontsize=9, framealpha=0.95,
          bbox_to_anchor=(0.02, 0.02))

# Title
ax.text(7, 9.5, "Modified CoCoMo Architecture", ha="center", va="center",
        fontsize=14, fontweight="bold", color="#2c3e50")
ax.text(7, 9.0, "Orange boxes (★) are the three architectural additions",
        ha="center", va="center", fontsize=10, color="#555555", style="italic")

plt.tight_layout()
out_path = "modified_cocomo_architecture.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Modified architecture diagram saved to {out_path}")
