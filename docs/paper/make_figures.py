"""Figures for the SAHARA paper. Every number is copied from the repo (docs/dataset_card.md,
data/metadata/class_thresholds.json) or computed analytically; nothing is simulated."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from math import comb
from pathlib import Path

OUT = Path(__file__).parent / "figures"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6})
INK, MUTED, BLUE, ORANGE = "#222222", "#777777", "#2a6fb0", "#c8641e"

# Fig. 2: per-class test AP of the frozen-YAMNet baseline (docs/dataset_card.md, 698 test clips).
classes = ["smoke\nalarm", "door-\nbell", "siren", "knock-\ning", "dog\nbark", "baby\ncry", "glass\nbreak", "appl.\nbeep"]
ap = [0.090, 0.632, 0.250, 0.751, 0.956, 0.669, 0.937, 0.923]
low = {0, 5}  # AudioSet-only classes (3 test clips each)
fig, ax = plt.subplots(figsize=(3.45, 1.9))
x = np.arange(len(ap))
bars = ax.bar(x, ap, width=0.62, color=[ORANGE if i in low else BLUE for i in x],
              hatch=None, edgecolor="white", linewidth=0.6)
for i in low:
    bars[i].set_hatch("////")
for i, v in enumerate(ap):
    ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color=INK)
ax.axhline(0.651, color=MUTED, lw=0.8, ls="--")
ax.text(2.0, 0.635, "mAP 0.651", ha="center", va="top", fontsize=6.5, color=MUTED)
ax.set_xticks(x, classes, fontsize=6.5)
ax.set_ylim(0, 1.08)
ax.set_ylabel("Average precision")
ax.grid(axis="y", lw=0.3, color="#dddddd"); ax.set_axisbelow(True)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=BLUE, label="FSD50K + AudioSet"),
                   Patch(facecolor=ORANGE, hatch="////", edgecolor="white", label="AudioSet only (n=20)")],
          fontsize=6, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.13), ncol=2)
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "baseline_ap.pdf"); fig.savefig(OUT / "baseline_ap.png", dpi=300)

# Fig. 3: analytic k-of-n window-level false-alarm probability (independence assumption).
p = np.logspace(-3, -0.3, 200)
def kofn(p, k, n):
    # DetectionPolicy fires only when the CURRENT window is a hit and at least k hits are in the
    # last n windows: P = p * P(>= k-1 hits among the previous n-1 windows).
    return p * sum(comb(n - 1, j) * p**j * (1 - p)**(n - 1 - j) for j in range(k - 1, n))
fig, ax = plt.subplots(figsize=(3.45, 1.9))
ax.loglog(p, kofn(p, 1, 3), color=ORANGE, lw=1.6, label="1-of-3 (safety-critical)")
ax.loglog(p, kofn(p, 2, 3), color=BLUE, lw=1.6, label="2-of-3 (other classes)")
ax.loglog(p, kofn(p, 3, 3), color=MUTED, lw=1.2, ls="--", label="3-of-3 (reference)")
ax.set_xlabel("Per-window false-positive probability $p$")
ax.set_ylabel("Per-decision false alarm")
ax.grid(which="major", lw=0.3, color="#dddddd")
ax.legend(fontsize=6, frameon=False, loc="lower right")
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "kofn.pdf"); fig.savefig(OUT / "kofn.png", dpi=300)
print("ok")
