"""Figures for the SAHARA paper. Every number is read from the selected run folder
(models/logs/<run>/test_metrics.json, policy_tuning.json) or computed analytically; nothing is typed in.

    python docs/paper/make_figures.py
"""
from math import comb
from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "models" / "logs" / (ROOT / "models" / "logs" / "selected_run.txt").read_text().strip().split("/")[-1]
OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6})
# Greyscale for print (Springer converts colour to greyscale); hatching separates the groups.
INK, MUTED, DARK, LIGHT = "#111111", "#666666", "#4d4d4d", "#bdbdbd"
CLASSES = ["smoke_alarm", "doorbell", "siren", "knocking", "dog_bark", "baby_cry", "glass_break", "appliance_beep"]
TICKS = ["smoke\nalarm", "door-\nbell", "siren", "knock-\ning", "dog\nbark", "baby\ncry", "glass\nbreak", "appl.\nbeep"]
SAFETY = {0, 2, 6}


def save(fig, name):
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)


# Fig. 2: per-class test AP and share of test clips that raise the right alert (public test set).
pub = json.loads((RUN / "test_metrics.json").read_text())["sets"]["public"]
ap = [pub["per_class"][c]["ap"] for c in CLASSES]
det = [pub["clip_detection_with_app_policy"][c]["recall"] for c in CLASSES]
fig, ax = plt.subplots(figsize=(4.6, 2.4))
x = np.arange(len(CLASSES))
ax.bar(x - 0.19, ap, width=0.36, color=DARK, edgecolor="white", linewidth=0.5)
ax.bar(x + 0.19, det, width=0.36, color=LIGHT, edgecolor="#333333", linewidth=0.5, hatch="////")
for i in x:
    ax.text(i - 0.19, ap[i] + 0.02, f"{ap[i]:.2f}", ha="center", va="bottom", fontsize=5.5, color=INK)
    ax.text(i + 0.19, det[i] + 0.02, f"{det[i]:.2f}", ha="center", va="bottom", fontsize=5.5, color=INK)
ax.axhline(pub["mAP"], color=MUTED, lw=0.8, ls="--")
ax.text(7.45, pub["mAP"] + 0.015, f"mAP {pub['mAP']:.3f}", ha="right", va="bottom", fontsize=6, color=MUTED)
ax.set_xticks(x, [("$^\\dagger$" if i in SAFETY else "") + t for i, t in enumerate(TICKS)], fontsize=6.5)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Test score")
ax.grid(axis="y", lw=0.3, color="#dddddd")
ax.set_axisbelow(True)
ax.legend(handles=[Patch(facecolor=DARK, label="Average precision (windows)"),
                   Patch(facecolor=LIGHT, hatch="////", edgecolor="#333333", label="Test clips alerted (app policy)")],
          fontsize=6, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.14), ncol=2)
save(fig, "test_ap")

# Fig. 3: detection against the false-alarm budget on validation (how the 4/h budget was chosen).
tr = json.loads((RUN / "policy_tuning.json").read_text())["tradeoff"]
fa = [r["total_fa_per_hour"] for r in tr]
md = [r["mean_detection"] for r in tr]
fig, ax = plt.subplots(figsize=(4.6, 2.2))
ax.semilogx([max(f, 0.1) for f in fa], md, color=INK, lw=1.4, marker="o", ms=3)
for r, f, m in zip(tr, fa, md):
    if r["budget_per_hour"] == 4:
        ax.annotate(f"chosen: {f:.1f}/h, {m:.2f}", (f, m), xytext=(-70, 8), textcoords="offset points", fontsize=6,
                    arrowprops={"arrowstyle": "-", "lw": 0.5, "color": MUTED})
ax.set_xlabel("False alarms per hour on validation background (log)")
ax.set_ylabel("Mean share of\nclips alerted")
ax.grid(which="major", lw=0.3, color="#dddddd")
save(fig, "budget")


# Fig. 4: analytic k-of-3 false-alarm probability (independence assumption).
def kofn(p, k, n=3):
    # DetectionPolicy fires only when the CURRENT window is a hit and at least k hits are in the
    # last n windows: P = p * P(>= k-1 hits among the previous n-1 windows).
    return p * sum(comb(n - 1, j) * p**j * (1 - p) ** (n - 1 - j) for j in range(k - 1, n))


p = np.logspace(-3, -0.3, 200)
fig, ax = plt.subplots(figsize=(4.6, 2.2))
ax.loglog(p, kofn(p, 1), color=INK, lw=1.6, label="1 of 3")
ax.loglog(p, kofn(p, 2), color=DARK, lw=1.6, ls="-.", label="2 of 3")
ax.loglog(p, kofn(p, 3), color=MUTED, lw=1.2, ls="--", label="3 of 3")
ax.set_xlabel("Per-window false-positive probability $p$")
ax.set_ylabel("Per-decision false alarm")
ax.grid(which="major", lw=0.3, color="#dddddd")
ax.legend(fontsize=6, frameon=False, loc="lower right")
save(fig, "kofn")
print("ok")
