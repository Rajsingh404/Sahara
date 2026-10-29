"""Tune per-class alert thresholds and hits-required against a false-alarm budget (validation only).

Window-level thresholds ignore how the app actually alerts. Here every candidate setting is
scored through the app policy (src/evaluation/policy.py, one class at a time: the policy treats
classes independently):

  detection rate  share of validation clips of the class that raise its alert at least once
  false alarms/h  alerts of the class on validation audio that does not contain it
                  (background clips + other classes' clips, played back to back)

Search space per class: threshold in POLICY_THRESHOLD_GRID (0.05-0.95 by 0.01, then 0.96-0.999,
since a confident model needs thresholds above 0.95) x hits_required in {1, 2, 3} (of 3 windows).
Detection mixes the two kinds of validation audio: when a class has Indian team clips in validation, its detection is
INDIAN_SHARE x (Indian clips alerted) + (1 - INDIAN_SHARE) x (public clips alerted), so a handful of Indian clips is not
outvoted by a hundred public ones. False alarms pool all validation audio.

Objective: maximise sum_c w_c * detection_c subject to sum_c false_alarms_c <= budget,
with w_c = POSITIVE_CLASS_WEIGHTS (safety classes weigh 3). Solved by a Lagrangian sweep.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np

from src.config import (
    METADATA_DIR,
    POSITIVE_CLASS_WEIGHTS,
    SOUND_CLASSES,
    THRESHOLD_GRID_END,
    THRESHOLD_GRID_START,
    THRESHOLD_GRID_STEP,
)
from src.evaluation.policy import COOLDOWN_MS, WINDOW_COUNT
from src.evaluation.streaming import HOP_MS, hop_scores
from src.preprocessing.features import hop_cache_path, load_hop_cache

HITS_OPTIONS = (1, 2, 3)
INDIAN_SHARE = 0.5
COOLDOWN_HOPS = COOLDOWN_MS // HOP_MS


def count_alerts(scores: np.ndarray, threshold: float, hits: int) -> int:
    """Alerts the policy raises for one class over a hop score sequence (one stream, fresh state)."""
    hit = scores >= threshold
    if not hit.any():
        return 0
    # Hits in the last WINDOW_COUNT hops, including this one.
    padded = np.concatenate([np.zeros(WINDOW_COUNT - 1, bool), hit])
    recent = np.lib.stride_tricks.sliding_window_view(padded, WINDOW_COUNT).sum(axis=1)
    candidates = np.flatnonzero(hit & (recent >= hits))
    alerts, last = 0, -(10**9)
    for i in candidates:
        if i - last >= COOLDOWN_HOPS:
            alerts += 1
            last = i
    return alerts


@dataclass
class ClipScores:
    filepath: str
    labels: set[str]
    scores: np.ndarray  # [n_hops, 8], zeros on silent hops
    indian: bool = False  # team recording (source_type self_recorded)


def load_split_clip_scores(model, split: str = "val") -> list[ClipScores]:
    with (METADATA_DIR / f"{split}_manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    clips: dict[str, set[str]] = {}
    indian = set()
    for r in rows:
        if r["source_type"] == "synthetic_mixed":
            continue
        if r["source_type"] == "self_recorded":
            indian.add(r["filepath"])
        clips.setdefault(r["filepath"], set())
        if r["label"] in SOUND_CLASSES:
            clips[r["filepath"]].add(r["label"])
    out = []
    for fp, labels in sorted(clips.items()):
        path = hop_cache_path(fp)
        if path.exists():
            out.append(ClipScores(fp, labels, hop_scores(model, load_hop_cache(path)), fp in indian))
    return out


def class_curves(clips: list[ClipScores], grid: np.ndarray) -> dict[str, dict]:
    """Per class: detection rate and false alarms/hour for every (threshold, hits) setting."""
    curves = {}
    for c_idx, c in enumerate(SOUND_CLASSES):
        pos = [cl.scores[:, c_idx] for cl in clips if c in cl.labels]
        n_indian = sum(1 for cl in clips if c in cl.labels and cl.indian)
        share = INDIAN_SHARE if 0 < n_indian < len(pos) else (1.0 if n_indian else 0.0)
        is_indian = np.array([cl.indian for cl in clips if c in cl.labels])
        # Negative audio for this class, played back to back as one stream.
        neg = np.concatenate([cl.scores[:, c_idx] for cl in clips if c not in cl.labels] or [np.zeros(0)])
        hours = len(neg) * HOP_MS / 3_600_000
        det = np.zeros((len(HITS_OPTIONS), len(grid)))
        fa = np.zeros_like(det)
        for h_i, hits in enumerate(HITS_OPTIONS):
            for t_i, t in enumerate(grid):
                if pos:
                    hit = np.array([count_alerts(s, t, hits) > 0 for s in pos])
                    det[h_i, t_i] = (share * hit[is_indian].mean() if share else 0.0) + (
                        (1 - share) * hit[~is_indian].mean() if share < 1 else 0.0)
                fa[h_i, t_i] = count_alerts(neg, t, hits) / hours if hours else 0.0
        curves[c] = {"detection": det, "fa_per_hour": fa, "clips": len(pos), "indian_clips": n_indian, "negative_hours": hours}
    return curves


def choose_settings(curves: dict[str, dict], grid: np.ndarray, budget_per_hour: float) -> dict[str, dict]:
    """Lagrangian sweep: per class maximise w*det - lam*fa; smallest lam meeting the budget wins."""
    weights = {c: POSITIVE_CLASS_WEIGHTS.get(c, 1.0) for c in SOUND_CLASSES}

    def pick(lam: float) -> dict[str, tuple[int, int]]:
        best = {}
        for c, cv in curves.items():
            value = weights[c] * cv["detection"] - lam * cv["fa_per_hour"]
            # Ties: prefer fewer false alarms, then the higher threshold (more conservative).
            order = np.lexsort((-np.broadcast_to(grid, value.shape).ravel(), cv["fa_per_hour"].ravel(), -value.ravel()))
            best[c] = np.unravel_index(order[0], value.shape)
        return best

    def total_fa(sel) -> float:
        return float(sum(curves[c]["fa_per_hour"][sel[c]] for c in sel))

    lo, hi = 0.0, 1.0
    while total_fa(pick(hi)) > budget_per_hour and hi < 1e6:
        hi *= 2
    if total_fa(pick(lo)) <= budget_per_hour:
        hi = lo
    for _ in range(60):
        mid = (lo + hi) / 2
        if total_fa(pick(mid)) <= budget_per_hour:
            hi = mid
        else:
            lo = mid
    sel = pick(hi)
    return {
        c: {
            "threshold": float(grid[t_i]),
            "hits_required": int(HITS_OPTIONS[h_i]),
            "val_clip_detection": round(float(curves[c]["detection"][h_i, t_i]), 4),
            "val_false_alarms_per_hour": round(float(curves[c]["fa_per_hour"][h_i, t_i]), 3),
            "val_clips": curves[c]["clips"],
        }
        for c, (h_i, t_i) in sel.items()
    }


HIGH_THRESHOLDS = (0.96, 0.97, 0.98, 0.985, 0.99, 0.993, 0.995, 0.997, 0.998, 0.999)


def grid() -> np.ndarray:
    base = np.round(np.arange(THRESHOLD_GRID_START, THRESHOLD_GRID_END + 1e-9, THRESHOLD_GRID_STEP), 2)
    return np.unique(np.concatenate([base, HIGH_THRESHOLDS]))


def tradeoff(curves: dict[str, dict], budgets=(0.25, 0.5, 1, 2, 4, 8, 16)) -> list[dict]:
    """Weighted detection achievable at each total false-alarm budget (for choosing the budget)."""
    g = grid()
    rows = []
    for b in budgets:
        sel = choose_settings(curves, g, b)
        rows.append({"budget_per_hour": b,
                     "total_fa_per_hour": round(sum(s["val_false_alarms_per_hour"] for s in sel.values()), 3),
                     "mean_detection": round(float(np.mean([s["val_clip_detection"] for s in sel.values()])), 4),
                     "per_class": {c: (s["val_clip_detection"], s["threshold"], s["hits_required"]) for c, s in sel.items()}})
    return rows
