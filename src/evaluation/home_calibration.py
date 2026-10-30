"""Home calibration: lower a class's alert threshold so that sounds recorded in the user's own home fire.

The user records a sound a few times where it happens (rings their doorbell, knocks on their door). For each
recording, ``fire_level`` is the highest threshold at which the deployed rule would still alert on it. The
calibrated threshold sits a margin below the lowest of these, but never below the class's threshold in the most
permissive validated sensitivity step (alert_profiles.csv, 16 false alarms/h) and never above the current one, so
calibration can only move a class within a range whose false-alarm cost was measured. An optional recording of
ordinary home sound raises it again if that sound would fire.

Mirrors app/.../core/HomeCalibration.kt; scripts/eval_home_calibration.py measures it on recording sessions the
model never heard.
"""
from __future__ import annotations

import numpy as np

from src.evaluation.policy_tuning import WINDOW_COUNT

# How far below the weakest calibration recording the threshold goes, in log-odds.
CALIBRATION_MARGIN_LOGIT = 0.5
CALIBRATION_RECORDINGS = 3


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return float(np.log(p / (1 - p)))


def _sigmoid(x: float) -> float:
    return float(1 / (1 + np.exp(-x)))


def fire_level(scores: np.ndarray, hits: int) -> float:
    """Highest threshold at which a hop score sequence raises an alert (current hop a hit, ``hits`` of the last 3)."""
    if len(scores) == 0:
        return 0.0
    padded = np.concatenate([np.zeros(WINDOW_COUNT - 1, np.float32), scores.astype(np.float32)])
    windows = np.lib.stride_tricks.sliding_window_view(padded, WINDOW_COUNT)
    kth = np.sort(windows, axis=1)[:, -hits]
    return float(np.max(np.minimum(scores, kth)))


def calibrate(current: float, floor: float, levels: list[float], background_level: float | None = None,
              margin: float = CALIBRATION_MARGIN_LOGIT) -> float:
    """New threshold for one class from the fire levels of its calibration recordings."""
    if not levels:
        return current
    t = _sigmoid(_logit(min(levels)) - margin)
    t = max(t, floor)
    if background_level is not None and t <= background_level:
        t = _sigmoid(_logit(background_level) + 0.1)
    return min(t, current)
