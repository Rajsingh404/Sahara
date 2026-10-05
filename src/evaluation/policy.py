"""Python port of the app's alert policy (app/.../core/DetectionPolicy.kt).

A class fires when its score clears its threshold in N of the last ``window_count`` windows and
it has not fired within ``cooldown_ms``. N = ``hits_per_class`` (tuned, from class_thresholds.json),
else 1 for safety-critical classes and ``hits_required`` for the rest. Sensitivity (-1..1) shifts
each threshold by up to 2 in log-odds; thresholds are clamped to [0.01, 0.999].
Keep this file and DetectionPolicy.kt in step: tests/test_policy.py and the Kotlin
DetectionPolicyFixtureTest both check app/android/app/src/test/resources/policy_fixture.csv.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from src.config import SAFETY_CRITICAL_CLASSES

DEFAULT_THRESHOLD = 0.5
MIN_THRESHOLD = 0.01
MAX_THRESHOLD = 0.999
SENSITIVITY_LOGIT_RANGE = 2.0
HITS_REQUIRED = 2
WINDOW_COUNT = 3
COOLDOWN_MS = 10_000


@dataclass(frozen=True)
class Detection:
    class_id: int
    label: str
    confidence: float
    timestamp_ms: int
    safety_critical: bool


class DetectionPolicy:
    def __init__(
        self,
        labels: list[str],
        base_thresholds: dict[str, float],
        safety_critical: tuple[str, ...] | set[str] = SAFETY_CRITICAL_CLASSES,
        hits_required: int = HITS_REQUIRED,
        window_count: int = WINDOW_COUNT,
        cooldown_ms: int = COOLDOWN_MS,
        sensitivity: float = 0.0,
        hits_per_class: dict[str, int] | None = None,
    ):
        self.labels = list(labels)
        self.base = np.array([base_thresholds.get(lbl, DEFAULT_THRESHOLD) for lbl in labels], dtype=np.float32)
        self.safety = set(safety_critical)
        self.hits_required = hits_required
        self.window_count = window_count
        self.cooldown_ms = cooldown_ms
        self.sensitivity = float(np.clip(sensitivity, -1.0, 1.0))
        self.hits_per_class = dict(hits_per_class or {})
        self.enabled = set(labels)
        self.history = [deque(maxlen=window_count) for _ in labels]
        self.last_fired = [-(2**62)] * len(labels)

    def threshold_for(self, i: int) -> float:
        # Same steps as Kotlin: double-precision logit shift of the float32 base, then float32.
        t = min(max(float(self.base[i]), 1e-4), 1 - 1e-4)
        logit = np.log(t / (1 - t)) - SENSITIVITY_LOGIT_RANGE * float(np.float32(self.sensitivity))
        return float(np.clip(np.float32(1.0 / (1.0 + np.exp(-logit))), np.float32(MIN_THRESHOLD), np.float32(MAX_THRESHOLD)))

    def hits_for(self, i: int) -> int:
        label = self.labels[i]
        default = 1 if label in self.safety else self.hits_required
        return int(min(max(self.hits_per_class.get(label, default), 1), self.window_count))

    def update(self, scores, now_ms: int) -> list[Detection]:
        if len(scores) != len(self.labels):
            raise ValueError(f"expected {len(self.labels)} scores, got {len(scores)}")
        fired = []
        for i, label in enumerate(self.labels):
            hit = float(scores[i]) >= self.threshold_for(i)
            self.history[i].append(hit)
            if not hit or label not in self.enabled:
                continue
            if sum(self.history[i]) < self.hits_for(i):
                continue
            if now_ms - self.last_fired[i] < self.cooldown_ms:
                continue
            self.last_fired[i] = now_ms
            fired.append(Detection(i, label, float(scores[i]), now_ms, label in self.safety))
        return sorted(fired, key=lambda d: -d.confidence)

    def reset(self) -> None:
        for h in self.history:
            h.clear()
        self.last_fired = [-(2**62)] * len(self.labels)
