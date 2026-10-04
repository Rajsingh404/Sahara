"""Python port of the app's alert policy (app/.../core/DetectionPolicy.kt).

A class fires when its score clears its threshold in ``hits_required`` of the last
``window_count`` windows and it has not fired within ``cooldown_ms``. Safety-critical classes
need one hit. Thresholds scale with sensitivity (-1..1) by up to 40 %, clamped to [0.05, 0.95].
Keep this file and DetectionPolicy.kt in step: tests/test_policy.py and the Kotlin
DetectionPolicyFixtureTest both check app/android/app/src/test/resources/policy_fixture.csv.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from src.config import SAFETY_CRITICAL_CLASSES

DEFAULT_THRESHOLD = 0.5
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
    ):
        self.labels = list(labels)
        self.base = np.array([base_thresholds.get(lbl, DEFAULT_THRESHOLD) for lbl in labels], dtype=np.float32)
        self.safety = set(safety_critical)
        self.hits_required = hits_required
        self.window_count = window_count
        self.cooldown_ms = cooldown_ms
        self.sensitivity = float(np.clip(sensitivity, -1.0, 1.0))
        self.enabled = set(labels)
        self.history = [deque(maxlen=window_count) for _ in labels]
        self.last_fired = [-(2**62)] * len(labels)

    def threshold_for(self, i: int) -> float:
        # Float32 arithmetic, as in Kotlin.
        return float(np.clip(np.float32(self.base[i]) * np.float32(1.0 - 0.4 * self.sensitivity), 0.05, 0.95))

    def update(self, scores, now_ms: int) -> list[Detection]:
        if len(scores) != len(self.labels):
            raise ValueError(f"expected {len(self.labels)} scores, got {len(scores)}")
        fired = []
        for i, label in enumerate(self.labels):
            hit = float(scores[i]) >= self.threshold_for(i)
            self.history[i].append(hit)
            if not hit or label not in self.enabled:
                continue
            needed = 1 if label in self.safety else self.hits_required
            if sum(self.history[i]) < needed:
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
