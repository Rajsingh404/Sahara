"""The Python alert policy must make the same decisions as DetectionPolicy.kt.

Both this test and app/android/app/src/test/kotlin/com/sahara/app/core/DetectionPolicyFixtureTest.kt
replay app/android/app/src/test/resources/policy_fixture.csv and compare with its `expected` column.
"""
from pathlib import Path

import numpy as np

from src.config import SOUND_CLASSES
from src.evaluation.policy import DetectionPolicy

FIXTURE = Path(__file__).resolve().parents[1] / "app/android/app/src/test/resources/policy_fixture.csv"


def _read_fixture():
    thresholds, hits, rows = None, {}, []
    for line in FIXTURE.read_text().splitlines():
        if line.startswith("#thresholds,"):
            thresholds = dict(zip(SOUND_CLASSES, map(float, line.split(",")[1:])))
        elif line.startswith("#hits,"):
            hits = dict(zip(SOUND_CLASSES, map(int, line.split(",")[1:])))
        elif line and not line.startswith("#") and not line.startswith("t_ms"):
            cells = line.split(",")
            rows.append((int(cells[0]), float(cells[1]), np.array(cells[2:10], dtype=np.float32),
                         [c for c in cells[10].split("|") if c]))
    return thresholds, hits, rows


def test_python_policy_matches_fixture():
    thresholds, hits, rows = _read_fixture()
    policy = DetectionPolicy(SOUND_CLASSES, thresholds, hits_per_class=hits)
    fired_total = 0
    for t_ms, sensitivity, scores, expected in rows:
        policy.sensitivity = sensitivity
        got = [d.label for d in policy.update(scores, t_ms)]
        assert got == expected, f"t={t_ms}"
        fired_total += len(got)
    assert fired_total > 10  # the fixture really exercises firing


def test_cooldown_and_two_of_three():
    policy = DetectionPolicy(SOUND_CLASSES, {c: 0.5 for c in SOUND_CLASSES})
    door = np.zeros(8, np.float32)
    door[SOUND_CLASSES.index("doorbell")] = 0.9
    silent = np.zeros(8, np.float32)
    assert policy.update(door, 0) == []
    assert policy.update(silent, 500) == []
    assert [d.label for d in policy.update(door, 1000)] == ["doorbell"]
    assert policy.update(door, 1500) == []  # cooldown
    assert [d.label for d in policy.update(door, 11_000)] == ["doorbell"]
