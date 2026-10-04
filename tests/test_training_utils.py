import numpy as np

from src.config import SOUND_CLASSES
from src.evaluation.metrics import safety_floor_recall, window_confusion
from src.training.train import balanced_sample_weights


def _y(rows):
    out = np.zeros((len(rows), len(SOUND_CLASSES)), np.float32)
    for i, label in enumerate(rows):
        if label:
            out[i, SOUND_CLASSES.index(label)] = 1
    return out


def test_rare_class_windows_weigh_more():
    Y = _y(["dog_bark"] * 100 + ["smoke_alarm"] * 4 + [None] * 100)
    w = balanced_sample_weights(Y, power=0.5)
    assert np.isclose(w.mean(), 1.0)
    assert w[100] / w[0] == np.sqrt(100 / 4)  # inverse sqrt frequency
    assert np.allclose(balanced_sample_weights(Y, power=0.0), 1.0)


def test_safety_floor_recall_respects_precision():
    Y = _y(["smoke_alarm"] * 4 + [None] * 4)
    scores = np.zeros_like(Y)
    i = SOUND_CLASSES.index("smoke_alarm")
    scores[:, i] = [0.9, 0.8, 0.3, 0.2, 0.85, 0.1, 0.1, 0.1]
    floor = safety_floor_recall(Y, scores, floor=0.7)["smoke_alarm"]
    # thresholds: 0.8 -> P=2/3 (fails), 0.2 -> 4/5 (meets, recall 1.0)
    assert floor["recall"] == 1.0 and floor["precision"] >= 0.7


def test_window_confusion_counts(tmp_path):
    Y = _y(["doorbell", "doorbell", None])
    scores = np.zeros_like(Y)
    scores[0, SOUND_CLASSES.index("doorbell")] = 0.9
    scores[1, SOUND_CLASSES.index("siren")] = 0.7
    m = window_confusion(Y, scores, tmp_path / "cm")
    d, s, bg = SOUND_CLASSES.index("doorbell"), SOUND_CLASSES.index("siren"), len(SOUND_CLASSES)
    assert m[d, d] == 1 and m[d, s] == 1 and m[bg, bg] == 1
    assert (tmp_path / "cm.csv").exists()
