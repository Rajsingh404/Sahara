"""Calibrate per-class alert thresholds on the held-out validation split."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import (
    METADATA_DIR, PROCESSED_DATA_DIR, SAFETY_CRITICAL_CLASSES,
    SAFETY_PRECISION_FLOOR, SOUND_CLASSES, THRESHOLD_GRID_END,
    THRESHOLD_GRID_START, THRESHOLD_GRID_STEP,
)
from src.training.losses import weighted_binary_crossentropy


def _model_path() -> Path:
    path = Path("models/checkpoints/best_model.keras")
    return path if path.exists() else Path("models/checkpoints/best_classifier.keras")


def _grid() -> np.ndarray:
    return np.round(np.arange(THRESHOLD_GRID_START, THRESHOLD_GRID_END + 0.001, THRESHOLD_GRID_STEP), 2)


def _scores(y: np.ndarray, p: np.ndarray, threshold: float) -> tuple[float, float, float]:
    binary = (p >= threshold).astype(int)
    return (
        float(precision_score(y, binary, zero_division=0)),
        float(recall_score(y, binary, zero_division=0)),
        float(f1_score(y, binary, zero_division=0)),
    )


def run() -> dict[str, float]:
    data = np.load(PROCESSED_DATA_DIR / "val_embeddings.npy")
    labels = np.load(PROCESSED_DATA_DIR / "val_labels.npy")
    model = tf.keras.models.load_model(_model_path(), custom_objects={"weighted_binary_crossentropy": weighted_binary_crossentropy})
    predictions = model.predict(data, verbose=0)
    results: dict[str, dict] = {}
    for index, name in enumerate(SOUND_CLASSES):
        default = _scores(labels[:, index], predictions[:, index], 0.5)
        candidates = [(threshold, *_scores(labels[:, index], predictions[:, index], float(threshold))) for threshold in _grid()]
        if name in SAFETY_CRITICAL_CLASSES:
            eligible = [candidate for candidate in candidates if candidate[1] >= SAFETY_PRECISION_FLOOR]
            # If none meet the safety floor, preserve a conservative threshold
            # by maximizing precision then recall instead of claiming a fit.
            chosen = max(eligible or candidates, key=lambda item: (item[2], item[1]) if eligible else (item[1], item[2]))
        else:
            chosen = max(candidates, key=lambda item: item[3])
        results[name] = {
            "threshold": float(chosen[0]), "precision": float(chosen[1]),
            "recall": float(chosen[2]), "f1": float(chosen[3]),
            "default_precision": default[0], "default_recall": default[1], "default_f1": default[2],
        }
    output = METADATA_DIR / "class_thresholds.json"
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("class\tdefault\tcalibrated\tprecision\trecall\tf1")
    for name, item in results.items():
        print(f"{name}\t0.50\t{item['threshold']:.2f}\t{item['precision']:.3f}\t{item['recall']:.3f}\t{item['f1']:.3f}")
    return {name: item["threshold"] for name, item in results.items()}


if __name__ == "__main__":
    run()
