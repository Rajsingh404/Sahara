"""Replay cached clips through the head and the app's alert policy, the way the phone runs.

Per hop (every 0.5 s): a silent hop (VAD) sends all-zero scores to the policy; any other hop
sends the head's scores for the pooled window vector (windowing.device_window_vectors).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.config import HOP_SAMPLES, METADATA_DIR, MODELS_DIR, SAMPLE_RATE, SOUND_CLASSES
from src.evaluation.policy import Detection, DetectionPolicy
from src.models.classifier_head import head_predict
from src.preprocessing.features import hop_cache_path, load_hop_cache
from src.preprocessing.windowing import device_window_vectors

HOP_MS = 1000 * HOP_SAMPLES // SAMPLE_RATE


def selected_run_dir(run: str | Path | None = None) -> Path:
    """The run named on the command line, else the one scripts/compare_runs.py selected."""
    logs = MODELS_DIR / "logs"
    if run:
        path = Path(run)
        return path if path.is_dir() else logs / str(run)
    return logs / (logs / "selected_run.txt").read_text().strip()


def load_head(run_dir: Path):
    import tensorflow as tf

    from src.models.classifier_head import HEAD_DEVICE

    with tf.device(HEAD_DEVICE):
        return tf.keras.models.load_model(run_dir / "model.keras", compile=False)


def load_thresholds(path: Path = METADATA_DIR / "class_thresholds.json") -> dict[str, float]:
    data = json.loads(path.read_text())
    return {c: float(data[c]["threshold"]) for c in SOUND_CLASSES if c in data}


def load_hits(path: Path = METADATA_DIR / "class_thresholds.json") -> dict[str, int]:
    data = json.loads(path.read_text())
    return {c: int(data[c]["hits_required"]) for c in SOUND_CLASSES if c in data and "hits_required" in data[c]}


def hop_scores(model, hops: dict[str, np.ndarray]) -> np.ndarray:
    """[n_hops, 8] scores as the policy sees them: zeros on silent hops."""
    scores = np.zeros((len(hops["vad"]), len(SOUND_CLASSES)), dtype=np.float32)
    win = device_window_vectors(hops)
    if len(win["vectors"]):
        scores[win["hop_index"]] = head_predict(model, win["vectors"])
    return scores


def clip_hop_scores(model, filepath: str, variant: str = "") -> np.ndarray | None:
    path = hop_cache_path(filepath, variant)
    return hop_scores(model, load_hop_cache(path)) if path.exists() else None


def run_policy(policy: DetectionPolicy, scores: np.ndarray, start_ms: int = 0) -> tuple[list[Detection], int]:
    """Feed hop scores to ``policy`` (state carries over between calls); returns detections and end time."""
    detections: list[Detection] = []
    t = start_ms
    for row in scores:
        detections.extend(policy.update(row, t))
        t += HOP_MS
    return detections, t
