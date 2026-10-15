"""Few-shot personalization: a user-enrolled sound as a prototype over YAMNet window vectors.

The same 1 s window vectors the 8-class head sees (windowing.device_window_vectors) are used, so
the phone needs no extra model. Enrolment: from each sample keep its loudest windows (the enrolled
sound, not the silence around it), L2-normalise them and average -> one unit-length prototype.
Matching: cosine similarity between each live window and the prototype; a window at or above the
threshold is a hit. Only the prototype (1,024 floats) is ever stored; enrolment audio is discarded.
"""
from __future__ import annotations

import numpy as np

from src.preprocessing.windowing import device_window_vectors


def background_center(train_embeddings, train_labels) -> np.ndarray:
    """Mean of the training background window vectors (rows with no positive label)."""
    background = np.asarray(train_labels).sum(axis=1) == 0
    return np.asarray(train_embeddings)[background].mean(axis=0).astype(np.float32)

ENROL_WINDOWS_PER_SAMPLE = 2  # loudest windows kept from each enrolment sample

# Mean window vector of training background audio, subtracted before cosine similarity so that
# only what is distinctive about a sound counts (all YAMNet embeddings share a large common
# component). Set by set_center(); None = raw cosine. Shipped to the app as one 1,024-float file.
_CENTER: np.ndarray | None = None


def set_center(center: np.ndarray | None) -> None:
    global _CENTER
    _CENTER = None if center is None else np.asarray(center, np.float32)


def _centered(x: np.ndarray) -> np.ndarray:
    return x - _CENTER if _CENTER is not None else x


def _unit(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, 1e-9)


def sample_vectors(hops: dict[str, np.ndarray], top_k: int = ENROL_WINDOWS_PER_SAMPLE) -> np.ndarray:
    """Unit window vectors of the loudest ``top_k`` windows of one enrolment sample."""
    win = device_window_vectors(hops, pools=(2,))
    if len(win["vectors"]) == 0:
        return np.zeros((0, win["vectors"].shape[1] if win["vectors"].ndim == 2 else 1024), np.float32)
    keep = np.argsort(-win["energy"], kind="stable")[:top_k]
    return _unit(_centered(win["vectors"][np.sort(keep)]))


def build_prototype(samples: list[dict[str, np.ndarray]], top_k: int = ENROL_WINDOWS_PER_SAMPLE) -> np.ndarray:
    """Unit-length prototype from enrolment samples (each a hop dict from features.extract_hop_embeddings)."""
    vecs = [sample_vectors(h, top_k) for h in samples]
    vecs = [v for v in vecs if len(v)]
    if not vecs:
        raise ValueError("no usable enrolment windows (all samples silent?)")
    return _unit(np.concatenate(vecs).mean(axis=0)).astype(np.float32)


def window_similarities(hops: dict[str, np.ndarray], prototype: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(cosine similarity per non-silent window, hop index of each window) for one clip or stream."""
    win = device_window_vectors(hops, pools=(2,))
    if len(win["vectors"]) == 0:
        return np.zeros(0, np.float32), np.zeros(0, np.int32)
    return _unit(_centered(win["vectors"])) @ prototype, win["hop_index"]
