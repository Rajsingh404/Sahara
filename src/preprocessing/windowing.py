"""Turn per-hop YAMNet embeddings into the window vectors the phone classifies.

The phone (AudioCaptureService.kt + InferenceEngine.kt + EmbeddingPool.kt) runs every 0.5 s:
quiet hops (VAD) are skipped and do not enter the pool; every other hop is embedded and the
head sees, for each size in ``WINDOW_POOLS`` (= 2, 6), the mean of that many most recent
non-silent hop embeddings, concatenated. ``device_window_vectors`` reproduces that exactly.

Training examples (``clip_examples``):
- background clips: every window, all-zero target;
- target clips: only the windows most likely to hold the event, per class (``select_event_windows``);
  the remaining windows of a target clip are dropped, not used as background, because a
  clip-level label does not say where in the clip the event is.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from src.config import (
    EMBEDDING_POOL,
    WINDOW_POOL_OP,
    WINDOW_POOLS,
    EVENT_MIN_RELATIVE_SCORE,
    EVENT_SCORE_FLOOR,
    EVENT_TOPK_FRACTION,
    EVENT_TOPK_MAX,
    SOUND_CLASSES,
)


def device_window_vectors(
    hops: dict[str, np.ndarray], pool_size: int | None = None, pools: tuple[int, ...] = WINDOW_POOLS,
) -> dict[str, np.ndarray]:
    """Window vectors exactly as the phone forms them.

    Returns ``vectors`` [m, 1024 * len(pools)] (means over the last p non-silent hops, for each
    p in ``pools``; ``pool_size`` = a single pool), ``class_scores`` [m, 8] and ``energy`` [m]
    over the short pool (EMBEDDING_POOL), and ``hop_index`` [m] (the hop at which the phone
    classifies this vector). Only VAD-passing hops produce a window.
    """
    if pool_size is not None:
        pools = (pool_size,)
    emb, scores, energy, vad = hops["embeddings"], hops["class_scores"], hops["energy"], hops["vad"]
    pool: deque[int] = deque(maxlen=max(pools))
    vectors, pooled_scores, pooled_energy, index = [], [], [], []
    for i in range(len(emb)):
        if not vad[i]:
            continue
        pool.append(i)
        members = list(pool)
        combine = np.max if WINDOW_POOL_OP == "max" else np.mean
        vectors.append(np.concatenate([combine(emb[members[-p:]], axis=0) for p in pools]))
        short = members[-min(EMBEDDING_POOL, max(pools)):]
        pooled_scores.append(scores[short].mean(axis=0))
        pooled_energy.append(energy[short].max())
        index.append(i)
    dim = (emb.shape[1] if emb.ndim == 2 else 1024) * len(pools)
    return {
        "vectors": np.asarray(vectors, dtype=np.float32).reshape(-1, dim),
        "class_scores": np.asarray(pooled_scores, dtype=np.float32).reshape(-1, len(SOUND_CLASSES)),
        "energy": np.asarray(pooled_energy, dtype=np.float32),
        "hop_index": np.asarray(index, dtype=np.int32),
    }


def select_event_windows(class_score: np.ndarray, energy: np.ndarray) -> np.ndarray:
    """Indices of the windows most likely to contain a weakly labelled event.

    Ranked by YAMNet's own score for the class; when YAMNet barely recognises the class in
    this clip (best score < EVENT_SCORE_FLOOR) the ranking falls back to energy. Keeps the
    top ``ceil(EVENT_TOPK_FRACTION * m)`` windows (at least 1, at most EVENT_TOPK_MAX) that
    also reach EVENT_MIN_RELATIVE_SCORE of the best window.
    """
    m = len(class_score)
    if m == 0:
        return np.zeros(0, dtype=np.int64)
    rank = class_score if class_score.max() >= EVENT_SCORE_FLOOR else energy
    k = int(min(EVENT_TOPK_MAX, max(1, np.ceil(EVENT_TOPK_FRACTION * m))))
    order = np.argsort(-rank, kind="stable")[:k]
    best = rank[order[0]]
    keep = order[rank[order] >= EVENT_MIN_RELATIVE_SCORE * best] if best > 0 else order[:1]
    return np.sort(keep)


def clip_examples(hops: dict[str, np.ndarray], labels: set[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(X [k, 1024], Y [k, 8], hop_index [k]) training examples for one clip.

    ``labels`` is the clip's set of target classes; empty for a background clip.
    """
    win = device_window_vectors(hops)
    n_classes = len(SOUND_CLASSES)
    if not labels:
        m = len(win["vectors"])
        return win["vectors"], np.zeros((m, n_classes), dtype=np.float32), win["hop_index"]
    targets: dict[int, np.ndarray] = {}
    for label in labels:
        c = SOUND_CLASSES.index(label)
        for w in select_event_windows(win["class_scores"][:, c], win["energy"]):
            targets.setdefault(int(w), np.zeros(n_classes, dtype=np.float32))[c] = 1.0
    keep = np.array(sorted(targets), dtype=np.int64)
    if keep.size == 0:
        return np.zeros((0, win["vectors"].shape[1]), np.float32), np.zeros((0, n_classes), np.float32), np.zeros(0, np.int32)
    return win["vectors"][keep], np.stack([targets[int(w)] for w in keep]), win["hop_index"][keep]


def clip_mean_example(hops: dict[str, np.ndarray], labels: set[str]) -> tuple[np.ndarray, np.ndarray]:
    """Old behaviour for comparison: one vector per clip = mean over all hops (``--mode clip_mean``)."""
    y = np.zeros((1, len(SOUND_CLASSES)), dtype=np.float32)
    for label in labels:
        y[0, SOUND_CLASSES.index(label)] = 1.0
    return hops["embeddings"].mean(axis=0, keepdims=True).astype(np.float32), y
