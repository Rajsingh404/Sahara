"""Name spotting head: "is someone saying Aryan in the last 1.3 s?", for any speaker.

Input: the last WINDOW speech embeddings (src/personalization/speech_embedding.py), flattened.
Model: standardise, then L2-regularised logistic regression. It is tiny (768 weights), ports to
Kotlin as a dot product (app/android/.../core/NameSpotter.kt), and with ~40 positive clips a
bigger model would only memorise the four recorders.

Training windows:
- positives: windows inside the spoken part of each "Aryan" clip (energy endpointing), plus the
  same clips mixed into background scenes at 5-20 dB SNR;
- negatives: every window of the similar-name clips and of the background scenes (TV, kitchen,
  hawkers, temple, rain), which is where false calls would come from in a home.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

WINDOW = 8
HOP = 160
FRAME = 400


def speech_bounds(wave: np.ndarray, db_below_peak: float = 20.0, pad_s: float = 0.15) -> tuple[int, int]:
    """Sample range of the loud core of an utterance (simple energy endpointing)."""
    n = 1 + max(0, (len(wave) - FRAME) // HOP)
    db = np.array([10 * np.log10(np.mean(wave[i * HOP : i * HOP + FRAME] ** 2) + 1e-10) for i in range(n)])
    keep = np.flatnonzero(db >= db.max() - db_below_peak)
    if len(keep) == 0:
        return 0, len(wave)
    pad = int(pad_s * 16_000)
    return max(0, keep[0] * HOP - pad), min(len(wave), keep[-1] * HOP + FRAME + pad)


def windows(emb: np.ndarray, window: int = WINDOW) -> np.ndarray:
    if len(emb) < window:
        emb = np.pad(emb, ((0, window - len(emb)), (0, 0)), mode="edge")
    return np.stack([emb[i : i + window].ravel() for i in range(len(emb) - window + 1)])


def mix(wave: np.ndarray, noise: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    if len(noise) < len(wave):
        noise = np.tile(noise, int(np.ceil(len(wave) / max(len(noise), 1))))
    start = rng.integers(0, len(noise) - len(wave) + 1)
    n = noise[start : start + len(wave)]
    p_s, p_n = np.mean(wave**2) + 1e-10, np.mean(n**2) + 1e-10
    return wave + n * np.sqrt(p_s / (p_n * 10 ** (snr_db / 10)))


@dataclass
class NameHead:
    name: str
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    threshold: float = 0.5
    window: int = WINDOW

    def score_windows(self, x: np.ndarray) -> np.ndarray:
        z = ((x - self.mean) / self.scale) @ self.coef + self.intercept
        return 1.0 / (1.0 + np.exp(-z))

    def score_clip(self, emb: np.ndarray) -> float:
        """Highest probability over the clip, as the phone would see it while streaming."""
        return float(self.score_windows(windows(emb, self.window)).max())

    def to_json(self, path: Path, extra: dict | None = None) -> None:
        payload = {
            "name": self.name,
            "window": self.window,
            "dim": int(len(self.mean) // self.window),
            "threshold": float(self.threshold),
            "intercept": float(self.intercept),
            "mean": [round(float(v), 6) for v in self.mean],
            "scale": [round(float(v), 6) for v in self.scale],
            "coef": [round(float(v), 7) for v in self.coef],
        }
        payload.update(extra or {})
        path.write_text(json.dumps(payload), encoding="utf-8")


def train_head(
    name: str,
    pos_embs: list[np.ndarray],
    neg_embs: list[np.ndarray],
    c: float = 1e-4,
) -> NameHead:
    from sklearn.linear_model import LogisticRegression

    xp = np.concatenate([windows(e) for e in pos_embs])
    xn = np.concatenate([windows(e) for e in neg_embs])
    x = np.concatenate([xp, xn])
    y = np.concatenate([np.ones(len(xp)), np.zeros(len(xn))])
    mean = x.mean(0)
    scale = x.std(0) + 1e-6
    clf = LogisticRegression(C=c, max_iter=5000, class_weight="balanced").fit((x - mean) / scale, y)
    return NameHead(name, mean, scale, clf.coef_[0], float(clf.intercept_[0]))
