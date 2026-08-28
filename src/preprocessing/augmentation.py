"""Config-toggleable audio augmentations for training."""
from __future__ import annotations

import numpy as np
from audiomentations import (
    AddBackgroundNoise,
    Compose,
    Gain,
    PitchShift,
    TimeStretch,
)

from src.config import SAMPLE_RATE


def build_augmentation_pipeline(
    background_noise_paths: list[str] | None = None,
    p: float = 0.5,
) -> Compose:
    transforms = [
        TimeStretch(min_rate=0.9, max_rate=1.1, p=p, leave_length_unchanged=False),
        PitchShift(min_semitones=-2, max_semitones=2, p=p),
        Gain(min_gain_db=-6, max_gain_db=6, p=p),
    ]
    if background_noise_paths:
        transforms.append(
            AddBackgroundNoise(
                sounds_path=background_noise_paths,
                min_snr_db=5,
                max_snr_db=20,
                p=p,
            )
        )
    return Compose(transforms)


def augment_waveform(
    waveform: np.ndarray,
    pipeline: Compose | None = None,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    pipeline = pipeline or build_augmentation_pipeline()
    augmented = pipeline(samples=waveform, sample_rate=sample_rate)
    return augmented.astype(np.float32)


def augment_embedding_batch(
    embeddings: np.ndarray,
    noise_std: float = 0.01,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Light embedding-space jitter when waveform-level aug is too slow."""
    rng = rng or np.random.default_rng()
    return embeddings + rng.normal(0, noise_std, embeddings.shape).astype(np.float32)
