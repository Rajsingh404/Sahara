"""Log-mel and YAMNet embedding extraction."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import librosa
import numpy as np
import tensorflow_hub as hub

from src.config import PROCESSED_DATA_DIR, SAMPLE_RATE

logger = logging.getLogger(__name__)

YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
_yamnet_model = None


def _get_yamnet():
    global _yamnet_model
    if _yamnet_model is None:
        logger.info("Loading YAMNet from TF-Hub …")
        _yamnet_model = hub.load(YAMNET_URL)
    return _yamnet_model


def extract_logmel(
    waveform: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mels: int = 64,
    n_fft: int = 1024,
    hop_length: int = 160,
) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def extract_yamnet_embedding(waveform: np.ndarray) -> np.ndarray:
    """Mean-pool frame-level YAMNet embeddings into a single 1024-d vector."""
    model = _get_yamnet()
    scores, embeddings, _ = model(waveform.astype(np.float32))
    del scores
    pooled = np.mean(embeddings.numpy(), axis=0)
    return pooled.astype(np.float32)


def _cache_key(filepath: str) -> str:
    return hashlib.sha256(filepath.encode()).hexdigest()[:16]


def embedding_cache_path(filepath: str) -> Path:
    return PROCESSED_DATA_DIR / "embedding_cache" / f"{_cache_key(filepath)}.npy"


def extract_yamnet_embedding_cached(filepath: str, waveform: np.ndarray) -> np.ndarray:
    cache = embedding_cache_path(filepath)
    if cache.exists():
        return np.load(cache)
    embedding = extract_yamnet_embedding(waveform)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, embedding)
    return embedding
