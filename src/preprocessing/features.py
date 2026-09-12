"""Log-mel and YAMNet embedding extraction."""
from __future__ import annotations

import hashlib
import logging
import csv
from pathlib import Path

import librosa
import numpy as np
try:
    import tensorflow_hub as hub
except ImportError:
    hub = None

from src.config import PROCESSED_DATA_DIR, SAMPLE_RATE

logger = logging.getLogger(__name__)

YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
_yamnet_model = None


def _get_yamnet():
    global _yamnet_model
    if _yamnet_model is None:
        if hub is None:
            raise ImportError("tensorflow_hub is required to load YAMNet")
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


def _self_test_file() -> Path | None:
    from src.config import METADATA_DIR
    manifest = METADATA_DIR / "full_manifest.csv"
    if not manifest.exists():
        return None
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            path = Path(row["filepath"])
            if path.is_file():
                return path
    return None


if __name__ == "__main__":
    from src.preprocessing.audio_utils import load_audio

    path = _self_test_file()
    if path is None:
        raise SystemExit("No readable manifest audio file found for self-test.")
    audio = load_audio(path)
    if audio is None:
        raise SystemExit(f"Could not load self-test audio: {path}")
    print(f"file: {path}")
    print(f"log-mel: {extract_logmel(audio).shape}")
    print(f"YAMNet embedding: {extract_yamnet_embedding(audio).shape}")
