"""Log-mel and YAMNet embedding extraction.

Two paths:
- ``extract_hop_embeddings`` (default since Phase 2): reproduces the phone. The app embeds
  the last 15,600 samples of its 1 s buffer every 0.5 s hop (AudioCaptureService.kt,
  InferenceEngine.kt). Here the same patches are cut from the clip and embedded in one
  batched YAMNet call. ``src/preprocessing/windowing.py`` turns the hop embeddings into the
  exact vectors the phone feeds the head.
- ``extract_yamnet_embedding``: the old whole-clip mean, kept for comparison (``--mode clip_mean``).
"""
from __future__ import annotations

import csv
import hashlib
import logging
import os
from pathlib import Path

import librosa
import numpy as np
try:
    import tensorflow_hub as hub
except ImportError:
    hub = None

from src.config import (
    HOP_SAMPLES,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    SAMPLE_RATE,
    SOUND_CLASSES,
    VAD_ENERGY_THRESHOLD,
    WINDOW_SAMPLES,
    YAMNET_PATCH_SAMPLES,
)

# Keep the TF-Hub download out of /var/folders so it survives temp cleanup.
os.environ.setdefault("TFHUB_CACHE_DIR", str(MODELS_DIR / "tfhub_cache"))

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


_batched_embed = None


def _get_batched_embed():
    """YAMNet over a batch of patches [n, 15600] -> (embeddings [n, 1024], scores [n, 521]).

    Bit-identical to one YAMNet call per patch (checked 2026-10-04), ~10x faster.
    """
    global _batched_embed
    if _batched_embed is None:
        import tensorflow as tf

        model = _get_yamnet()

        @tf.function(input_signature=[tf.TensorSpec([None, YAMNET_PATCH_SAMPLES], tf.float32)])
        @tf.autograph.experimental.do_not_convert
        def embed(patches):
            def one(patch):
                scores, embeddings, _ = model(patch)
                return embeddings[0], scores[0]

            return tf.map_fn(one, patches, fn_output_signature=(tf.float32, tf.float32))

        _batched_embed = embed
    return _batched_embed


def yamnet_class_indices() -> dict[str, list[int]]:
    """Indices into YAMNet's 521 outputs for each target class's AudioSet mids."""
    from src.data.label_map import audioset_mids_for_class

    model = _get_yamnet()
    with open(model.class_map_path().numpy().decode(), newline="", encoding="utf-8") as handle:
        mid_to_index = {row["mid"]: int(row["index"]) for row in csv.DictReader(handle)}
    return {c: sorted(mid_to_index[m] for m in audioset_mids_for_class(c) if m in mid_to_index) for c in SOUND_CLASSES}


def device_hops(waveform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Patch start indices and VAD energies for every hop at which the phone would run.

    The phone pushes 8,000-sample hops into a 16,000-sample buffer and, once the buffer is
    full, measures the mean square of the newest hop (VAD) and embeds the newest 15,600
    samples. Clips shorter than 1 s are left-padded with silence to one full buffer.
    """
    if len(waveform) < WINDOW_SAMPLES:
        waveform = np.pad(waveform, (WINDOW_SAMPLES - len(waveform), 0))
    ends = np.arange(WINDOW_SAMPLES, len(waveform) + 1, HOP_SAMPLES)
    starts = ends - YAMNET_PATCH_SAMPLES
    energy = np.array([np.mean(np.square(waveform[e - HOP_SAMPLES : e], dtype=np.float64)) for e in ends])
    return starts, energy.astype(np.float32)


def extract_hop_embeddings(waveform: np.ndarray) -> dict[str, np.ndarray]:
    """Per-hop YAMNet output for a 16 kHz mono clip at its natural level (no normalisation).

    Returns ``embeddings`` [n, 1024], ``class_scores`` [n, 8] (YAMNet's own score for each
    target class = max over its mids), ``energy`` [n] and ``vad`` [n] (True = the phone would
    run inference on this hop).
    """
    waveform = np.asarray(waveform, dtype=np.float32)
    if len(waveform) < WINDOW_SAMPLES:
        waveform = np.pad(waveform, (WINDOW_SAMPLES - len(waveform), 0))
    starts, energy = device_hops(waveform)
    patches = np.stack([waveform[s : s + YAMNET_PATCH_SAMPLES] for s in starts])
    embeddings, scores = (t.numpy() for t in _get_batched_embed()(patches))
    indices = _class_indices_cached()
    class_scores = np.stack([scores[:, indices[c]].max(axis=1) if indices[c] else np.zeros(len(scores))
                             for c in SOUND_CLASSES], axis=1)
    return {
        "embeddings": embeddings.astype(np.float32),
        "class_scores": class_scores.astype(np.float32),
        "energy": energy,
        "vad": energy >= VAD_ENERGY_THRESHOLD,
    }


_class_indices: dict[str, list[int]] | None = None


def _class_indices_cached() -> dict[str, list[int]]:
    global _class_indices
    if _class_indices is None:
        _class_indices = yamnet_class_indices()
    return _class_indices


def hop_cache_path(filepath: str, variant: str = "") -> Path:
    """Per-clip cache of extract_hop_embeddings (float16 embeddings), keyed by path + variant."""
    key = _cache_key(filepath + (f"#{variant}" if variant else ""))
    return PROCESSED_DATA_DIR / "hop_cache" / f"{key}.npz"


def save_hop_cache(path: Path, hops: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, embeddings=hops["embeddings"].astype(np.float16), class_scores=hops["class_scores"].astype(np.float16),
             energy=hops["energy"], vad=hops["vad"])


def load_hop_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {
            "embeddings": data["embeddings"].astype(np.float32),
            "class_scores": data["class_scores"].astype(np.float32),
            "energy": data["energy"],
            "vad": data["vad"],
        }


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
