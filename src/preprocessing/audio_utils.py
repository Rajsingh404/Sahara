"""Audio loading utilities for the SAHARA preprocessing pipeline."""
from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np

from src.config import SAMPLE_RATE

logger = logging.getLogger(__name__)


def load_audio(filepath: str | Path, target_sr: int = SAMPLE_RATE) -> np.ndarray | None:
    """Load audio, resample to mono 16 kHz, peak-normalize. Returns None on failure."""
    path = Path(filepath)
    try:
        waveform, _ = librosa.load(path, sr=target_sr, mono=True)
        if waveform.size == 0:
            logger.warning("Empty audio: %s", path)
            return None
        peak = np.max(np.abs(waveform))
        if peak > 0:
            waveform = waveform / peak
        return waveform.astype(np.float32)
    except Exception as exc:  # noqa: BLE001 — corrupt files must not crash the pipeline
        logger.warning("Failed to load %s: %s", path, exc)
        return None


def get_duration_sec(filepath: str | Path, target_sr: int = SAMPLE_RATE) -> float:
    """Return clip duration in seconds.

    Uses soundfile.info() (header-only read) as a fast-path; this avoids
    decoding the full PCM waveform, which is critical when scanning thousands
    of files over a network mount (e.g. Google Drive FUSE).  Falls back to
    librosa on any failure.
    """
    path = Path(filepath)
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return float(info.frames / info.samplerate)
    except Exception:
        pass
    try:
        return float(librosa.get_duration(path=path, sr=target_sr))
    except Exception:
        return 0.0
