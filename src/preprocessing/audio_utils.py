"""Audio loading utilities for the SAHARA preprocessing pipeline."""
from __future__ import annotations

import logging
import csv
from pathlib import Path

import librosa
import numpy as np

from src.config import METADATA_DIR, SAMPLE_RATE

logger = logging.getLogger(__name__)


def _log_load_failure(path: Path, error: Exception) -> None:
    """Append a batch-safe, inspectable record without raising another error."""
    try:
        METADATA_DIR.mkdir(parents=True, exist_ok=True)
        failure_log = METADATA_DIR / "load_failures.csv"
        needs_header = not failure_log.exists() or failure_log.stat().st_size == 0
        with failure_log.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if needs_header:
                writer.writerow(["filepath", "error"])
            writer.writerow([str(path), str(error)])
    except OSError:
        logger.exception("Could not write audio load failure log")


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
        _log_load_failure(path, exc)
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


def get_duration(filepath: str | Path, target_sr: int = SAMPLE_RATE) -> float:
    """Compatibility-friendly public duration helper used by manifests."""
    return get_duration_sec(filepath, target_sr)
