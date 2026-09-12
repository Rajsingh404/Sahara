"""iNoise chunking module.

Slices raw concatenated iNoise audio files into non-overlapping 5-second windows,
filters out silent or clipped audio, and saves the output to $SAHARA_DRIVE_DATASETS/iNoise_chunks/{category}/.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import INOISE_CHUNK_DURATION_SEC, SAMPLE_RATE, get_raw_dir
from src.preprocessing.audio_utils import load_audio

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SILENCE_RMS_THRESHOLD = 1e-4
MAX_AMPLITUDE_CLIPPING = 0.999


def is_valid_chunk(chunk: np.ndarray) -> bool:
    """Return True if chunk is not silent and not severely clipped."""
    if len(chunk) == 0:
        return False
    rms = np.sqrt(np.mean(chunk**2))
    if rms < SILENCE_RMS_THRESHOLD:
        return False
    if np.max(np.abs(chunk)) >= MAX_AMPLITUDE_CLIPPING:
        return False
    return True


def chunk_audio_file(
    filepath: Path,
    output_category_dir: Path,
    chunk_sec: float = INOISE_CHUNK_DURATION_SEC,
) -> int:
    """Load, slice, and save valid chunks from a single audio file."""
    audio = load_audio(filepath, target_sr=SAMPLE_RATE)
    if audio is None or len(audio) == 0:
        logger.warning("Could not load audio from %s", filepath)
        return 0

    chunk_len = int(chunk_sec * SAMPLE_RATE)
    num_chunks = len(audio) // chunk_len
    saved_count = 0

    output_category_dir.mkdir(parents=True, exist_ok=True)
    stem = filepath.stem

    for i in range(num_chunks):
        chunk = audio[i * chunk_len : (i + 1) * chunk_len]
        if is_valid_chunk(chunk):
            out_name = f"{stem}_chunk{i:04d}.wav"
            out_path = output_category_dir / out_name
            sf.write(str(out_path), chunk, SAMPLE_RATE)
            saved_count += 1

    return saved_count


def process_all_inoise_categories() -> dict[str, int]:
    raw_inoise_dir = get_raw_dir("inoise")
    chunks_dir = get_raw_dir("inoise_chunks")

    logger.info("Chunking iNoise files from %s to %s", raw_inoise_dir, chunks_dir)
    results: dict[str, int] = {}

    if not raw_inoise_dir.exists():
        logger.warning("Raw iNoise directory not found: %s", raw_inoise_dir)
        return results

    for root, _, files in os.walk(raw_inoise_dir):
        rel_path = Path(root).relative_to(raw_inoise_dir)
        cat_name = rel_path.as_posix().lower() if str(rel_path) != "." else "uncategorized"

        out_cat_dir = chunks_dir / cat_name
        cat_saved = 0

        for file in files:
            if file.lower().endswith((".wav", ".flac", ".mp3", ".ogg")):
                file_path = Path(root) / file
                n_saved = chunk_audio_file(file_path, out_cat_dir)
                cat_saved += n_saved

        if cat_saved > 0:
            results[cat_name] = cat_saved
            logger.info("Category '%s': created %d valid chunks", cat_name, cat_saved)

    logger.info("iNoise chunking complete. Total categories processed: %d", len(results))
    return results


if __name__ == "__main__":
    process_all_inoise_categories()
