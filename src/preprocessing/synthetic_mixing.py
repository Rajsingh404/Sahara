"""Synthetic Audio Mixing Pipeline.

Mixes target sound clips (from FSD50K/AudioSet in train_manifest.csv) with background chunks
(from iNoise_chunks or DESED) at controlled signal-to-noise ratios (SNR), saving outputs
to Google Drive under $SAHARA_DRIVE_DATASETS/SyntheticMixed/{class_name}/.
"""
from __future__ import annotations

import csv
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import (
    METADATA_DIR,
    RANDOM_SEED,
    SAMPLE_RATE,
    SOUND_CLASSES,
    SYNTHETIC_MIXES_PER_TARGET,
    SYNTHETIC_SNR_MAX_DB,
    SYNTHETIC_SNR_MIN_DB,
    get_raw_dir,
)
from src.preprocessing.audio_utils import load_audio

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def mix_target_with_background(
    target_clip_path: Path | str,
    background_chunk_path: Path | str,
    snr_db: float,
) -> np.ndarray:
    """Load target and background at 16kHz, scale background to snr_db, sum, and peak-normalize."""
    target = load_audio(target_clip_path, target_sr=SAMPLE_RATE)
    background = load_audio(background_chunk_path, target_sr=SAMPLE_RATE)

    if target is None or len(target) == 0:
        raise ValueError(f"Could not load target clip: {target_clip_path}")
    if background is None or len(background) == 0:
        raise ValueError(f"Could not load background chunk: {background_chunk_path}")

    # Match background length to target clip length
    t_len = len(target)
    b_len = len(background)
    if b_len < t_len:
        # Tile background if shorter than target
        repeats = (t_len // b_len) + 1
        background = np.tile(background, repeats)[:t_len]
    elif b_len > t_len:
        # Random start crop if background is longer
        start = random.randint(0, b_len - t_len)
        background = background[start : start + t_len]

    # Calculate signal powers
    target_power = np.mean(target**2)
    bg_power = np.mean(background**2)

    if target_power < 1e-10 or bg_power < 1e-10:
        # Handle near-silent audio gracefully
        mixed = target + background
    else:
        # snr_db = 10 * log10(target_power / (scale^2 * bg_power))
        scale = np.sqrt(target_power / (bg_power * (10 ** (snr_db / 10.0))))
        mixed = target + scale * background

    # Peak normalization to prevent clipping
    max_amp = np.max(np.abs(mixed))
    if max_amp > 0.99:
        mixed = (mixed / max_amp) * 0.99

    return mixed.astype(np.float32)


def find_available_background_chunks() -> list[Path]:
    """Find background chunks in iNoise_chunks and DESED background set."""
    chunks: list[Path] = []

    # iNoise_chunks
    inoise_chunks_dir = get_raw_dir("inoise_chunks")
    if inoise_chunks_dir.exists():
        for root, _, files in os.walk(inoise_chunks_dir):
            for f in files:
                if f.lower().endswith(".wav"):
                    chunks.append(Path(root) / f)

    # DESED background
    desed_real_dir = get_raw_dir("desed") / "real"
    if desed_real_dir.exists():
        for root, _, files in os.walk(desed_real_dir):
            for f in files:
                if f.lower().endswith(".wav"):
                    chunks.append(Path(root) / f)

    return chunks


def generate_synthetic_dataset(
    mixes_per_target: int = SYNTHETIC_MIXES_PER_TARGET,
    snr_min: float = SYNTHETIC_SNR_MIN_DB,
    snr_max: float = SYNTHETIC_SNR_MAX_DB,
) -> int:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    train_manifest = METADATA_DIR / "train_manifest.csv"
    if not train_manifest.exists():
        logger.error("train_manifest.csv not found in %s", METADATA_DIR)
        return 0

    bg_chunks = find_available_background_chunks()
    if not bg_chunks:
        logger.warning("No background chunks found in iNoise_chunks or DESED.")
        return 0

    output_dir = get_raw_dir("synthetic_mixed")
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_csv_path = output_dir / "synthetic_metadata.csv"

    # Read train_manifest rows matching target sound classes
    target_rows: list[dict] = []
    with train_manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["label"] in SOUND_CLASSES:
                target_rows.append(row)

    logger.info("Found %d target clips in train_manifest.csv", len(target_rows))
    generated_count = 0

    metadata_rows: list[dict] = []

    for row in target_rows:
        target_path = Path(row["filepath"])
        label = row["label"]
        class_out_dir = output_dir / label
        class_out_dir.mkdir(parents=True, exist_ok=True)

        for mix_idx in range(mixes_per_target):
            bg_path = random.choice(bg_chunks)
            snr_db = round(random.uniform(snr_min, snr_max), 1)

            try:
                mixed_audio = mix_target_with_background(target_path, bg_path, snr_db)
                bg_cat = bg_path.parent.name
                out_filename = f"syn_{target_path.stem}_{bg_cat}_snr{snr_db:+.1f}_v{mix_idx}.wav"
                out_path = class_out_dir / out_filename

                sf.write(str(out_path), mixed_audio, SAMPLE_RATE)
                generated_count += 1

                metadata_rows.append(
                    {
                        "filename": out_filename,
                        "source_target": str(target_path),
                        "source_background": str(bg_path),
                        "snr_db": snr_db,
                        "label": label,
                        "relative_filepath": f"{label}/{out_filename}",
                    }
                )
            except Exception as exc:
                logger.debug("Skipping mix for %s: %s", target_path, exc)

    # Write synthetic_metadata.csv
    if metadata_rows:
        fieldnames = ["filename", "source_target", "source_background", "snr_db", "label", "relative_filepath"]
        with meta_csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metadata_rows)
        logger.info("Saved synthetic metadata (%d rows) to %s", len(metadata_rows), meta_csv_path)

    logger.info("Synthetic dataset generation complete: %d clips created", generated_count)
    return generated_count


if __name__ == "__main__":
    generate_synthetic_dataset()
