"""Synthetic Audio Mixing Pipeline.

Mixes real train-split target clips with train-split background beds (DESED, team
recordings, iNoise chunks when present) at SNRs in [SYNTHETIC_SNR_MIN_DB, SYNTHETIC_SNR_MAX_DB],
saving outputs locally under data/raw/synthetic_mixed/{class_name}/. Mixes are train-only.
"""
from __future__ import annotations

import csv
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import logging
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
from src.data.label_map import BACKGROUND_LABEL
from src.preprocessing.audio_utils import load_audio

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def mix_target_with_background(
    target_clip_path: Path | str,
    background_chunk_path: Path | str,
    snr_db: float,
    crop_fraction: float | None = None,
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
        start = random.randint(0, b_len - t_len) if crop_fraction is None else int(crop_fraction * (b_len - t_len))
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


def _read_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def find_available_background_chunks(train_rows: list[dict] | None = None) -> list[Path]:
    """Background beds: train-split background clips (DESED, team recordings) + iNoise chunks.

    Only train-split clips are used, so no validation/test audio reaches training through a mix.
    iNoise chunks are only used when iNoise is present (it needs a manual IEEE DataPort download).
    """
    if train_rows is None:
        train_rows = _read_manifest(METADATA_DIR / "train_manifest.csv")
    chunks = [Path(r["filepath"]) for r in train_rows
              if r["label"] == BACKGROUND_LABEL and r["source_type"] != "synthetic_mixed"]
    inoise_chunks_dir = get_raw_dir("inoise_chunks")
    if inoise_chunks_dir.exists():
        chunks.extend(sorted(inoise_chunks_dir.rglob("*.wav")))
    return chunks


def _bed_source(path: Path) -> str:
    text = str(path)
    for name in ("inoise", "desed", "indian_ambient"):
        if f"/{name}" in text:
            return name
    return "other"


def generate_synthetic_dataset(
    mixes_per_target: int = SYNTHETIC_MIXES_PER_TARGET,
    snr_min: float = SYNTHETIC_SNR_MIN_DB,
    snr_max: float = SYNTHETIC_SNR_MAX_DB,
    workers: int = 4,
) -> int:
    """Mix every real train-split target clip onto ``mixes_per_target`` random beds.

    Mixes are written to data/raw/synthetic_mixed/<class>/ with synthetic_metadata.csv,
    which records the source clip's recording_id so the manifest builder keeps every mix
    in train next to its source. Existing mixes are kept (the run resumes).
    """
    train_manifest = METADATA_DIR / "train_manifest.csv"
    if not train_manifest.exists():
        logger.error("train_manifest.csv not found in %s", METADATA_DIR)
        return 0
    train_rows = _read_manifest(train_manifest)
    bg_chunks = find_available_background_chunks(train_rows)
    if not bg_chunks:
        logger.warning("No train-split background clips found.")
        return 0
    logger.info("Background beds: %d (%s)", len(bg_chunks), dict(Counter(_bed_source(b) for b in bg_chunks)))

    output_dir = get_raw_dir("synthetic_mixed")
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_csv_path = output_dir / "synthetic_metadata.csv"

    target_rows = [r for r in train_rows if r["label"] in SOUND_CLASSES and r["source_type"] != "synthetic_mixed"]
    logger.info("Found %d real target clips in train_manifest.csv", len(target_rows))

    # Plan every mix up front from one seeded RNG, so results do not depend on thread timing.
    rng = random.Random(RANDOM_SEED)
    plans = []
    for row in target_rows:
        for mix_idx in range(mixes_per_target):
            plans.append((row, mix_idx, rng.choice(bg_chunks), round(rng.uniform(snr_min, snr_max), 1), rng.random()))

    def make(plan) -> dict | None:
        row, mix_idx, bg_path, snr_db, crop = plan
        target_path = Path(row["filepath"])
        label = row["label"]
        source = row["source_dataset"]
        out_filename = f"syn_{source}_{target_path.stem}_{_bed_source(bg_path)}_snr{snr_db:+.1f}_v{mix_idx}.wav"
        out_path = output_dir / label / out_filename
        if not out_path.exists():
            try:
                mixed_audio = mix_target_with_background(target_path, bg_path, snr_db, crop_fraction=crop)
            except Exception as exc:
                logger.debug("Skipping mix for %s: %s", target_path, exc)
                return None
            out_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out_path), mixed_audio, SAMPLE_RATE)
        return {
            "filename": out_filename,
            "label": label,
            "relative_filepath": f"{label}/{out_filename}",
            "source_target": str(target_path),
            "source_dataset": source,
            "source_recording_id": row["recording_id"],
            "source_background": str(bg_path),
            "bed_source": _bed_source(bg_path),
            "snr_db": snr_db,
            "duration_sec": round(sf.info(str(out_path)).duration, 3),
        }

    metadata_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, result in enumerate(pool.map(make, plans), 1):
            if result:
                metadata_rows.append(result)
            if i % 500 == 0:
                logger.info("%d / %d mixes", i, len(plans))

    if metadata_rows:
        with meta_csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metadata_rows[0]))
            writer.writeheader()
            writer.writerows(metadata_rows)
        logger.info("Saved synthetic metadata (%d rows) to %s", len(metadata_rows), meta_csv_path)
    logger.info("Synthetic dataset: %d mixes from %d target clips", len(metadata_rows), len(target_rows))
    return len(metadata_rows)


if __name__ == "__main__":
    generate_synthetic_dataset()
