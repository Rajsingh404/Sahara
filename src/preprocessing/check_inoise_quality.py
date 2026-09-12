"""Quality check script for iNoise dataset (IEEE DataPort DOI: 10.21227/w3xm-jn45).

Inspects raw audio parameters (sample rate, bit depth), generates temporary
A/B listening clips (iNoise 'Home' vs DESED background), and provides recommendations
regarding telephony bandpass filtering (~300-3400Hz).
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import REPO_ROOT, SAMPLE_RATE, get_raw_dir
from src.preprocessing.audio_utils import load_audio

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TEMP_LISTEN_DIR = REPO_ROOT / "data" / "interim" / "ab_listening_samples"


def inspect_file(filepath: Path) -> dict:
    """Read raw metadata from soundfile without full resampling."""
    info = sf.info(str(filepath))
    return {
        "filepath": str(filepath),
        "sample_rate": info.samplerate,
        "subtype": info.subtype,
        "channels": info.channels,
        "duration": info.duration,
    }


def find_sample_files(inoise_dir: Path) -> dict[str, Path]:
    """Find one sample audio file for each category in inoise_dir."""
    samples: dict[str, Path] = {}
    if not inoise_dir.exists():
        return samples

    for root, _, files in os.walk(inoise_dir):
        for file in files:
            if file.lower().endswith((".wav", ".flac", ".mp3", ".ogg")):
                category = Path(root).name.lower()
                if category not in samples and category != Path(inoise_dir).name.lower():
                    samples[category] = Path(root) / file
                elif "default" not in samples:
                    samples[file.stem] = Path(root) / file
    return samples


def find_desed_bg_sample() -> Path | None:
    """Locate a sample background file from DESED."""
    desed_dir = get_raw_dir("desed") / "real"
    if not desed_dir.exists():
        return None
    for root, _, files in os.walk(desed_dir):
        for f in files:
            if f.lower().endswith(".wav"):
                return Path(root) / f
    return None


def run_quality_check() -> dict:
    inoise_dir = get_raw_dir("inoise")
    logger.info("Running iNoise quality check on: %s", inoise_dir)

    samples = find_sample_files(inoise_dir)
    results = {}

    if not samples:
        logger.warning("No sample files found in %s. Run download_inoise.py or place raw files first.", inoise_dir)
        print("\nQuality Check Recommendation:")
        print("  - iNoise files not found. Please download/place iNoise raw WAV files in Google Drive / data/raw/inoise.")
        print("  - Note: Indoor categories require human A/B listening check due to telephony bandpass (~300-3400 Hz).")
        return results

    print("\niNoise Category Sample Inspection:")
    print("-" * 70)
    for cat, path in samples.items():
        info = inspect_file(path)
        results[cat] = info
        print(f"  Category '{cat}': SR={info['sample_rate']}Hz, Subtype={info['subtype']}, Channels={info['channels']}, Duration={info['duration']:.1f}s")

    # A/B Sample Generation for "Home" vs DESED
    TEMP_LISTEN_DIR.mkdir(parents=True, exist_ok=True)
    home_sample = samples.get("home") or list(samples.values())[0]
    desed_sample = find_desed_bg_sample()

    home_audio = load_audio(home_sample)
    if home_audio is not None:
        clip_len = 5 * SAMPLE_RATE
        home_5s = home_audio[:clip_len] if len(home_audio) >= clip_len else np.pad(home_audio, (0, max(0, clip_len - len(home_audio))))
        sf.write(str(TEMP_LISTEN_DIR / "inoise_home_16k_5s.wav"), home_5s, SAMPLE_RATE)
        logger.info("Saved A/B sample: %s", TEMP_LISTEN_DIR / "inoise_home_16k_5s.wav")

    if desed_sample:
        desed_audio = load_audio(desed_sample)
        if desed_audio is not None:
            clip_len = 5 * SAMPLE_RATE
            desed_5s = desed_audio[:clip_len] if len(desed_audio) >= clip_len else np.pad(desed_audio, (0, max(0, clip_len - len(desed_audio))))
            sf.write(str(TEMP_LISTEN_DIR / "desed_bg_16k_5s.wav"), desed_5s, SAMPLE_RATE)
            logger.info("Saved A/B sample: %s", TEMP_LISTEN_DIR / "desed_bg_16k_5s.wav")

    print("\nQuality Check Recommendations:")
    print("-" * 70)
    print("  1. Outdoor categories (autorickshaw, bus, highway, railway_station, street): Usable as background noise.")
    print("  2. Indoor categories (airport, cafeteria, home, train, workplace): Pending human review.")
    print(f"     Please listen to A/B WAV files in: {TEMP_LISTEN_DIR}")
    print("     Check if telephony bandpass filtering (~300-3400 Hz) introduces audible degradation.")

    return results


if __name__ == "__main__":
    run_quality_check()
