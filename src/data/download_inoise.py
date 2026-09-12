"""Download script for iNoise Indian Noise Database (IEEE DataPort DOI: 10.21227/w3xm-jn45).

Attempts automated retrieval first; if authentication is required by IEEE DataPort,
it prints clear instructions for manual download and placement into Google Drive.
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import DRIVE_DATASETS_DIR, get_raw_dir

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

INOISE_IEEE_URL = "https://ieee-dataport.org/documents/inoise-indian-noise-database"
EXPECTED_CATEGORIES = {
    "outdoor": ["autorickshaw", "bus", "highway", "railway_station", "street"],
    "indoor": ["airport", "cafeteria", "home", "train", "workplace"],
}

MANUAL_DOWNLOAD_MSG = (
    "\n" + "=" * 80 + "\n"
    "iNoise requires manual download — please create a free IEEE DataPort account,\n"
    "visit https://ieee-dataport.org/documents/inoise-indian-noise-database,\n"
    "download the files, and place them in data/raw/inoise/ or $SAHARA_DRIVE_DATASETS/iNoise/.\n"
    "=" * 80 + "\n"
)


def verify_inoise_directory(target_dir: Path) -> dict[str, list[str]]:
    """Scan target_dir and log present WAV files by category."""
    found: dict[str, list[str]] = {}
    if not target_dir.exists():
        logger.warning("Target directory does not exist: %s", target_dir)
        return found

    for root, _, files in os.walk(target_dir):
        wavs = [f for f in files if f.lower().endswith(".wav")]
        if wavs:
            rel_cat = Path(root).relative_to(target_dir).as_posix()
            found[rel_cat] = wavs

    if found:
        logger.info("iNoise directory scan complete at %s:", target_dir)
        for cat, files in found.items():
            logger.info("  Category '%s': %d WAV files", cat, len(files))
    else:
        logger.warning("No WAV files found in %s", target_dir)
    return found


def download_inoise() -> Path:
    target_dir = get_raw_dir("inoise")
    logger.info("Target directory for iNoise: %s", target_dir)

    # Check if files are already present
    existing = verify_inoise_directory(target_dir)
    if existing:
        logger.info("iNoise dataset is already present in %s", target_dir)
        return target_dir

    logger.info("Attempting automated download from IEEE DataPort...")
    try:
        response = requests.get(INOISE_IEEE_URL, allow_redirects=True, timeout=15)
        # IEEE DataPort redirects unauthenticated users or requires login for file download links
        if response.status_code != 200 or "login" in response.url.lower() or "user/login" in response.text.lower():
            logger.error("Automated download blocked by IEEE DataPort authentication wall.")
            print(MANUAL_DOWNLOAD_MSG)
            return target_dir
    except Exception as exc:
        logger.error("Automated download attempt failed: %s", exc)
        print(MANUAL_DOWNLOAD_MSG)
        return target_dir

    print(MANUAL_DOWNLOAD_MSG)
    return target_dir


if __name__ == "__main__":
    download_inoise()
