"""Shared, filesystem-independent project configuration.

Dataset modes
-------------
LOCAL mode (default):
    Audio files are read from  data/raw/<dataset>/  inside the repo.

GOOGLE DRIVE STREAM mode:
    Set the environment variable SAHARA_DRIVE_DATASETS to the local
    filesystem path exposed by Google Drive for Desktop, e.g.:

        export SAHARA_DRIVE_DATASETS="/Users/rajsingh/Library/CloudStorage/\
GoogleDrive-singhraj04.jan@gmail.com/My Drive/SAHARA/Datasets"

    When set, the manifest builder and cache builder will look for
    FSD50K / AudioSet / DESED / Indian_Ambient under that path instead of
    data/raw/.  The repo's data/raw/ directory is still used for any
    datasets that are NOT present under SAHARA_DRIVE_DATASETS.

    Files are streamed on-demand by Google Drive for Desktop (Stream Files
    mode).  Only the small per-clip .npy embedding cache (~4 KB each) is
    written to local disk — the full audio never needs to be downloaded.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"
MODELS_DIR = REPO_ROOT / "models"

# Optional: override per-dataset raw dirs with Google Drive stream path.
# Set SAHARA_DRIVE_DATASETS to the Datasets folder inside your Drive.
_DRIVE_DATASETS = os.environ.get("SAHARA_DRIVE_DATASETS", "").strip()
DRIVE_DATASETS_DIR: Path | None = Path(_DRIVE_DATASETS) if _DRIVE_DATASETS else None

def get_raw_dir(dataset: str) -> Path:
    """Return the raw audio directory for *dataset*.

    Priority:
      1. SAHARA_DRIVE_DATASETS/<Dataset>  if env var is set and folder exists.
      2. data/raw/<dataset>               (local / repo default).

    Dataset name mapping (Drive uses Title-Case; local uses lower_snake):
        fsd50k       → FSD50K
        audioset     → AudioSet
        desed        → DESED
        indian_ambient → Indian_Ambient
    """
    _drive_name_map = {
        "fsd50k": "FSD50K",
        "audioset": "AudioSet",
        "desed": "DESED",
        "indian_ambient": "Indian_Ambient",
    }
    if DRIVE_DATASETS_DIR is not None:
        drive_name = _drive_name_map.get(dataset, dataset)
        drive_path = DRIVE_DATASETS_DIR / drive_name
        if drive_path.exists() and any(drive_path.iterdir()):
            return drive_path
    return RAW_DATA_DIR / dataset

SAMPLE_RATE = 16_000
YAMNET_EMBEDDING_DIM = 1_024
SOUND_CLASSES = [
    "smoke_alarm", "doorbell", "siren", "knocking", "dog_bark", "baby_cry",
    "glass_break", "appliance_beep",
]

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42
