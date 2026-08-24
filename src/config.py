"""Shared, filesystem-independent project configuration."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"
MODELS_DIR = REPO_ROOT / "models"

SAMPLE_RATE = 16_000
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
