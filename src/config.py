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

# Generated or organised locally; never read from / written to Drive.
_LOCAL_ONLY = {"indian_ambient", "esc50", "inoise_chunks", "synthetic_mixed"}


def get_raw_dir(dataset: str) -> Path:
    """Return the raw audio directory for *dataset*.

    Priority:
      1. SAHARA_DRIVE_DATASETS/<Dataset>  if env var is set and folder exists.
      2. data/raw/<dataset>               (local / repo default).

    Dataset name mapping (Drive uses Title-Case; local uses lower_snake):
        fsd50k          → FSD50K
        audioset        → AudioSet
        desed           → DESED
        indian_ambient  → always data/raw/indian_ambient (organise_recordings.py output;
                          Drive's Indian_Ambient holds the unsorted originals)
        inoise          → iNoise
        inoise_chunks   → iNoise_chunks
        synthetic_mixed → SyntheticMixed
    """
    _drive_name_map = {
        "fsd50k": "FSD50K",
        "audioset": "AudioSet",
        "desed": "DESED",
        "inoise": "iNoise",
        "inoise_chunks": "iNoise_chunks",
        "synthetic_mixed": "SyntheticMixed",
    }
    if DRIVE_DATASETS_DIR is not None and dataset not in _LOCAL_ONLY:
        drive_name = _drive_name_map.get(dataset, dataset)
        drive_path = DRIVE_DATASETS_DIR / drive_name
        if drive_path.exists() and any(drive_path.iterdir()):
            return drive_path
        # Return Drive path as default for drive-resident new datasets if DRIVE_DATASETS_DIR is configured
        if dataset in {"inoise", "inoise_chunks", "synthetic_mixed"}:
            return drive_path
    return RAW_DATA_DIR / dataset

SAMPLE_RATE = 16_000
YAMNET_EMBEDDING_DIM = 1_024
SOUND_CLASSES = [
    "smoke_alarm", "doorbell", "siren", "knocking", "dog_bark", "baby_cry",
    "glass_break", "appliance_beep",
]

# Synthetic mixing and iNoise chunking configuration
SYNTHETIC_MIXES_PER_TARGET = 3
SYNTHETIC_SNR_MIN_DB = -5.0
SYNTHETIC_SNR_MAX_DB = 10.0
INOISE_CHUNK_DURATION_SEC = 5.0

# Seeded sample of non-target FSD50K clips used as background (label_map.FSD50K_BACKGROUND_EXCLUDE).
FSD50K_BACKGROUND_SAMPLE = 10000

# AudioSet top-up target per class (clips on disk, after exact-mid filtering).
AUDIOSET_TARGET_PER_CLASS = 200

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# Raw-waveform augmentation is deliberately opt-in: validation and test clips
# are never augmented, and baseline cache generation remains deterministic.
AUGMENTATION_ENABLED = False
AUGMENTATION_VARIANTS_PER_CLIP = 1

# Asymmetric safety costs for the eight classifier outputs.  These weights are
# used only by the optional cost-sensitive BCE during training.
POSITIVE_CLASS_WEIGHTS = {
    "smoke_alarm": 3.0,
    "doorbell": 1.0,
    "siren": 3.0,
    "knocking": 1.0,
    "dog_bark": 1.0,
    "baby_cry": 1.5,
    "glass_break": 3.0,
    "appliance_beep": 1.5,
}

# Deployment calibration and streaming inference defaults. Values are tuned on
# validation data by threshold_search.py; the precision floor protects against
# high-cost false positives for safety-critical events.
SAFETY_CRITICAL_CLASSES = ("smoke_alarm", "siren", "glass_break")
SAFETY_PRECISION_FLOOR = 0.70
FALSE_ALARM_RATE_LIMIT = 0.10
# Total alerts per hour allowed on validation audio that holds none of the alerted sounds, summed
# over the 8 classes (src/evaluation/tune_policy.py). Chosen in Phase 4, see docs/evaluation_report.md.
FALSE_ALARM_BUDGET_PER_HOUR = 4.0
THRESHOLD_GRID_START = 0.05
THRESHOLD_GRID_END = 0.95
THRESHOLD_GRID_STEP = 0.01
VAD_ENERGY_THRESHOLD = 1e-4
INFERENCE_WINDOW_SEC = 1.0
INFERENCE_HOP_SEC = 0.5

# What the phone computes (mirrors app/.../core/ModelConfig.kt; change both together).
WINDOW_SAMPLES = int(INFERENCE_WINDOW_SEC * SAMPLE_RATE)  # 16,000-sample ring buffer
HOP_SAMPLES = int(INFERENCE_HOP_SEC * SAMPLE_RATE)  # inference every 8,000 samples
YAMNET_PATCH_SAMPLES = 15_600  # newest 0.975 s of the buffer = one YAMNet patch
EMBEDDING_POOL = 2  # short pool: mean of the last 2 non-silent hop embeddings
# Head input = concatenated means over each pool size. (2, 6) (~1.5 s and ~3.5 s of context,
# 2,048 values) was tried in Phase 4 and did not beat (2,) across seeds, so the head input stays
# 1,024. The app picks the pools from the head's input size (ModelConfig.kt).
WINDOW_POOLS = (2,)
HEAD_INPUT_DIM = YAMNET_EMBEDDING_DIM * len(WINDOW_POOLS)

# Event-window selection for weakly labelled target clips (windowing.py). A clip is labelled
# as a whole, but the event may fill only part of it, so only the windows most likely to hold
# the event are kept as positives; the rest of a target clip is dropped (not used as background).
EVENT_TOPK_FRACTION = 0.3  # keep up to this share of a clip's windows ...
EVENT_TOPK_MAX = 8  # ... but never more than this many per clip and class
EVENT_MIN_RELATIVE_SCORE = 0.25  # and only windows scoring >= 25 % of the clip's best window
EVENT_SCORE_FLOOR = 0.05  # if YAMNet's best score for the class is below this, rank by energy instead
