# SAHARA

**S**ound-based **A**lert & **H**aptic **A**ssistance for real-time **R**eal-world **A**wareness is an on-device safety-sound detector for Deaf and Hard-of-Hearing users. It targets smoke alarms, doorbells, sirens, knocks, dog barks, baby cries, glass breaks, and appliance beeps.

This repository currently implements the offline data and baseline-training phase only: a frozen [YAMNet](https://tfhub.dev/google/yamnet/1) embedding backbone and trainable classifier head. Android, ESP32 BLE wearable, TFLite export, Indian ambient recordings, and personalization are intentionally out of scope for this milestone.

## Current honest status

- Environment, modular data pipeline, manifests, preprocessing, baseline classifier, evaluation utilities, documentation, and unit tests: implemented.
- Baseline artifacts are present in `models/checkpoints/best_classifier.keras` and `docs/confusion_matrix.png`.
- Current baseline used the locally available AudioSet subset because the full FSD50K audio is not available in this workspace. The reported metrics are therefore small-sample baseline results, not final project performance. See [docs/dataset_card.md](docs/dataset_card.md).
- AudioSet has locally downloaded clips and DESED has background soundscapes. Re-running the downloader is resumable.

## Setup

Apple Silicon is required for the intended TensorFlow configuration.

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/verify_env.py
```

`tensorflow-macos` and `tensorflow-metal` are pinned in `requirements.txt`. On macOS, install FFmpeg before downloading AudioSet clips:

```bash
brew install ffmpeg sox
```

## Pipeline

```bash
# Download or resume one source (all is supported too)
python src/data/download.py --dataset fsd50k
python src/data/download.py --dataset audioset --retry-failed
python src/data/download.py --dataset desed

# Build source-agnostic manifests from whatever raw data is present
python src/data/manifest_builder.py

# Cache YAMNet embeddings; unavailable/corrupt audio is logged and skipped
python src/preprocessing/build_features_cache.py

# Train and evaluate the classifier head
python src/training/train.py --augment
python src/evaluation/confusion_matrix.py

# Run correctness tests
pytest -q
```

Use `python src/data/clean.py` to inspect removable FSD50K archives after successful feature caching. It only deletes archives when `--apply` is supplied.

## Dataset layout

`data/raw/indian_ambient/` is deliberately a field-recording placeholder. Add real WAV files under `background/` or `target_classes/` and rows to `metadata.csv`; `manifest_builder.py` will include them without code changes.

Detailed data status and coverage: [docs/dataset_card.md](docs/dataset_card.md). Intended offline-training and on-device-inference design: [docs/architecture.md](docs/architecture.md).

## Next phase (not implemented)

1. Record and validate Indian ambient audio.
2. Rebuild embeddings with the full FSD50K set and retrain/evaluate on a substantially larger holdout.
3. Export and validate TensorFlow Lite/INT8 inference.
4. Build the native Kotlin Android foreground audio service and BLE client.
5. Build ESP32 wristband firmware and the personalization module.
