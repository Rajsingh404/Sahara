# SAHARA — Claude Code project guide

Sound-based Alert & Haptic Assistance for Real-time Awareness. Final-year B.Tech capstone, NMIMS MPSTME.
Earlier name: SonicAlert. Treat any `sonicalert` reference as legacy and rename it when touched.

## Goal

Real-time, 100% on-device detection of safety-critical sounds for Deaf and Hard-of-Hearing users, with visual (phone) and haptic (wristband) alerts. Focus: Indian acoustic environments.

**Positioning rule (never break this):** the user never notices, records or uploads a sound. The only data flow is
continuous background capture → on-device VAD / windowing / inference → automatic alert.
Do not add any upload endpoint, cloud inference, or "record and analyze" flow.

## Classes (8, fixed order)

`smoke_alarm, doorbell, siren, knocking, dog_bark, baby_cry, glass_break, appliance_beep`

The single source of truth is `SOUND_CLASSES` in `src/config.py` (`src/data/label_map.py` maps each class to dataset labels).
The app reads the same list from `app/assets/labels.txt`, which the export step generates. Never hand-edit the order in two places.
Non-target audio uses the label `background`.

## Stack (locked)

| Layer | Choice |
|---|---|
| App | Flutter (Dart), Android only. UI, alerts, BLE. |
| Background audio | Native Kotlin platform channel: Foreground Service + `AudioRecord`. Dart alone cannot keep capture alive with the screen off. |
| Model | Frozen YAMNet backbone + small trainable Dense head (transfer learning). |
| Inference | TensorFlow Lite, on-device only. |
| Wristband | Custom ESP32 over BLE, haptic motor. BOM under ₹1000. Not Wear OS. |
| Personalization | Few-shot prototypical network over YAMNet embeddings for user-enrolled sounds. |
| Training | Python + TensorFlow (`tensorflow-macos` + `tensorflow-metal`, Apple Silicon). |

## Repo layout

The ML pipeline lives at the repo root (it came first). App and firmware sit beside it.

```
src/                    Python ML package (import as `src.*`)
  config.py             paths, SOUND_CLASSES, SAHARA_DRIVE_DATASETS switch
  data/                 downloaders (fsd50k, audioset, desed, inoise), manifest_builder, label_map
  preprocessing/        audio utils, YAMNet features + .npy cache, augmentation, iNoise chunking, synthetic mixing
  models/               classifier_head.py (Dense head)
  training/             train.py, losses.py (weighted BCE)
  evaluation/           metrics, confusion matrix, threshold search, false-alarm eval
  export/               to_tflite.py, quantize.py, validate_tflite.py
scripts/                pipeline runners, env check, status
tests/                  pytest suite for the ML pipeline
models/                 checkpoints/, logs/, exported/ (contents git-ignored)
data/
  metadata/             manifests (train/val/test/full), class_thresholds.json — committed
  raw/, interim/, processed/   local only, git-ignored
docs/                   dataset card, evaluation report, roadmap, architecture, reviews, paper

app/                    Flutter app (Android) — to be scaffolded
  assets/               sahara_classifier.tflite + labels.txt, written by the export step
  lib/                  audio/, detection/, alerts/, ble/ (uuids.dart), personalization/, ui/
  android/app/src/main/kotlin/…   AudioCaptureService.kt, InferenceEngine.kt, SaharaChannel.kt
firmware/esp32_wristband/   PlatformIO project: include/ble_uuids.h, src/main.cpp, hardware/ (BOM, schematic)
android/                empty legacy placeholder; native code goes in app/android/ once Flutter is scaffolded
```

## Conventions

- Branch per feature: `feat/<area>-<short-name>`, `fix/...`, `exp/...` for training experiments. Open a PR; never commit to `main` directly.
- Commit prefix by area: `app:`, `android:`, `ml:`, `fw:`, `docs:`.
- Python: format with `ruff format`, lint with `ruff check`, type hints on public functions. Run modules from the repo root so `src.*` imports resolve.
- Dart: `dart format`, `flutter analyze` clean before commit.
- Kotlin: keep the service minimal. It captures audio, runs inference, and posts results over the platform channel. UI logic stays in Dart.
- Firmware: PlatformIO project. BLE service and characteristic UUIDs live in `firmware/esp32_wristband/include/ble_uuids.h` and are mirrored in `app/lib/ble/uuids.dart`. Change both in the same PR.
- Never commit audio, `.npy` embeddings, datasets, or credentials. Model artifacts (`.keras`, `.tflite`) go in releases, or in `app/assets/` only when final.
- Manifests in `data/metadata/` store absolute paths from the machine that built them. Rebuild with `manifest_builder.py` rather than editing paths by hand.
- Every training run should record its config, metrics and label order (today: `models/logs/` + `docs/evaluation_report.md`).
- Ask before changing the class list, the YAMNet input contract, or the BLE protocol. These cross all three subprojects.

## How to run, train and test

Dev machine: macOS (Apple Silicon). Run commands from the repo root.

### ML (Python)

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg sox
python scripts/verify_env.py

# Optional: stream datasets from Google Drive for Desktop instead of data/raw/
export SAHARA_DRIVE_DATASETS="<path to My Drive/SAHARA/Datasets>"

python src/data/download.py --dataset fsd50k          # also: audioset, desed, inoise, all
python src/data/manifest_builder.py                   # build manifests from whatever raw data is present
python src/preprocessing/build_features_cache.py      # YAMNet embeddings → .npy cache (--force --augment to augment)
python src/training/train.py --batch-size 32 --epochs 30 --learning-rate 0.001
python src/evaluation/confusion_matrix.py
python src/export/to_tflite.py                        # FP32 TFLite; quantize.py / validate_tflite.py follow
pytest -q
```

### App (Flutter)

```bash
cd app
flutter pub get
flutter analyze
flutter test
flutter run            # Android device with USB debugging on
```

### Firmware (ESP32)

```bash
cd firmware/esp32_wristband
pio run                # build
pio run -t upload      # flash
pio device monitor
```

macOS prerequisites: Homebrew, Python 3.11, Flutter SDK, Android Studio (Android SDK + platform tools), PlatformIO (`brew install platformio`). The ESP32 board may need a USB-serial driver (CP210x or CH340), depending on the board.

## On-device constraints

- Target: Android phone + ESP32 wristband. No network needed at runtime.
- Must keep detecting with the screen off (Foreground Service, persistent notification).
- Per-class thresholds start from `data/metadata/class_thresholds.json` (output of `threshold_search.py`).
- TODO: minimum Android version / API level.
- TODO: target test phone(s).
- TODO: max `.tflite` size (MB).
- TODO: end-to-end latency budget, sound onset → phone alert → wristband buzz (ms).
- TODO: battery budget per hour of continuous listening.
- TODO: false-alarm budget.
- TODO: ESP32 board variant, haptic motor, battery capacity.
- Audio input contract follows YAMNet: 16 kHz mono float waveform, 0.96 s frames with 0.48 s hop. Do not change this without retraining.

## Dataset layout

Sources: FSD50K (primary), AudioSet (via yt-dlp), DESED (domestic hard negatives), iNoise Indian Noise Database, self-recorded Indian ambient clips, and synthetic mixes.

- Raw audio comes from `data/raw/<dataset>/` or, when `SAHARA_DRIVE_DATASETS` is set, from Google Drive for Desktop (streamed, cold fetch about 2.3 s). Only the `.npy` embedding cache (about 4 KB per clip) is written locally.
- Synthetic mixing (`src/preprocessing/synthetic_mixing.py`) overlays clean clips on Indian background beds at varied SNR. Those rows are tagged `source_type: synthetic_mixed`.
- The pipeline is dataset-agnostic. Add a source with a downloader in `src/data/`, its entry in `get_raw_dir()` and `label_map.py`; the manifest builder picks it up.
- Self-recorded clips: WAVs under `data/raw/indian_ambient/background/` or `target_classes/`, plus rows in `data/raw/indian_ambient/metadata.csv`.
- `source_type` values: `real`, `synthetic_mixed`, `self_recorded`.

**Recording protocol:** 4–6 s clips with ambient padding around the event, including naturally short sounds (knocking, appliance_beep). Do not trim to the event; YAMNet windowing handles it.

**Personalization dev-set:** 10–20 samples per enrolled sound, across 3–4 sounds.

## Current status (update as it changes)

- Implemented: data pipeline, manifests, YAMNet embedding cache, weighted Dense head training, evaluation, threshold search, TFLite export scripts, iNoise + synthetic mixing.
- `data/metadata/full_manifest.csv`: 4,494 FSD50K + 188 AudioSet target-class rows, 905 DESED background rows.
- Baseline: about 171 AudioSet clips, mAP 0.618 (handoff figure). `docs/evaluation_report.md` currently shows macro AP 0.783 from a later small-sample run; confirm which is current before quoting either.
- Pending: full retrain on the combined ~4,655 clips (Phase 2–3 result).
- Not started: Flutter app, Kotlin service, ESP32 firmware, personalization.
- Not yet collected: self-recorded Indian ambient dataset, personalization dev-set.
- Next milestone: 50–60% completion review. Current roadmap phase: TODO.

## Working with Claude

- Give complete, ready-to-use changes. No extras beyond what was asked.
- Lead with the shortest viable version; expand only on request.
- Report status honestly: say what ran, what passed, and what was not checked.
- For any training result, quote the run folder and metric file, not memory.
