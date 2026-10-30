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
| Personalization | Name calls: pretrained speech embeddings (openWakeWord) + a small "Aryan" head, and a pretrained speaker encoder (Resemblyzer GE2E, run in pure Kotlin) for enrolled callers. Custom household sounds: few-shot prototypes over YAMNet window vectors (`core/CustomSoundMatcher.kt`, experimental). |
| Open set | YAMNet's own 521 AudioSet classes as low-priority "other sounds" when none of the 8 fires. |
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
  evaluation/           metrics, policy.py (port of DetectionPolicy.kt), policy_tuning/tune_policy (alert settings
                        against a false-alarm budget), evaluate.py, false_alarm_eval.py, team_cv.py
  export/               to_tflite.py, quantize.py, validate_tflite.py
  personalization/      custom sounds: prototypes.py, evaluate.py, export_center.py;
                        name calls + caller ID: speech_embedding.py, name_head.py, speaker.py (docs/name_call.md)
scripts/                pipeline runners, env check, status
tests/                  pytest suite for the ML pipeline
models/                 checkpoints/, logs/, exported/ (contents git-ignored)
data/
  metadata/             manifests (train/val/test/full), class_thresholds.json — committed
  raw/, interim/, processed/   local only, git-ignored
docs/                   dataset card, evaluation report, roadmap, architecture, reviews, paper

app/                    Flutter app (Android)
  assets/               yamnet.tflite, sahara_classifier.tflite, labels.txt, class_thresholds.json,
                        personal_center.bin (copied by app/tool/sync_assets.py)
  lib/                  audio/, detection/, alerts/, ble/ (uuids.dart), personalization/, ui/
  android/app/src/main/kotlin/…   AudioCaptureService.kt, InferenceEngine.kt, SaharaChannel.kt, CustomEnrolment.kt,
                                  core/ (DetectionPolicy, EmbeddingPool, AudioWindow, CustomSoundMatcher, BlePacket)
firmware/esp32_wristband/   PlatformIO project: include/ble_uuids.h, src/main.cpp, hardware/ (BOM, schematic)
android/                empty legacy placeholder; native code lives in app/android/
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
pip install -r requirements-dev.txt
brew install ffmpeg sox
python scripts/verify_env.py

# Optional: stream datasets from Google Drive for Desktop instead of data/raw/
export SAHARA_DRIVE_DATASETS="<path to My Drive/SAHARA/Datasets>"

python src/data/download.py --dataset esc50           # also: fsd50k, audioset, desed, all
python -m src.data.download_audioset_hf --classes smoke_alarm --splits eval bal_train   # AudioSet without YouTube
python scripts/run_full_pipeline.py                   # data → synthetic → features → train → eval → export → report
python scripts/run_full_pipeline.py --from train      # reuse data and features; --sync-assets copies models to the app
python src/personalization/evaluate.py                # custom-sound (personalization) evaluation
pytest -q

# Name calls (docs/name_call.md): extra deps, then evaluate + export app/assets/name_head.json, speaker_encoder.bin
pip install -r requirements-personalization.txt
python scripts/eval_name_call.py --wav-dir data/interim/name_wav --report docs/name_call_eval.md --export app/assets
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
- Per-class alert settings (threshold + windows needed) come from `data/metadata/class_thresholds.json`, written by
  `src/evaluation/tune_policy.py` and copied to `app/assets/`.
- Minimum Android version: API 26 (Android 8.0), `minSdk` in `app/android/app/build.gradle.kts`. Tested on API 36.
- Test phone: Samsung Galaxy S24 (SM-S921B), Android 16, measured 2026-10-05 (`models/logs/phone/`).
- Model size: no budget was set. Shipped: `yamnet.tflite` 15.0 MB (embeddings + 521 scores) + FP32 head 1.2 MB + speech models 8.1 MB; arm64 release APK 45.8 MB.
  The INT8 head (0.3 MB) passes validation but is not shipped (thresholds sit at 0.93–0.98, see the report).
- Latency: no budget was set. Measured inference 20–29 ms per window (screen on), 47–49 ms (screen off) on the S24;
  sound start → phone alert 1.0–2.5 s in a speaker test. Phone → wristband buzz: not measured (no hardware).
- Battery per hour of listening: not measured yet (needs the phone unplugged for about an hour).
- False-alarm budget: 4 alerts per hour on validation background audio (`FALSE_ALARM_BUDGET_PER_HOUR`), chosen in
  Phase 4 because no target was given; test result 3.5/h on 3.4 h of busy background.
- ESP32 board, haptic motor, battery: not chosen/measured (no hardware yet); the BOM in
  `firmware/esp32_wristband/README.md` is the plan (about ₹870).
- Audio input contract follows YAMNet: 16 kHz mono float waveform, 0.96 s frames with 0.48 s hop. Do not change this without retraining.

## Dataset layout

Sources: FSD50K (Google Drive), AudioSet (YouTube via yt-dlp, and the Hugging Face mirror `agkphysics/AudioSet`
via `download_audioset_hf.py`), ESC-50, DESED (background), self-recorded team clips, and train-only synthetic mixes.
iNoise is supported but not downloaded (needs an IEEE DataPort login). Counts: `data/metadata/coverage.txt`.

- Raw audio comes from `data/raw/<dataset>/` or, when `SAHARA_DRIVE_DATASETS` is set, from Google Drive for Desktop (streamed, cold fetch about 2.3 s). Only the `.npy` embedding cache (about 4 KB per clip) is written locally.
- Synthetic mixing (`src/preprocessing/synthetic_mixing.py`) overlays clean clips on Indian background beds at varied SNR. Those rows are tagged `source_type: synthetic_mixed`.
- The pipeline is dataset-agnostic. Add a source with a downloader in `src/data/`, its entry in `get_raw_dir()` and `label_map.py`; the manifest builder picks it up.
- Self-recorded clips: `scripts/organise_recordings.py` builds `data/raw/indian_ambient/` from `data/recordings/label_map.csv`.
  The team set is split per label (~40 % test), spread across recording conditions, not per person.
- `source_type` values: `real`, `synthetic_mixed`, `self_recorded`.

**Recording protocol:** 4–6 s clips with ambient padding around the event, including naturally short sounds (knocking, appliance_beep). Do not trim to the event; YAMNet windowing handles it.

**Personalization dev-set:** 10–20 samples per enrolled sound, across 3–4 sounds.

## Current status (update as it changes)

- Branch `feat/pr5-name-calls` (on top of `feat/integration`; PR #5 merged: name calls, caller ID, other sounds); not pushed.
- Phases 0–5 and 7 done; Phase 6 mostly done; Phase 8 not started (no hardware); Phase 9 docs/CI done.
- Selected model: `models/logs/20261030-185320_a_real_s3/` (in `models/logs/selected_run.txt`; chosen on validation among
  runs trained on the current data). Public test mAP 0.778, false alarms 3.5/h, Indian test mAP 0.831, Indian 5-fold CV
  mAP 0.621 → 0.822 (329 clips; 0.600 with whole recording sessions held out: no gain in unseen homes). All numbers: `docs/evaluation_report.md` (generated, do not hand-edit).
- Superseded figures (0.618, 0.651, 0.783, 0.913) are listed in the report; never quote them.
- The classifier head always runs on the CPU (`classifier_head.HEAD_DEVICE`): Keras `predict()` on the Metal GPU
  returns wrong scores on this Mac.
- Pending on the phone: battery per hour, release-build memory/latency, real-sound tests, custom-sound and caller enrolment,
  live name calls.
- Custom sounds work for distinctive household sounds, not names; names use the separate name-call model
  (reproduced here: 79% of calls caught at EER, caller named 76%; `docs/name_call_eval.md`).
- BLE id 0x20 = name call (firmware plays its own pattern); other sounds never go over BLE.

## Working with Claude

- Give complete, ready-to-use changes. No extras beyond what was asked.
- Lead with the shortest viable version; expand only on request.
- Report status honestly: say what ran, what passed, and what was not checked.
- For any training result, quote the run folder and metric file, not memory.
