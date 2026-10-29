# SAHARA

**S**ound-based **A**lert & **H**aptic **A**ssistance for **R**eal-time **A**wareness: an Android app that listens
continuously, on the phone and offline, for eight safety sounds and alerts Deaf and Hard-of-Hearing users with a
full-screen alert, phone vibration and a custom ESP32 vibrating wristband. Focus: Indian homes and streets.

Final-year B.Tech capstone, NMIMS MPSTME.

**Sounds:** smoke alarm, doorbell, siren, knocking, dog bark, baby cry, glass breaking, appliance beep, plus
user-enrolled custom sounds (experimental), a quiet notice for other sounds YAMNet recognises (e.g. a pressure-cooker
whistle or a phone ringing), and **name calls**: someone calling the user's name, with who called if they enrolled.

## Status

| Part | State |
|---|---|
| ML pipeline (`src/`, `scripts/`) | Done: verified data, phone-exact features, recorded runs, policy-aware tuning, evaluation, export |
| Android app (`app/`) | Done and running on a Galaxy S24; battery and real-sound checks still pending |
| Personalization (`src/personalization/`, `app/lib/personalization/`) | Done: evaluated; offered as experimental for household sounds (it cannot tell names apart) |
| Name calls and caller ID (`app/android/.../speech/`, `scripts/eval_name_call.py`) | Done: evaluated on team recordings (79% of calls caught at the equal-error threshold; enrolled caller named 76%); not yet tried live on the phone |
| Other sounds (`OtherSoundsPolicy.kt`) | Done: quiet notice from YAMNet's own 521 classes, no vibration or wristband |
| Wristband firmware (`firmware/esp32_wristband/`) | Builds; **not tested on hardware** (no parts yet) |

Results, with the run folder each number comes from: [docs/evaluation_report.md](docs/evaluation_report.md).
Headline (test data, run `20261005-195620_b_synthetic`): public test mAP 0.789; false alarms on 3.4 h of busy
background audio 3.5 per hour; smoke alarm 23/27 and siren 28/34 test clips alerted; Indian recordings (168 clips) 5-fold
cross-validation mAP 0.797 without them in training, 0.840 with them.

## How it works

Microphone → 1 s buffer every 0.5 s → YAMNet (TFLite) → small Dense head → per-class alert rules → full-screen alert,
vibration and wristband. YAMNet's own scores also drive the quiet "other sounds" notice, and when they say speech, a
separate speech model listens for the user's name and a speaker encoder names the caller. All on the phone; audio is never stored or uploaded. Diagrams and the shared contracts:
[docs/architecture.md](docs/architecture.md).

## Repository

```
src/          Python ML package: data/, preprocessing/, training/, evaluation/, export/, personalization/
scripts/      run_full_pipeline.py, compare_runs.py, write_evaluation_report.py, organise_recordings.py, ...
tests/        pytest suite
app/          Flutter app (Android): lib/ (UI), android/ (Kotlin service, inference, BLE), assets/ (models), tool/
firmware/     ESP32 wristband (PlatformIO)
data/         metadata/ (manifests, thresholds, coverage) and recordings/ (team label map); audio stays local
models/logs/  one folder per training run (config, metrics, test results); models themselves are not in git
docs/         architecture, dataset card, evaluation report, BLE protocol, reviews
```

## One command per part

macOS (Apple Silicon), from the repo root.

```bash
# ML: data -> features -> train -> evaluate -> export -> report
/opt/homebrew/bin/python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt && brew install ffmpeg sox
export SAHARA_DRIVE_DATASETS="$HOME/Library/CloudStorage/GoogleDrive-<account>/My Drive/SAHARA/Datasets"   # FSD50K
python scripts/run_full_pipeline.py            # add --sync-assets to copy the final models into app/assets/
pytest -q

# App (Flutter stable >= 3.47, JDK 17, Android SDK 35+)
cd app && flutter pub get && flutter analyze && flutter test
flutter build apk --release --split-per-abi     # or: flutter run (USB debugging on)

# Wristband (brew install platformio)
cd firmware/esp32_wristband && pio run          # pio run -t upload to flash
```

Data sources and how to fetch each (FSD50K on Google Drive, AudioSet via `src/data/download_audioset_hf.py`,
ESC-50, DESED, team recordings via `scripts/organise_recordings.py`): [docs/dataset_card.md](docs/dataset_card.md).

## Privacy

Detection runs entirely on the phone. Audio lives about one second in memory and is overwritten; it is never written
to storage, and the release app has no internet permission. The wristband only receives which sound was detected.
Custom-sound enrolment keeps a short list of numbers per sound, not audio.

## Known limits

- Knocking, appliance beeps and doorbells are the weakest classes; the team has no recordings of smoke alarms,
  sirens or glass breaking, so the Indian-context check covers five classes.
- Custom sounds cannot recognise names or words; name calls use a separate model trained for one name ("Aryan") and
  confuse similar names (Arya, Aryanshi) about a third of the time.
- The speech models (openWakeWord) are CC BY-NC-SA 4.0: fine for this academic project, not for commercial use.
- Battery use, real-sound tests on the phone and the wristband on hardware are not measured yet.

Full list: [docs/evaluation_report.md](docs/evaluation_report.md#known-limitations).
