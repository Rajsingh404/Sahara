# SAHARA app

Flutter UI (Android only) with a native Kotlin foreground service that listens continuously,
runs the TFLite model on the phone, and raises visual, vibration and wristband alerts.
Nothing is recorded or uploaded: audio lives in a 1 s in-memory ring buffer, and release
builds have no `INTERNET` permission.

## Run it (macOS)

Needs Flutter stable 3.47+ and Android Studio (Android SDK, platform tools). Then, from the repo root:

```bash
# 1. Put the model into the app (after training + src/export/to_tflite.py [+ quantize.py])
python app/tool/export_yamnet_tflite.py              # once: YAMNet backbone -> models/exported/yamnet.tflite
python app/tool/sync_assets.py --yamnet models/exported/yamnet.tflite

# 2. Build and run on a phone with USB debugging on
cd app
flutter pub get
flutter analyze
flutter test
flutter run
```

Kotlin unit tests (detection policy, BLE packets, audio window): `cd app/android && ./gradlew :app:testDebugUnitTest`.

Without the `.tflite` files the app still opens, says the model is missing, and the
**Settings → Test alerts** chips fire the whole alert path (full-screen alert, vibration, wristband).

## How it works

```
mic (AudioRecord 16 kHz mono PCM16)
  → AudioWindow: 1.0 s window, 0.5 s hop            (INFERENCE_WINDOW_SEC / INFERENCE_HOP_SEC)
  → VAD: skip windows quieter than 1e-4 mean square  (VAD_ENERGY_THRESHOLD)
  → InferenceEngine: yamnet.tflite (0.975 s → 1024-d embedding), mean of last 2 embeddings
                     → sahara_classifier.tflite (Dense head, FP32 or INT8) → 8 sigmoid scores
  → DetectionPolicy: per-class threshold × sensitivity, 2-of-3 windows (1 hit for smoke alarm,
                     siren, glass break), 10 s cooldown per class
  → AlertDispatcher: history, high-priority notification + full-screen intent, per-class
                     vibration, BLE write to the wristband, event to the Flutter UI
```

Everything from the mic to the alert runs in `AudioCaptureService` (foreground service, type
`microphone`, partial wake lock), so it keeps working with the screen off and the UI closed.
Dart owns the screens, settings and permission prompts and talks to the service over
`sahara/control` (methods) and `sahara/events` (stream). That keeps more in Kotlin than the
"minimal service" note in CLAUDE.md, because a Dart isolate is not alive to raise alerts when the
screen is off.

If `sahara_classifier.tflite` already takes raw audio (a combined YAMNet + head export), the engine
detects that from its input shape and `yamnet.tflite` is not needed.

| Path | What |
|---|---|
| `assets/labels.txt` | class order, generated from `SOUND_CLASSES` in `src/config.py` by `tool/sync_assets.py` |
| `assets/class_thresholds.json` | copy of `data/metadata/class_thresholds.json` |
| `assets/*.tflite` | model files, copied in by `tool/sync_assets.py` (not committed until final) |
| `lib/audio/sahara_service.dart` | platform-channel client and event types |
| `lib/app_state.dart` | UI state mirrored from the service |
| `lib/ui/` | home (start/stop, live scores, per-sound switches), settings (sensitivity, wristband pairing, test alerts), history |
| `lib/alerts/alert_screen.dart` | flashing full-screen alert |
| `lib/ble/uuids.dart` | wristband UUIDs, see `docs/ble_protocol.md` |
| `android/app/src/main/kotlin/com/sahara/app/` | `AudioCaptureService`, `InferenceEngine`, `SaharaChannel`, `Wristband` (GATT client), `AlertNotifier`, `core/` (pure-Kotlin logic with JVM tests) |
| `tool/` | `sync_assets.py`, `export_yamnet_tflite.py` |

## Permissions

Microphone (required), notifications (Android 13+), Bluetooth scan/connect (Android 12+) or
location (Android 11 and below, scanning only). Android does not allow a microphone foreground
service to start at boot, so after a reboot the user opens the app and taps Start.

## Not done yet

- Personalization (few-shot enrollment) — `lib/personalization/` is reserved for it.
- Battery, latency and false-alarm numbers on a real phone (the CLAUDE.md TODOs).
