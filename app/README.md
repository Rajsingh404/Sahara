# SAHARA app

Flutter (Android only). Not scaffolded yet; the app PR runs `flutter create` here and keeps this layout.

- `assets/` — `sahara_classifier.tflite` and `labels.txt`, written by the ML export step.
- `lib/audio/` — platform-channel client for the Kotlin foreground service.
- `lib/detection/` — thresholds, debouncing, event model.
- `lib/alerts/` — visual alerts, flash, notifications.
- `lib/ble/` — wristband scan / connect / send; `uuids.dart` mirrors `firmware/esp32_wristband/include/ble_uuids.h`.
- `lib/personalization/` — enrollment UI and prototype store.
- `lib/ui/` — screens and widgets.
- `android/app/src/main/kotlin/…` — `AudioCaptureService.kt`, `InferenceEngine.kt`, `SaharaChannel.kt`.
