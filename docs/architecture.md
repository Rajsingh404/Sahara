# SAHARA architecture

SAHARA has three parts that share a few fixed contracts: the **ML pipeline** (Python, offline, on a Mac), the
**phone app** (Flutter UI + native Kotlin service, Android), and the **wristband** (ESP32 firmware). Everything that
touches sound runs on the phone; nothing is uploaded, and the release app has no internet permission.

## 1. On the phone: from microphone to alert

```mermaid
flowchart LR
    mic["Microphone<br/>AudioRecord 16 kHz mono<br/>(VOICE_RECOGNITION; UNPROCESSED if supported)"]
    ring["1 s ring buffer<br/>new 0.5 s hop each step<br/>(AudioWindow.kt)"]
    vad{"VAD: newest hop<br/>mean square ≥ 1e-4?"}
    yam["yamnet.tflite<br/>newest 15,600 samples<br/>→ 1,024-d embedding"]
    pool["Embedding pool<br/>mean of last 2 non-silent<br/>embeddings (EmbeddingPool.kt)"]
    head["sahara_classifier.tflite<br/>Dense 256→128→8 sigmoid"]
    policy["DetectionPolicy.kt<br/>per-class threshold + hits (1–3 of 3)<br/>10 s cooldown, sensitivity slider"]
    custom["CustomSoundMatcher.kt<br/>cosine vs enrolled prototypes"]
    other["OtherSoundsPolicy.kt<br/>YAMNet's 521 scores ≥ 0.35,<br/>2 of 3 windows → quiet notice"]
    name["NameSpotter.kt<br/>speech score ≥ 0.2 → openWakeWord<br/>embeddings → 'Aryan' head"]
    who["SpeakerEncoder.kt + PrototypeBank<br/>GE2E, cosine ≥ 0.70 → caller,<br/>else 'Someone'"]
    alert["AlertDispatcher.kt"]
    screen["Full-screen alert +<br/>phone vibration (AlertNotifier)"]
    band["Wristband over BLE<br/>(Wristband.kt)"]
    hist["History in SharedPreferences<br/>(label, score, time only)"]

    mic --> ring --> vad
    vad -- "silent: zero scores" --> policy
    vad -- "sound" --> yam --> pool --> head --> policy --> alert
    pool --> custom --> alert
    yam -- "521 scores" --> other --> alert
    yam -- "speech" --> name --> who --> alert
    alert --> screen
    alert --> band
    alert --> hist
```

- **Foreground service** (`AudioCaptureService.kt`): keeps capturing with the screen off and after the app is swiped
  from recents, with a persistent notification. All detection runs in Kotlin; the Flutter UI only displays state and
  settings over the platform channel (`SaharaChannel.kt` ↔ `lib/audio/sahara_service.dart`), so detection works with
  no Dart isolate alive.
- **Audio is never stored.** The ring buffer is overwritten every 0.5 s; enrolment samples are zeroed right after use.
  App storage holds only settings, alert history and custom-sound prototypes (checked on the phone, see
  `docs/evaluation_report.md`).
- **Cost** on a Galaxy S24: 20–29 ms per window with the screen on, 47–49 ms with it off (budget: 500 ms).

## 2. Offline: training and export

```mermaid
flowchart TB
    subgraph sources["Data sources (audio never in git)"]
        fsd["FSD50K<br/>(Google Drive, streamed)"]
        as["AudioSet<br/>(YouTube + Hugging Face mirror)"]
        esc["ESC-50"]
        desed["DESED"]
        team["Team recordings<br/>(Indian context)"]
    end
    lm["label_map.py<br/>verified label ids"]
    man["manifest_builder.py<br/>split by upstream recording;<br/>team clips split per label"]
    syn["synthetic_mixing.py<br/>train-only mixes, −5…10 dB"]
    cache["build_features_cache.py<br/>per-hop YAMNet cache, exactly as the phone<br/>(parity test: max diff 0.0)"]
    win["windowing.py<br/>phone window vectors;<br/>event windows for weak labels"]
    train["train.py (CPU)<br/>background ×8, team ×10,<br/>class-balanced weights"]
    sel["compare_runs.py<br/>select on validation policy score"]
    tune["tune_policy.py<br/>threshold + hits per class<br/>at 4 false alarms/h (validation)"]
    eval["evaluate.py / false_alarm_eval.py / team_cv.py<br/>test: public · Indian · synthetic"]
    exp["to_tflite.py / quantize.py / validate_tflite.py<br/>+ export_yamnet_tflite.py"]
    assets["app/assets/<br/>yamnet.tflite · sahara_classifier.tflite<br/>labels.txt · class_thresholds.json · personal_center.bin"]
    chk["check_app_pipeline.py<br/>replay through the app's exact path<br/>(90/90 same alerts)"]

    sources --> man
    lm --> man
    man --> syn --> man
    man --> cache --> win --> train --> sel --> tune --> eval
    sel --> exp --> assets --> chk
    tune --> assets
```

One command reproduces it: `python scripts/run_full_pipeline.py` (data → synthetic → features → train → eval →
export → report); `docs/evaluation_report.md` is generated from the selected run's files.

## 3. Personalization: enrolling a custom sound

```mermaid
flowchart LR
    rec["User records the sound<br/>5–10 × 3 s (in memory)"]
    wv["Same window loop as listening<br/>(InferenceEngine.enrolmentVectors)"]
    top["Keep the 2 loudest windows<br/>per sample; zero the audio"]
    proto["Prototype = mean of centred,<br/>normalised windows (1,024 floats)"]
    store["SharedPreferences<br/>(name + prototype)"]
    live["Live listening:<br/>cosine(window − center, prototype) ≥ 0.73<br/>→ alert 'custom:&lt;name&gt;'"]
    rec --> wv --> top --> proto --> store --> live
```

## 4. Name calls and caller ID

A separate speech path, used only when YAMNet's speech score is at least 0.2: openWakeWord's melspectrogram and
embedding models (pretrained) turn the last 2 s into speech embeddings, and a small logistic head scores the
name "Aryan" for any voice. When it fires, a pretrained GE2E speaker encoder (Resemblyzer weights, run in plain Kotlin)
compares the voice with the callers the family enrolled (a few taps of them saying the name); a match at cosine ≥ 0.70
names the caller, otherwise the alert says "Someone". The alert goes to the screen, vibration and the wristband
(BLE id 0x20, its own pattern). Results: `docs/name_call_eval.md`; design: `docs/name_call.md`.

## 5. Custom sounds: evaluation

Evaluated in `src/personalization/` (see the report): it separates distinctive household sounds from background, but
cannot tell names apart, so the app offers it as experimental and for household sounds only. On the wristband a custom
sound uses the doorbell pattern; BLE protocol v1 has no custom-sound id.

## Contracts shared across the three parts

| Contract | Value | Defined in | Mirrored in |
|---|---|---|---|
| Class order | `smoke_alarm, doorbell, siren, knocking, dog_bark, baby_cry, glass_break, appliance_beep` | `SOUND_CLASSES`, `src/config.py` | `app/assets/labels.txt` (generated), `firmware/.../sound_classes.h` (build-checked) |
| Audio input | 16 kHz mono, YAMNet patch 15,600 samples (0.975 s), 0.96 s frames / 0.48 s hop inside YAMNet | `src/config.py` | `ModelConfig.kt` |
| Inference cadence | 1 s buffer, 0.5 s hop, VAD 1e-4 on the newest hop, mean of last 2 non-silent embeddings | `src/config.py`, `windowing.py` | `ModelConfig.kt`, `AudioWindow.kt`, `EmbeddingPool.kt` |
| Alert policy | threshold + hits (1–3 of 3) per class, 10 s cooldown, sensitivity ±2 log-odds | `class_thresholds.json`, `src/evaluation/policy.py` | `DetectionPolicy.kt` (shared fixture test) |
| BLE | service `c7a10000-…`, 4-byte alert `[ver, cmd, class_id, intensity]`; class ids 0–7, name call 0x20; other sounds are never sent | `docs/ble_protocol.md` | `uuids.dart`, `BleUuids.kt`, `ble_uuids.h`, `sound_classes.h` |
| YAMNet export | outputs `embeddings` (1,024) and `scores` (521), class order of `yamnet_class_map.csv` | `app/tool/export_yamnet_tflite.py` | `InferenceEngine.kt`, `other_sounds.json` |
