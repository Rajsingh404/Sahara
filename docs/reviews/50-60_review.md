# SAHARA: 50–60 % completion review

**Project:** SAHARA (Sound-based Alert & Haptic Assistance for Real-time Awareness), B.Tech capstone, NMIMS MPSTME
**Team:** Advika, Mahi, Prachi, Raj · **Guide:** _to confirm (Prof. Priti Rumao / Dr. Kanikar)_
**Code:** branch `feat/integration` · **All numbers:** [`docs/evaluation_report.md`](../evaluation_report.md), generated from
the run folders named below.

## 1. What SAHARA does

An Android app that listens continuously, entirely on the phone, and alerts Deaf and Hard-of-Hearing users to eight
safety sounds (smoke alarm, doorbell, siren, knocking, dog bark, baby cry, glass breaking, appliance beep) with a
full-screen alert, vibration and a custom ESP32 wristband. Audio is never stored or uploaded; the release app has no
internet permission. Architecture: [`docs/architecture.md`](../architecture.md).

## 2. Done so far

| Area | What is in place | Evidence |
|---|---|---|
| Data | 8 classes, each ≥ 150 real clips and ≥ 20 in test (smoke alarm 183/27 … dog bark 597/94); 12,425 background clips; 123 team recordings from Indian homes | `data/metadata/coverage.txt`, [`docs/dataset_card.md`](../dataset_card.md) |
| Data quality | Wrong label ids fixed (e.g. AudioSet "glass" was *Whoop*; 64 of 188 old clips mislabelled); splits with no recording on two sides | dataset card, `tests/test_dataset_split.py` |
| Model | Frozen YAMNet + small Dense head, trained on exactly the 1 s windows the phone computes (parity test: max difference 0.0) | `tests/test_windowing.py` |
| Alert rules | Per-class threshold and "windows needed", tuned on validation to ≤ 4 false alarms/hour | `data/metadata/class_thresholds.json` |
| App | Flutter UI + Kotlin foreground service, TFLite on device, full-screen alerts, vibration, BLE client, history, settings, custom sounds | `app/`; 15 Flutter + 34 Kotlin tests |
| On a real phone | Galaxy S24, Android 16: listens with the screen off and after being swiped away; no audio files stored; 20–49 ms per window | `models/logs/phone/20261005_galaxy_s24.json` |
| Personalization | Few-shot custom sounds, evaluated; offered as experimental | `models/logs/personalization/` |
| Name calls | "Aryan" spotted in any voice; enrolled caller named, else "Someone"; own wristband pattern | `docs/name_call_eval.md` |
| Other sounds | Quiet notice for sounds YAMNet knows outside the eight (cooker whistle, phone ringing...) | `OtherSoundsPolicy.kt` |
| Engineering | One-command ML pipeline, every training run recorded, CI for Python, Flutter, Kotlin and firmware | `scripts/run_full_pipeline.py`, `.github/workflows/ci.yml` |

## 3. Measured results

Model `models/logs/20261005-195620_b_synthetic/`, selected on validation only; test data used once.

| Measure | Result |
|---|---|
| Public test set mAP (FSD50K, AudioSet, ESC-50) | **0.789** |
| False alarms on 3.4 h of busy background audio | **3.5 per hour** |
| Alert tuning, same model, all 4.4 h of test audio | false alarms 43.7 → **6.1 per hour**; share of clips alerted 0.67 → 0.59 |
| Test clips that raise the right alert | smoke alarm 23/27, siren 28/34, dog bark 76/90, baby cry 19/25, glass 47/97, doorbell 17/35, appliance beep 20/78, knocking 19/73 |
| Indian team recordings, 5-fold cross-validation | mAP 0.721 without team data → **0.757** with it |
| Phone app vs evaluated model | same alerts on 90/90 test clips (`app/tool/check_app_pipeline.py`) |
| Model size on the phone | YAMNet 15.0 MB + head 1.2 MB + name-call models 8.1 MB; arm64 APK 45.8 MB |
| Inference on Galaxy S24 | 20–29 ms per window (screen on), 47–49 ms (screen off); one window every 500 ms |
| Custom sounds | separates distinctive household sounds from background (toilet flush perfectly); cannot tell names apart |
| Name calls (team recordings, leave one recorder out) | AUC 0.895 vs background, 0.719 vs similar names; 76% of calls caught at the equal-error threshold |
| Caller ID | enrolled caller named 29/38 = 76%; unenrolled shown as "Someone" 82% at cosine 0.70 |

Earlier figures (mAP 0.618, 0.651, 0.783, 0.913) are superseded and should not be quoted; the report explains why.

## 4. Demo script (about 5 minutes)

Setup: release APK installed, phone unlocked, laptop volume about 70 %, clips ready from the held-out test set
(`afplay` on macOS or any player). Keep the phone within 30 cm of the laptop speaker.

1. **Privacy first (30 s).** Settings → Privacy card: audio stays on the phone; show there is no internet permission.
2. **Start listening (30 s).** Tap *Start listening*; point out the live score bars and the persistent notification.
3. **Screen off (1 min).** Lock the phone. Play a **smoke alarm** clip → the screen wakes with the full-screen red
   alert and vibration. Dismiss.
4. **Siren (30 s).** Play a **siren** clip → safety alert. (Smoke alarm and siren alerted in every phone test.)
5. **Alert path for the other sounds (1 min).** Settings → *Test alerts* → tap *Doorbell* and *Knocking*: the same
   alert, vibration pattern and (if paired) wristband buzz, without sound. Use a **real** knock on a door or a real
   doorbell for a live test: laptop speakers barely reproduce knocks, barks and doorbells.
6. **History (30 s).** Open the history screen: label, confidence and time only.
7. **Custom sound (1 min, rehearse first).** Settings → Custom sounds → add a distinctive sound (a bell or whistle),
   record it 5 times, play it again. Explain it suits household sounds, not names.
8. **Name call (1 min, rehearse first).** Settings → People who call you → enrol one teammate (a few taps of them saying
   "Aryan"); they call the name from across the room → alert naming them. Someone not enrolled → "Someone".
9. **Wristband.** Only if hardware is ready (see section 5); otherwise show the firmware build and the BLE protocol.

## 5. Known limitations and what is left

- **Weakest classes:** knocking, appliance beep and doorbell (lowest AP); limited by the model, not the alert rules.
- **Indian context:** no team recordings of smoke alarms, sirens or glass breaking; the team background caused a few
  "siren" alerts in minutes of audio (too little to give a rate).
- **Custom sounds** cannot recognise names; few-shot detection is modest at a strict false-alarm limit.
- **Name calls** are trained for one name and confuse similar names about a third of the time; caller ID was
  measured with each recorder on their own phone and room, so it may be optimistic. Not yet tried live on the phone.
- **Not measured yet:** battery use per hour of listening; release-build memory; real-sound tests on the phone.
- **Wristband (Phase 8):** firmware builds, but not tested on hardware (parts not yet available): latency to buzz,
  current draw and battery life are not measured.
- **Data not obtained:** iNoise (Indian noise database) needs an IEEE DataPort login.

**Remaining work:** phone battery and real-sound tests; wristband assembly and tests; more team recordings of the
weak and missing classes; then the final report and paper.
