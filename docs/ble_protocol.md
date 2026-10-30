# SAHARA wristband BLE protocol (v1)

Contract between the phone app (`app/`) and the ESP32 wristband (`firmware/esp32_wristband/`).
The phone is the GATT **central/client**; the wristband is the **peripheral/server**.
Change this file, `firmware/esp32_wristband/include/ble_uuids.h` and `app/lib/ble/uuids.dart`
(plus `app/android/.../BleUuids.kt`) in the same PR.

## Advertising

- Device name: `SAHARA-Band` (the app filters scan results on the `SAHARA` prefix **or** the service UUID).
- The service UUID below is included in the advertisement payload.

## GATT layout

| Item | UUID | Properties |
|---|---|---|
| SAHARA service | `c7a10000-5a5a-4b3d-9a8e-5348415241a1` | primary |
| Alert characteristic | `c7a10001-5a5a-4b3d-9a8e-5348415241a1` | Write, Write Without Response |
| Status characteristic | `c7a10002-5a5a-4b3d-9a8e-5348415241a1` | Read, Notify (CCCD `0x2902`) |

All multi-byte values are little-endian. Unknown versions or commands are ignored by the receiver.

## Alert characteristic (phone → band), 4 bytes

| Byte | Field | Values |
|---|---|---|
| 0 | `version` | `0x01` |
| 1 | `command` | `0x01` ALERT, `0x02` TEST, `0x03` STOP |
| 2 | `class_id` | `0..7` index into `SOUND_CLASSES` (`src/config.py`); `0x20` = name call; `0xFF` = none (TEST/STOP) |
| 3 | `intensity` | `0..255` motor strength. The app sends `255` for safety-critical classes, otherwise `round(confidence × 255)` clamped to `128..255` |

Class ids (fixed order, same as `src/config.py` and `app/assets/labels.txt`):

| id | class | safety-critical | haptic pattern (phone and band) |
|---|---|---|---|
| 0 | smoke_alarm | yes | Three long buzzes, the fire-alarm "temporal three" rhythm, ×3 |
| 1 | doorbell | no | Short strong, then longer soft ("ding-dong"), ×2 |
| 2 | siren | yes | Rising and falling swell, ×3 |
| 3 | knocking | no | Four quick taps, ×2 |
| 4 | dog_bark | no | Two sharp pulses, ×2 |
| 5 | baby_cry | no | Long, short, long, ×2 |
| 6 | glass_break | yes | One hard hit, then a flutter, ×2 |
| 7 | appliance_beep | no | Three light ticks, ×1 |
| 0x20 | name call ("someone is calling you", docs/name_call.md) | no | Two medium buzzes, then two short, ×2 |

"Other sounds" (YAMNet's open-set layer) never reach the band. Firmware that does not know `0x20` yet should play its
default non-critical pattern rather than ignore it.

- **ALERT**: play the pattern for `class_id` at `intensity`. A new ALERT interrupts the current pattern.
- **TEST**: one 300 ms pulse at `intensity` (used by the app's "Test wristband" button).
- **STOP**: stop any running pattern immediately (user dismissed the alert on the phone).

Example: siren alert = `01 01 02 FF`; test = `01 02 FF C8`; stop = `01 03 FF 00`.

Exact timings live in `firmware/esp32_wristband/src/patterns.cpp`; the phone plays the same patterns from
`app/android/.../core/HapticPatterns.kt`, and `tests/test_haptic_patterns.py` fails if the two differ. The app only sends ids.

## Status characteristic (band → phone), 4 bytes

| Byte | Field | Values |
|---|---|---|
| 0 | `version` | `0x01` |
| 1 | `battery_pct` | `0..100`, `0xFF` unknown |
| 2 | `flags` | bit0 = charging, bit1 = motor busy |
| 3 | `fw_minor` | firmware build number |

The band notifies on battery change (≥ 1 %) and on charging/busy transitions.

## Connection behaviour

- The app remembers the paired band's MAC address and reconnects automatically (the foreground service owns the connection so it survives the screen being off).
- The band accepts one central at a time and restarts advertising on disconnect.
- No pairing/bonding in v1 (no PIN). Writes carry no personal data: only a class id and an intensity.
- Positioning rule: no audio, embeddings or scores ever cross BLE.
