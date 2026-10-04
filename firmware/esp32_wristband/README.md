# SAHARA wristband firmware

ESP32 BLE peripheral. The phone app sends a sound class id; the band plays a distinct vibration pattern and flashes its LED. The protocol is in [`docs/ble_protocol.md`](../../docs/ble_protocol.md); the UUIDs are in `include/ble_uuids.h`.

## Build and flash (macOS)

```bash
brew install platformio
cd firmware/esp32_wristband
pio run                 # build (first run downloads the ESP32 toolchain, ~500 MB)
pio run -t upload       # flash over USB
pio device monitor      # serial log at 115200
```

If the board does not show up as a serial port, install the USB-serial driver for its chip (CP210x or CH340, printed on the chip next to the USB socket).

The build fails if `include/sound_classes.h` and `SOUND_CLASSES` in `src/config.py` disagree (`scripts/check_classes.py`).

## Quick test without the app

1. On a bare DevKit with nothing on GPIO34, set `BATTERY_DIVIDER_FITTED 0` in `include/config.h` first; a floating ADC pin reads noise and can put the board to sleep.
2. Flash, open the serial monitor, and install **nRF Connect** on a phone.
3. Connect to `SAHARA-Band`, open the service `c7a10000-…`, and write hex to the Alert characteristic `c7a10001-…`:
   - `01 01 00 FF` smoke alarm, full strength
   - `01 01 03 C8` knocking
   - `01 02 FF C8` test buzz
   - `01 03 FF 00` stop
4. Enable notifications on Status `c7a10002-…` to see `[version, battery %, flags, fw]`.

Without a motor wired, the on-board LED shows the pattern.

## Behaviour

| Class (id) | Pattern |
|---|---|
| smoke_alarm (0) | Three long buzzes, the "temporal three" rhythm real fire alarms use. ×3 |
| doorbell (1) | Short strong, then longer soft: "ding-dong". ×2 |
| siren (2) | Rising and falling swell. ×3 |
| knocking (3) | Four quick taps. ×2 |
| dog_bark (4) | Two sharp pulses. ×2 |
| baby_cry (5) | Long, short, long wavering pulse. ×2 |
| glass_break (6) | One hard hit, then a flutter. ×2 |
| appliance_beep (7) | Three light ticks. ×1 |

Timings are in `src/patterns.cpp`. The app's `intensity` byte scales every step. A new alert from the phone interrupts the one playing.

Device cues:

- Boot, button press when idle: one short buzz.
- Connected to the phone: two quick taps.
- Phone disconnected: two long soft buzzes, so the user knows alerts have stopped.
- Button press during an alert: stops it.

Battery-aware idle:

- Not connected: advertises every 100 ms for 30 s, then every 1 s. The LED blinks for 20 ms every 5 s.
- CPU runs at 80 MHz, BLE TX at +3 dBm. The band asks the phone for a 30–50 ms connection interval with slave latency 2 (alert delay under about 150 ms).
- Battery is read every 30 s, only while the motor is off, so motor current does not drag the reading down.
- Low (≤ 20 %, leaves above 25 %): one low-battery buzz, motor at 70 %, LED off, TX 0 dBm, advertising every 2 s when disconnected.
- Critical (≤ 3.4 V on two readings in a row, not charging): one long buzz, then deep sleep. The button wakes it; with the battery still flat it blinks once and sleeps again.

All pins and thresholds are in `include/config.h`.

## Wiring (ESP32 DevKit V1)

```
LiPo + ──┬── TP4056 B+          TP4056 OUT+ ── switch ── LDO IN    LDO OUT (3.3 V) ── DevKit 3V3
         │                      TP4056 OUT- ─────────────────────── GND
         │
         ├── 100k ──┬── 100k ── GND
         │          └── GPIO34                     (battery sense)
         │
         └── motor + ;  motor - ── MOSFET drain    (1N4148 across the motor, cathode to +)
                                   MOSFET source ── GND
                                   MOSFET gate ── 100 Ω ── GPIO25,  100k gate ── GND

GPIO2  ── 330 Ω ── LED ── GND   (or use the on-board LED, already on GPIO2)
GPIO0  ── BOOT button on the DevKit (dismiss / wake)
GPIO27 ── 1N4148 anode;  cathode ── TP4056 CHRG    (optional, set CHARGE_DETECT_FITTED 1)
```

Notes:

- Do not feed the battery into the DevKit's 5V/VIN pin: its AMS1117 regulator drops about 1.1 V and the board browns out. Use a low-dropout 3.3 V regulator into the 3V3 pin. Turn the slide switch off before plugging in USB to flash, so the two regulators never drive 3V3 together.
- The motor runs straight from the cell through the MOSFET, not from the 3.3 V rail.
- The TP4056 module must be the version with protection (DW01 + 8205A), which cuts off an over-discharged cell.

## Bill of materials (under ₹1000)

Typical Indian online prices (Robu, Robocraze, Amazon.in), 2026. Prices vary; the total leaves headroom.

| Part | Qty | Approx. ₹ |
|---|---|---|
| ESP32 DevKit V1 (ESP32-WROOM-32, CP2102 or CH340) | 1 | 350 |
| 3.7 V LiPo, 500 mAh (e.g. 502540) | 1 | 180 |
| TP4056 USB-C charger module with protection | 1 | 35 |
| LDO regulator 3.3 V, ≥ 500 mA (AP2112K-3.3 / ME6211 / HT7833) | 1 | 20 |
| Coin vibration motor, 10 mm, 3 V (1027) | 1 | 40 |
| AO3400 logic-level N-MOSFET (or 2N7000) | 1 | 10 |
| 1N4148 diode | 2 | 4 |
| Resistors: 100k ×3, 100 Ω, 330 Ω | 5 | 5 |
| 5 mm LED (if not using the on-board one) | 1 | 2 |
| Slide switch | 1 | 10 |
| Perfboard, wires, header pins | — | 60 |
| Strap (velcro watch strap) and small enclosure / 3D print | 1 | 150 |
| **Total** | | **~₹870** |

The DevKit is the right board for development. For a smaller, lower-power final band, an ESP32-C3 SuperMini (~₹250) fits the same code with different pins in `include/config.h`, but it has not been built or tested here.

Expected battery life with the DevKit: the ESP32 radio plus the DevKit's USB-serial chip and regulator draw roughly 30–50 mA while connected, so a 500 mAh cell lasts about 10–15 hours. Measure it on the real build before quoting a number.
