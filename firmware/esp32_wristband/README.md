# SAHARA wristband firmware

ESP32 PlatformIO project (not written yet): BLE peripheral that receives a class id from the phone and plays a haptic pattern per class.

- `include/ble_uuids.h` — service and characteristic UUIDs, mirrored in `app/lib/ble/uuids.dart`.
- `src/main.cpp` — BLE peripheral + haptic patterns.
- `hardware/` — BOM (under ₹1000), schematic, enclosure.
