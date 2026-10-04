// BLE contract between the SAHARA app and the wristband.
// Mirrored in app/lib/ble/uuids.dart and app/android/.../BleUuids.kt, specified in
// docs/ble_protocol.md. Change all of them in the same PR.
#pragma once

#define SAHARA_DEVICE_NAME         "SAHARA-Band"

#define SAHARA_SERVICE_UUID        "c7a10000-5a5a-4b3d-9a8e-5348415241a1"
#define SAHARA_ALERT_CHAR_UUID     "c7a10001-5a5a-4b3d-9a8e-5348415241a1"  // Write, write without response.
#define SAHARA_STATUS_CHAR_UUID    "c7a10002-5a5a-4b3d-9a8e-5348415241a1"  // Read, notify.

#define SAHARA_PROTOCOL_VERSION    0x01

// Alert payload (phone -> band): [version, command, class_id, intensity]
#define ALERT_LEN                  4
#define CMD_ALERT                  0x01
#define CMD_TEST                   0x02
#define CMD_STOP                   0x03
#define CLASS_NONE                 0xFF

// Status payload (band -> phone): [version, battery_pct, flags, fw_minor]
#define STATUS_LEN                 4
#define STATUS_BATTERY_UNKNOWN     0xFF
#define STATUS_FLAG_CHARGING       0x01
#define STATUS_FLAG_MOTOR_BUSY     0x02
