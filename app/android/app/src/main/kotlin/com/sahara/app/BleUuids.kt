package com.sahara.app

import java.util.UUID

/** Mirrors app/lib/ble/uuids.dart and firmware/esp32_wristband/include/ble_uuids.h. */
object BleUuids {
    const val DEVICE_NAME_PREFIX = "SAHARA"
    val SERVICE: UUID = UUID.fromString("c7a10000-5a5a-4b3d-9a8e-5348415241a1")
    val ALERT_CHAR: UUID = UUID.fromString("c7a10001-5a5a-4b3d-9a8e-5348415241a1")
    val STATUS_CHAR: UUID = UUID.fromString("c7a10002-5a5a-4b3d-9a8e-5348415241a1")
    val CCCD: UUID = UUID.fromString("00002902-0000-1000-8000-00805f9b34fb")
}
