package com.sahara.app.core

import kotlin.math.roundToInt

/** Encodes alert-characteristic writes. See docs/ble_protocol.md. */
object BlePacket {
    const val VERSION: Byte = 0x01
    const val CMD_ALERT: Byte = 0x01
    const val CMD_TEST: Byte = 0x02
    const val CMD_STOP: Byte = 0x03
    const val NO_CLASS: Int = 0xFF

    fun alert(classId: Int, confidence: Float, safetyCritical: Boolean): ByteArray {
        require(classId in 0..254) { "class id out of range: $classId" }
        val intensity = if (safetyCritical) 255 else (confidence * 255f).roundToInt().coerceIn(128, 255)
        return bytes(CMD_ALERT, classId, intensity)
    }

    fun test(intensity: Int = 200): ByteArray = bytes(CMD_TEST, NO_CLASS, intensity.coerceIn(0, 255))

    fun stop(): ByteArray = bytes(CMD_STOP, NO_CLASS, 0)

    private fun bytes(cmd: Byte, classId: Int, intensity: Int) =
        byteArrayOf(VERSION, cmd, classId.toByte(), intensity.toByte())

    data class Status(val batteryPct: Int?, val charging: Boolean, val motorBusy: Boolean, val fwMinor: Int)

    /** Parses a status-characteristic value; null if it is not a v1 packet. */
    fun parseStatus(value: ByteArray?): Status? {
        if (value == null || value.size < 4 || value[0] != VERSION) return null
        val battery = value[1].toInt() and 0xFF
        val flags = value[2].toInt() and 0xFF
        return Status(
            batteryPct = if (battery == 0xFF) null else battery.coerceAtMost(100),
            charging = flags and 0x01 != 0,
            motorBusy = flags and 0x02 != 0,
            fwMinor = value[3].toInt() and 0xFF,
        )
    }
}
