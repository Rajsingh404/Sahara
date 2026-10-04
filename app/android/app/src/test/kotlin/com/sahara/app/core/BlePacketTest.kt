package com.sahara.app.core

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/** Byte layouts must match docs/ble_protocol.md. */
class BlePacketTest {
    private fun hex(b: ByteArray) = b.joinToString(" ") { "%02X".format(it) }

    @Test
    fun safetyAlertUsesFullIntensity() = assertEquals("01 01 02 FF", hex(BlePacket.alert(2, 0.4f, true)))

    @Test
    fun ordinaryAlertScalesWithConfidence() {
        assertEquals("01 01 01 FF", hex(BlePacket.alert(1, 1.0f, false)))
        assertEquals("01 01 01 80", hex(BlePacket.alert(1, 0.1f, false))) // floor 128
        assertEquals("01 01 03 BF", hex(BlePacket.alert(3, 0.75f, false)))
    }

    @Test
    fun testAndStopPackets() {
        assertEquals("01 02 FF C8", hex(BlePacket.test()))
        assertEquals("01 03 FF 00", hex(BlePacket.stop()))
    }

    @Test
    fun parsesStatus() {
        val s = BlePacket.parseStatus(byteArrayOf(0x01, 87, 0x03, 4))!!
        assertEquals(87, s.batteryPct)
        assertEquals(true, s.charging)
        assertEquals(true, s.motorBusy)
        assertEquals(4, s.fwMinor)
        assertNull(BlePacket.parseStatus(byteArrayOf(0x01, 0xFF.toByte(), 0, 0))!!.batteryPct)
    }

    @Test
    fun rejectsUnknownStatus() {
        assertNull(BlePacket.parseStatus(byteArrayOf(0x02, 50, 0, 0)))
        assertNull(BlePacket.parseStatus(byteArrayOf(0x01, 50)))
        assertNull(BlePacket.parseStatus(null))
    }

    @Test
    fun packetsAreFourBytes() {
        assertArrayEquals(byteArrayOf(1, 1, 7, -1), BlePacket.alert(7, 0.5f, true))
    }
}
