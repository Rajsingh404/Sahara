package com.sahara.app.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class AudioWindowTest {
    @Test
    fun keepsLatestSamplesInOrder() {
        val w = AudioWindow(4)
        w.push(shortArrayOf(1, 2, 3))
        assertFalse(w.isFull)
        w.push(shortArrayOf(4, 5, 6))
        assertTrue(w.isFull)
        val out = w.snapshot().map { Math.round(it * 32768f) }
        assertEquals(listOf(3, 4, 5, 6), out)
    }

    @Test
    fun scalesPcm16ToUnitRange() {
        val w = AudioWindow(2)
        w.push(shortArrayOf(Short.MIN_VALUE, 16384))
        val out = w.snapshot()
        assertEquals(-1f, out[0], 0f)
        assertEquals(0.5f, out[1], 0f)
    }

    @Test
    fun hopSizedPushesMatchServiceUsage() {
        val w = AudioWindow(ModelConfig.WINDOW_SAMPLES)
        w.push(ShortArray(ModelConfig.HOP_SAMPLES) { 100 })
        assertFalse(w.isFull)
        w.push(ShortArray(ModelConfig.HOP_SAMPLES) { 200 })
        assertTrue(w.isFull)
        val s = w.snapshot()
        assertEquals(100f / 32768f, s.first(), 0f)
        assertEquals(200f / 32768f, s.last(), 0f)
    }

    @Test
    fun meanSquareEnergy() {
        assertEquals(0.25f, AudioWindow.meanSquare(floatArrayOf(0.5f, -0.5f)), 1e-7f)
        assertEquals(0f, AudioWindow.meanSquare(floatArrayOf()), 0f)
        assertEquals(1f, AudioWindow.meanSquare(floatArrayOf(0f, 1f), 1, 2), 0f)
    }
}
