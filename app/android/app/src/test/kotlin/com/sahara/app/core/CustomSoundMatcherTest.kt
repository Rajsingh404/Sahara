package com.sahara.app.core

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.random.Random

class CustomSoundMatcherTest {
    private val dim = 1024
    private fun noise(seed: Int, scale: Float = 1f) = Random(seed).let { r -> FloatArray(dim) { (r.nextFloat() - 0.5f) * scale } }
    private fun plus(a: FloatArray, b: FloatArray) = FloatArray(dim) { a[it] + b[it] }

    @Test
    fun enrolledSoundMatchesAndOthersDoNot() {
        val sound = noise(1, 2f)
        val samples = (0 until 5).map { plus(sound, noise(10 + it, 0.1f)) }
        val proto = CustomSoundMatcher.prototype(samples, center = null)
        val m = CustomSoundMatcher(center = null).apply { sounds = listOf(CustomSound("cooker", proto)) }
        assertEquals("cooker", m.match(plus(sound, noise(99, 0.1f)), 0).single().first.name)
        assertTrue(m.match(noise(500, 2f), 20_000).isEmpty())
    }

    @Test
    fun cooldownSuppressesRepeats() {
        val sound = noise(2, 2f)
        val m = CustomSoundMatcher(center = null).apply {
            sounds = listOf(CustomSound("bell", CustomSoundMatcher.prototype(listOf(sound), null)))
        }
        assertEquals(1, m.match(sound, 0).size)
        assertTrue(m.match(sound, 5_000).isEmpty())
        assertEquals(1, m.match(sound, 10_000).size)
    }

    @Test
    fun centeringRemovesTheSharedComponent() {
        val common = FloatArray(dim) { 10f }
        val a = plus(common, noise(3)); val b = plus(common, noise(4))
        val raw = CustomSoundMatcher(null).similarity(b, CustomSoundMatcher.prototype(listOf(a), null))
        val centred = CustomSoundMatcher(common).similarity(b, CustomSoundMatcher.prototype(listOf(a), common))
        assertTrue("raw $raw", raw > 0.9f)
        assertTrue("centred $centred", kotlin.math.abs(centred) < 0.2f)
    }

    @Test
    fun loudestKeepsTopWindowsInOrder() {
        val w = (0 until 5).map { FloatArray(dim) { _ -> it.toFloat() } to listOf(0.1f, 5f, 0.2f, 4f, 0.3f)[it] }
        val picked = CustomSoundMatcher.loudest(w, 2)
        assertEquals(listOf(1f, 3f), picked.map { it[0] })
    }

    @Test
    fun encodeDecodeRoundTripAndCenterParsing() {
        val v = noise(5)
        assertArrayEquals(v, CustomSoundMatcher.decode(CustomSoundMatcher.encode(v)), 0f)
        assertNull(CustomSoundMatcher.parseCenter(ByteArray(10), dim))
        val bytes = java.nio.ByteBuffer.allocate(dim * 4).order(java.nio.ByteOrder.LITTLE_ENDIAN)
            .apply { v.forEach { putFloat(it) } }.array()
        assertArrayEquals(v, CustomSoundMatcher.parseCenter(bytes, dim), 0f)
    }
}
