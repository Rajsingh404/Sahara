package com.sahara.app.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

class NameCallDetectorTest {
    /** 2-window, 2-d head that fires when dimension 0 is high in both embeddings. */
    private val head = NameSpotter.Head(
        name = "Aryan", window = 2, dim = 2,
        mean = FloatArray(4), scale = FloatArray(4) { 1f },
        coef = floatArrayOf(4f, 0f, 4f, 0f), intercept = -4f, threshold = 0.6f,
    )

    /** The fake embedder reads the "audio" as embeddings: wave[2i], wave[2i+1]. */
    private val embedder = NameCallDetector.SpeechEmbedder { w -> Array(w.size / 2) { floatArrayOf(w[2 * it], w[2 * it + 1]) } }

    @Test
    fun firesForAnyoneAndThenCoolsDown() {
        val d = NameCallDetector(NameSpotter(head), embedder, encoder = null, callers = PrototypeBank(2))
        assertNull(d.process(floatArrayOf(0f, 0f, 0f, 0f, 0f, 0f), 0))
        val call = d.process(floatArrayOf(0f, 0f, 1f, 0f, 1f, 0f), 1_000)
        assertNotNull(call)
        assertNull("no encoder, so the caller is unknown", call!!.caller)
        assertNull("cooldown", d.process(floatArrayOf(1f, 0f, 1f, 0f), 2_000))
        assertNotNull(d.process(floatArrayOf(1f, 0f, 1f, 0f), 10_000))
    }

    @Test
    fun sensitivityMovesTheThreshold() {
        val d = NameCallDetector(NameSpotter(head), embedder, encoder = null, callers = PrototypeBank(2))
        d.sensitivity = 1f
        assertEquals(0.45f, d.threshold, 1e-6f)
        d.sensitivity = -1f
        assertEquals(0.75f, d.threshold, 1e-6f)
    }

    @Test
    fun bundledNameHeadParses() {
        val file = listOf("../../assets/name_head.json", "app/assets/name_head.json", "../assets/name_head.json")
            .map(::File).firstOrNull { it.exists() } ?: return
        val parsed = AssetParsers.parseNameHead(file.readText())
        assertEquals("Aryan", parsed.name)
        assertEquals(8, parsed.window)
        assertEquals(96, parsed.dim)
        assertTrue(parsed.threshold in 0.05f..0.95f)
    }
}
