package com.sahara.app.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test

class OtherSoundsPolicyTest {
    private val names = listOf("Speech", "Cat", "Water tap, faucet", "Doorbell")

    private fun policy() = OtherSoundsPolicy(names, ignored = setOf("Speech"), coveredByHead = setOf("Doorbell"))

    private fun scores(vararg pairs: Pair<String, Float>) = FloatArray(names.size).also { s ->
        for ((n, v) in pairs) s[names.indexOf(n)] = v
    }

    @Test
    fun reportsATopClassSeenInTwoOfThreeWindows() {
        val p = policy()
        assertNull(p.update(scores("Cat" to 0.6f), 0, headFired = false))
        val hit = p.update(scores("Cat" to 0.7f), 500, headFired = false)
        assertNotNull(hit)
        assertEquals("Cat", hit!!.name)
    }

    @Test
    fun neverReportsIgnoredOrHeadCoveredClasses() {
        val p = policy()
        repeat(4) { assertNull(p.update(scores("Speech" to 0.9f, "Doorbell" to 0.9f), it * 500L, headFired = false)) }
    }

    @Test
    fun staysQuietWhenTheTrainedHeadFired() {
        val p = policy()
        p.update(scores("Cat" to 0.6f), 0, headFired = false)
        assertNull(p.update(scores("Cat" to 0.6f), 500, headFired = true))
    }

    @Test
    fun ignoresLowScoresAndRespectsCooldown() {
        val p = policy()
        repeat(3) { assertNull(p.update(scores("Cat" to 0.2f), it * 500L, headFired = false)) }
        p.update(scores("Water tap, faucet" to 0.5f), 2_000, headFired = false)
        assertNotNull(p.update(scores("Water tap, faucet" to 0.5f), 2_500, headFired = false))
        assertNull(p.update(scores("Water tap, faucet" to 0.5f), 3_000, headFired = false))
        assertNotNull(p.update(scores("Water tap, faucet" to 0.5f), 40_000, headFired = false))
    }
}
