package com.sahara.app.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class DetectionPolicyTest {
    private val labels = listOf(
        "smoke_alarm", "doorbell", "siren", "knocking", "dog_bark", "baby_cry", "glass_break", "appliance_beep",
    )
    private val thresholds = labels.associateWith { 0.5f }

    private fun scores(vararg pairs: Pair<String, Float>) =
        FloatArray(labels.size).also { arr -> pairs.forEach { (l, v) -> arr[labels.indexOf(l)] = v } }

    @Test
    fun safetyCriticalFiresOnFirstHit() {
        val p = DetectionPolicy(labels, thresholds)
        val fired = p.update(scores("siren" to 0.9f), 0)
        assertEquals(listOf("siren"), fired.map { it.label })
        assertTrue(fired.single().safetyCritical)
    }

    @Test
    fun ordinaryClassNeedsTwoOfThreeWindows() {
        val p = DetectionPolicy(labels, thresholds)
        assertTrue(p.update(scores("doorbell" to 0.8f), 0).isEmpty())
        assertTrue(p.update(scores(), 500).isEmpty())
        assertEquals("doorbell", p.update(scores("doorbell" to 0.8f), 1000).single().label)
    }

    @Test
    fun hitsOutsideTheWindowDoNotCount() {
        val p = DetectionPolicy(labels, thresholds)
        p.update(scores("doorbell" to 0.8f), 0)
        p.update(scores(), 500)
        p.update(scores(), 1000)
        assertTrue(p.update(scores("doorbell" to 0.8f), 1500).isEmpty())
    }

    @Test
    fun cooldownSuppressesRepeats() {
        val p = DetectionPolicy(labels, thresholds, cooldownMs = 10_000)
        assertEquals(1, p.update(scores("smoke_alarm" to 0.9f), 0).size)
        assertTrue(p.update(scores("smoke_alarm" to 0.9f), 5_000).isEmpty())
        assertEquals(1, p.update(scores("smoke_alarm" to 0.9f), 10_000).size)
    }

    @Test
    fun disabledClassesNeverFire() {
        val p = DetectionPolicy(labels, thresholds)
        p.enabled = labels.toSet() - "siren"
        assertTrue(p.update(scores("siren" to 0.99f), 0).isEmpty())
    }

    @Test
    fun usesPerClassThresholdsAndSensitivity() {
        val p = DetectionPolicy(labels, thresholds + ("glass_break" to 0.25f))
        assertTrue(p.update(scores("glass_break" to 0.2f), 0).isEmpty())
        p.sensitivity = 1f // 0.25 * 0.6 = 0.15
        assertEquals(0.15f, p.thresholdFor(labels.indexOf("glass_break")), 1e-6f)
        assertEquals(1, p.update(scores("glass_break" to 0.2f), 500).size)
    }

    @Test
    fun missingThresholdFallsBackToDefault() {
        val p = DetectionPolicy(labels, emptyMap())
        assertEquals(DetectionPolicy.DEFAULT_THRESHOLD, p.thresholdFor(0), 0f)
    }

    @Test
    fun multipleDetectionsSortedByConfidence() {
        val p = DetectionPolicy(labels, thresholds)
        val fired = p.update(scores("siren" to 0.7f, "smoke_alarm" to 0.95f), 0)
        assertEquals(listOf("smoke_alarm", "siren"), fired.map { it.label })
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsWrongScoreCount() {
        DetectionPolicy(labels, thresholds).update(FloatArray(3), 0)
    }
}
