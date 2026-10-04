package com.sahara.app.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Replays src/test/resources/policy_fixture.csv, shared with tests/test_policy.py, so the Python
 * port used for false-alarm evaluation (src/evaluation/policy.py) and this policy stay identical.
 */
class DetectionPolicyFixtureTest {
    private val labels = listOf(
        "smoke_alarm", "doorbell", "siren", "knocking", "dog_bark", "baby_cry", "glass_break", "appliance_beep",
    )

    @Test
    fun matchesSharedFixture() {
        val lines = javaClass.classLoader!!.getResourceAsStream("policy_fixture.csv")!!
            .bufferedReader().readLines()
        val thresholds = lines.first { it.startsWith("#thresholds,") }.split(",").drop(1)
            .mapIndexed { i, v -> labels[i] to v.toFloat() }.toMap()
        val policy = DetectionPolicy(labels, thresholds)
        var fired = 0
        for (line in lines) {
            if (line.isBlank() || line.startsWith("#") || line.startsWith("t_ms")) continue
            val cells = line.split(",")
            policy.sensitivity = cells[1].toFloat()
            val scores = FloatArray(labels.size) { cells[2 + it].toFloat() }
            val expected = cells.getOrElse(10) { "" }.split("|").filter { it.isNotEmpty() }
            val got = policy.update(scores, cells[0].toLong()).map { it.label }
            assertEquals("t=${cells[0]}", expected, got)
            fired += got.size
        }
        assertTrue(fired > 10)
    }
}
