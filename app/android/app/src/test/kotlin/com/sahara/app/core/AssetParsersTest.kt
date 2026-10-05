package com.sahara.app.core

import org.junit.Assert.assertEquals
import org.junit.Test

class AssetParsersTest {
    @Test
    fun parsesLabelsSkippingCommentsAndBlanks() {
        val labels = AssetParsers.parseLabels("# generated\nsmoke_alarm\n\ndoorbell\n  siren  \n")
        assertEquals(listOf("smoke_alarm", "doorbell", "siren"), labels)
    }

    @Test
    fun parsesThresholdSearchOutput() {
        val json = """
            {
              "smoke_alarm": {"threshold": 0.15, "precision": 0.25, "recall": 0.33},
              "siren": {
                "precision": 1.0,
                "threshold": 0.4
              }
            }
        """.trimIndent()
        assertEquals(mapOf("smoke_alarm" to 0.15f, "siren" to 0.4f), AssetParsers.parseThresholds(json))
    }

    @Test
    fun parsesHitsRequired() {
        val json = """{"siren": {"threshold": 0.99, "hits_required": 3}, "doorbell": {"threshold": 0.6},
            "_meta": {"run": "x"}}"""
        assertEquals(mapOf("siren" to 3), AssetParsers.parseHitsRequired(json))
        assertEquals(mapOf("siren" to 0.99f, "doorbell" to 0.6f), AssetParsers.parseThresholds(json))
    }

    @Test
    fun parsesFlatThresholds() {
        assertEquals(mapOf("doorbell" to 0.2f), AssetParsers.parseThresholds("""{"doorbell": 0.2}"""))
    }
}
