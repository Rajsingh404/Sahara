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
    fun parsesFlatThresholds() {
        assertEquals(mapOf("doorbell" to 0.2f), AssetParsers.parseThresholds("""{"doorbell": 0.2}"""))
    }

    @Test
    fun parsesYamnetClassMapWithQuotedNames() {
        val csv = "index,mid,display_name\n0,/m/09x0r,Speech\n3,/m/02zsn,\"Female speech, woman speaking\"\n"
        assertEquals(listOf("Speech", "Female speech, woman speaking"), AssetParsers.parseYamnetClassMap(csv))
    }

    @Test
    fun parsesOtherSoundsConfig() {
        val json = """{"min_score": 0.4, "ignored": ["Speech", "Inside, small room"], "covered_by_head": ["Police car (siren)"]}"""
        val cfg = AssetParsers.parseOtherSounds(json)
        assertEquals(0.4f, cfg.minScore)
        assertEquals(setOf("Speech", "Inside, small room"), cfg.ignored)
        assertEquals(setOf("Police car (siren)"), cfg.coveredByHead)
    }
}
