package com.sahara.app.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** Same cases as tests/test_home_calibration.py. */
class HomeCalibrationTest {
    private val s = floatArrayOf(0.1f, 0.7f, 0.9f, 0.6f, 0.2f)

    @Test
    fun fireLevelIsTheHighestThresholdThatStillAlerts() {
        assertEquals(0.9f, HomeCalibration.fireLevel(s, 1))
        assertEquals(0.7f, HomeCalibration.fireLevel(s, 2))
        assertEquals(0.6f, HomeCalibration.fireLevel(s, 3))
    }

    @Test
    fun calibrateStaysBetweenFloorAndCurrent() {
        assertTrue(HomeCalibration.calibrate(0.95f, 0.6f, listOf(0.9f, 0.8f)) < 0.8f)
        assertEquals(0.79f, HomeCalibration.calibrate(0.95f, 0.79f, listOf(0.9f, 0.8f)))
        assertEquals(0.95f, HomeCalibration.calibrate(0.95f, 0.6f, listOf(0.99f)))
        assertEquals(0.95f, HomeCalibration.calibrate(0.95f, 0.6f, emptyList()))
    }

    @Test
    fun backgroundRaisesTheThresholdAgain() {
        val t = HomeCalibration.calibrate(0.95f, 0.5f, listOf(0.8f), backgroundLevel = 0.85f)
        assertTrue(t > 0.85f && t <= 0.95f)
    }

    @Test
    fun calibrationOnlyLowersThePolicyThreshold() {
        val labels = listOf("doorbell", "knocking")
        val p = DetectionPolicy(labels, mapOf("doorbell" to 0.95f, "knocking" to 0.9f))
        p.calibrated = mapOf("doorbell" to 0.8f, "knocking" to 0.97f)
        assertEquals(0.8f, p.thresholdFor(0), 1e-6f)
        assertEquals(0.9f, p.thresholdFor(1), 1e-6f)
    }
}
