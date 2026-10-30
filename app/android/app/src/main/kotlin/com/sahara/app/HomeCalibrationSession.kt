package com.sahara.app

import android.content.Context
import com.sahara.app.core.HomeCalibration

/**
 * Home calibration (see core/HomeCalibration.kt). For one sound at a time the user records it
 * [HomeCalibration.RECORDINGS] times where it happens (rings their own doorbell, knocks on their own door) and,
 * optionally, a few seconds of ordinary home sound. Each recording is scored in memory and zeroed; only the
 * resulting threshold is saved. Listening pauses meanwhile, as in custom-sound enrolment.
 */
object HomeCalibrationSession {
    const val SAMPLE_SECONDS = 5

    private var engine: InferenceEngine? = null
    private var resumeListening = false
    private var label: String? = null
    private val levels = mutableListOf<Float>()
    private var backgroundLevel: Float? = null

    @Synchronized
    fun recordSample(context: Context, forLabel: String, background: Boolean = false): Map<String, Any?> {
        val app = context.applicationContext
        if (label != forLabel) reset()
        label = forLabel
        val e = engine ?: run {
            if (AudioCaptureService.isRunning) {
                resumeListening = true
                AudioCaptureService.stop(app)
                val until = System.currentTimeMillis() + 3_000
                while (AudioCaptureService.isRunning && System.currentTimeMillis() < until) Thread.sleep(50)
            }
            InferenceEngine.load(app)?.also { engine = it } ?: return state(ModelStatus.reason ?: "Model not available")
        }
        val index = e.labels.indexOf(forLabel).takeIf { it >= 0 } ?: return state("Unknown sound")
        val pcm = CustomEnrolment.record(SAMPLE_SECONDS) ?: return state("Microphone unavailable")
        val scores = e.recordingScores(pcm).map { it[index] }.toFloatArray()
        pcm.fill(0) // the audio is not kept, not even in memory
        val level = HomeCalibration.fireLevel(scores, hits(app, forLabel))
        if (background) {
            backgroundLevel = level
        } else {
            if (scores.isEmpty()) return state("Too quiet: make the sound closer to the phone and try again")
            levels += level
        }
        return state(null)
    }

    /** Saves the calibrated threshold for the current sound. Returns it with the previous one. */
    @Synchronized
    fun finish(context: Context): Map<String, Any?> {
        val app = context.applicationContext
        val l = label ?: return state("Nothing recorded")
        if (levels.size < HomeCalibration.RECORDINGS) return state("Record the sound ${HomeCalibration.RECORDINGS} times")
        val current = InferenceEngine.loadThresholds(app)[l] ?: return state("Unknown sound")
        val floor = InferenceEngine.loadAlertProfiles(app).lastOrNull()?.thresholds?.get(l) ?: current
        val t = HomeCalibration.calibrate(current, floor, levels, backgroundLevel)
        val prefs = SaharaPrefs(app)
        prefs.calibratedThresholds = prefs.calibratedThresholds + (l to t)
        cleanup(app)
        return mapOf("saved" to true, "label" to l, "before" to current.toDouble(), "after" to t.toDouble())
    }

    @Synchronized
    fun clear(context: Context, forLabel: String?) {
        val prefs = SaharaPrefs(context.applicationContext)
        prefs.calibratedThresholds = if (forLabel == null) emptyMap() else prefs.calibratedThresholds - forLabel
        AudioCaptureService.instance?.applySettings()
    }

    @Synchronized
    fun cancel(context: Context) = cleanup(context.applicationContext)

    private fun hits(context: Context, l: String): Int =
        (InferenceEngine.loadHitsRequired(context)[l] ?: 1).coerceIn(1, 3)

    private fun reset() {
        levels.clear()
        backgroundLevel = null
    }

    private fun cleanup(app: Context) {
        reset()
        label = null
        engine?.close()
        engine = null
        if (resumeListening) AudioCaptureService.start(app)
        resumeListening = false
        AudioCaptureService.instance?.applySettings()
    }

    private fun state(error: String?) = mapOf(
        "label" to label,
        "samples" to levels.size,
        "needed" to HomeCalibration.RECORDINGS,
        "background" to (backgroundLevel != null),
        "error" to error,
    )
}
