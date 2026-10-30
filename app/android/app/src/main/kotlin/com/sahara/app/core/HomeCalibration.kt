package com.sahara.app.core

import kotlin.math.exp
import kotlin.math.ln

/**
 * Home calibration maths, mirroring src/evaluation/home_calibration.py (scripts/eval_home_calibration.py
 * measures it on recording sessions the model never heard).
 *
 * The user records a sound a few times where it happens. [fireLevel] is the highest threshold at which the
 * alert rule would still fire on one recording; [calibrate] puts the class's threshold a margin below the
 * weakest recording, never below [floor] (the class's threshold in the most permissive tuned sensitivity
 * step) and never above the current one. Pure Kotlin so it runs in JVM unit tests.
 */
object HomeCalibration {
    const val RECORDINGS = 3
    const val MARGIN_LOGIT = 0.5
    private const val WINDOW_COUNT = 3

    /** Highest threshold at which [scores] (one per hop) alert: current hop a hit and [hits] of the last 3. */
    fun fireLevel(scores: FloatArray, hits: Int): Float {
        var best = 0f
        for (t in scores.indices) {
            val window = (maxOf(0, t - WINDOW_COUNT + 1)..t).map { scores[it] }.sortedDescending()
            val kth = if (window.size >= hits) window[hits - 1] else 0f
            best = maxOf(best, minOf(scores[t], kth))
        }
        return best
    }

    fun calibrate(current: Float, floor: Float, levels: List<Float>, backgroundLevel: Float? = null): Float {
        if (levels.isEmpty()) return current
        var t = sigmoid(logit(levels.min()) - MARGIN_LOGIT)
        t = maxOf(t, floor)
        if (backgroundLevel != null && t <= backgroundLevel) t = sigmoid(logit(backgroundLevel) + 0.1)
        return minOf(t, current)
    }

    private fun logit(p: Float): Double {
        val q = p.toDouble().coerceIn(1e-4, 1 - 1e-4)
        return ln(q / (1 - q))
    }

    private fun sigmoid(x: Double): Float = (1.0 / (1.0 + exp(-x))).toFloat()
}
