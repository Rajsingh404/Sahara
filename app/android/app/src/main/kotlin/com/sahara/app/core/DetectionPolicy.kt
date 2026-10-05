package com.sahara.app.core

import kotlin.math.exp
import kotlin.math.ln

/**
 * Turns per-window model scores into alert decisions.
 *
 * A class fires when its score clears its threshold in N of the last [windowCount] windows
 * (hop 0.5 s), and it has not fired within [cooldownMs]. N comes from [hitsPerClass]
 * (`hits_required` in class_thresholds.json, tuned against a false-alarm budget by
 * src/evaluation/tune_policy.py); without an entry, safety-critical classes need 1 hit and
 * the others [hitsRequired].
 *
 * Pure Kotlin (no Android imports) so it runs in JVM unit tests.
 */
class DetectionPolicy(
    private val labels: List<String>,
    baseThresholds: Map<String, Float>,
    private val safetyCritical: Set<String> = DEFAULT_SAFETY_CRITICAL,
    private val hitsRequired: Int = 2,
    private val windowCount: Int = 3,
    private val cooldownMs: Long = 10_000L,
    private val hitsPerClass: Map<String, Int> = emptyMap(),
) {
    private val base = FloatArray(labels.size) { i -> baseThresholds[labels[i]] ?: DEFAULT_THRESHOLD }
    private val history = Array(labels.size) { ArrayDeque<Boolean>() }
    private val lastFired = LongArray(labels.size) { Long.MIN_VALUE / 2 }

    /** Classes the user switched off are never reported. */
    var enabled: Set<String> = labels.toSet()

    /**
     * -1.0 (fewer alerts) .. +1.0 (more alerts). Shifts every threshold by up to 2 in log-odds
     * (0.5 -> 0.12..0.88, 0.99 -> 0.93..0.9986), which stays meaningful for thresholds near 1.
     */
    var sensitivity: Float = 0f
        set(value) {
            field = value.coerceIn(-1f, 1f)
        }

    fun thresholdFor(index: Int): Float {
        val t = base[index].toDouble().coerceIn(1e-4, 1 - 1e-4)
        val logit = ln(t / (1 - t)) - SENSITIVITY_LOGIT_RANGE * sensitivity
        return (1.0 / (1.0 + exp(-logit))).toFloat().coerceIn(MIN_THRESHOLD, MAX_THRESHOLD)
    }

    fun hitsFor(index: Int): Int =
        (hitsPerClass[labels[index]] ?: if (labels[index] in safetyCritical) 1 else hitsRequired).coerceIn(1, windowCount)

    fun update(scores: FloatArray, nowMs: Long): List<Detection> {
        require(scores.size == labels.size) { "expected ${labels.size} scores, got ${scores.size}" }
        val fired = mutableListOf<Detection>()
        for (i in labels.indices) {
            val hit = scores[i] >= thresholdFor(i)
            val h = history[i]
            h.addLast(hit)
            while (h.size > windowCount) h.removeFirst()
            if (!hit || labels[i] !in enabled) continue
            if (h.count { it } < hitsFor(i)) continue
            if (nowMs - lastFired[i] < cooldownMs) continue
            lastFired[i] = nowMs
            fired += Detection(i, labels[i], scores[i], nowMs, labels[i] in safetyCritical)
        }
        return fired.sortedByDescending { it.confidence }
    }

    fun reset() {
        history.forEach { it.clear() }
        lastFired.fill(Long.MIN_VALUE / 2)
    }

    companion object {
        const val DEFAULT_THRESHOLD = 0.5f
        const val MIN_THRESHOLD = 0.01f
        const val MAX_THRESHOLD = 0.999f
        const val SENSITIVITY_LOGIT_RANGE = 2.0
        /** Mirrors SAFETY_CRITICAL_CLASSES in src/config.py. */
        val DEFAULT_SAFETY_CRITICAL = setOf("smoke_alarm", "siren", "glass_break")
    }
}

data class Detection(
    val classId: Int,
    val label: String,
    val confidence: Float,
    val timestampMs: Long,
    val safetyCritical: Boolean,
)
