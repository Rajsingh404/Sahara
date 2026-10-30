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
    /** Sensitivity steps tuned at different false-alarm budgets (alert_profiles.csv), lowest budget first. */
    private val profiles: List<AssetParsers.AlertProfile> = emptyList(),
) {
    private val base = FloatArray(labels.size) { i -> baseThresholds[labels[i]] ?: DEFAULT_THRESHOLD }
    private val history = Array(labels.size) { ArrayDeque<Boolean>() }
    private val lastFired = LongArray(labels.size) { Long.MIN_VALUE / 2 }

    /** Classes the user switched off are never reported. */
    var enabled: Set<String> = labels.toSet()

    /**
     * -1.0 (fewer alerts) .. +1.0 (more alerts). With [profiles], picks the nearest tuned step (0 = the
     * middle one, the default budget), so every position has a measured detection / false-alarm trade-off.
     * Without them, shifts every threshold by up to 2 in log-odds (0.5 -> 0.12..0.88, 0.99 -> 0.93..0.9986).
     */
    var sensitivity: Float = 0f
        set(value) {
            field = value.coerceIn(-1f, 1f)
        }

    /** Per-class thresholds from home calibration (HomeCalibration); they can only lower a threshold. */
    var calibrated: Map<String, Float> = emptyMap()

    /** The tuned step in use, or null without profiles. */
    val activeProfile: AssetParsers.AlertProfile?
        get() = if (profiles.isEmpty()) null
        else profiles[Math.round((sensitivity + 1f) / 2f * (profiles.size - 1)).coerceIn(0, profiles.size - 1)]

    fun thresholdFor(index: Int): Float {
        val own = uncalibratedThreshold(index)
        return calibrated[labels[index]]?.let { minOf(own, it) } ?: own
    }

    private fun uncalibratedThreshold(index: Int): Float {
        activeProfile?.let { profile ->
            // A class the profiles do not cover keeps its own threshold: no tuned step to move it to.
            return (profile.thresholds[labels[index]] ?: base[index]).coerceIn(MIN_THRESHOLD, MAX_THRESHOLD)
        }
        val t = base[index].toDouble().coerceIn(1e-4, 1 - 1e-4)
        val logit = ln(t / (1 - t)) - SENSITIVITY_LOGIT_RANGE * sensitivity
        return (1.0 / (1.0 + exp(-logit))).toFloat().coerceIn(MIN_THRESHOLD, MAX_THRESHOLD)
    }

    fun hitsFor(index: Int): Int =
        (activeProfile?.hits?.get(labels[index]) ?: hitsPerClass[labels[index]] ?: if (labels[index] in safetyCritical) 1 else hitsRequired).coerceIn(1, windowCount)

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
    /** Name calls: the caller's name (null = "Someone"). Other sounds: YAMNet's class name. */
    val detail: String? = null,
) {
    companion object {
        const val NAME_CALL = "name_call"
        const val OTHER_SOUND = "other_sound"

        fun nameCall(call: NameCall) =
            Detection(ModelConfig.NAME_CALL_ID, NAME_CALL, call.nameScore, call.timestampMs, safetyCritical = false, detail = call.caller)

        fun otherSound(s: OtherSound) =
            Detection(ModelConfig.OTHER_SOUND_ID, OTHER_SOUND, s.score, s.timestampMs, safetyCritical = false, detail = s.name)
    }

    val isOtherSound: Boolean get() = label == OTHER_SOUND
}
