package com.sahara.app.core

/**
 * Turns per-window model scores into alert decisions.
 *
 * A class fires when its score clears its threshold in [hitsRequired] of the last [windowCount]
 * windows (hop 0.5 s), and it has not fired within [cooldownMs]. Safety-critical classes need a
 * single hit so a smoke alarm or siren is never delayed by smoothing.
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
) {
    private val base = FloatArray(labels.size) { i -> baseThresholds[labels[i]] ?: DEFAULT_THRESHOLD }
    private val history = Array(labels.size) { ArrayDeque<Boolean>() }
    private val lastFired = LongArray(labels.size) { Long.MIN_VALUE / 2 }

    /** Classes the user switched off are never reported. */
    var enabled: Set<String> = labels.toSet()

    /**
     * -1.0 (fewer alerts) .. +1.0 (more alerts). Scales every threshold by up to ±40 %.
     */
    var sensitivity: Float = 0f
        set(value) {
            field = value.coerceIn(-1f, 1f)
        }

    fun thresholdFor(index: Int): Float =
        (base[index] * (1f - 0.4f * sensitivity)).coerceIn(0.05f, 0.95f)

    fun update(scores: FloatArray, nowMs: Long): List<Detection> {
        require(scores.size == labels.size) { "expected ${labels.size} scores, got ${scores.size}" }
        val fired = mutableListOf<Detection>()
        for (i in labels.indices) {
            val hit = scores[i] >= thresholdFor(i)
            val h = history[i]
            h.addLast(hit)
            while (h.size > windowCount) h.removeFirst()
            if (!hit || labels[i] !in enabled) continue
            val needed = if (labels[i] in safetyCritical) 1 else hitsRequired
            if (h.count { it } < needed) continue
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
