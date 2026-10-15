package com.sahara.app.core

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Base64
import kotlin.math.sqrt

/** A user-enrolled sound: a name and a unit-length prototype over 1,024-d window vectors. */
class CustomSound(val name: String, val prototype: FloatArray)

/**
 * Few-shot matching of user-enrolled sounds (Phase 7), mirroring src/personalization/prototypes.py.
 *
 * A window vector (the same 2-hop pooled YAMNet embedding the 8-class head sees) is centred on the
 * mean background window ([center], assets/personal_center.bin), L2-normalised and compared with
 * each prototype by cosine similarity. A sound fires at or above [threshold], at most once per
 * [cooldownMs]. Pure Kotlin so it runs in JVM tests.
 */
class CustomSoundMatcher(
    private val center: FloatArray?,
    var threshold: Float = DEFAULT_THRESHOLD,
    private val cooldownMs: Long = 10_000L,
) {
    var sounds: List<CustomSound> = emptyList()
        set(value) {
            field = value
            lastFired.keys.retainAll(value.map { it.name }.toSet())
        }
    private val lastFired = HashMap<String, Long>()

    /** Sounds that fire for [window] at [nowMs], with their similarity. */
    fun match(window: FloatArray, nowMs: Long): List<Pair<CustomSound, Float>> {
        if (sounds.isEmpty()) return emptyList()
        val v = unitCentered(window, center)
        val fired = mutableListOf<Pair<CustomSound, Float>>()
        for (s in sounds) {
            val sim = dot(v, s.prototype)
            if (sim < threshold) continue
            if (nowMs - (lastFired[s.name] ?: Long.MIN_VALUE / 2) < cooldownMs) continue
            lastFired[s.name] = nowMs
            fired += s to sim
        }
        return fired.sortedByDescending { it.second }
    }

    fun similarity(window: FloatArray, prototype: FloatArray): Float = dot(unitCentered(window, center), prototype)

    companion object {
        /** Median of the thresholds tuned in src/personalization/evaluate.py (Phase 7). */
        const val DEFAULT_THRESHOLD = 0.73f
        const val WINDOWS_PER_SAMPLE = 2
        const val MIN_SAMPLES = 5
        const val MAX_SAMPLES = 10
        /**
         * Class id sent with custom-sound alerts: the doorbell's (1), so the wristband plays the doorbell
         * pattern. BLE protocol v1 has no id for custom sounds (docs/ble_protocol.md); a dedicated id would
         * need a protocol change across app and firmware.
         */
        const val ALERT_CLASS_ID = 1
        const val LABEL_PREFIX = "custom:"

        fun unitCentered(v: FloatArray, center: FloatArray?): FloatArray {
            val out = FloatArray(v.size) { i -> v[i] - (center?.getOrNull(i) ?: 0f) }
            var norm = 0.0
            for (x in out) norm += x * x
            val n = sqrt(norm).coerceAtLeast(1e-9).toFloat()
            for (i in out.indices) out[i] /= n
            return out
        }

        /** The loudest [k] windows of one enrolment sample: (window vector, energy) pairs. */
        fun loudest(windows: List<Pair<FloatArray, Float>>, k: Int = WINDOWS_PER_SAMPLE): List<FloatArray> =
            windows.withIndex().sortedByDescending { it.value.second }.take(k).sortedBy { it.index }.map { it.value.first }

        /** Unit-length prototype: mean of the centred, normalised selected windows of all samples. */
        fun prototype(selected: List<FloatArray>, center: FloatArray?): FloatArray {
            require(selected.isNotEmpty()) { "no enrolment windows" }
            val dim = selected[0].size
            val mean = FloatArray(dim)
            for (w in selected) {
                val u = unitCentered(w, center)
                for (i in 0 until dim) mean[i] += u[i]
            }
            for (i in 0 until dim) mean[i] /= selected.size
            return unitCentered(mean, null)
        }

        fun encode(v: FloatArray): String {
            val buf = ByteBuffer.allocate(v.size * 4).order(ByteOrder.LITTLE_ENDIAN)
            v.forEach { buf.putFloat(it) }
            return Base64.getEncoder().encodeToString(buf.array())
        }

        fun decode(s: String): FloatArray {
            val buf = ByteBuffer.wrap(Base64.getDecoder().decode(s)).order(ByteOrder.LITTLE_ENDIAN)
            return FloatArray(buf.remaining() / 4) { buf.getFloat() }
        }

        /** Little-endian float32 file (assets/personal_center.bin), or null if absent/malformed. */
        fun parseCenter(bytes: ByteArray?, dim: Int): FloatArray? {
            if (bytes == null || bytes.size != dim * 4) return null
            val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            return FloatArray(dim) { buf.getFloat() }
        }

        private fun dot(a: FloatArray, b: FloatArray): Float {
            var s = 0f
            for (i in a.indices) s += a[i] * b[i]
            return s
        }
    }
}
