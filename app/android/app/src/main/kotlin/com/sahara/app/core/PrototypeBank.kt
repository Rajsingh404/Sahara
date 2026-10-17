package com.sahara.app.core

import kotlin.math.sqrt

/**
 * Few-shot matching by cosine similarity to per-name centroids.
 *
 * Used twice: for enrolled callers (voice vectors from [SpeakerEncoder]) and for sounds the user
 * teaches the app (YAMNet embeddings of their own doorbell, pressure cooker, ...). Each name keeps
 * its samples; the centroid is the mean of the L2-normalised samples.
 */
class PrototypeBank(private val dim: Int) {
    private val samples = LinkedHashMap<String, MutableList<FloatArray>>()
    private val centroids = HashMap<String, FloatArray>()

    val names: Set<String> get() = samples.keys

    fun count(name: String): Int = samples[name]?.size ?: 0

    fun add(name: String, vector: FloatArray) {
        require(vector.size == dim) { "expected $dim-d vector, got ${vector.size}" }
        samples.getOrPut(name) { mutableListOf() } += normalised(vector)
        centroids[name] = mean(samples.getValue(name))
    }

    fun remove(name: String) {
        samples.remove(name)
        centroids.remove(name)
    }

    fun samplesOf(name: String): List<FloatArray> = samples[name].orEmpty()

    /** Best (name, cosine similarity), or null when the bank is empty. */
    fun best(vector: FloatArray): Pair<String, Float>? {
        if (centroids.isEmpty()) return null
        val v = normalised(vector)
        return centroids.map { (name, c) -> name to dot(v, c) / (norm(c) + 1e-9f) }.maxByOrNull { it.second }
    }

    /** Name when the best similarity reaches [threshold], else null ("Someone" / unknown sound). */
    fun identify(vector: FloatArray, threshold: Float): Match? {
        val (name, sim) = best(vector) ?: return null
        return Match(if (sim >= threshold) name else null, sim)
    }

    data class Match(val name: String?, val similarity: Float)

    companion object {
        fun dot(a: FloatArray, b: FloatArray): Float {
            var s = 0f
            for (i in a.indices) s += a[i] * b[i]
            return s
        }

        fun norm(a: FloatArray): Float = sqrt(dot(a, a))

        fun normalised(a: FloatArray): FloatArray {
            val n = norm(a) + 1e-9f
            return FloatArray(a.size) { a[it] / n }
        }

        private fun mean(items: List<FloatArray>): FloatArray {
            val out = FloatArray(items[0].size)
            for (v in items) for (i in out.indices) out[i] += v[i]
            for (i in out.indices) out[i] /= items.size
            return out
        }
    }
}
