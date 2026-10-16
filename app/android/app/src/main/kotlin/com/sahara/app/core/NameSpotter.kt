package com.sahara.app.core

import kotlin.math.exp

/**
 * Scores whether the user's name ("Aryan") is in a run of 96-d speech embeddings, whoever says it.
 *
 * The embeddings come from the pretrained speech-embedding model (openWakeWord's
 * melspectrogram.tflite + embedding_model.tflite, one embedding per 80 ms). This class is the
 * small head from src/personalization/name_head.py: standardise [Head.window] embeddings, flatten,
 * logistic regression. `scripts/eval_name_call.py --export` writes it as assets/name_head.json.
 *
 * Pure Kotlin so it runs in JVM unit tests.
 */
class NameSpotter(private val head: Head) {
    val window: Int get() = head.window
    val headThreshold: Float get() = head.threshold
    val displayName: String get() = head.name

    /** Name probability for exactly [window] consecutive embeddings. */
    fun score(frames: List<FloatArray>): Float {
        require(frames.size == head.window) { "expected ${head.window} embeddings, got ${frames.size}" }
        var z = head.intercept
        var k = 0
        for (f in frames) {
            require(f.size == head.dim) { "expected ${head.dim}-d embedding, got ${f.size}" }
            for (d in 0 until head.dim) {
                z += head.coef[k] * (f[d] - head.mean[k]) / head.scale[k]
                k++
            }
        }
        return (1.0 / (1.0 + exp(-z.toDouble()))).toFloat()
    }

    class Head(
        val name: String,
        val window: Int,
        val dim: Int,
        val mean: FloatArray,
        val scale: FloatArray,
        val coef: FloatArray,
        val intercept: Float,
        val threshold: Float,
    ) {
        init {
            val n = window * dim
            require(mean.size == n && scale.size == n && coef.size == n) { "name head expects $n weights" }
        }
    }
}
