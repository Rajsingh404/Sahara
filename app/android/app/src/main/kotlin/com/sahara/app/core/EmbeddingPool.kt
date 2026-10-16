package com.sahara.app.core

/** Running mean of the last [capacity] embeddings, fed to the classifier head. */
class EmbeddingPool(private val dim: Int, private val capacity: Int) {
    private val items = ArrayDeque<FloatArray>()

    fun add(embedding: FloatArray): FloatArray {
        require(embedding.size == dim) { "expected $dim-d embedding, got ${embedding.size}" }
        items.addLast(embedding.copyOf())
        while (items.size > capacity) items.removeFirst()
        val mean = FloatArray(dim)
        for (e in items) for (i in 0 until dim) mean[i] += e[i]
        for (i in 0 until dim) mean[i] /= items.size
        return mean
    }

    fun clear() = items.clear()
}
