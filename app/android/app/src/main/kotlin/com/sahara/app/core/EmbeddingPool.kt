package com.sahara.app.core

/**
 * Running means of the last embeddings, fed to the classifier head: for each size in [sizes],
 * the mean of that many most recent embeddings (fewer at the start), concatenated in order.
 * EmbeddingPool(dim, 2) is the plain 2-embedding mean; EmbeddingPool(dim, 2, 6) gives 2*dim values.
 */
class EmbeddingPool(private val dim: Int, vararg sizes: Int) {
    private val sizes = sizes.copyOf().also { require(it.isNotEmpty() && it.all { s -> s > 0 }) }
    private val capacity = this.sizes.max()
    private val items = ArrayDeque<FloatArray>()

    val outputDim: Int get() = dim * sizes.size

    fun add(embedding: FloatArray): FloatArray {
        require(embedding.size == dim) { "expected $dim-d embedding, got ${embedding.size}" }
        items.addLast(embedding.copyOf())
        while (items.size > capacity) items.removeFirst()
        val out = FloatArray(outputDim)
        for ((k, size) in sizes.withIndex()) {
            val n = minOf(size, items.size)
            for (j in items.size - n until items.size) {
                val e = items[j]
                for (i in 0 until dim) out[k * dim + i] += e[i]
            }
            for (i in 0 until dim) out[k * dim + i] /= n
        }
        return out
    }

    fun clear() = items.clear()
}
