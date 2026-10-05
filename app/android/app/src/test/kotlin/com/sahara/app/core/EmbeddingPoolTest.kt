package com.sahara.app.core

import org.junit.Assert.assertArrayEquals
import org.junit.Test

class EmbeddingPoolTest {
    @Test
    fun averagesLastNEmbeddings() {
        val pool = EmbeddingPool(2, 2)
        assertArrayEquals(floatArrayOf(1f, 2f), pool.add(floatArrayOf(1f, 2f)), 0f)
        assertArrayEquals(floatArrayOf(2f, 3f), pool.add(floatArrayOf(3f, 4f)), 0f)
        assertArrayEquals(floatArrayOf(4f, 5f), pool.add(floatArrayOf(5f, 6f)), 0f)
    }

    @Test
    fun concatenatesShortAndLongMeans() {
        val pool = EmbeddingPool(1, 2, 6)
        val values = (1..7).map { pool.add(floatArrayOf(it.toFloat())) }
        assertArrayEquals(floatArrayOf(1f, 1f), values[0], 0f)            // one item so far
        assertArrayEquals(floatArrayOf(6.5f, 4.5f), values[6], 1e-6f)      // mean(6,7), mean(2..7)
    }

    @Test
    fun headInputSizeSelectsPools() {
        org.junit.Assert.assertArrayEquals(intArrayOf(2), ModelConfig.poolsForHeadInput(1024))
        org.junit.Assert.assertArrayEquals(intArrayOf(2, 6), ModelConfig.poolsForHeadInput(2048))
        org.junit.Assert.assertNull(ModelConfig.poolsForHeadInput(15_600))
    }
}
