package com.sahara.app.core

import org.junit.Assert.assertArrayEquals
import org.junit.Test

class EmbeddingPoolTest {
    @Test
    fun averagesLastNEmbeddings() {
        val pool = EmbeddingPool(dim = 2, capacity = 2)
        assertArrayEquals(floatArrayOf(1f, 2f), pool.add(floatArrayOf(1f, 2f)), 0f)
        assertArrayEquals(floatArrayOf(2f, 3f), pool.add(floatArrayOf(3f, 4f)), 0f)
        assertArrayEquals(floatArrayOf(4f, 5f), pool.add(floatArrayOf(5f, 6f)), 0f)
    }
}
