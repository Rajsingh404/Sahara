package com.sahara.app.core

/** Values shared with the Python pipeline (src/config.py). Change both together. */
object ModelConfig {
    const val SAMPLE_RATE = 16_000
    /** INFERENCE_WINDOW_SEC = 1.0 */
    const val WINDOW_SAMPLES = 16_000
    /** INFERENCE_HOP_SEC = 0.5 */
    const val HOP_SAMPLES = 8_000
    /** One YAMNet patch: 0.975 s of audio. */
    const val YAMNET_INPUT_SAMPLES = 15_600
    const val EMBEDDING_DIM = 1_024
    /** VAD_ENERGY_THRESHOLD: windows quieter than this skip inference. */
    const val VAD_ENERGY_THRESHOLD = 1e-4f
    /** Embeddings averaged before the head, mirroring the mean-pooling used in training. */
    const val EMBEDDING_POOL = 2
    /**
     * Long-context layout supported by the app: the mean of the last 2 and of the last 6
     * non-silent embeddings, concatenated (2,048 values). Used only when the head has 2,048
     * inputs (src/config.py WINDOW_POOLS = (2, 6)); the current head has 1,024 and uses [2].
     */
    val WINDOW_POOLS = intArrayOf(2, 6)

    /** Pool sizes for a head with [inputLen] inputs, or null if the size matches no layout. */
    fun poolsForHeadInput(inputLen: Int): IntArray? = when (inputLen) {
        EMBEDDING_DIM -> intArrayOf(EMBEDDING_POOL)
        EMBEDDING_DIM * WINDOW_POOLS.size -> WINDOW_POOLS
        else -> null
    }
}
