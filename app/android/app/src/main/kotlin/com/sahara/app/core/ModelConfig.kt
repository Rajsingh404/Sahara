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
}
