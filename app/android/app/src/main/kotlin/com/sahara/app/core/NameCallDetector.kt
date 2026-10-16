package com.sahara.app.core

/**
 * "Raj is calling you" / "Someone is calling you".
 *
 * Every hop (0.5 s) it looks at the last [ModelConfig.NAME_WINDOW_SAMPLES] of audio:
 * 1. [embedder] turns it into speech embeddings (one per 80 ms) and [spotter] scores every run of
 *    [NameSpotter.Head.window] of them. The name fires when the best score clears the threshold.
 * 2. Only then, [encoder] turns the same audio into a voice vector and [callers] names the closest
 *    enrolled caller, or nobody when the similarity is below [callerThreshold].
 *
 * Works for anyone: an unenrolled caller still triggers step 1, and step 2 reports them as unknown.
 */
class NameCallDetector(
    private val spotter: NameSpotter,
    private val embedder: SpeechEmbedder,
    private val encoder: SpeakerEncoder?,
    callers: PrototypeBank,
    var callerThreshold: Float = DEFAULT_CALLER_THRESHOLD,
    private val cooldownMs: Long = 8_000L,
) {
    fun interface SpeechEmbedder {
        /** [n][96] speech embeddings for 16 kHz mono audio, one per 80 ms. */
        fun embed(wave: FloatArray): Array<FloatArray>
    }

    /** Replaced as a whole when someone enrols, so a running identification never sees half a bank. */
    @Volatile var callers: PrototypeBank = callers

    val displayName: String get() = spotter.displayName
    val canIdentify: Boolean get() = encoder != null

    var enabled = true
    /** -1 (fewer false calls) .. +1 (catch more calls). Moves the threshold by up to ±0.15. */
    var sensitivity = 0f
    private var lastFired = Long.MIN_VALUE / 2

    val threshold: Float get() = (spotter.headThreshold - 0.15f * sensitivity).coerceIn(0.05f, 0.95f)

    fun process(wave: FloatArray, nowMs: Long): NameCall? {
        val score = spot(wave, nowMs) ?: return null
        val match = identify(wave)
        return NameCall(match?.name, score, match?.similarity ?: 0f, nowMs)
    }

    /** Step 1, cheap enough for every hop: the name score when it fires (and starts the cooldown), else null. */
    fun spot(wave: FloatArray, nowMs: Long): Float? {
        if (!enabled || nowMs - lastFired < cooldownMs) return null
        val score = nameScore(wave)
        if (score < threshold) return null
        lastFired = nowMs
        return score
    }

    /** Step 2, only after step 1 fired (~0.5 s, so the service runs it off the capture thread). */
    fun identify(wave: FloatArray): PrototypeBank.Match? =
        encoder?.embed(wave)?.let { callers.identify(it, callerThreshold) }

    fun nameScore(wave: FloatArray): Float {
        val emb = embedder.embed(wave)
        val w = spotter.window
        if (emb.size < w) return 0f
        var best = 0f
        for (i in 0..emb.size - w) best = maxOf(best, spotter.score(emb.asList().subList(i, i + w)))
        return best
    }

    /** Voice vector for enrolment, from a short recording of the caller saying the name. */
    fun voiceOf(wave: FloatArray): FloatArray? = encoder?.embed(wave)

    companion object {
        /** docs/name_call_eval.md: at 0.70 most unenrolled callers are shown as "Someone". */
        const val DEFAULT_CALLER_THRESHOLD = 0.70f
    }
}

data class NameCall(
    /** Enrolled caller's name, or null for "Someone". */
    val caller: String?,
    val nameScore: Float,
    val callerSimilarity: Float,
    val timestampMs: Long,
)
