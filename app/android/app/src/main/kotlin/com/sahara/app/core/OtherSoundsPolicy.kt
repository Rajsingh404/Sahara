package com.sahara.app.core

/**
 * Open-set layer: reports any of YAMNet's 521 AudioSet classes as a low-priority "other sound"
 * when none of the eight trained safety classes is firing.
 *
 * The eight classes stay the only ones that can raise a full alert. This layer only says what
 * else is going on ("Water tap", "Cat", "Shout"), so a Deaf user is not left with silence for
 * every sound the head was never trained on. YAMNet's AudioSet labels are noisier than the
 * trained head, so the policy needs [hitsRequired] of the last [windowCount] windows and repeats
 * a label at most once per [cooldownMs].
 *
 * Pure Kotlin so it runs in JVM unit tests. The ignored and covered names come from
 * assets/other_sounds.json, written by app/tool/sync_assets.py from src/config.py.
 */
class OtherSoundsPolicy(
    private val classNames: List<String>,
    ignored: Set<String>,
    /** YAMNet classes already covered by the trained head (e.g. "Doorbell"); never reported twice. */
    coveredByHead: Set<String> = emptySet(),
    private val minScore: Float = 0.35f,
    private val hitsRequired: Int = 2,
    private val windowCount: Int = 3,
    private val cooldownMs: Long = 30_000L,
) {
    private val skip = BooleanArray(classNames.size) { classNames[it] in ignored || classNames[it] in coveredByHead }
    private val recentTop = ArrayDeque<Int>()
    private val lastReported = HashMap<Int, Long>()

    var enabled: Boolean = true

    /**
     * [scores]: YAMNet's 521 scores for this window (mean over its frames).
     * [headFired]: a trained class fired on this window; the other-sounds layer stays quiet.
     */
    fun update(scores: FloatArray, nowMs: Long, headFired: Boolean): OtherSound? {
        require(scores.size == classNames.size) { "expected ${classNames.size} YAMNet scores, got ${scores.size}" }
        var best = -1
        var bestScore = minScore
        for (i in scores.indices) {
            if (!skip[i] && scores[i] >= bestScore) {
                best = i
                bestScore = scores[i]
            }
        }
        recentTop.addLast(best)
        while (recentTop.size > windowCount) recentTop.removeFirst()
        if (!enabled || headFired || best < 0) return null
        if (recentTop.count { it == best } < hitsRequired) return null
        val last = lastReported[best]
        if (last != null && nowMs - last < cooldownMs) return null
        lastReported[best] = nowMs
        return OtherSound(best, classNames[best], bestScore, nowMs)
    }

    fun reset() {
        recentTop.clear()
        lastReported.clear()
    }
}

data class OtherSound(val yamnetIndex: Int, val name: String, val score: Float, val timestampMs: Long)
