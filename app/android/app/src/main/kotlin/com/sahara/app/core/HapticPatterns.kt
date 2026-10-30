package com.sahara.app.core

/**
 * Per-class vibration patterns, identical to the wristband's (firmware/esp32_wristband/src/patterns.cpp),
 * so a sound feels the same on the phone and on the band. tests/test_haptic_patterns.py checks they match.
 */
object HapticPatterns {
    /** One step: motor strength 0..255 for [ms] milliseconds. */
    data class Step(val level: Int, val ms: Long)

    /** [steps] played [repeats] times with [gapMs] of stillness between repeats. */
    data class Pattern(val steps: List<Step>, val repeats: Int, val gapMs: Long)

    private fun p(repeats: Int, gapMs: Long, vararg s: Pair<Int, Int>) =
        Pattern(s.map { Step(it.first, it.second.toLong()) }, repeats, gapMs)

    val BY_CLASS: Map<String, Pattern> = mapOf(
        "smoke_alarm" to p(3, 1500, 255 to 500, 0 to 500, 255 to 500, 0 to 500, 255 to 500),
        "doorbell" to p(2, 700, 255 to 150, 0 to 120, 160 to 350),
        "siren" to p(3, 300, 100 to 150, 150 to 150, 200 to 150, 255 to 300, 200 to 150, 150 to 150, 100 to 150),
        "knocking" to p(2, 600, 255 to 60, 0 to 100, 255 to 60, 0 to 100, 255 to 60, 0 to 100, 255 to 60),
        "dog_bark" to p(2, 600, 220 to 120, 0 to 150, 220 to 120),
        "baby_cry" to p(2, 600, 180 to 400, 0 to 150, 120 to 250, 0 to 150, 180 to 400),
        "glass_break" to p(2, 800, 255 to 400, 0 to 120, 200 to 40, 0 to 40, 200 to 40, 0 to 40, 200 to 40, 0 to 40, 200 to 40),
        "appliance_beep" to p(1, 0, 130 to 80, 0 to 150, 130 to 80, 0 to 150, 130 to 80),
        Detection.NAME_CALL to p(2, 800, 220 to 400, 0 to 150, 220 to 400, 0 to 150, 220 to 150, 0 to 150, 220 to 150),
    )

    /** Anything else (custom sounds): one gentle pulse, never a safety pattern; the band does the same. */
    val DEFAULT: Pattern = p(1, 0, 255 to 300)

    // Below this a phone motor, like the band's coin motor, is barely felt; non-zero steps are raised to it.
    private const val MIN_LEVEL = 90

    fun forLabel(label: String): Pattern = BY_CLASS[label] ?: DEFAULT

    /** Timings and amplitudes for VibrationEffect.createWaveform, repeats and gaps unrolled. */
    fun waveform(pattern: Pattern): Pair<LongArray, IntArray> {
        val ms = ArrayList<Long>()
        val amp = ArrayList<Int>()
        repeat(pattern.repeats) { r ->
            for (s in pattern.steps) {
                ms += s.ms
                amp += if (s.level == 0) 0 else s.level.coerceIn(MIN_LEVEL, 255)
            }
            if (r < pattern.repeats - 1 && pattern.gapMs > 0) {
                ms += pattern.gapMs
                amp += 0
            }
        }
        return ms.toLongArray() to amp.toIntArray()
    }

    /** On/off timings (off first) for phones without amplitude control. */
    fun onOff(pattern: Pattern): LongArray {
        val (ms, amp) = waveform(pattern)
        val out = arrayListOf(0L)
        var on = false
        for (i in ms.indices) {
            val stepOn = amp[i] > 0
            if (stepOn == on) out[out.size - 1] = out.last() + ms[i] else { out += ms[i]; on = stepOn }
        }
        return out.toLongArray()
    }
}
