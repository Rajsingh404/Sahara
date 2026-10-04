package com.sahara.app.core

/**
 * Fixed-size sliding window over 16 kHz mono audio, fed in hop-sized PCM16 chunks.
 * [snapshot] returns the latest [size] samples as floats in [-1, 1], oldest first.
 */
class AudioWindow(val size: Int) {
    private val buffer = FloatArray(size)
    private var writePos = 0
    private var filled = 0

    val isFull: Boolean get() = filled >= size

    fun push(pcm: ShortArray, count: Int = pcm.size) {
        for (k in 0 until count) {
            buffer[writePos] = pcm[k] / 32768f
            writePos = (writePos + 1) % size
        }
        filled = (filled + count).coerceAtMost(size)
    }

    fun snapshot(out: FloatArray = FloatArray(size)): FloatArray {
        val tail = size - writePos
        System.arraycopy(buffer, writePos, out, 0, tail)
        System.arraycopy(buffer, 0, out, tail, writePos)
        return out
    }

    fun clear() {
        buffer.fill(0f)
        writePos = 0
        filled = 0
    }

    companion object {
        /** Mean square energy, the same measure as VAD_ENERGY_THRESHOLD in src/config.py. */
        fun meanSquare(samples: FloatArray, from: Int = 0, to: Int = samples.size): Float {
            if (to <= from) return 0f
            var acc = 0.0
            for (i in from until to) acc += samples[i] * samples[i]
            return (acc / (to - from)).toFloat()
        }
    }
}
