package com.sahara.app.core

import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.math.tanh

/**
 * Caller voice vector: the pretrained GE2E speaker encoder (Resemblyzer, 3-layer LSTM, 256 units)
 * in pure Kotlin, so no extra runtime is needed. Weights come from assets/speaker_encoder.bin,
 * written by `scripts/eval_name_call.py --export` (src/personalization/speaker.py has the same
 * maths in NumPy and is checked against PyTorch to 1e-7).
 *
 * Runs only after a name call fires (~0.2 s on a mid-range phone), never continuously.
 */
class SpeakerEncoder(private val tensors: Map<String, Tensor>) {
    class Tensor(val shape: IntArray, val data: FloatArray)

    private val melFb = tensors.getValue("mel_fb")
    private val nMels = melFb.shape[0]
    private val nBins = melFb.shape[1]
    private val window = FloatArray(N_FFT) { (0.5 - 0.5 * cos(2 * PI * it / N_FFT)).toFloat() }
    private val cosT = Array(nBins) { k -> FloatArray(N_FFT) { n -> cos(2 * PI * k * n / N_FFT).toFloat() } }
    private val sinT = Array(nBins) { k -> FloatArray(N_FFT) { n -> sin(2 * PI * k * n / N_FFT).toFloat() } }

    val dim: Int = tensors.getValue("linear.bias").data.size

    /** 256-d unit vector for the voice in [wave] (16 kHz mono floats). */
    fun embed(wave: FloatArray): FloatArray {
        val (a, b) = Endpoint.bounds(wave, dbBelowPeak = 30f)
        var w = normaliseVolume(wave.copyOfRange(a, b))
        val need = (PARTIAL_FRAMES - 1) * HOP
        if (w.size < need) w = w.copyOf(need)
        var mel = mel(w)
        if (mel.size > PARTIAL_FRAMES) {
            val start = (mel.size - PARTIAL_FRAMES) / 2
            mel = mel.copyOfRange(start, start + PARTIAL_FRAMES)
        }
        return forward(mel)
    }

    /** [frames][40] power mel spectrogram, centred STFT with zero padding (librosa-compatible). */
    fun mel(wave: FloatArray): Array<FloatArray> {
        val padded = FloatArray(wave.size + N_FFT)
        System.arraycopy(wave, 0, padded, N_FFT / 2, wave.size)
        val frames = 1 + (padded.size - N_FFT) / HOP
        val frame = FloatArray(N_FFT)
        val power = FloatArray(nBins)
        return Array(frames) { f ->
            for (n in 0 until N_FFT) frame[n] = padded[f * HOP + n] * window[n]
            for (k in 0 until nBins) {
                var re = 0f
                var im = 0f
                val c = cosT[k]
                val s = sinT[k]
                for (n in 0 until N_FFT) {
                    re += frame[n] * c[n]
                    im -= frame[n] * s[n]
                }
                power[k] = re * re + im * im
            }
            FloatArray(nMels) { m ->
                var acc = 0f
                val row = m * nBins
                for (k in 0 until nBins) acc += melFb.data[row + k] * power[k]
                acc
            }
        }
    }

    fun forward(mel: Array<FloatArray>): FloatArray {
        var x = mel
        for (layer in 0 until LAYERS) {
            val wIh = tensors.getValue("lstm.weight_ih_l$layer")
            val wHh = tensors.getValue("lstm.weight_hh_l$layer")
            val bIh = tensors.getValue("lstm.bias_ih_l$layer").data
            val bHh = tensors.getValue("lstm.bias_hh_l$layer").data
            val gates = wIh.shape[0]
            val inDim = wIh.shape[1]
            val h = FloatArray(gates / 4)
            val c = FloatArray(gates / 4)
            val hd = h.size
            val g = FloatArray(gates)
            val out = Array(x.size) { FloatArray(hd) }
            for (t in x.indices) {
                val xt = x[t]
                for (r in 0 until gates) {
                    var acc = bIh[r] + bHh[r]
                    val ri = r * inDim
                    for (i in 0 until inDim) acc += wIh.data[ri + i] * xt[i]
                    val rh = r * hd
                    for (i in 0 until hd) acc += wHh.data[rh + i] * h[i]
                    g[r] = acc
                }
                // PyTorch gate order: input, forget, cell, output.
                for (j in 0 until hd) {
                    val ig = sigmoid(g[j])
                    val fg = sigmoid(g[hd + j])
                    val cg = tanh(g[2 * hd + j])
                    val og = sigmoid(g[3 * hd + j])
                    c[j] = fg * c[j] + ig * cg
                    h[j] = og * tanh(c[j])
                }
                h.copyInto(out[t])
            }
            x = out
        }
        val lw = tensors.getValue("linear.weight")
        val lb = tensors.getValue("linear.bias").data
        val last = x.last()
        val e = FloatArray(lb.size) { r ->
            var acc = lb[r]
            for (i in last.indices) acc += lw.data[r * last.size + i] * last[i]
            max(acc, 0f)
        }
        return PrototypeBank.normalised(e)
    }

    companion object {
        const val N_FFT = 400
        const val HOP = 160
        const val PARTIAL_FRAMES = 160
        const val LAYERS = 3
        private const val TARGET_DBFS = -30f
        private val ORDER = listOf("mel_fb") +
            (0 until LAYERS).flatMap { l -> listOf("weight_ih", "weight_hh", "bias_ih", "bias_hh").map { "lstm.${it}_l$l" } } +
            listOf("linear.weight", "linear.bias")

        private fun sigmoid(v: Float): Float = (1.0 / (1.0 + exp(-v.toDouble().coerceIn(-30.0, 30.0)))).toFloat()

        fun normaliseVolume(w: FloatArray): FloatArray {
            var acc = 0.0
            for (v in w) acc += v.toDouble() * v
            val rms = sqrt(acc / max(w.size, 1)) + 1e-10
            val gain = 10.0.pow((TARGET_DBFS - 20 * log10(rms)) / 20).toFloat()
            return if (gain > 1f) FloatArray(w.size) { w[it] * gain } else w
        }

        /** Parses speaker_encoder.bin (see SpeakerEncoder.export in src/personalization/speaker.py). */
        fun parse(bytes: ByteArray): SpeakerEncoder {
            val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            val magic = ByteArray(4).also { buf.get(it) }
            require(String(magic, Charsets.US_ASCII) == "SPK1") { "not a speaker_encoder.bin" }
            val count = buf.int
            require(count == ORDER.size) { "expected ${ORDER.size} tensors, found $count" }
            val tensors = ORDER.associateWith {
                val shape = IntArray(buf.int) { buf.int }
                val data = FloatArray(shape.fold(1) { a, b -> a * b })
                buf.asFloatBuffer().get(data)
                buf.position(buf.position() + data.size * 4)
                Tensor(shape, data)
            }
            return SpeakerEncoder(tensors)
        }
    }
}

/** Energy endpointing shared by the name spotter and the speaker encoder (name_head.speech_bounds). */
object Endpoint {
    private const val FRAME = 400
    private const val HOP = 160

    /** Sample range of the loud core of an utterance: frames within [dbBelowPeak] of the peak, padded. */
    fun bounds(wave: FloatArray, dbBelowPeak: Float = 20f, padSec: Float = 0.15f): Pair<Int, Int> {
        if (wave.size < FRAME) return 0 to wave.size
        val n = 1 + (wave.size - FRAME) / HOP
        val db = FloatArray(n) { f ->
            var acc = 0.0
            for (i in f * HOP until f * HOP + FRAME) acc += wave[i].toDouble() * wave[i]
            (10 * log10(acc / FRAME + 1e-10)).toFloat()
        }
        val peak = db.max()
        var first = -1
        var last = -1
        for (f in 0 until n) if (db[f] >= peak - dbBelowPeak) {
            if (first < 0) first = f
            last = f
        }
        if (first < 0) return 0 to wave.size
        val pad = (padSec * 16_000).toInt()
        return max(0, first * HOP - pad) to min(wave.size, last * HOP + FRAME + pad)
    }
}
