package com.sahara.app

import android.content.Context
import android.util.Log
import com.sahara.app.core.AssetParsers
import com.sahara.app.core.EmbeddingPool
import com.sahara.app.core.ModelConfig
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Tensor
import java.io.Closeable
import java.io.FileNotFoundException
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * On-device TFLite inference. Loads from the Flutter asset bundle (app/assets/):
 *
 * - `sahara_classifier.tflite`: either the Dense head exported by src/export (input = 1024-d
 *   YAMNet embedding, output = 8 sigmoid scores, FP32 or INT8), or a combined waveform model.
 * - `yamnet.tflite`: YAMNet with an embeddings output. Needed only when the classifier is the head.
 *   When it also has YAMNet's 521-class scores output (app/tool/export_yamnet_tflite.py), those
 *   feed the open-set "other sounds" layer and the speech gate for name calls.
 * - `labels.txt`, `class_thresholds.json`.
 *
 * Audio never leaves this class except as class scores.
 */
class InferenceEngine private constructor(
    val labels: List<String>,
    val thresholds: Map<String, Float>,
    val hitsRequired: Map<String, Int>,
    private val classifier: Interpreter,
    private val yamnet: Interpreter?,
    /** Mean background window (assets/personal_center.bin) for custom-sound matching, or null. */
    val personalCenter: FloatArray?,
) : Closeable {

    /** The 2-hop window vector of the last [predict] call (first 1,024 values of the head input). */
    @Volatile var lastWindowVector: FloatArray? = null
        private set

    private val headIsEmbeddingModel = yamnet != null
    private val classifierInputLen = classifier.getInputTensor(0).numElements()
    private val pool = EmbeddingPool(
        ModelConfig.EMBEDDING_DIM,
        *(ModelConfig.poolsForHeadInput(classifierInputLen) ?: intArrayOf(ModelConfig.EMBEDDING_POOL)),
    )

    /** Scores for one window: the trained head's, and YAMNet's own 521 when the model exports them. */
    class Prediction(val scores: FloatArray, val yamnetScores: FloatArray?)

    /** [window] is the latest audio, 16 kHz mono floats, oldest first. */
    fun predict(window: FloatArray): Prediction {
        var yamnetScores: FloatArray? = null
        val scores = if (headIsEmbeddingModel) {
            val (embedding, yScores) = embed(window)
            yamnetScores = yScores
            val headInput = pool.add(embedding)
            lastWindowVector = headInput.copyOf(ModelConfig.EMBEDDING_DIM)
            runSingle(classifier, headInput)
        } else {
            runSingle(classifier, tail(window, classifierInputLen))
        }
        return Prediction(if (scores.size == labels.size) scores else scores.copyOf(labels.size), yamnetScores)
    }

    /** True when yamnet.tflite has the 521-class scores output (needed for "other sounds"). */
    val hasYamnetScores: Boolean = yamnet?.let { y -> scoresIndex(y) >= 0 } ?: false

    fun resetState() = pool.clear()

    /**
     * Window vectors of a recorded enrolment sample, formed exactly as live listening forms them
     * (1 s ring buffer, 0.5 s hops, VAD, newest 15,600 samples, 2-embedding pool), each with the
     * energy of its newest hop. The PCM is only read here; nothing is stored.
     */
    fun enrolmentVectors(pcm: ShortArray): List<Pair<FloatArray, Float>> {
        if (yamnet == null) return emptyList()
        val padded = if (pcm.size >= ModelConfig.WINDOW_SAMPLES) pcm
        else ShortArray(ModelConfig.WINDOW_SAMPLES).also { System.arraycopy(pcm, 0, it, it.size - pcm.size, pcm.size) }
        val window = com.sahara.app.core.AudioWindow(ModelConfig.WINDOW_SAMPLES)
        val samplePool = EmbeddingPool(ModelConfig.EMBEDDING_DIM, ModelConfig.EMBEDDING_POOL)
        val snapshot = FloatArray(ModelConfig.WINDOW_SAMPLES)
        val out = mutableListOf<Pair<FloatArray, Float>>()
        var start = 0
        while (start + ModelConfig.HOP_SAMPLES <= padded.size) {
            window.push(padded.copyOfRange(start, start + ModelConfig.HOP_SAMPLES))
            start += ModelConfig.HOP_SAMPLES
            if (!window.isFull) continue
            window.snapshot(snapshot)
            val energy = com.sahara.app.core.AudioWindow.meanSquare(
                snapshot, snapshot.size - ModelConfig.HOP_SAMPLES, snapshot.size,
            )
            if (energy < ModelConfig.VAD_ENERGY_THRESHOLD) continue
            out += samplePool.add(embed(snapshot).first) to energy
        }
        return out
    }

    /** Mean YAMNet embedding over the window's frames, plus mean class scores when exported. */
    private fun embed(window: FloatArray): Pair<FloatArray, FloatArray?> {
        val net = yamnet!!
        val inputLen = net.getInputTensor(0).numElements()
        val input = tail(window, inputLen)
        val outIndex = (0 until net.outputTensorCount).first { i ->
            net.getOutputTensor(i).shape().lastOrNull() == ModelConfig.EMBEDDING_DIM
        }
        val scoresIdx = scoresIndex(net)
        val inBuf = floatBuffer(input)
        val outputs = HashMap<Int, Any>()
        for (i in 0 until net.outputTensorCount) {
            outputs[i] = ByteBuffer.allocateDirect(net.getOutputTensor(i).numBytes()).order(ByteOrder.nativeOrder())
        }
        net.runForMultipleInputsOutputs(arrayOf(inBuf), outputs)
        val embedding = frameMean(outputs[outIndex] as ByteBuffer, net.getOutputTensor(outIndex).numElements(), ModelConfig.EMBEDDING_DIM)
        val scores = if (scoresIdx >= 0) {
            frameMean(outputs[scoresIdx] as ByteBuffer, net.getOutputTensor(scoresIdx).numElements(), ModelConfig.YAMNET_CLASSES)
        } else null
        return embedding to scores
    }

    /** [frames, dim] -> mean over frames. */
    private fun frameMean(buf: ByteBuffer, elements: Int, dim: Int): FloatArray {
        buf.rewind()
        val frames = FloatArray(elements)
        buf.asFloatBuffer().get(frames)
        val n = frames.size / dim
        val mean = FloatArray(dim)
        for (f in 0 until n) for (d in mean.indices) mean[d] += frames[f * dim + d]
        for (d in mean.indices) mean[d] /= n.coerceAtLeast(1)
        return mean
    }

    private fun scoresIndex(net: Interpreter): Int = (0 until net.outputTensorCount).firstOrNull { i ->
        net.getOutputTensor(i).shape().lastOrNull() == ModelConfig.YAMNET_CLASSES
    } ?: -1

    private fun runSingle(net: Interpreter, values: FloatArray): FloatArray {
        val inT = net.getInputTensor(0)
        val outT = net.getOutputTensor(0)
        val inBuf = when (inT.dataType()) {
            DataType.INT8 -> quantize(values, inT)
            else -> floatBuffer(values)
        }
        val outBuf = ByteBuffer.allocateDirect(outT.numBytes()).order(ByteOrder.nativeOrder())
        net.run(inBuf, outBuf)
        outBuf.rewind()
        return when (outT.dataType()) {
            DataType.INT8 -> {
                val q = outT.quantizationParams()
                FloatArray(outT.numElements()) { (outBuf.get().toInt() - q.zeroPoint) * q.scale }
            }
            else -> FloatArray(outT.numElements()).also { outBuf.asFloatBuffer().get(it) }
        }
    }

    private fun quantize(values: FloatArray, tensor: Tensor): ByteBuffer {
        val q = tensor.quantizationParams()
        val buf = ByteBuffer.allocateDirect(values.size).order(ByteOrder.nativeOrder())
        for (v in values) buf.put((Math.round(v / q.scale) + q.zeroPoint).coerceIn(-128, 127).toByte())
        buf.rewind()
        return buf
    }

    private fun floatBuffer(values: FloatArray): ByteBuffer {
        val buf = ByteBuffer.allocateDirect(values.size * 4).order(ByteOrder.nativeOrder())
        buf.asFloatBuffer().put(values)
        return buf
    }

    private fun tail(window: FloatArray, len: Int): FloatArray = when {
        window.size == len -> window
        window.size > len -> window.copyOfRange(window.size - len, window.size)
        else -> FloatArray(len).also { System.arraycopy(window, 0, it, len - window.size, window.size) }
    }

    override fun close() {
        classifier.close()
        yamnet?.close()
    }

    companion object {
        private const val TAG = "SaharaInference"
        const val CLASSIFIER_ASSET = "assets/sahara_classifier.tflite"
        const val YAMNET_ASSET = "assets/yamnet.tflite"
        const val LABELS_ASSET = "assets/labels.txt"
        const val THRESHOLDS_ASSET = "assets/class_thresholds.json"
        const val PERSONAL_CENTER_ASSET = "assets/personal_center.bin"

        fun readAssetText(context: Context, asset: String): String? = readAsset(context, asset)?.toString(Charsets.UTF_8)

        fun readAssetBytes(context: Context, asset: String): ByteArray? = readAsset(context, asset)

        fun loadLabels(context: Context): List<String> =
            readAssetText(context, LABELS_ASSET)?.let(AssetParsers::parseLabels).orEmpty()

        fun loadThresholds(context: Context): Map<String, Float> =
            readAssetText(context, THRESHOLDS_ASSET)?.let(AssetParsers::parseThresholds).orEmpty()

        fun loadHitsRequired(context: Context): Map<String, Int> =
            readAssetText(context, THRESHOLDS_ASSET)?.let(AssetParsers::parseHitsRequired).orEmpty()

        /** Returns null (with [ModelStatus.reason] set) when the model assets are missing or invalid. */
        fun load(context: Context): InferenceEngine? {
            val labels = loadLabels(context)
            if (labels.isEmpty()) return fail("labels.txt missing or empty")
            val classifierBytes = readAsset(context, CLASSIFIER_ASSET)
                ?: return fail("sahara_classifier.tflite not bundled. Run app/tool/sync_assets.py")
            return try {
                val options = Interpreter.Options().setNumThreads(2)
                val classifier = Interpreter(direct(classifierBytes), options)
                val needsYamnet = ModelConfig.poolsForHeadInput(classifier.getInputTensor(0).numElements()) != null
                val yamnet = if (needsYamnet) {
                    val bytes = readAsset(context, YAMNET_ASSET)
                    if (bytes == null) {
                        classifier.close()
                        return fail("yamnet.tflite not bundled (the classifier is the embedding head)")
                    }
                    Interpreter(direct(bytes), options).also { y ->
                        val hasEmbeddings = (0 until y.outputTensorCount).any { i ->
                            y.getOutputTensor(i).shape().lastOrNull() == ModelConfig.EMBEDDING_DIM
                        }
                        if (!hasEmbeddings) {
                            y.close()
                            classifier.close()
                            return fail("yamnet.tflite has no 1024-d embeddings output. Use app/tool/export_yamnet_tflite.py")
                        }
                    }
                } else null
                ModelStatus.reason = null
                val center = com.sahara.app.core.CustomSoundMatcher.parseCenter(
                    readAsset(context, PERSONAL_CENTER_ASSET), ModelConfig.EMBEDDING_DIM,
                )
                InferenceEngine(labels, loadThresholds(context), loadHitsRequired(context), classifier, yamnet, center)
            } catch (e: Exception) {
                Log.e(TAG, "model load failed", e)
                fail("model failed to load: ${e.message}")
            }
        }

        private fun fail(reason: String): InferenceEngine? {
            Log.w(TAG, reason)
            ModelStatus.reason = reason
            return null
        }

        private fun direct(bytes: ByteArray): ByteBuffer =
            ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder()).apply { put(bytes); rewind() }

        private fun readAsset(context: Context, asset: String): ByteArray? = try {
            // Flutter bundles pubspec assets under flutter_assets/ in the APK. Reading the path
            // directly works even when the service is restarted without a Flutter engine.
            context.assets.open("flutter_assets/$asset").use { it.readBytes() }
        } catch (_: FileNotFoundException) {
            null
        } catch (e: Exception) {
            Log.w(TAG, "could not read $asset", e)
            null
        }
    }
}

/** Last model-load problem, surfaced to the UI. */
object ModelStatus {
    @Volatile var reason: String? = "not loaded yet"
}
