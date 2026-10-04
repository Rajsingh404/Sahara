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
 * - `labels.txt`, `class_thresholds.json`.
 *
 * Audio never leaves this class except as 8 class scores.
 */
class InferenceEngine private constructor(
    val labels: List<String>,
    val thresholds: Map<String, Float>,
    private val classifier: Interpreter,
    private val yamnet: Interpreter?,
) : Closeable {

    private val headIsEmbeddingModel = yamnet != null
    private val pool = EmbeddingPool(ModelConfig.EMBEDDING_DIM, ModelConfig.EMBEDDING_POOL)
    private val classifierInputLen = classifier.getInputTensor(0).numElements()

    /** [window] is the latest audio, 16 kHz mono floats, oldest first. Returns one score per label. */
    fun predict(window: FloatArray): FloatArray {
        val scores = if (headIsEmbeddingModel) {
            val embedding = embed(window)
            runSingle(classifier, pool.add(embedding))
        } else {
            runSingle(classifier, tail(window, classifierInputLen))
        }
        return if (scores.size == labels.size) scores else scores.copyOf(labels.size)
    }

    fun resetState() = pool.clear()

    private fun embed(window: FloatArray): FloatArray {
        val net = yamnet!!
        val inputLen = net.getInputTensor(0).numElements()
        val input = tail(window, inputLen)
        val outIndex = (0 until net.outputTensorCount).first { i ->
            net.getOutputTensor(i).shape().lastOrNull() == ModelConfig.EMBEDDING_DIM
        }
        val outTensor = net.getOutputTensor(outIndex)
        val inBuf = floatBuffer(input)
        val outBuf = ByteBuffer.allocateDirect(outTensor.numBytes()).order(ByteOrder.nativeOrder())
        val outputs = HashMap<Int, Any>()
        for (i in 0 until net.outputTensorCount) {
            outputs[i] = if (i == outIndex) outBuf
            else ByteBuffer.allocateDirect(net.getOutputTensor(i).numBytes()).order(ByteOrder.nativeOrder())
        }
        net.runForMultipleInputsOutputs(arrayOf(inBuf), outputs)
        outBuf.rewind()
        val frames = FloatArray(outTensor.numElements())
        outBuf.asFloatBuffer().get(frames)
        // [frames, 1024] -> mean over frames.
        val n = frames.size / ModelConfig.EMBEDDING_DIM
        val mean = FloatArray(ModelConfig.EMBEDDING_DIM)
        for (f in 0 until n) for (d in mean.indices) mean[d] += frames[f * ModelConfig.EMBEDDING_DIM + d]
        for (d in mean.indices) mean[d] /= n.coerceAtLeast(1)
        return mean
    }

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

        fun readAssetText(context: Context, asset: String): String? = readAsset(context, asset)?.toString(Charsets.UTF_8)

        fun loadLabels(context: Context): List<String> =
            readAssetText(context, LABELS_ASSET)?.let(AssetParsers::parseLabels).orEmpty()

        fun loadThresholds(context: Context): Map<String, Float> =
            readAssetText(context, THRESHOLDS_ASSET)?.let(AssetParsers::parseThresholds).orEmpty()

        /** Returns null (with [ModelStatus.reason] set) when the model assets are missing or invalid. */
        fun load(context: Context): InferenceEngine? {
            val labels = loadLabels(context)
            if (labels.isEmpty()) return fail("labels.txt missing or empty")
            val classifierBytes = readAsset(context, CLASSIFIER_ASSET)
                ?: return fail("sahara_classifier.tflite not bundled. Run app/tool/sync_assets.py")
            return try {
                val options = Interpreter.Options().setNumThreads(2)
                val classifier = Interpreter(direct(classifierBytes), options)
                val needsYamnet = classifier.getInputTensor(0).numElements() == ModelConfig.EMBEDDING_DIM
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
                InferenceEngine(labels, loadThresholds(context), classifier, yamnet)
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
