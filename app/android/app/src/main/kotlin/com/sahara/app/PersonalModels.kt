package com.sahara.app

import android.content.Context
import android.util.Log
import com.sahara.app.core.AssetParsers
import com.sahara.app.core.NameCallDetector
import com.sahara.app.core.NameSpotter
import com.sahara.app.core.OtherSoundsPolicy
import com.sahara.app.core.PrototypeBank
import com.sahara.app.core.SpeakerEncoder
import org.tensorflow.lite.Interpreter
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Everything beyond the eight trained classes, loaded from app/assets/ (see app/tool/sync_assets.py):
 *
 * - Name calls: `melspectrogram.tflite` + `embedding_model.tflite` (pretrained speech embeddings),
 *   `name_head.json` (the "Aryan" spotter) and `speaker_encoder.bin` (caller voices).
 * - Other sounds: `yamnet_class_map.csv` + `other_sounds.json`, using the scores yamnet.tflite exports.
 *
 * Each part is optional: a missing asset switches that feature off and [status] says why.
 */
class PersonalModels private constructor(
    val nameCalls: NameCallDetector?,
    val otherSounds: OtherSoundsPolicy?,
    /** YAMNet indices that mean someone is talking: name spotting runs only then. */
    val speechIndices: IntArray,
    private val speech: TfliteSpeechEmbedder?,
) : Closeable {

    /** Highest YAMNet speech score, or 1 when YAMNet scores are unavailable (gate open). */
    fun speechScore(yamnetScores: FloatArray?): Float {
        if (yamnetScores == null || speechIndices.isEmpty()) return 1f
        var best = 0f
        for (i in speechIndices) best = maxOf(best, yamnetScores[i])
        return best
    }

    override fun close() {
        speech?.close()
    }

    companion object {
        private const val TAG = "SaharaPersonal"
        const val SPEECH_GATE = 0.2f
        private val SPEECH_NAMES = setOf("Speech", "Shout", "Yell", "Child speech, kid speaking", "Conversation",
            "Male speech, man speaking", "Female speech, woman speaking", "Narration, monologue", "Children shouting")

        @Volatile var status: Map<String, Any?> = mapOf("nameCalls" to "not loaded yet", "otherSounds" to "not loaded yet")
            private set

        fun load(context: Context, hasYamnetScores: Boolean): PersonalModels {
            val prefs = SaharaPrefs(context)
            val classMap = InferenceEngine.readAssetText(context, "assets/yamnet_class_map.csv")
                ?.let(AssetParsers::parseYamnetClassMap).orEmpty()

            var otherReason: String? = null
            val other = when {
                !hasYamnetScores -> null.also { otherReason = "yamnet.tflite has no class scores. Re-run app/tool/export_yamnet_tflite.py" }
                classMap.size != 521 -> null.also { otherReason = "yamnet_class_map.csv missing. Run app/tool/sync_assets.py" }
                else -> {
                    val cfg = InferenceEngine.readAssetText(context, "assets/other_sounds.json")?.let(AssetParsers::parseOtherSounds)
                    OtherSoundsPolicy(classMap, cfg?.ignored.orEmpty(), cfg?.coveredByHead.orEmpty(), cfg?.minScore ?: 0.35f)
                        .also { it.enabled = prefs.otherSounds }
                }
            }

            var nameReason: String? = null
            var speech: TfliteSpeechEmbedder? = null
            val name = try {
                val headJson = InferenceEngine.readAssetText(context, "assets/name_head.json")
                val mel = InferenceEngine.readAssetBytes(context, "assets/melspectrogram.tflite")
                val emb = InferenceEngine.readAssetBytes(context, "assets/embedding_model.tflite")
                when {
                    headJson == null -> null.also { nameReason = "name_head.json missing. Run scripts/eval_name_call.py --export app/assets" }
                    mel == null || emb == null -> null.also { nameReason = "speech embedding models missing. Run app/tool/sync_assets.py" }
                    else -> {
                        speech = TfliteSpeechEmbedder(mel, emb)
                        val encoder = InferenceEngine.readAssetBytes(context, "assets/speaker_encoder.bin")?.let(SpeakerEncoder::parse)
                        if (encoder == null) nameReason = "speaker_encoder.bin missing: name calls work, callers are not identified"
                        NameCallDetector(NameSpotter(AssetParsers.parseNameHead(headJson)), speech!!, encoder, prefs.loadCallers())
                            .also { it.enabled = prefs.nameCalls }
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "name-call models failed to load", e)
                speech?.close()
                speech = null
                nameReason = "name-call models failed to load: ${e.message}"
                null
            }
            val speechIdx = classMap.withIndex().filter { it.value in SPEECH_NAMES }.map { it.index }.toIntArray()
            status = mapOf(
                "nameCalls" to nameReason,
                "nameReady" to (name != null),
                "name" to name?.displayName,
                "callerIdReady" to (name?.canIdentify == true),
                "otherSounds" to otherReason,
                "otherSoundsReady" to (other != null),
            )
            return PersonalModels(name, other, speechIdx, speech)
        }
    }
}

/** openWakeWord's two TFLite models: 16 kHz audio -> 32-bin mel -> 96-d embedding every 80 ms. */
class TfliteSpeechEmbedder(melModel: ByteArray, embeddingModel: ByteArray) : NameCallDetector.SpeechEmbedder, Closeable {
    private val mel = Interpreter(direct(melModel), Interpreter.Options().setNumThreads(1))
    private val emb = Interpreter(direct(embeddingModel), Interpreter.Options().setNumThreads(1))
    private var melInputLen = -1
    private var embBatch = -1

    @Synchronized
    override fun embed(wave: FloatArray): Array<FloatArray> {
        if (wave.size != melInputLen) {
            mel.resizeInput(0, intArrayOf(1, wave.size))
            mel.allocateTensors()
            melInputLen = wave.size
        }
        val inBuf = ByteBuffer.allocateDirect(wave.size * 4).order(ByteOrder.nativeOrder())
        for (v in wave) inBuf.putFloat(v.coerceIn(-1f, 1f) * 32767f)
        inBuf.rewind()
        val melOut = mel.getOutputTensor(0)
        val melBuf = ByteBuffer.allocateDirect(melOut.numBytes()).order(ByteOrder.nativeOrder())
        mel.run(inBuf, melBuf)
        melBuf.rewind()
        val spec = FloatArray(melOut.numElements()).also { melBuf.asFloatBuffer().get(it) }
        val frames = spec.size / MEL_BINS
        val count = if (frames >= MEL_WINDOW) 1 + (frames - MEL_WINDOW) / MEL_STEP else 0
        if (count == 0) return emptyArray()
        if (count != embBatch) {
            emb.resizeInput(0, intArrayOf(count, MEL_WINDOW, MEL_BINS, 1))
            emb.allocateTensors()
            embBatch = count
        }
        val embIn = ByteBuffer.allocateDirect(count * MEL_WINDOW * MEL_BINS * 4).order(ByteOrder.nativeOrder())
        for (w in 0 until count) {
            val start = w * MEL_STEP * MEL_BINS
            // openWakeWord's transform (x / 10 + 2) matches Google's speech_embedding front end.
            for (k in 0 until MEL_WINDOW * MEL_BINS) embIn.putFloat(spec[start + k] / 10f + 2f)
        }
        embIn.rewind()
        val out = ByteBuffer.allocateDirect(count * DIM * 4).order(ByteOrder.nativeOrder())
        emb.run(embIn, out)
        out.rewind()
        val fb = out.asFloatBuffer()
        return Array(count) { FloatArray(DIM).also { fb.get(it) } }
    }

    override fun close() {
        mel.close()
        emb.close()
    }

    companion object {
        const val MEL_BINS = 32
        const val MEL_WINDOW = 76
        const val MEL_STEP = 8
        const val DIM = 96

        private fun direct(bytes: ByteArray): ByteBuffer =
            ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder()).apply { put(bytes); rewind() }
    }
}
