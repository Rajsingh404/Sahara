package com.sahara.app

import android.annotation.SuppressLint
import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import com.sahara.app.core.AudioWindow
import com.sahara.app.core.ModelConfig
import com.sahara.app.core.SpeakerEncoder

/**
 * Enrols a caller: they say the user's name once into the phone, and only the resulting 256-number
 * voice vector is stored (SaharaPrefs.saveCallers). Three or more recordings per caller work best.
 */
object CallerEnrolment {
    @Volatile private var encoder: SpeakerEncoder? = null

    sealed class Result {
        data class Enrolled(val name: String, val samples: Int) : Result()
        data class Failed(val code: String, val message: String) : Result()
    }

    /** Blocks for ~2.5 s while recording; call off the main thread. */
    fun enrol(context: Context, name: String): Result {
        val enc = encoder(context) ?: return Result.Failed("no_model", "speaker_encoder.bin is not bundled. Run app/tool/sync_assets.py")
        val audio = record(context, ModelConfig.ENROL_SAMPLES)
            ?: return Result.Failed("mic", "The microphone is busy or not allowed")
        if (AudioWindow.meanSquare(audio) < ModelConfig.VAD_ENERGY_THRESHOLD * 4) {
            return Result.Failed("too_quiet", "We could not hear anyone. Hold the phone closer and say the name again.")
        }
        val voice = enc.embed(audio)
        val prefs = SaharaPrefs(context)
        val bank = prefs.loadCallers()
        bank.add(name, voice)
        prefs.saveCallers(bank)
        AudioCaptureService.instance?.reloadCallers()
        return Result.Enrolled(name, bank.count(name))
    }

    fun remove(context: Context, name: String) {
        val prefs = SaharaPrefs(context)
        val bank = prefs.loadCallers()
        bank.remove(name)
        prefs.saveCallers(bank)
        AudioCaptureService.instance?.reloadCallers()
    }

    private fun encoder(context: Context): SpeakerEncoder? = encoder ?: synchronized(this) {
        encoder ?: InferenceEngine.readAssetBytes(context, "assets/speaker_encoder.bin")?.let(SpeakerEncoder::parse)?.also { encoder = it }
    }

    /** Uses the running service's audio when it is listening (one mic client), else opens the mic briefly. */
    private fun record(context: Context, samples: Int): FloatArray? {
        AudioCaptureService.instance?.takeIf { AudioCaptureService.isRunning }?.let { service ->
            return service.captureNext(samples, timeoutMs = 6_000)
        }
        return recordDirect(samples)
    }

    @SuppressLint("MissingPermission") // SaharaChannel checks RECORD_AUDIO first
    private fun recordDirect(samples: Int): FloatArray? {
        val minBuf = AudioRecord.getMinBufferSize(ModelConfig.SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val record = try {
            AudioRecord(
                MediaRecorder.AudioSource.VOICE_RECOGNITION, ModelConfig.SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT, maxOf(minBuf, samples * 2),
            )
        } catch (_: Exception) {
            return null
        }
        if (record.state != AudioRecord.STATE_INITIALIZED) {
            record.release()
            return null
        }
        val pcm = ShortArray(samples)
        return try {
            record.startRecording()
            var read = 0
            while (read < samples) {
                val n = record.read(pcm, read, samples - read)
                if (n < 0) return null
                read += n
            }
            FloatArray(samples) { pcm[it] / 32768f }
        } finally {
            try { record.stop() } catch (_: Exception) {}
            record.release()
        }
    }
}
