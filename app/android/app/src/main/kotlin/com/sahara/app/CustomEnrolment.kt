package com.sahara.app

import android.annotation.SuppressLint
import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import com.sahara.app.core.CustomSound
import com.sahara.app.core.CustomSoundMatcher
import com.sahara.app.core.ModelConfig

/**
 * Enrolment of a user-chosen sound (Phase 7). One-time setup, like teaching a fingerprint:
 * the user records MIN_SAMPLES..MAX_SAMPLES short samples of the sound. Each sample is held in
 * memory, turned into window vectors exactly as live listening does, its loudest windows kept,
 * and the audio buffer zeroed straight away. Only the resulting prototype (1,024 floats) is saved,
 * on the phone; no audio is ever written to storage or sent anywhere. Listening pauses during
 * enrolment (one microphone user at a time) and resumes afterwards.
 */
object CustomEnrolment {
    const val SAMPLE_SECONDS = 3

    private val selected = mutableListOf<FloatArray>()
    private var engine: InferenceEngine? = null
    private var resumeListening = false

    @Volatile var samples = 0
        private set

    /** Records one sample (blocking: call off the main thread). Returns the state for the UI. */
    @Synchronized
    fun recordSample(context: Context): Map<String, Any?> {
        val app = context.applicationContext
        if (samples >= CustomSoundMatcher.MAX_SAMPLES) return state("Enough samples recorded")
        val e = engine ?: run {
            if (AudioCaptureService.isRunning) {
                resumeListening = true
                AudioCaptureService.stop(app)
                val until = System.currentTimeMillis() + 3_000
                while (AudioCaptureService.isRunning && System.currentTimeMillis() < until) Thread.sleep(50)
            }
            InferenceEngine.load(app)?.also { engine = it } ?: return state(ModelStatus.reason ?: "Model not available")
        }
        val pcm = record() ?: return state("Microphone unavailable")
        val windows = e.enrolmentVectors(pcm)
        pcm.fill(0) // the audio is not kept, not even in memory
        val picked = CustomSoundMatcher.loudest(windows)
        if (picked.isEmpty()) return state("Too quiet: hold the phone closer to the sound and try again")
        selected += picked
        samples++
        return state(null)
    }

    /** Saves the prototype under [name] once at least MIN_SAMPLES were recorded. */
    @Synchronized
    fun finish(context: Context, name: String): Map<String, Any?> {
        val app = context.applicationContext
        val clean = name.trim().take(40)
        if (clean.isEmpty()) return state("Give the sound a name")
        if (samples < CustomSoundMatcher.MIN_SAMPLES) return state("Record at least ${CustomSoundMatcher.MIN_SAMPLES} samples")
        val prototype = CustomSoundMatcher.prototype(selected, engine?.personalCenter)
        SaharaPrefs(app).saveCustomSound(CustomSound(clean, prototype))
        cleanup(app)
        return mapOf("saved" to true, "name" to clean)
    }

    @Synchronized
    fun cancel(context: Context) = cleanup(context.applicationContext)

    private fun cleanup(app: Context) {
        selected.clear()
        samples = 0
        engine?.close()
        engine = null
        if (resumeListening) AudioCaptureService.start(app)
        resumeListening = false
        AudioCaptureService.instance?.applySettings()
    }

    private fun state(error: String?) = mapOf(
        "samples" to samples,
        "minSamples" to CustomSoundMatcher.MIN_SAMPLES,
        "maxSamples" to CustomSoundMatcher.MAX_SAMPLES,
        "error" to error,
    )

    @SuppressLint("MissingPermission") // the UI only offers enrolment once the microphone is granted
    private fun record(): ShortArray? {
        val n = ModelConfig.SAMPLE_RATE * SAMPLE_SECONDS
        val minBuf = AudioRecord.getMinBufferSize(ModelConfig.SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        for (source in intArrayOf(MediaRecorder.AudioSource.VOICE_RECOGNITION, MediaRecorder.AudioSource.MIC)) {
            val r = try {
                AudioRecord(source, ModelConfig.SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT,
                    maxOf(minBuf, n * 2))
            } catch (_: Exception) { null } ?: continue
            if (r.state != AudioRecord.STATE_INITIALIZED) { r.release(); continue }
            val pcm = ShortArray(n)
            try {
                r.startRecording()
                var read = 0
                while (read < n) {
                    val got = r.read(pcm, read, n - read)
                    if (got < 0) return null
                    read += got
                }
            } finally {
                try { r.stop() } catch (_: Exception) {}
                r.release()
            }
            return pcm
        }
        return null
    }
}
