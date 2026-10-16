package com.sahara.app

import android.annotation.SuppressLint
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.os.SystemClock
import android.util.Log
import com.sahara.app.core.AudioWindow
import com.sahara.app.core.Detection
import com.sahara.app.core.DetectionPolicy
import com.sahara.app.core.ModelConfig
import com.sahara.app.core.NameCall
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

/**
 * Foreground service: continuous mic capture → VAD → on-device TFLite → thresholds → alert.
 *
 * Every 0.5 s hop runs three layers on the same audio:
 * 1. the eight trained classes (full alerts, safety-critical ones first);
 * 2. name calls: when YAMNet hears speech, the name spotter checks the last 2 s, and if the name
 *    is there the caller is identified off the capture thread ("Raj is calling you" / "Someone");
 * 3. other sounds: YAMNet's own 521 classes as low-priority notices when (1) is quiet.
 *
 * Audio stays in 1 s and 2 s in-memory ring buffers, overwritten every 0.5 s. Nothing is written
 * to disk or sent anywhere; only the class id of a detection leaves this service.
 */
class AudioCaptureService : Service() {

    @Volatile private var running = false
    private var worker: Thread? = null
    private var wakeLock: PowerManager.WakeLock? = null
    @Volatile private var policy: DetectionPolicy? = null
    @Volatile private var personal: PersonalModels? = null
    private val identifyThread = Executors.newSingleThreadExecutor { Thread(it, "sahara-caller-id") }
    private val taps = CopyOnWriteArrayList<Tap>()

    /** A request for the next [buffer].size samples of live audio (caller enrolment). */
    private class Tap(val buffer: FloatArray) {
        var filled = 0
        val done = ArrayBlockingQueue<FloatArray>(1)
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            stopSelf()
            return START_NOT_STICKY
        }
        if (running) return START_STICKY
        try {
            if (Build.VERSION.SDK_INT >= 30) {
                startForeground(NOTIFICATION_ID, buildNotification(), ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE)
            } else {
                startForeground(NOTIFICATION_ID, buildNotification())
            }
        } catch (e: Exception) {
            // Android 14+ refuses a microphone FGS without RECORD_AUDIO or from the background.
            Log.e(TAG, "could not start foreground", e)
            lastError = "Could not start listening: ${e.message}"
            stopSelf()
            return START_NOT_STICKY
        }
        startCapture()
        return START_STICKY
    }

    private fun startCapture() {
        val engine = InferenceEngine.load(this)
        if (engine == null) {
            lastError = ModelStatus.reason
            emitState()
            stopSelf()
            return
        }
        policy = DetectionPolicy(engine.labels, engine.thresholds)
        personal = PersonalModels.load(this, engine.hasYamnetScores)
        applySettings()
        running = true
        lastError = null
        instance = this
        wakeLock = getSystemService(PowerManager::class.java)
            ?.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "sahara:capture")
            ?.apply { setReferenceCounted(false); acquire() }
        Wristband.get(this).connectSaved()
        worker = Thread({ captureLoop(engine) }, "sahara-capture").apply { start() }
        emitState()
    }

    @SuppressLint("MissingPermission") // checked in SaharaChannel before starting the service
    private fun captureLoop(engine: InferenceEngine) {
        val minBuf = AudioRecord.getMinBufferSize(
            ModelConfig.SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT,
        )
        val record = try {
            createRecord(MediaRecorder.AudioSource.VOICE_RECOGNITION, minBuf)
                ?: createRecord(MediaRecorder.AudioSource.MIC, minBuf)
        } catch (e: SecurityException) {
            null
        }
        if (record == null) {
            lastError = "Microphone unavailable"
            running = false
            engine.close()
            emitState()
            stopSelf()
            return
        }
        val window = AudioWindow(ModelConfig.WINDOW_SAMPLES)
        val speechWindow = AudioWindow(ModelConfig.NAME_WINDOW_SAMPLES)
        val hop = ShortArray(ModelConfig.HOP_SAMPLES)
        val snapshot = FloatArray(ModelConfig.WINDOW_SAMPLES)
        val speechSnapshot = FloatArray(ModelConfig.NAME_WINDOW_SAMPLES)
        val silent = FloatArray(engine.labels.size)
        try {
            record.startRecording()
            while (running) {
                var read = 0
                while (read < hop.size && running) {
                    val n = record.read(hop, read, hop.size - read)
                    if (n < 0) throw IllegalStateException("AudioRecord.read error $n")
                    read += n
                }
                if (!running) break
                window.push(hop, read)
                speechWindow.push(hop, read)
                feedTaps(hop, read)
                if (!window.isFull) continue
                window.snapshot(snapshot)
                val energy = AudioWindow.meanSquare(snapshot, snapshot.size - hop.size, snapshot.size)
                val now = System.currentTimeMillis()
                val p = policy ?: break
                if (energy < ModelConfig.VAD_ENERGY_THRESHOLD) {
                    p.update(silent, now)
                    emitScores(silent, energy)
                    continue
                }
                val started = SystemClock.elapsedRealtime()
                val prediction = engine.predict(snapshot)
                val fired = p.update(prediction.scores, now)
                for (d in fired) AlertDispatcher.dispatch(this, d)
                personal?.let { extra ->
                    extra.otherSounds?.update(prediction.yamnetScores ?: return@let, now, headFired = fired.isNotEmpty())
                        ?.let { AlertDispatcher.dispatch(this, Detection.otherSound(it)) }
                }
                personal?.let { extra ->
                    val names = extra.nameCalls ?: return@let
                    if (!speechWindow.isFull || extra.speechScore(prediction.yamnetScores) < PersonalModels.SPEECH_GATE) return@let
                    speechWindow.snapshot(speechSnapshot)
                    val score = names.spot(speechSnapshot, now) ?: return@let
                    val audio = speechSnapshot.copyOf()
                    identifyThread.execute {
                        val match = names.identify(audio)
                        AlertDispatcher.dispatch(this, Detection.nameCall(NameCall(match?.name, score, match?.similarity ?: 0f, now)))
                    }
                }
                lastInferenceMs = SystemClock.elapsedRealtime() - started
                emitScores(prediction.scores, energy)
            }
        } catch (e: Exception) {
            Log.e(TAG, "capture loop stopped", e)
            lastError = "Listening stopped: ${e.message}"
        } finally {
            try { record.stop() } catch (_: Exception) {}
            record.release()
            engine.close()
            personal?.close()
            personal = null
            if (running) {
                // Loop died on an error: drop the foreground notification instead of lying about listening.
                running = false
                stopSelf()
            }
        }
    }

    @SuppressLint("MissingPermission")
    private fun createRecord(source: Int, minBuf: Int): AudioRecord? {
        val r = AudioRecord(
            source, ModelConfig.SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT,
            maxOf(minBuf, ModelConfig.HOP_SAMPLES * 2 * 2),
        )
        return if (r.state == AudioRecord.STATE_INITIALIZED) r else r.release().let { null }
    }

    fun applySettings() {
        val prefs = SaharaPrefs(this)
        policy?.let { p ->
            prefs.enabledClasses?.let { p.enabled = it }
            p.sensitivity = prefs.sensitivity
        }
        personal?.nameCalls?.let {
            it.enabled = prefs.nameCalls
            it.sensitivity = prefs.sensitivity
        }
        personal?.otherSounds?.enabled = prefs.otherSounds
    }

    fun reloadCallers() {
        personal?.nameCalls?.callers = SaharaPrefs(this).loadCallers()
    }

    /** Next [samples] of live audio, or null if the service stops or [timeoutMs] passes first. */
    fun captureNext(samples: Int, timeoutMs: Long): FloatArray? {
        val tap = Tap(FloatArray(samples))
        taps += tap
        return try {
            tap.done.poll(timeoutMs, TimeUnit.MILLISECONDS)
        } finally {
            taps -= tap
        }
    }

    private fun feedTaps(pcm: ShortArray, count: Int) {
        for (t in taps) {
            val n = minOf(count, t.buffer.size - t.filled)
            for (k in 0 until n) t.buffer[t.filled + k] = pcm[k] / 32768f
            t.filled += n
            if (t.filled == t.buffer.size) {
                taps -= t
                t.done.offer(t.buffer)
            }
        }
    }

    private fun emitScores(scores: FloatArray, energy: Float) {
        if (!SaharaEvents.hasListeners) return
        SaharaEvents.emit(
            mapOf(
                "type" to "scores",
                "scores" to scores.map { it.toDouble() },
                "energy" to energy.toDouble(),
                "inferenceMs" to lastInferenceMs,
            ),
        )
    }

    override fun onDestroy() {
        running = false
        worker?.join(1500)
        worker = null
        identifyThread.shutdown()
        wakeLock?.let { if (it.isHeld) it.release() }
        wakeLock = null
        if (instance === this) instance = null
        emitState()
        super.onDestroy()
    }

    private fun buildNotification(): Notification {
        val nm = getSystemService(NotificationManager::class.java)
        if (nm != null && nm.getNotificationChannel(LISTEN_CHANNEL) == null) {
            nm.createNotificationChannel(
                NotificationChannel(LISTEN_CHANNEL, "Listening", NotificationManager.IMPORTANCE_LOW).apply {
                    description = "Shown while SAHARA listens for sounds on this phone"
                    setShowBadge(false)
                },
            )
        }
        val open = PendingIntent.getActivity(
            this, 0, Intent(this, MainActivity::class.java), PendingIntent.FLAG_IMMUTABLE,
        )
        val stop = PendingIntent.getService(
            this, 1, Intent(this, AudioCaptureService::class.java).setAction(ACTION_STOP), PendingIntent.FLAG_IMMUTABLE,
        )
        val builder = Notification.Builder(this, LISTEN_CHANNEL)
            .setSmallIcon(R.mipmap.ic_launcher)
            .setContentTitle("SAHARA is listening")
            .setContentText("Sounds are analysed on this phone only. Nothing is recorded.")
            .setOngoing(true)
            .setContentIntent(open)
            .addAction(Notification.Action.Builder(null, "Stop", stop).build())
        if (Build.VERSION.SDK_INT >= 31) builder.setForegroundServiceBehavior(Notification.FOREGROUND_SERVICE_IMMEDIATE)
        return builder.build()
    }

    companion object {
        private const val TAG = "SaharaCapture"
        private const val LISTEN_CHANNEL = "sahara_listening"
        private const val NOTIFICATION_ID = 1001
        const val ACTION_STOP = "com.sahara.app.STOP"

        @Volatile var instance: AudioCaptureService? = null
            private set
        @Volatile var lastError: String? = null
        @Volatile var lastInferenceMs: Long = 0

        val isRunning: Boolean get() = instance?.running == true

        fun start(context: Context) {
            context.startForegroundService(Intent(context, AudioCaptureService::class.java))
        }

        fun stop(context: Context) {
            context.stopService(Intent(context, AudioCaptureService::class.java))
        }

        fun stateMap(): Map<String, Any?> = mapOf(
            "listening" to isRunning,
            "error" to lastError,
            "modelReason" to ModelStatus.reason,
        )

        fun emitState() = SaharaEvents.emit(mapOf("type" to "state") + stateMap())
    }
}
