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
import com.sahara.app.core.DetectionPolicy
import com.sahara.app.core.ModelConfig

/**
 * Foreground service: continuous mic capture → VAD → on-device TFLite → thresholds → alert.
 *
 * Audio stays in a 1 s in-memory ring buffer and is overwritten every 0.5 s. Nothing is written
 * to disk or sent anywhere; only the class id of a detection leaves this service.
 */
class AudioCaptureService : Service() {

    @Volatile private var running = false
    private var worker: Thread? = null
    private var wakeLock: PowerManager.WakeLock? = null
    @Volatile private var policy: DetectionPolicy? = null

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
        policy = DetectionPolicy(engine.labels, engine.thresholds).also { applySettings(it) }
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
        val hop = ShortArray(ModelConfig.HOP_SAMPLES)
        val snapshot = FloatArray(ModelConfig.WINDOW_SAMPLES)
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
                val scores = engine.predict(snapshot)
                lastInferenceMs = SystemClock.elapsedRealtime() - started
                emitScores(scores, energy)
                for (d in p.update(scores, now)) AlertDispatcher.dispatch(this, d)
            }
        } catch (e: Exception) {
            Log.e(TAG, "capture loop stopped", e)
            lastError = "Listening stopped: ${e.message}"
        } finally {
            try { record.stop() } catch (_: Exception) {}
            record.release()
            engine.close()
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

    fun applySettings(p: DetectionPolicy? = policy) {
        p ?: return
        val prefs = SaharaPrefs(this)
        prefs.enabledClasses?.let { p.enabled = it }
        p.sensitivity = prefs.sensitivity
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
