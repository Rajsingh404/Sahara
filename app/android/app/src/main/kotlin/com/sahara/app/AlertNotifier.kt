package com.sahara.app

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.media.AudioAttributes
import android.os.Build
import android.os.VibrationAttributes
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import com.sahara.app.core.Detection

/** Phone-side visual + haptic alert: heads-up / full-screen notification and a vibration pattern. */
object AlertNotifier {
    const val ALERT_CHANNEL = "sahara_alerts"
    const val EXTRA_ALERT_CLASS_ID = "sahara_alert_class_id"
    const val EXTRA_ALERT_CONFIDENCE = "sahara_alert_confidence"
    const val EXTRA_ALERT_TIME = "sahara_alert_time"
    private const val ALERT_NOTIFICATION_ID = 2001

    fun ensureChannel(context: Context) {
        val nm = context.getSystemService(NotificationManager::class.java) ?: return
        if (nm.getNotificationChannel(ALERT_CHANNEL) != null) return
        val channel = NotificationChannel(ALERT_CHANNEL, "Sound alerts", NotificationManager.IMPORTANCE_HIGH).apply {
            description = "Shown when SAHARA detects a sound around you"
            // Vibration is played explicitly with a per-class pattern.
            enableVibration(false)
            setSound(null, null)
            enableLights(true)
            lockscreenVisibility = Notification.VISIBILITY_PUBLIC
        }
        nm.createNotificationChannel(channel)
    }

    fun show(context: Context, d: Detection, vibrate: Boolean) {
        ensureChannel(context)
        val title = displayName(d.label)
        val open = Intent(context, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_SINGLE_TOP
            putExtra(EXTRA_ALERT_CLASS_ID, d.classId)
            putExtra(EXTRA_ALERT_CONFIDENCE, d.confidence)
            putExtra(EXTRA_ALERT_TIME, d.timestampMs)
        }
        val pi = PendingIntent.getActivity(
            context, d.classId, open, PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val notification = Notification.Builder(context, ALERT_CHANNEL)
            .setSmallIcon(R.mipmap.ic_launcher)
            .setContentTitle(if (d.safetyCritical) "⚠ $title" else title)
            .setContentText("Detected near you · ${(d.confidence * 100).toInt()}% confidence")
            .setCategory(if (d.safetyCritical) Notification.CATEGORY_ALARM else Notification.CATEGORY_EVENT)
            .setVisibility(Notification.VISIBILITY_PUBLIC)
            .setContentIntent(pi)
            .setFullScreenIntent(pi, true)
            .setAutoCancel(true)
            .build()
        try {
            context.getSystemService(NotificationManager::class.java)?.notify(ALERT_NOTIFICATION_ID, notification)
        } catch (_: SecurityException) {
            // POST_NOTIFICATIONS denied: the vibration and wristband still fire.
        }
        if (vibrate) vibrate(context, d)
    }

    fun cancel(context: Context) {
        context.getSystemService(NotificationManager::class.java)?.cancel(ALERT_NOTIFICATION_ID)
        vibrator(context)?.cancel()
    }

    /** Per-class phone vibration; safety-critical classes get a long repeated pattern. */
    fun pattern(classLabel: String, safetyCritical: Boolean): LongArray = when {
        safetyCritical -> longArrayOf(0, 600, 200, 600, 200, 600, 400, 600, 200, 600, 200, 600)
        classLabel == "doorbell" -> longArrayOf(0, 150, 150, 150)
        classLabel == "knocking" -> longArrayOf(0, 100, 100, 100, 100, 100)
        classLabel == "dog_bark" -> longArrayOf(0, 300, 200, 300)
        classLabel == "baby_cry" -> longArrayOf(0, 800, 200, 150, 150, 150)
        else -> longArrayOf(0, 150, 150, 400)
    }

    private fun vibrate(context: Context, d: Detection) {
        val effect = VibrationEffect.createWaveform(pattern(d.label, d.safetyCritical), -1)
        val v = vibrator(context) ?: return
        if (Build.VERSION.SDK_INT >= 33) {
            v.vibrate(effect, VibrationAttributes.createForUsage(VibrationAttributes.USAGE_ALARM))
        } else {
            @Suppress("DEPRECATION")
            v.vibrate(effect, AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_ALARM).build())
        }
    }

    private fun vibrator(context: Context): Vibrator? =
        if (Build.VERSION.SDK_INT >= 31) {
            context.getSystemService(VibratorManager::class.java)?.defaultVibrator
        } else {
            @Suppress("DEPRECATION")
            context.getSystemService(Vibrator::class.java)
        }

    fun displayName(label: String): String =
        label.split('_').joinToString(" ") { it.replaceFirstChar(Char::uppercase) }
}
