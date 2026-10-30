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
import com.sahara.app.core.HapticPatterns

/** Phone-side visual + haptic alert: heads-up / full-screen notification and a vibration pattern. */
object AlertNotifier {
    const val ALERT_CHANNEL = "sahara_alerts"
    const val NOTICE_CHANNEL = "sahara_other_sounds"
    const val EXTRA_ALERT_CLASS_ID = "sahara_alert_class_id"
    const val EXTRA_ALERT_CONFIDENCE = "sahara_alert_confidence"
    const val EXTRA_ALERT_TIME = "sahara_alert_time"
    const val EXTRA_ALERT_LABEL = "sahara_alert_label"
    const val EXTRA_ALERT_DETAIL = "sahara_alert_detail"
    private const val ALERT_NOTIFICATION_ID = 2001
    private const val NOTICE_NOTIFICATION_ID = 2002

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
        nm.createNotificationChannel(
            NotificationChannel(NOTICE_CHANNEL, "Other sounds", NotificationManager.IMPORTANCE_LOW).apply {
                description = "Quiet notices for sounds outside the main alerts (water tap, cat, shouting...)"
                enableVibration(false)
                setSound(null, null)
            },
        )
    }

    /** "Raj is calling you", "Someone is calling you", "Doorbell", "Water tap, faucet". */
    fun title(d: Detection): String = when (d.label) {
        Detection.NAME_CALL -> "${d.detail ?: "Someone"} is calling you"
        Detection.OTHER_SOUND -> d.detail ?: "A sound"
        else -> displayName(d.label)
    }

    /** Low-priority notice for an "other sound": no vibration, no full-screen, replaced by the next one. */
    fun showNotice(context: Context, d: Detection) {
        ensureChannel(context)
        val open = PendingIntent.getActivity(
            context, NOTICE_NOTIFICATION_ID, Intent(context, MainActivity::class.java), PendingIntent.FLAG_IMMUTABLE,
        )
        val notification = Notification.Builder(context, NOTICE_CHANNEL)
            .setSmallIcon(R.mipmap.ic_launcher)
            .setContentTitle(title(d))
            .setContentText("Heard near you · not one of the main alert sounds")
            .setCategory(Notification.CATEGORY_STATUS)
            .setVisibility(Notification.VISIBILITY_PUBLIC)
            .setContentIntent(open)
            .setAutoCancel(true)
            .build()
        try {
            context.getSystemService(NotificationManager::class.java)?.notify(NOTICE_NOTIFICATION_ID, notification)
        } catch (_: SecurityException) {
        }
    }

    fun show(context: Context, d: Detection, vibrate: Boolean) {
        ensureChannel(context)
        val title = title(d)
        val open = Intent(context, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_SINGLE_TOP
            putExtra(EXTRA_ALERT_CLASS_ID, d.classId)
            putExtra(EXTRA_ALERT_CONFIDENCE, d.confidence)
            putExtra(EXTRA_ALERT_TIME, d.timestampMs)
            putExtra(EXTRA_ALERT_LABEL, d.label)
            putExtra(EXTRA_ALERT_DETAIL, d.detail)
        }
        val pi = PendingIntent.getActivity(
            context, d.classId, open, PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val notification = Notification.Builder(context, ALERT_CHANNEL)
            .setSmallIcon(R.mipmap.ic_launcher)
            .setContentTitle(if (d.safetyCritical) "⚠ $title" else title)
            .setContentText(
                if (d.label == Detection.NAME_CALL) "Look around: someone is trying to get your attention"
                else "Detected near you · ${(d.confidence * 100).toInt()}% confidence",
            )
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
    private fun vibrate(context: Context, d: Detection) {
        val v = vibrator(context) ?: return
        // Same pattern as the wristband for this sound (core/HapticPatterns.kt).
        val pattern = HapticPatterns.forLabel(d.label)
        val effect = if (v.hasAmplitudeControl()) {
            val (timings, amplitudes) = HapticPatterns.waveform(pattern)
            VibrationEffect.createWaveform(timings, amplitudes, -1)
        } else {
            VibrationEffect.createWaveform(HapticPatterns.onOff(pattern), -1)
        }
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
        if (label.startsWith(com.sahara.app.core.CustomSoundMatcher.LABEL_PREFIX)) {
            "Custom sound: " + label.removePrefix(com.sahara.app.core.CustomSoundMatcher.LABEL_PREFIX)
        } else {
            label.split('_').joinToString(" ") { it.replaceFirstChar(Char::uppercase) }
        }
}
