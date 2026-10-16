package com.sahara.app

import android.content.Context
import com.sahara.app.core.Detection

/**
 * Everything that happens when a sound is detected. Runs with or without the Flutter UI.
 *
 * Trained classes and name calls are full alerts (heads-up notification, vibration, wristband).
 * Other sounds are notices only: a silent notification and the history, no vibration or band, so
 * the open-set layer can never drown out a safety alert.
 */
object AlertDispatcher {
    @Synchronized
    fun dispatch(context: Context, d: Detection) {
        val app = context.applicationContext
        val prefs = SaharaPrefs(app)
        prefs.addHistory(d)
        if (d.isOtherSound) {
            AlertNotifier.showNotice(app, d)
        } else {
            AlertNotifier.show(app, d, vibrate = prefs.vibrate)
            Wristband.get(app).sendAlert(d)
        }
        SaharaEvents.emit(mapOf("type" to "detection") + d.toMap())
    }
}
