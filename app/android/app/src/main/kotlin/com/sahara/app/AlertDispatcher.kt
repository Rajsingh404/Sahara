package com.sahara.app

import android.content.Context
import com.sahara.app.core.Detection

/** Everything that happens when a sound is detected. Runs with or without the Flutter UI. */
object AlertDispatcher {
    fun dispatch(context: Context, d: Detection) {
        val app = context.applicationContext
        val prefs = SaharaPrefs(app)
        prefs.addHistory(d)
        AlertNotifier.show(app, d, vibrate = prefs.vibrate)
        Wristband.get(app).sendAlert(d)
        SaharaEvents.emit(mapOf("type" to "detection") + d.toMap())
    }
}
