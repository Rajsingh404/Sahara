package com.sahara.app

import android.content.Intent
import android.os.Build
import android.os.Bundle
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine

class MainActivity : FlutterActivity() {
    private var channel: SaharaChannel? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        // Before super: configureFlutterEngine consumes the alert extras.
        showOverLockIfAlert(intent)
        super.onCreate(savedInstanceState)
        AlertNotifier.ensureChannel(this)
    }

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        channel = SaharaChannel(applicationContext, flutterEngine.dartExecutor.binaryMessenger).also {
            it.handleIntent(intent)
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        showOverLockIfAlert(intent)
        channel?.handleIntent(intent)
    }

    /** Only an alert opens over the lock screen and wakes the display; normal launches do not. */
    private fun showOverLockIfAlert(intent: Intent?) {
        val isAlert = (intent?.getIntExtra(AlertNotifier.EXTRA_ALERT_CLASS_ID, -1) ?: -1) >= 0
        if (Build.VERSION.SDK_INT >= 27) {
            setShowWhenLocked(isAlert)
            setTurnScreenOn(isAlert)
        }
    }

    override fun cleanUpFlutterEngine(flutterEngine: FlutterEngine) {
        channel?.dispose()
        channel = null
        super.cleanUpFlutterEngine(flutterEngine)
    }
}
