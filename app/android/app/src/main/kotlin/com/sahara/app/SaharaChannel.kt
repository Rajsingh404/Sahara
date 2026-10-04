package com.sahara.app

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import com.sahara.app.core.DetectionPolicy
import com.sahara.app.core.Detection
import io.flutter.plugin.common.BinaryMessenger
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel

/**
 * Platform channel between Dart (UI) and the native service.
 * Method channel `sahara/control`, event channel `sahara/events`. Keep in sync with
 * app/lib/audio/sahara_service.dart.
 */
class SaharaChannel(private val context: Context, messenger: BinaryMessenger) : MethodChannel.MethodCallHandler {
    private val methods = MethodChannel(messenger, "sahara/control")
    private val events = EventChannel(messenger, "sahara/events")
    private var listener: SaharaEvents.Listener? = null
    private val prefs = SaharaPrefs(context)
    private var launchAlert: Map<String, Any?>? = null

    init {
        methods.setMethodCallHandler(this)
        events.setStreamHandler(object : EventChannel.StreamHandler {
            override fun onListen(arguments: Any?, sink: EventChannel.EventSink) {
                listener = SaharaEvents.Listener { sink.success(it) }.also(SaharaEvents::add)
            }

            override fun onCancel(arguments: Any?) {
                listener?.let(SaharaEvents::remove)
                listener = null
            }
        })
    }

    fun dispose() {
        methods.setMethodCallHandler(null)
        events.setStreamHandler(null)
        listener?.let(SaharaEvents::remove)
    }

    /** Called by MainActivity when it is opened from an alert notification. */
    fun handleIntent(intent: Intent?) {
        val id = intent?.getIntExtra(AlertNotifier.EXTRA_ALERT_CLASS_ID, -1) ?: -1
        if (id < 0) return
        val labels = InferenceEngine.loadLabels(context)
        val label = labels.getOrNull(id) ?: return
        val alert = mapOf(
            "classId" to id,
            "label" to label,
            "confidence" to intent!!.getFloatExtra(AlertNotifier.EXTRA_ALERT_CONFIDENCE, 0f).toDouble(),
            "timestampMs" to intent.getLongExtra(AlertNotifier.EXTRA_ALERT_TIME, System.currentTimeMillis()),
            "safetyCritical" to (label in DetectionPolicy.DEFAULT_SAFETY_CRITICAL),
        )
        intent.removeExtra(AlertNotifier.EXTRA_ALERT_CLASS_ID)
        if (listener != null) SaharaEvents.emit(mapOf("type" to "openAlert") + alert) else launchAlert = alert
    }

    override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
        when (call.method) {
            "getStatus" -> result.success(status())
            "startListening" -> {
                if (!granted(Manifest.permission.RECORD_AUDIO)) {
                    result.error("permission", "Microphone permission is required", null)
                } else {
                    AudioCaptureService.lastError = null
                    AudioCaptureService.start(context)
                    result.success(true)
                }
            }
            "stopListening" -> {
                AudioCaptureService.stop(context)
                result.success(true)
            }
            "updateSettings" -> {
                call.argument<List<String>>("enabledClasses")?.let { prefs.enabledClasses = it.toSet() }
                call.argument<Double>("sensitivity")?.let { prefs.sensitivity = it.toFloat() }
                call.argument<Boolean>("vibrate")?.let { prefs.vibrate = it }
                AudioCaptureService.instance?.applySettings()
                result.success(prefs.settingsMap())
            }
            "getHistory" -> result.success(prefs.history())
            "clearHistory" -> {
                prefs.clearHistory()
                result.success(true)
            }
            "scanWristbands" -> {
                val timeout = (call.argument<Int>("timeoutMs") ?: 6000).toLong()
                Wristband.get(context).scan(timeout) { result.success(it) }
            }
            "connectWristband" -> {
                val address = call.argument<String>("address")
                if (address == null) {
                    result.error("args", "address is required", null)
                } else {
                    Wristband.get(context).connect(address, call.argument<String>("name"))
                    result.success(Wristband.get(context).statusMap())
                }
            }
            "forgetWristband" -> {
                Wristband.get(context).forget()
                result.success(true)
            }
            "testWristband" -> result.success(Wristband.get(context).sendTest())
            "simulateAlert" -> {
                val labels = InferenceEngine.loadLabels(context)
                val id = call.argument<Int>("classId") ?: 0
                val label = labels.getOrNull(id)
                if (label == null) {
                    result.error("args", "unknown class id $id", null)
                } else {
                    val d = Detection(id, label, 0.99f, System.currentTimeMillis(), label in DetectionPolicy.DEFAULT_SAFETY_CRITICAL)
                    AlertDispatcher.dispatch(context, d)
                    result.success(true)
                }
            }
            "dismissAlert" -> {
                AlertNotifier.cancel(context)
                Wristband.get(context).sendStop()
                result.success(true)
            }
            "consumeLaunchAlert" -> {
                result.success(launchAlert)
                launchAlert = null
            }
            else -> result.notImplemented()
        }
    }

    private fun status(): Map<String, Any?> {
        if (!AudioCaptureService.isRunning) modelCheck(context)
        return AudioCaptureService.stateMap() + mapOf(
            "labels" to InferenceEngine.loadLabels(context),
            "thresholds" to InferenceEngine.loadThresholds(context).mapValues { it.value.toDouble() },
            "modelReady" to (ModelStatus.reason == null),
            "settings" to prefs.settingsMap(),
            "wristband" to Wristband.get(context).statusMap(),
            "inferenceMs" to AudioCaptureService.lastInferenceMs,
        )
    }

    private fun granted(permission: String) =
        context.checkSelfPermission(permission) == PackageManager.PERMISSION_GRANTED

    companion object {
        private var checked = false

        /** Loads the model once per process so the UI can say whether it is bundled. */
        fun modelCheck(context: Context) {
            if (checked) return
            checked = true
            InferenceEngine.load(context)?.close()
        }
    }
}
