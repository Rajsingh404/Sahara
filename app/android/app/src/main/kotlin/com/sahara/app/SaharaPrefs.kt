package com.sahara.app

import android.content.Context
import android.content.SharedPreferences
import com.sahara.app.core.Detection
import com.sahara.app.core.PrototypeBank
import org.json.JSONArray
import org.json.JSONObject

/** Settings and alert history, stored on the device only. */
class SaharaPrefs(context: Context) {
    private val prefs: SharedPreferences =
        context.applicationContext.getSharedPreferences("sahara", Context.MODE_PRIVATE)

    var enabledClasses: Set<String>?
        get() = prefs.getStringSet(KEY_ENABLED, null)
        set(value) = prefs.edit().putStringSet(KEY_ENABLED, value).apply()

    var sensitivity: Float
        get() = prefs.getFloat(KEY_SENSITIVITY, 0f)
        set(value) = prefs.edit().putFloat(KEY_SENSITIVITY, value).apply()

    var vibrate: Boolean
        get() = prefs.getBoolean(KEY_VIBRATE, true)
        set(value) = prefs.edit().putBoolean(KEY_VIBRATE, value).apply()

    /** "Someone is calling <name>" alerts. */
    var nameCalls: Boolean
        get() = prefs.getBoolean(KEY_NAME_CALLS, true)
        set(value) = prefs.edit().putBoolean(KEY_NAME_CALLS, value).apply()

    /** Low-priority "other sounds" from YAMNet's 521 classes. */
    var otherSounds: Boolean
        get() = prefs.getBoolean(KEY_OTHER_SOUNDS, true)
        set(value) = prefs.edit().putBoolean(KEY_OTHER_SOUNDS, value).apply()

    var wristbandAddress: String?
        get() = prefs.getString(KEY_BAND_ADDRESS, null)
        set(value) = prefs.edit().putString(KEY_BAND_ADDRESS, value).apply()

    var wristbandName: String?
        get() = prefs.getString(KEY_BAND_NAME, null)
        set(value) = prefs.edit().putString(KEY_BAND_NAME, value).apply()

    fun settingsMap(): Map<String, Any?> = mapOf(
        "enabledClasses" to enabledClasses?.toList(),
        "sensitivity" to sensitivity.toDouble(),
        "vibrate" to vibrate,
        "wristbandAddress" to wristbandAddress,
        "wristbandName" to wristbandName,
        "nameCalls" to nameCalls,
        "otherSounds" to otherSounds,
    )

    /**
     * Enrolled callers: per name, the voice vectors from their enrolment recordings. Only these
     * 256 numbers per recording are kept, never the audio.
     */
    @Synchronized
    fun loadCallers(): PrototypeBank {
        val bank = PrototypeBank(VOICE_DIM)
        val obj = JSONObject(prefs.getString(KEY_CALLERS, "{}") ?: "{}")
        for (name in obj.keys()) {
            val samples = obj.getJSONArray(name)
            for (i in 0 until samples.length()) {
                val arr = samples.getJSONArray(i)
                if (arr.length() == VOICE_DIM) bank.add(name, FloatArray(VOICE_DIM) { arr.getDouble(it).toFloat() })
            }
        }
        return bank
    }

    @Synchronized
    fun saveCallers(bank: PrototypeBank) {
        val obj = JSONObject()
        for (name in bank.names) {
            val samples = JSONArray()
            for (v in bank.samplesOf(name)) samples.put(JSONArray().apply { v.forEach { put(it.toDouble()) } })
            obj.put(name, samples)
        }
        prefs.edit().putString(KEY_CALLERS, obj.toString()).apply()
    }

    fun callersList(): List<Map<String, Any?>> {
        val bank = loadCallers()
        return bank.names.map { mapOf("name" to it, "samples" to bank.count(it)) }
    }

    @Synchronized
    fun addHistory(d: Detection) {
        val arr = JSONArray(prefs.getString(KEY_HISTORY, "[]"))
        val updated = JSONArray().put(toJson(d))
        for (i in 0 until minOf(arr.length(), MAX_HISTORY - 1)) updated.put(arr.get(i))
        prefs.edit().putString(KEY_HISTORY, updated.toString()).apply()
    }

    /** Newest first. */
    fun history(): List<Map<String, Any?>> {
        val arr = JSONArray(prefs.getString(KEY_HISTORY, "[]"))
        return (0 until arr.length()).map { i ->
            val o = arr.getJSONObject(i)
            mapOf(
                "classId" to o.getInt("classId"),
                "label" to o.getString("label"),
                "confidence" to o.getDouble("confidence"),
                "timestampMs" to o.getLong("timestampMs"),
                "safetyCritical" to o.getBoolean("safetyCritical"),
                "detail" to o.optString("detail").ifEmpty { null },
            )
        }
    }

    fun clearHistory() = prefs.edit().remove(KEY_HISTORY).apply()

    private fun toJson(d: Detection) = JSONObject()
        .put("classId", d.classId)
        .put("label", d.label)
        .put("confidence", d.confidence.toDouble())
        .put("timestampMs", d.timestampMs)
        .put("safetyCritical", d.safetyCritical)
        .put("detail", d.detail ?: "")

    companion object {
        private const val KEY_ENABLED = "enabled_classes"
        private const val KEY_SENSITIVITY = "sensitivity"
        private const val KEY_VIBRATE = "vibrate"
        private const val KEY_BAND_ADDRESS = "wristband_address"
        private const val KEY_BAND_NAME = "wristband_name"
        private const val KEY_HISTORY = "history"
        private const val KEY_NAME_CALLS = "name_calls"
        private const val KEY_OTHER_SOUNDS = "other_sounds"
        private const val KEY_CALLERS = "callers"
        private const val MAX_HISTORY = 100
        const val VOICE_DIM = 256
    }
}

fun Detection.toMap(): Map<String, Any?> = mapOf(
    "classId" to classId,
    "label" to label,
    "confidence" to confidence.toDouble(),
    "timestampMs" to timestampMs,
    "safetyCritical" to safetyCritical,
    "detail" to detail,
)
