package com.sahara.app

import android.content.Context
import android.content.SharedPreferences
import com.sahara.app.core.Detection
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
    )

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
            )
        }
    }

    fun clearHistory() = prefs.edit().remove(KEY_HISTORY).apply()

    /** User-enrolled sounds: names and prototypes (1,024 floats each). Never any audio. */
    @Synchronized
    fun customSounds(): List<com.sahara.app.core.CustomSound> {
        val arr = JSONArray(prefs.getString(KEY_CUSTOM, "[]"))
        return (0 until arr.length()).map { i ->
            val o = arr.getJSONObject(i)
            com.sahara.app.core.CustomSound(o.getString("name"), com.sahara.app.core.CustomSoundMatcher.decode(o.getString("prototype")))
        }
    }

    @Synchronized
    fun saveCustomSound(sound: com.sahara.app.core.CustomSound) {
        val kept = customSounds().filter { it.name != sound.name } + sound
        val arr = JSONArray()
        kept.forEach { arr.put(JSONObject().put("name", it.name).put("prototype", com.sahara.app.core.CustomSoundMatcher.encode(it.prototype))) }
        prefs.edit().putString(KEY_CUSTOM, arr.toString()).apply()
    }

    @Synchronized
    fun deleteCustomSound(name: String) {
        val arr = JSONArray()
        customSounds().filter { it.name != name }.forEach {
            arr.put(JSONObject().put("name", it.name).put("prototype", com.sahara.app.core.CustomSoundMatcher.encode(it.prototype)))
        }
        prefs.edit().putString(KEY_CUSTOM, arr.toString()).apply()
    }

    private fun toJson(d: Detection) = JSONObject()
        .put("classId", d.classId)
        .put("label", d.label)
        .put("confidence", d.confidence.toDouble())
        .put("timestampMs", d.timestampMs)
        .put("safetyCritical", d.safetyCritical)

    companion object {
        private const val KEY_ENABLED = "enabled_classes"
        private const val KEY_SENSITIVITY = "sensitivity"
        private const val KEY_VIBRATE = "vibrate"
        private const val KEY_BAND_ADDRESS = "wristband_address"
        private const val KEY_BAND_NAME = "wristband_name"
        private const val KEY_HISTORY = "history"
        private const val KEY_CUSTOM = "custom_sounds"
        private const val MAX_HISTORY = 100
    }
}

fun Detection.toMap(): Map<String, Any?> = mapOf(
    "classId" to classId,
    "label" to label,
    "confidence" to confidence.toDouble(),
    "timestampMs" to timestampMs,
    "safetyCritical" to safetyCritical,
)
