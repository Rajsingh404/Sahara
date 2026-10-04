package com.sahara.app.core

/** Parses labels.txt and class_thresholds.json without Android's org.json (keeps this JVM-testable). */
object AssetParsers {
    fun parseLabels(text: String): List<String> =
        text.lineSequence().map { it.trim() }.filter { it.isNotEmpty() && !it.startsWith("#") }.toList()

    /**
     * Reads `{"class": {"threshold": 0.3, ...}, ...}` (threshold_search.py output) or a flat
     * `{"class": 0.3}` map. Only the threshold numbers are extracted.
     */
    fun parseThresholds(json: String): Map<String, Float> {
        val result = mutableMapOf<String, Float>()
        val nested = Regex("\"([A-Za-z0-9_]+)\"\\s*:\\s*\\{[^{}]*?\"threshold\"\\s*:\\s*([0-9.eE+-]+)")
        nested.findAll(json).forEach { result[it.groupValues[1]] = it.groupValues[2].toFloat() }
        if (result.isEmpty()) {
            val flat = Regex("\"([A-Za-z0-9_]+)\"\\s*:\\s*([0-9.eE+-]+)")
            flat.findAll(json).forEach { result[it.groupValues[1]] = it.groupValues[2].toFloat() }
        }
        return result
    }
}
