package com.sahara.app.core

/** Parses the bundled assets without Android's org.json (keeps this JVM-testable). */
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

    /** YAMNet's class map CSV (`index,mid,display_name`, names may be quoted): display names in index order. */
    fun parseYamnetClassMap(csv: String): List<String> =
        csv.lineSequence().drop(1).filter { it.isNotBlank() }.map { line ->
            val rest = line.substringAfter(',').substringAfter(',').trim()
            if (rest.startsWith("\"")) rest.removeSurrounding("\"").replace("\"\"", "\"") else rest
        }.toList()

    data class OtherSoundsConfig(val minScore: Float, val ignored: Set<String>, val coveredByHead: Set<String>)

    /** other_sounds.json, written by app/tool/sync_assets.py from OTHER_SOUND_* in src/config.py. */
    fun parseOtherSounds(json: String): OtherSoundsConfig = OtherSoundsConfig(
        minScore = number(json, "min_score") ?: 0.35f,
        ignored = strings(json, "ignored").toSet(),
        coveredByHead = strings(json, "covered_by_head").toSet(),
    )

    /** name_head.json, written by scripts/eval_name_call.py --export. */
    fun parseNameHead(json: String): NameSpotter.Head = NameSpotter.Head(
        name = Regex("\"display_name\"\\s*:\\s*\"([^\"]*)\"").find(json)?.groupValues?.get(1) ?: "your name",
        window = number(json, "window")!!.toInt(),
        dim = number(json, "dim")!!.toInt(),
        mean = numbers(json, "mean"),
        scale = numbers(json, "scale"),
        coef = numbers(json, "coef"),
        intercept = number(json, "intercept")!!,
        threshold = number(json, "threshold")!!,
    )

    private fun number(json: String, key: String): Float? =
        Regex("\"$key\"\\s*:\\s*([0-9.eE+-]+)").find(json)?.groupValues?.get(1)?.toFloat()

    private fun array(json: String, key: String): String =
        Regex("\"$key\"\\s*:\\s*\\[(.*?)]", RegexOption.DOT_MATCHES_ALL).find(json)?.groupValues?.get(1).orEmpty()

    private fun numbers(json: String, key: String): FloatArray =
        array(json, key).split(',').map { it.trim() }.filter { it.isNotEmpty() }.map { it.toFloat() }.toFloatArray()

    private fun strings(json: String, key: String): List<String> =
        Regex("\"((?:[^\"\\\\]|\\\\.)*)\"").findAll(array(json, key)).map { it.groupValues[1].replace("\\\"", "\"") }.toList()
}
