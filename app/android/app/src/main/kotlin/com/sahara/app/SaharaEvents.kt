package com.sahara.app

import android.os.Handler
import android.os.Looper
import java.util.concurrent.CopyOnWriteArraySet

/** Fan-out from the service to the Flutter EventChannel (when the UI is alive). */
object SaharaEvents {
    fun interface Listener {
        fun onEvent(event: Map<String, Any?>)
    }

    private val listeners = CopyOnWriteArraySet<Listener>()
    private val main = Handler(Looper.getMainLooper())

    val hasListeners: Boolean get() = listeners.isNotEmpty()

    fun add(listener: Listener) = listeners.add(listener)

    fun remove(listener: Listener) = listeners.remove(listener)

    fun emit(event: Map<String, Any?>) {
        if (listeners.isEmpty()) return
        main.post { listeners.forEach { it.onEvent(event) } }
    }
}
