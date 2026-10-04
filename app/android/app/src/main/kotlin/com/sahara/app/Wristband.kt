package com.sahara.app

import android.annotation.SuppressLint
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothGatt
import android.bluetooth.BluetoothGattCallback
import android.bluetooth.BluetoothGattCharacteristic
import android.bluetooth.BluetoothGattDescriptor
import android.bluetooth.BluetoothManager
import android.bluetooth.BluetoothProfile
import android.bluetooth.le.ScanCallback
import android.bluetooth.le.ScanResult
import android.bluetooth.le.ScanSettings
import android.content.Context
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.util.Log
import com.sahara.app.core.BlePacket
import com.sahara.app.core.Detection

/**
 * GATT client for the ESP32 wristband (docs/ble_protocol.md). App-scoped so the foreground
 * service keeps the link while the screen is off. All BLE calls catch SecurityException:
 * the Flutter side requests BLUETOOTH_SCAN / BLUETOOTH_CONNECT before using these.
 */
@SuppressLint("MissingPermission")
class Wristband private constructor(private val context: Context) {
    enum class State { DISCONNECTED, CONNECTING, CONNECTED }

    private val main = Handler(Looper.getMainLooper())
    private val prefs = SaharaPrefs(context)
    private val adapter get() = context.getSystemService(BluetoothManager::class.java)?.adapter

    @Volatile var state: State = State.DISCONNECTED
        private set
    @Volatile var status: BlePacket.Status? = null
        private set

    private var gatt: BluetoothGatt? = null
    private var alertChar: BluetoothGattCharacteristic? = null

    fun statusMap(): Map<String, Any?> = mapOf(
        "state" to state.name.lowercase(),
        "address" to prefs.wristbandAddress,
        "name" to prefs.wristbandName,
        "batteryPct" to status?.batteryPct,
        "charging" to status?.charging,
    )

    /** Scans for [timeoutMs] and returns bands whose name starts with SAHARA or that advertise the service. */
    fun scan(timeoutMs: Long, onDone: (List<Map<String, Any?>>) -> Unit) {
        val scanner = adapter?.bluetoothLeScanner
        if (scanner == null) {
            onDone(emptyList()); return
        }
        val found = LinkedHashMap<String, Map<String, Any?>>()
        val callback = object : ScanCallback() {
            override fun onScanResult(callbackType: Int, result: ScanResult) {
                val name = result.scanRecord?.deviceName ?: safeName(result.device)
                val advertisesService = result.scanRecord?.serviceUuids?.any { it.uuid == BleUuids.SERVICE } == true
                if (advertisesService || name?.startsWith(BleUuids.DEVICE_NAME_PREFIX) == true) {
                    found[result.device.address] = mapOf(
                        "address" to result.device.address,
                        "name" to (name ?: "SAHARA band"),
                        "rssi" to result.rssi,
                    )
                }
            }
        }
        try {
            val settings = ScanSettings.Builder().setScanMode(ScanSettings.SCAN_MODE_LOW_LATENCY).build()
            scanner.startScan(null, settings, callback)
        } catch (e: SecurityException) {
            Log.w(TAG, "scan not permitted", e)
            onDone(emptyList()); return
        }
        main.postDelayed({
            try { scanner.stopScan(callback) } catch (_: Exception) {}
            onDone(found.values.sortedByDescending { it["rssi"] as Int })
        }, timeoutMs)
    }

    fun connect(address: String, name: String?) {
        prefs.wristbandAddress = address
        prefs.wristbandName = name
        disconnectInternal()
        connectSaved()
    }

    /** Reconnects to the remembered band, if any. Safe to call repeatedly. */
    fun connectSaved() {
        val address = prefs.wristbandAddress ?: return
        if (state != State.DISCONNECTED) return
        val device = try { adapter?.getRemoteDevice(address) } catch (_: IllegalArgumentException) { null } ?: return
        try {
            setState(State.CONNECTING)
            // autoConnect = true: Android reconnects whenever the band comes back in range.
            gatt = device.connectGatt(context, true, callback, BluetoothDevice.TRANSPORT_LE)
        } catch (e: SecurityException) {
            Log.w(TAG, "connect not permitted", e)
            setState(State.DISCONNECTED)
        }
    }

    fun forget() {
        disconnectInternal()
        prefs.wristbandAddress = null
        prefs.wristbandName = null
        emitStatus()
    }

    fun sendAlert(d: Detection) = write(BlePacket.alert(d.classId, d.confidence, d.safetyCritical))

    fun sendTest(): Boolean = write(BlePacket.test())

    fun sendStop() = write(BlePacket.stop())

    private fun write(value: ByteArray): Boolean {
        val g = gatt ?: return false.also { connectSaved() }
        val c = alertChar ?: return false
        return try {
            if (Build.VERSION.SDK_INT >= 33) {
                g.writeCharacteristic(c, value, BluetoothGattCharacteristic.WRITE_TYPE_DEFAULT) == BluetoothGatt.GATT_SUCCESS
            } else {
                @Suppress("DEPRECATION")
                c.writeType = BluetoothGattCharacteristic.WRITE_TYPE_DEFAULT
                @Suppress("DEPRECATION")
                c.value = value
                @Suppress("DEPRECATION")
                g.writeCharacteristic(c)
            }
        } catch (e: SecurityException) {
            Log.w(TAG, "write not permitted", e)
            false
        }
    }

    private fun disconnectInternal() {
        try {
            gatt?.disconnect()
            gatt?.close()
        } catch (_: SecurityException) {}
        gatt = null
        alertChar = null
        status = null
        setState(State.DISCONNECTED)
    }

    private val callback = object : BluetoothGattCallback() {
        override fun onConnectionStateChange(g: BluetoothGatt, statusCode: Int, newState: Int) {
            when (newState) {
                BluetoothProfile.STATE_CONNECTED -> {
                    setState(State.CONNECTING)
                    try { g.discoverServices() } catch (_: SecurityException) {}
                }
                BluetoothProfile.STATE_DISCONNECTED -> {
                    alertChar = null
                    // With autoConnect the same BluetoothGatt reconnects by itself.
                    setState(if (gatt != null) State.CONNECTING else State.DISCONNECTED)
                }
            }
        }

        override fun onServicesDiscovered(g: BluetoothGatt, statusCode: Int) {
            val service = g.getService(BleUuids.SERVICE)
            alertChar = service?.getCharacteristic(BleUuids.ALERT_CHAR)
            if (alertChar == null) {
                Log.w(TAG, "connected device has no SAHARA alert characteristic")
                return
            }
            setState(State.CONNECTED)
            service.getCharacteristic(BleUuids.STATUS_CHAR)?.let { enableNotify(g, it) }
        }

        @Deprecated("Deprecated in Java")
        @Suppress("DEPRECATION")
        override fun onCharacteristicChanged(g: BluetoothGatt, c: BluetoothGattCharacteristic) {
            onStatus(c.uuid, c.value)
        }

        override fun onCharacteristicChanged(g: BluetoothGatt, c: BluetoothGattCharacteristic, value: ByteArray) {
            onStatus(c.uuid, value)
        }
    }

    private fun enableNotify(g: BluetoothGatt, c: BluetoothGattCharacteristic) {
        try {
            g.setCharacteristicNotification(c, true)
            val cccd = c.getDescriptor(BleUuids.CCCD) ?: return
            if (Build.VERSION.SDK_INT >= 33) {
                g.writeDescriptor(cccd, BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE)
            } else {
                @Suppress("DEPRECATION")
                cccd.value = BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE
                @Suppress("DEPRECATION")
                g.writeDescriptor(cccd)
            }
        } catch (_: SecurityException) {}
    }

    private fun onStatus(uuid: java.util.UUID, value: ByteArray?) {
        if (uuid != BleUuids.STATUS_CHAR) return
        BlePacket.parseStatus(value)?.let {
            status = it
            emitStatus()
        }
    }

    private fun setState(s: State) {
        state = s
        emitStatus()
    }

    private fun emitStatus() = SaharaEvents.emit(mapOf("type" to "wristband") + statusMap())

    private fun safeName(device: BluetoothDevice): String? = try { device.name } catch (_: SecurityException) { null }

    companion object {
        private const val TAG = "SaharaBand"
        @Volatile private var instance: Wristband? = null

        fun get(context: Context): Wristband =
            instance ?: synchronized(this) {
                instance ?: Wristband(context.applicationContext).also { instance = it }
            }
    }
}
