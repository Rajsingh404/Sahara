import 'package:permission_handler/permission_handler.dart';

/// Runtime permissions. Microphone is required; the rest degrade gracefully.
class Permissions {
  static Future<bool> ensureListening() async {
    final mic = await Permission.microphone.request();
    // Android 13+: without this the alert notification is silent, but vibration and the
    // wristband still work, so it is requested and not enforced.
    await Permission.notification.request();
    return mic.isGranted;
  }

  static Future<bool> ensureBluetooth() async {
    final results = await [
      Permission.bluetoothScan,
      Permission.bluetoothConnect,
      // Only used for scanning on Android 11 and below (maxSdkVersion=30 in the manifest).
      Permission.locationWhenInUse,
    ].request();
    final scan = results[Permission.bluetoothScan];
    final connect = results[Permission.bluetoothConnect];
    final location = results[Permission.locationWhenInUse];
    final modern = (scan?.isGranted ?? false) && (connect?.isGranted ?? false);
    return modern || (location?.isGranted ?? false);
  }

  static Future<void> openSettings() => openAppSettings();
}
