import 'dart:async';

import 'package:flutter/services.dart';

import '../detection/detection_event.dart';

/// Wristband connection as reported by the native service.
class WristbandInfo {
  const WristbandInfo({this.state = 'disconnected', this.address, this.name, this.batteryPct, this.charging});

  final String state;
  final String? address;
  final String? name;
  final int? batteryPct;
  final bool? charging;

  bool get paired => address != null;
  bool get connected => state == 'connected';

  factory WristbandInfo.fromMap(Map<dynamic, dynamic>? m) => m == null
      ? const WristbandInfo()
      : WristbandInfo(
          state: m['state'] as String? ?? 'disconnected',
          address: m['address'] as String?,
          name: m['name'] as String?,
          batteryPct: (m['batteryPct'] as num?)?.toInt(),
          charging: m['charging'] as bool?,
        );
}

class ScannedBand {
  const ScannedBand(this.address, this.name, this.rssi);
  final String address;
  final String name;
  final int rssi;
}

/// Snapshot of the native side.
class SaharaStatus {
  const SaharaStatus({
    this.listening = false,
    this.error,
    this.modelReady = false,
    this.modelReason,
    this.labels = const [],
    this.thresholds = const {},
    this.enabledClasses,
    this.sensitivity = 0,
    this.vibrate = true,
    this.wristband = const WristbandInfo(),
  });

  final bool listening;
  final String? error;
  final bool modelReady;
  final String? modelReason;
  final List<String> labels;
  final Map<String, double> thresholds;

  /// null means every class is enabled.
  final Set<String>? enabledClasses;
  final double sensitivity;
  final bool vibrate;
  final WristbandInfo wristband;

  bool isEnabled(String label) => enabledClasses?.contains(label) ?? true;

  SaharaStatus copyWith({
    bool? listening,
    String? error,
    bool clearError = false,
    String? modelReason,
    Set<String>? enabledClasses,
    double? sensitivity,
    bool? vibrate,
    WristbandInfo? wristband,
  }) => SaharaStatus(
    listening: listening ?? this.listening,
    error: clearError ? null : (error ?? this.error),
    modelReady: modelReady,
    modelReason: modelReason ?? this.modelReason,
    labels: labels,
    thresholds: thresholds,
    enabledClasses: enabledClasses ?? this.enabledClasses,
    sensitivity: sensitivity ?? this.sensitivity,
    vibrate: vibrate ?? this.vibrate,
    wristband: wristband ?? this.wristband,
  );

  factory SaharaStatus.fromMap(Map<dynamic, dynamic> m) {
    final settings = (m['settings'] as Map?) ?? const {};
    final enabled = settings['enabledClasses'] as List?;
    return SaharaStatus(
      listening: m['listening'] as bool? ?? false,
      error: m['error'] as String?,
      modelReady: m['modelReady'] as bool? ?? false,
      modelReason: m['modelReason'] as String?,
      labels: ((m['labels'] as List?) ?? const []).cast<String>(),
      thresholds: ((m['thresholds'] as Map?) ?? const {}).map((k, v) => MapEntry(k as String, (v as num).toDouble())),
      enabledClasses: enabled?.cast<String>().toSet(),
      sensitivity: (settings['sensitivity'] as num?)?.toDouble() ?? 0,
      vibrate: settings['vibrate'] as bool? ?? true,
      wristband: WristbandInfo.fromMap(m['wristband'] as Map?),
    );
  }
}

/// Native events pushed while the UI is open.
sealed class SaharaEvent {
  const SaharaEvent();

  static SaharaEvent? fromMap(Map<dynamic, dynamic> m) => switch (m['type']) {
    'detection' => DetectionArrived(DetectionEvent.fromMap(m), opened: false),
    'openAlert' => DetectionArrived(DetectionEvent.fromMap(m), opened: true),
    'scores' => ScoresArrived(
      ((m['scores'] as List?) ?? const []).map((e) => (e as num).toDouble()).toList(),
      (m['energy'] as num?)?.toDouble() ?? 0,
    ),
    'state' => StateChanged(m['listening'] as bool? ?? false, m['error'] as String?),
    'wristband' => WristbandChanged(WristbandInfo.fromMap(m)),
    _ => null,
  };
}

class DetectionArrived extends SaharaEvent {
  const DetectionArrived(this.event, {required this.opened});
  final DetectionEvent event;

  /// true when the user tapped the alert notification.
  final bool opened;
}

class ScoresArrived extends SaharaEvent {
  const ScoresArrived(this.scores, this.energy);
  final List<double> scores;
  final double energy;
}

class StateChanged extends SaharaEvent {
  const StateChanged(this.listening, this.error);
  final bool listening;
  final String? error;
}

class WristbandChanged extends SaharaEvent {
  const WristbandChanged(this.info);
  final WristbandInfo info;
}

/// Dart side of the `sahara/control` + `sahara/events` channels (SaharaChannel.kt).
abstract class SaharaApi {
  Stream<SaharaEvent> get events;
  Future<SaharaStatus> getStatus();
  Future<void> startListening();
  Future<void> stopListening();
  Future<void> updateSettings({Set<String>? enabledClasses, double? sensitivity, bool? vibrate});
  Future<List<DetectionEvent>> getHistory();
  Future<void> clearHistory();
  Future<List<ScannedBand>> scanWristbands({Duration timeout = const Duration(seconds: 6)});
  Future<void> connectWristband(ScannedBand band);
  Future<void> forgetWristband();
  Future<bool> testWristband();
  Future<void> simulateAlert(int classId);
  Future<void> dismissAlert();
  Future<DetectionEvent?> consumeLaunchAlert();
}

class ChannelSaharaApi implements SaharaApi {
  static const _methods = MethodChannel('sahara/control');
  static const _events = EventChannel('sahara/events');

  late final Stream<SaharaEvent> _stream = _events
      .receiveBroadcastStream()
      .map((e) => SaharaEvent.fromMap(e as Map))
      .where((e) => e != null)
      .cast<SaharaEvent>();

  @override
  Stream<SaharaEvent> get events => _stream;

  @override
  Future<SaharaStatus> getStatus() async =>
      SaharaStatus.fromMap((await _methods.invokeMethod<Map>('getStatus')) ?? const {});

  @override
  Future<void> startListening() => _methods.invokeMethod('startListening');

  @override
  Future<void> stopListening() => _methods.invokeMethod('stopListening');

  @override
  Future<void> updateSettings({Set<String>? enabledClasses, double? sensitivity, bool? vibrate}) =>
      _methods.invokeMethod('updateSettings', {
        if (enabledClasses != null) 'enabledClasses': enabledClasses.toList(),
        'sensitivity': ?sensitivity,
        'vibrate': ?vibrate,
      });

  @override
  Future<List<DetectionEvent>> getHistory() async {
    final list = await _methods.invokeMethod<List>('getHistory') ?? const [];
    return list.map((e) => DetectionEvent.fromMap(e as Map)).toList();
  }

  @override
  Future<void> clearHistory() => _methods.invokeMethod('clearHistory');

  @override
  Future<List<ScannedBand>> scanWristbands({Duration timeout = const Duration(seconds: 6)}) async {
    final list = await _methods.invokeMethod<List>('scanWristbands', {'timeoutMs': timeout.inMilliseconds}) ?? const [];
    return list
        .cast<Map>()
        .map((m) => ScannedBand(m['address'] as String, m['name'] as String, (m['rssi'] as num).toInt()))
        .toList();
  }

  @override
  Future<void> connectWristband(ScannedBand band) =>
      _methods.invokeMethod('connectWristband', {'address': band.address, 'name': band.name});

  @override
  Future<void> forgetWristband() => _methods.invokeMethod('forgetWristband');

  @override
  Future<bool> testWristband() async => await _methods.invokeMethod<bool>('testWristband') ?? false;

  @override
  Future<void> simulateAlert(int classId) => _methods.invokeMethod('simulateAlert', {'classId': classId});

  @override
  Future<void> dismissAlert() => _methods.invokeMethod('dismissAlert');

  @override
  Future<DetectionEvent?> consumeLaunchAlert() async {
    final m = await _methods.invokeMethod<Map>('consumeLaunchAlert');
    return m == null ? null : DetectionEvent.fromMap(m);
  }
}
