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

/// Someone enrolled to be recognised when they call the user.
class Caller {
  const Caller(this.name, this.samples);
  final String name;

  /// Enrolment recordings so far; three or more identify a caller best.
  final int samples;

  static List<Caller> listFrom(List? raw) =>
      (raw ?? const []).cast<Map>().map((m) => Caller(m['name'] as String, (m['samples'] as num).toInt())).toList();
}

/// What the name-call and other-sounds layers can do with the bundled assets.
class PersonalStatus {
  const PersonalStatus({
    this.nameReady = false,
    this.name,
    this.callerIdReady = false,
    this.nameReason,
    this.otherSoundsReady = false,
    this.otherSoundsReason,
  });

  final bool nameReady;

  /// The name the spotter listens for ("Aryan").
  final String? name;
  final bool callerIdReady;
  final String? nameReason;
  final bool otherSoundsReady;
  final String? otherSoundsReason;

  factory PersonalStatus.fromMap(Map<dynamic, dynamic>? m) => m == null
      ? const PersonalStatus()
      : PersonalStatus(
          nameReady: m['nameReady'] as bool? ?? false,
          name: m['name'] as String?,
          callerIdReady: m['callerIdReady'] as bool? ?? false,
          nameReason: m['nameCalls'] as String?,
          otherSoundsReady: m['otherSoundsReady'] as bool? ?? false,
          otherSoundsReason: m['otherSounds'] as String?,
        );
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
    this.nameCalls = true,
    this.otherSounds = true,
    this.wristband = const WristbandInfo(),
    this.personal = const PersonalStatus(),
    this.callers = const [],
    this.calibrated = const {},
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
  final bool nameCalls;
  final bool otherSounds;
  final WristbandInfo wristband;
  final PersonalStatus personal;
  final List<Caller> callers;

  /// Home calibration: per sound, the threshold fitted to this home (empty = none).
  final Map<String, double> calibrated;

  bool isEnabled(String label) => enabledClasses?.contains(label) ?? true;

  SaharaStatus copyWith({
    bool? listening,
    String? error,
    bool clearError = false,
    String? modelReason,
    Set<String>? enabledClasses,
    double? sensitivity,
    bool? vibrate,
    bool? nameCalls,
    bool? otherSounds,
    WristbandInfo? wristband,
    List<Caller>? callers,
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
    nameCalls: nameCalls ?? this.nameCalls,
    otherSounds: otherSounds ?? this.otherSounds,
    wristband: wristband ?? this.wristband,
    personal: personal,
    callers: callers ?? this.callers,
    calibrated: calibrated,
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
      nameCalls: settings['nameCalls'] as bool? ?? true,
      otherSounds: settings['otherSounds'] as bool? ?? true,
      wristband: WristbandInfo.fromMap(m['wristband'] as Map?),
      personal: PersonalStatus.fromMap(m['personal'] as Map?),
      callers: Caller.listFrom(m['callers'] as List?),
      calibrated: ((settings['calibrated'] as Map?) ?? const {})
          .map((k, v) => MapEntry(k as String, (v as num).toDouble())),
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

/// Progress of a custom-sound enrolment (CustomEnrolment.kt).
class EnrolmentState {
  const EnrolmentState({this.samples = 0, this.minSamples = 5, this.maxSamples = 10, this.error});

  final int samples;
  final int minSamples;
  final int maxSamples;
  final String? error;

  bool get canFinish => samples >= minSamples;
  bool get full => samples >= maxSamples;

  factory EnrolmentState.fromMap(Map<dynamic, dynamic> m) => EnrolmentState(
    samples: (m['samples'] as num?)?.toInt() ?? 0,
    minSamples: (m['minSamples'] as num?)?.toInt() ?? 5,
    maxSamples: (m['maxSamples'] as num?)?.toInt() ?? 10,
    error: m['error'] as String?,
  );
}

/// Dart side of the `sahara/control` + `sahara/events` channels (SaharaChannel.kt).
abstract class SaharaApi {
  Stream<SaharaEvent> get events;
  Future<SaharaStatus> getStatus();
  Future<void> startListening();
  Future<void> stopListening();
  Future<void> updateSettings({
    Set<String>? enabledClasses,
    double? sensitivity,
    bool? vibrate,
    bool? nameCalls,
    bool? otherSounds,
  });
  Future<List<DetectionEvent>> getHistory();

  /// Records ~2.5 s while [name] says the user's name, and enrols their voice.
  /// Throws a PlatformException (code `too_quiet`, `mic`, `no_model`, `permission`) on failure.
  Future<Caller> enrollCaller(String name);
  Future<List<Caller>> removeCaller(String name);
  Future<void> clearHistory();
  Future<List<ScannedBand>> scanWristbands({Duration timeout = const Duration(seconds: 6)});
  Future<void> connectWristband(ScannedBand band);
  Future<void> forgetWristband();
  Future<bool> testWristband();
  Future<void> simulateAlert(int classId, {String? caller});
  Future<void> dismissAlert();
  Future<DetectionEvent?> consumeLaunchAlert();

  // Custom sounds (Phase 7). Enrolment audio never leaves native memory; only a prototype is kept.
  Future<List<String>> listCustomSounds();
  Future<EnrolmentState> recordEnrolmentSample();
  Future<String?> finishEnrolment(String name);
  Future<void> cancelEnrolment();
  Future<List<String>> deleteCustomSound(String name);

  // Home calibration: record a sound where it happens; only a threshold per sound is kept.
  Future<CalibrationState> recordCalibrationSample(String label, {bool background = false});
  /// Returns (before, after) thresholds when saved, or throws a [StateError] with the reason.
  Future<(double, double)> finishCalibration();
  Future<void> cancelCalibration();
  Future<Map<String, double>> clearCalibration([String? label]);
}

class CalibrationState {
  const CalibrationState({this.samples = 0, this.needed = 3, this.background = false, this.error});

  final int samples;
  final int needed;
  final bool background;
  final String? error;

  bool get canFinish => samples >= needed;

  factory CalibrationState.fromMap(Map<dynamic, dynamic> m) => CalibrationState(
    samples: (m['samples'] as num?)?.toInt() ?? 0,
    needed: (m['needed'] as num?)?.toInt() ?? 3,
    background: m['background'] == true,
    error: m['error'] as String?,
  );
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
  Future<void> updateSettings({
    Set<String>? enabledClasses,
    double? sensitivity,
    bool? vibrate,
    bool? nameCalls,
    bool? otherSounds,
  }) => _methods.invokeMethod('updateSettings', {
    if (enabledClasses != null) 'enabledClasses': enabledClasses.toList(),
    'sensitivity': ?sensitivity,
    'vibrate': ?vibrate,
    'nameCalls': ?nameCalls,
    'otherSounds': ?otherSounds,
  });

  @override
  Future<Caller> enrollCaller(String name) async {
    final m = await _methods.invokeMethod<Map>('enrollCaller', {'name': name});
    return Caller(m!['name'] as String, (m['samples'] as num).toInt());
  }

  @override
  Future<List<Caller>> removeCaller(String name) async =>
      Caller.listFrom(await _methods.invokeMethod<List>('removeCaller', {'name': name}));

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
  Future<void> simulateAlert(int classId, {String? caller}) =>
      _methods.invokeMethod('simulateAlert', {'classId': classId, 'caller': ?caller});

  @override
  Future<void> dismissAlert() => _methods.invokeMethod('dismissAlert');

  @override
  Future<DetectionEvent?> consumeLaunchAlert() async {
    final m = await _methods.invokeMethod<Map>('consumeLaunchAlert');
    return m == null ? null : DetectionEvent.fromMap(m);
  }

  @override
  Future<List<String>> listCustomSounds() async =>
      (await _methods.invokeMethod<List>('listCustomSounds') ?? const []).cast<String>();

  @override
  Future<EnrolmentState> recordEnrolmentSample() async =>
      EnrolmentState.fromMap(await _methods.invokeMethod<Map>('recordEnrolmentSample') ?? const {});

  /// Returns null when saved, else the reason it was not.
  @override
  Future<String?> finishEnrolment(String name) async {
    final m = await _methods.invokeMethod<Map>('finishEnrolment', {'name': name}) ?? const {};
    return m['saved'] == true ? null : (m['error'] as String? ?? 'Could not save the sound');
  }

  @override
  Future<void> cancelEnrolment() => _methods.invokeMethod('cancelEnrolment');

  @override
  Future<List<String>> deleteCustomSound(String name) async =>
      (await _methods.invokeMethod<List>('deleteCustomSound', {'name': name}) ?? const []).cast<String>();

  @override
  Future<CalibrationState> recordCalibrationSample(String label, {bool background = false}) async =>
      CalibrationState.fromMap(await _methods.invokeMethod<Map>(
            'recordCalibrationSample', {'label': label, 'background': background}) ??
          const {});

  @override
  Future<(double, double)> finishCalibration() async {
    final m = await _methods.invokeMethod<Map>('finishCalibration') ?? const {};
    if (m['saved'] != true) throw StateError(m['error'] as String? ?? 'Could not save');
    return ((m['before'] as num).toDouble(), (m['after'] as num).toDouble());
  }

  @override
  Future<void> cancelCalibration() => _methods.invokeMethod('cancelCalibration');

  @override
  Future<Map<String, double>> clearCalibration([String? label]) async =>
      ((await _methods.invokeMethod<Map>('clearCalibration', {'label': label})) ?? const {})
          .map((k, v) => MapEntry(k as String, (v as num).toDouble()));
}
