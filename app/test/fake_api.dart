import 'dart:async';

import 'package:sahara/audio/sahara_service.dart';
import 'package:sahara/detection/detection_event.dart';

const testLabels = [
  'smoke_alarm',
  'doorbell',
  'siren',
  'knocking',
  'dog_bark',
  'baby_cry',
  'glass_break',
  'appliance_beep',
];

class FakeSaharaApi implements SaharaApi {
  FakeSaharaApi({this.modelReady = true});

  final bool modelReady;
  final controller = StreamController<SaharaEvent>.broadcast();
  final calls = <String>[];
  Map<String, Object?> lastSettings = {};

  @override
  Stream<SaharaEvent> get events => controller.stream;

  @override
  Future<SaharaStatus> getStatus() async => SaharaStatus.fromMap({
    'listening': false,
    'modelReady': modelReady,
    'modelReason': modelReady ? null : 'sahara_classifier.tflite not bundled',
    'labels': testLabels,
    'thresholds': {for (final l in testLabels) l: 0.5},
    'settings': {'sensitivity': 0.0, 'vibrate': true},
    'wristband': {'state': 'disconnected'},
    'personal': {'nameReady': true, 'name': 'Aryan', 'callerIdReady': true, 'otherSoundsReady': true},
    'callers': [],
  });

  @override
  Future<void> startListening() async => calls.add('start');
  @override
  Future<void> stopListening() async => calls.add('stop');
  @override
  Future<void> updateSettings({
    Set<String>? enabledClasses,
    double? sensitivity,
    bool? vibrate,
    bool? nameCalls,
    bool? otherSounds,
  }) async {
    calls.add('settings');
    lastSettings = {
      'enabledClasses': enabledClasses,
      'sensitivity': sensitivity,
      'vibrate': vibrate,
      'nameCalls': nameCalls,
      'otherSounds': otherSounds,
    };
  }

  final callers = <String, int>{};

  @override
  Future<Caller> enrollCaller(String name) async {
    calls.add('enroll:$name');
    callers[name] = (callers[name] ?? 0) + 1;
    return Caller(name, callers[name]!);
  }

  @override
  Future<List<Caller>> removeCaller(String name) async {
    calls.add('remove:$name');
    callers.remove(name);
    return [for (final e in callers.entries) Caller(e.key, e.value)];
  }

  @override
  Future<List<DetectionEvent>> getHistory() async => [];
  @override
  Future<void> clearHistory() async => calls.add('clearHistory');
  @override
  Future<List<ScannedBand>> scanWristbands({Duration timeout = const Duration(seconds: 6)}) async => [];
  @override
  Future<void> connectWristband(ScannedBand band) async => calls.add('connect');
  @override
  Future<void> forgetWristband() async => calls.add('forget');
  @override
  Future<bool> testWristband() async => true;
  @override
  Future<void> simulateAlert(int classId, {String? caller}) async => calls.add('simulate:$classId');
  @override
  Future<void> dismissAlert() async => calls.add('dismiss');
  @override
  Future<DetectionEvent?> consumeLaunchAlert() async => null;
}
