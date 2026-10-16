import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import 'audio/sahara_service.dart';
import 'detection/detection_event.dart';

/// UI state, fed by the native service. Detection itself runs natively so it keeps going
/// with the screen off; this class only mirrors it.
class AppState extends ChangeNotifier {
  AppState(this.api);

  final SaharaApi api;
  StreamSubscription<SaharaEvent>? _sub;

  SaharaStatus status = const SaharaStatus();
  List<double> liveScores = const [];
  double liveEnergy = 0;
  List<DetectionEvent> history = const [];

  /// Latest "other sound" (YAMNet class outside the eight alerts), for the home screen.
  DetectionEvent? lastOtherSound;
  bool loading = true;

  final _alerts = StreamController<DetectionEvent>.broadcast();

  /// Detections that should open the full-screen alert.
  Stream<DetectionEvent> get alerts => _alerts.stream;

  Future<void> init() async {
    _sub = api.events.listen(_onEvent);
    await refresh();
    history = await api.getHistory();
    loading = false;
    notifyListeners();
    final launched = await api.consumeLaunchAlert();
    if (launched != null) _alerts.add(launched);
  }

  Future<void> refresh() async {
    status = await api.getStatus();
    notifyListeners();
  }

  void _onEvent(SaharaEvent e) {
    switch (e) {
      case DetectionArrived(:final event):
        history = [event, ...history].take(100).toList();
        // Other sounds are notices: they show in the history and the live card, never full screen.
        if (event.isOtherSound) {
          lastOtherSound = event;
        } else {
          _alerts.add(event);
        }
      case ScoresArrived(:final scores, :final energy):
        liveScores = scores;
        liveEnergy = energy;
      case StateChanged(:final listening, :final error):
        status = status.copyWith(listening: listening, error: error, clearError: error == null);
        if (!listening) liveScores = const [];
      case WristbandChanged(:final info):
        status = status.copyWith(wristband: info);
    }
    notifyListeners();
  }

  Future<String?> setListening(bool on) async {
    try {
      if (on) {
        await api.startListening();
      } else {
        await api.stopListening();
      }
      return null;
    } on PlatformException catch (e) {
      return e.message ?? e.code;
    }
  }

  Future<void> setClassEnabled(String label, bool enabled) async {
    final current = status.enabledClasses ?? status.labels.toSet();
    final next = {...current};
    enabled ? next.add(label) : next.remove(label);
    status = status.copyWith(enabledClasses: next);
    notifyListeners();
    await api.updateSettings(enabledClasses: next);
  }

  Future<void> setSensitivity(double value) async {
    status = status.copyWith(sensitivity: value);
    notifyListeners();
    await api.updateSettings(sensitivity: value);
  }

  Future<void> setVibrate(bool value) async {
    status = status.copyWith(vibrate: value);
    notifyListeners();
    await api.updateSettings(vibrate: value);
  }

  Future<void> setNameCalls(bool value) async {
    status = status.copyWith(nameCalls: value);
    notifyListeners();
    await api.updateSettings(nameCalls: value);
  }

  Future<void> setOtherSounds(bool value) async {
    status = status.copyWith(otherSounds: value);
    notifyListeners();
    await api.updateSettings(otherSounds: value);
  }

  /// Records one enrolment sample for [name]. Returns null on success, else a message to show.
  Future<String?> enrollCaller(String name) async {
    try {
      final caller = await api.enrollCaller(name);
      final others = status.callers.where((c) => c.name != caller.name);
      status = status.copyWith(callers: [...others, caller]);
      notifyListeners();
      return null;
    } on PlatformException catch (e) {
      return e.message ?? e.code;
    }
  }

  Future<void> removeCaller(String name) async {
    status = status.copyWith(callers: await api.removeCaller(name));
    notifyListeners();
  }

  Future<void> clearHistory() async {
    await api.clearHistory();
    history = const [];
    notifyListeners();
  }

  @override
  void dispose() {
    _sub?.cancel();
    _alerts.close();
    super.dispose();
  }
}
