import 'package:flutter_test/flutter_test.dart';
import 'package:sahara/audio/sahara_service.dart';
import 'package:sahara/detection/sound_class.dart';

void main() {
  test('parses a detection event', () {
    final e = SaharaEvent.fromMap({
      'type': 'detection',
      'classId': 2,
      'label': 'siren',
      'confidence': 0.91,
      'timestampMs': 1700000000000,
      'safetyCritical': true,
    });
    expect(e, isA<DetectionArrived>());
    final d = (e as DetectionArrived).event;
    expect(d.label, 'siren');
    expect(d.sound.name, 'Siren');
    expect(d.safetyCritical, isTrue);
    expect(e.opened, isFalse);
  });

  test('parses scores, state and wristband events; ignores unknown', () {
    expect(
      (SaharaEvent.fromMap({
                'type': 'scores',
                'scores': [0.1, 0.2],
                'energy': 0.001,
              })
              as ScoresArrived)
          .scores,
      [0.1, 0.2],
    );
    final st = SaharaEvent.fromMap({'type': 'state', 'listening': true, 'error': null}) as StateChanged;
    expect(st.listening, isTrue);
    final wb =
        SaharaEvent.fromMap({'type': 'wristband', 'state': 'connected', 'address': 'AA', 'batteryPct': 80})
            as WristbandChanged;
    expect(wb.info.connected, isTrue);
    expect(wb.info.batteryPct, 80);
    expect(SaharaEvent.fromMap({'type': 'nope'}), isNull);
  });

  test('status: null enabledClasses means all on', () {
    final s = SaharaStatus.fromMap({
      'labels': ['doorbell', 'siren'],
      'settings': {'enabledClasses': null, 'sensitivity': 0.5},
    });
    expect(s.isEnabled('doorbell'), isTrue);
    expect(s.sensitivity, 0.5);
    final s2 = SaharaStatus.fromMap({
      'labels': ['doorbell', 'siren'],
      'settings': {
        'enabledClasses': ['siren'],
      },
    });
    expect(s2.isEnabled('doorbell'), isFalse);
    expect(s2.isEnabled('siren'), isTrue);
  });

  test('unknown labels get a generic display entry', () {
    final s = SoundClass.of('pressure_cooker');
    expect(s.name, 'Pressure Cooker');
    expect(s.safetyCritical, isFalse);
  });
}
