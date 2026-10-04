import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:sahara/ble/uuids.dart';
import 'package:sahara/detection/sound_class.dart';

/// Cross-project contracts: class order (src/config.py), thresholds and BLE UUIDs.
/// `flutter test` runs with app/ as the working directory.
void main() {
  List<String> labelsTxt() => File(
    'assets/labels.txt',
  ).readAsLinesSync().map((l) => l.trim()).where((l) => l.isNotEmpty && !l.startsWith('#')).toList();

  test('labels.txt matches SOUND_CLASSES in src/config.py', () {
    final config = File('../src/config.py').readAsStringSync();
    final block = RegExp(r'SOUND_CLASSES\s*=\s*\[([^\]]*)\]').firstMatch(config)!.group(1)!;
    final classes = RegExp(r'"([a-z_]+)"').allMatches(block).map((m) => m.group(1)).toList();
    expect(labelsTxt(), classes);
  });

  test('every label has display metadata and a threshold', () {
    final thresholds = jsonDecode(File('assets/class_thresholds.json').readAsStringSync()) as Map;
    for (final label in labelsTxt()) {
      expect(SoundClass.knownLabels, contains(label));
      expect(thresholds[label]?['threshold'], isA<num>(), reason: label);
    }
  });

  test('safety-critical set matches src/config.py', () {
    final config = File('../src/config.py').readAsStringSync();
    final block = RegExp(r'SAFETY_CRITICAL_CLASSES\s*=\s*\(([^)]*)\)').firstMatch(config)!.group(1)!;
    final safety = RegExp(r'"([a-z_]+)"').allMatches(block).map((m) => m.group(1)).toSet();
    final app = SoundClass.knownLabels.where((l) => SoundClass.of(l).safetyCritical).toSet();
    expect(app, safety);
  });

  test('BLE UUIDs match docs/ble_protocol.md and BleUuids.kt', () {
    final doc = File('../docs/ble_protocol.md').readAsStringSync();
    final kotlin = File('android/app/src/main/kotlin/com/sahara/app/BleUuids.kt').readAsStringSync();
    for (final uuid in [saharaServiceUuid, alertCharacteristicUuid, statusCharacteristicUuid]) {
      expect(doc, contains(uuid));
      expect(kotlin, contains(uuid));
    }
    final header = File('../firmware/esp32_wristband/include/ble_uuids.h');
    if (header.existsSync()) {
      final h = header.readAsStringSync().toLowerCase();
      for (final uuid in [saharaServiceUuid, alertCharacteristicUuid, statusCharacteristicUuid]) {
        expect(h, contains(uuid), reason: 'firmware ble_uuids.h is out of sync');
      }
    }
  });
}
