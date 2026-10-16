import 'sound_class.dart';

/// One alert raised by the native service.
class DetectionEvent {
  const DetectionEvent({
    required this.classId,
    required this.label,
    required this.confidence,
    required this.time,
    required this.safetyCritical,
    this.detail,
  });

  final int classId;
  final String label;
  final double confidence;
  final DateTime time;
  final bool safetyCritical;

  /// Name calls: who is calling (null = someone not enrolled). Other sounds: what YAMNet heard.
  final String? detail;

  SoundClass get sound => SoundClass.of(label);

  bool get isNameCall => label == SoundClass.nameCall;
  bool get isOtherSound => label == SoundClass.otherSound;

  /// "Raj is calling you", "Someone is calling you", "Water tap, faucet", "Doorbell".
  String get title => switch (label) {
    SoundClass.nameCall => '${detail ?? 'Someone'} is calling you',
    SoundClass.otherSound => detail ?? 'A sound',
    _ => sound.name,
  };

  factory DetectionEvent.fromMap(Map<dynamic, dynamic> m) => DetectionEvent(
    classId: (m['classId'] as num).toInt(),
    label: m['label'] as String,
    confidence: (m['confidence'] as num).toDouble(),
    time: DateTime.fromMillisecondsSinceEpoch((m['timestampMs'] as num).toInt()),
    safetyCritical: m['safetyCritical'] as bool? ?? false,
    detail: m['detail'] as String?,
  );
}
