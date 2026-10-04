import 'sound_class.dart';

/// One alert raised by the native service.
class DetectionEvent {
  const DetectionEvent({
    required this.classId,
    required this.label,
    required this.confidence,
    required this.time,
    required this.safetyCritical,
  });

  final int classId;
  final String label;
  final double confidence;
  final DateTime time;
  final bool safetyCritical;

  SoundClass get sound => SoundClass.of(label);

  factory DetectionEvent.fromMap(Map<dynamic, dynamic> m) => DetectionEvent(
    classId: (m['classId'] as num).toInt(),
    label: m['label'] as String,
    confidence: (m['confidence'] as num).toDouble(),
    time: DateTime.fromMillisecondsSinceEpoch((m['timestampMs'] as num).toInt()),
    safetyCritical: m['safetyCritical'] as bool? ?? false,
  );
}
