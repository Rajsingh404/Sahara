import 'package:flutter/material.dart';

/// Display metadata for a model class. The class order itself comes from
/// assets/labels.txt (generated from SOUND_CLASSES in src/config.py).
class SoundClass {
  const SoundClass({
    required this.label,
    required this.name,
    required this.icon,
    required this.color,
    required this.advice,
    this.safetyCritical = false,
  });

  final String label;
  final String name;
  final IconData icon;
  final Color color;
  final String advice;
  final bool safetyCritical;

  static const _catalog = <String, SoundClass>{
    'smoke_alarm': SoundClass(
      label: 'smoke_alarm',
      name: 'Smoke alarm',
      icon: Icons.local_fire_department,
      color: Color(0xFFD32F2F),
      advice: 'Check for smoke or fire and move to safety.',
      safetyCritical: true,
    ),
    'doorbell': SoundClass(
      label: 'doorbell',
      name: 'Doorbell',
      icon: Icons.doorbell,
      color: Color(0xFF1976D2),
      advice: 'Someone is at the door.',
    ),
    'siren': SoundClass(
      label: 'siren',
      name: 'Siren',
      icon: Icons.emergency,
      color: Color(0xFFC62828),
      advice: 'An emergency vehicle is nearby. Look around before moving.',
      safetyCritical: true,
    ),
    'knocking': SoundClass(
      label: 'knocking',
      name: 'Knocking',
      icon: Icons.front_hand,
      color: Color(0xFF5D4037),
      advice: 'Someone is knocking.',
    ),
    'dog_bark': SoundClass(
      label: 'dog_bark',
      name: 'Dog barking',
      icon: Icons.pets,
      color: Color(0xFF6D4C41),
      advice: 'A dog is barking nearby.',
    ),
    'baby_cry': SoundClass(
      label: 'baby_cry',
      name: 'Baby crying',
      icon: Icons.child_care,
      color: Color(0xFF8E24AA),
      advice: 'A baby is crying.',
    ),
    'glass_break': SoundClass(
      label: 'glass_break',
      name: 'Glass breaking',
      icon: Icons.broken_image,
      color: Color(0xFFE65100),
      advice: 'Glass may have broken. Check the area carefully.',
      safetyCritical: true,
    ),
    'appliance_beep': SoundClass(
      label: 'appliance_beep',
      name: 'Appliance beep',
      icon: Icons.kitchen,
      color: Color(0xFF00897B),
      advice: 'An appliance (microwave, washing machine, cooker) is beeping.',
    ),
  };

  /// Metadata for [label]; unknown labels (e.g. a retrained model with a new class) get a
  /// generic entry instead of crashing.
  static SoundClass of(String label) =>
      _catalog[label] ??
      SoundClass(
        label: label,
        name: label.split('_').map((w) => w.isEmpty ? w : '${w[0].toUpperCase()}${w.substring(1)}').join(' '),
        icon: Icons.hearing,
        color: const Color(0xFF455A64),
        advice: 'A sound was detected.',
      );

  static Iterable<String> get knownLabels => _catalog.keys;
}
