import 'dart:async';

import 'package:flutter/material.dart';

import '../detection/detection_event.dart';

/// Full-screen visual alert. Flashes the class colour; safety-critical sounds flash faster
/// and stay until dismissed.
class AlertScreen extends StatefulWidget {
  const AlertScreen({super.key, required this.event, required this.onDismiss});

  final DetectionEvent event;
  final VoidCallback onDismiss;

  @override
  State<AlertScreen> createState() => _AlertScreenState();
}

class _AlertScreenState extends State<AlertScreen> {
  Timer? _flash;
  bool _on = true;

  @override
  void initState() {
    super.initState();
    final period = Duration(milliseconds: widget.event.safetyCritical ? 350 : 700);
    _flash = Timer.periodic(period, (_) => setState(() => _on = !_on));
  }

  @override
  void dispose() {
    _flash?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final s = widget.event.sound;
    final bg = _on ? s.color : Colors.black;
    final t = widget.event.time;
    final time = '${t.hour.toString().padLeft(2, '0')}:${t.minute.toString().padLeft(2, '0')}';
    return Scaffold(
      backgroundColor: bg,
      body: SafeArea(
        child: Semantics(
          liveRegion: true,
          label: '${s.name} detected',
          child: Padding(
            padding: const EdgeInsets.all(24),
            child: Column(
              children: [
                const Spacer(),
                Icon(s.icon, size: 140, color: Colors.white),
                const SizedBox(height: 24),
                Text(
                  s.name.toUpperCase(),
                  textAlign: TextAlign.center,
                  style: const TextStyle(color: Colors.white, fontSize: 40, fontWeight: FontWeight.w900),
                ),
                const SizedBox(height: 12),
                Text(
                  s.advice,
                  textAlign: TextAlign.center,
                  style: const TextStyle(color: Colors.white, fontSize: 20),
                ),
                const SizedBox(height: 12),
                Text(
                  'Detected at $time · ${(widget.event.confidence * 100).round()}% confidence',
                  style: const TextStyle(color: Colors.white70, fontSize: 16),
                ),
                const Spacer(),
                SizedBox(
                  width: double.infinity,
                  height: 72,
                  child: FilledButton(
                    style: FilledButton.styleFrom(backgroundColor: Colors.white, foregroundColor: Colors.black),
                    onPressed: widget.onDismiss,
                    child: const Text('Dismiss', style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold)),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
