import 'package:flutter/material.dart';

import '../audio/sahara_service.dart';
import '../detection/sound_class.dart';
import '../ui/app_scope.dart';

/// Home calibration: record each sound where it actually happens, so the app catches this home's doorbell,
/// door and appliances. Each recording is analysed on the phone and discarded; only one number per sound is
/// kept. Calibration can only make a sound easier to catch, and never beyond the most sensitive setting that
/// was tested for false alarms.
class HomeCalibrationScreen extends StatefulWidget {
  const HomeCalibrationScreen({super.key});

  @override
  State<HomeCalibrationScreen> createState() => _HomeCalibrationScreenState();
}

class _HomeCalibrationScreenState extends State<HomeCalibrationScreen> {
  Future<void> _calibrate(String label) async {
    final saved = await Navigator.of(context).push<bool>(
      MaterialPageRoute(builder: (_) => CalibrateSoundScreen(label: label)),
    );
    if (saved == true && mounted) await AppScope.of(context).refresh();
  }

  Future<void> _reset([String? label]) async {
    await AppScope.of(context).api.clearCalibration(label);
    if (mounted) await AppScope.of(context).refresh();
  }

  @override
  Widget build(BuildContext context) {
    final s = AppScope.of(context).status;
    return Scaffold(
      appBar: AppBar(title: const Text('Calibrate for this home')),
      body: ListView(
        children: [
          const Card(
            margin: EdgeInsets.all(16),
            child: Padding(
              padding: EdgeInsets.all(16),
              child: Text(
                'Every home sounds different. Record your own doorbell, door, microwave and so on '
                'three times each, from where the phone usually sits, and the app will catch them more reliably.\n\n'
                'Each recording is analysed on this phone and deleted at once. Only one number per sound is kept.',
              ),
            ),
          ),
          for (final label in s.labels)
            ListTile(
              leading: Icon(SoundClass.of(label).icon, color: SoundClass.of(label).color),
              title: Text(SoundClass.of(label).name),
              subtitle: Text(s.calibrated.containsKey(label) ? 'Calibrated for this home' : 'Not calibrated'),
              trailing: s.calibrated.containsKey(label)
                  ? IconButton(
                      tooltip: 'Reset ${SoundClass.of(label).name}',
                      icon: const Icon(Icons.restart_alt),
                      onPressed: () => _reset(label),
                    )
                  : const Icon(Icons.chevron_right),
              onTap: () => _calibrate(label),
            ),
          if (s.calibrated.isNotEmpty)
            Padding(
              padding: const EdgeInsets.all(16),
              child: OutlinedButton(onPressed: _reset, child: const Text('Reset all calibration')),
            ),
        ],
      ),
    );
  }
}

/// Three recordings of one sound, plus an optional recording of ordinary home sound.
class CalibrateSoundScreen extends StatefulWidget {
  const CalibrateSoundScreen({super.key, required this.label});

  final String label;

  @override
  State<CalibrateSoundScreen> createState() => _CalibrateSoundScreenState();
}

class _CalibrateSoundScreenState extends State<CalibrateSoundScreen> {
  CalibrationState _state = const CalibrationState();
  bool _recording = false;
  bool _done = false;
  String? _message;

  SaharaApi get _api => AppScope.of(context).api;

  static const _how = {
    'smoke_alarm': 'Press the test button on your smoke alarm',
    'doorbell': 'Ring your doorbell',
    'siren': 'Play or wait for a siren outside',
    'knocking': 'Knock on your front door',
    'dog_bark': 'Record your dog barking',
    'baby_cry': 'Record the baby crying',
    'glass_break': 'Record glass breaking safely, e.g. an old bottle in a bag',
    'appliance_beep': 'Let your microwave or washing machine beep',
  };

  Future<void> _record({bool background = false}) async {
    setState(() {
      _recording = true;
      _message = null;
    });
    final s = await _api.recordCalibrationSample(widget.label, background: background);
    if (!mounted) return;
    setState(() {
      _recording = false;
      _state = s;
      _message = s.error;
    });
  }

  Future<void> _save() async {
    try {
      final (before, after) = await _api.finishCalibration();
      if (!mounted) return;
      _done = true;
      final name = SoundClass.of(widget.label).name;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(
        content: Text(after < before
            ? '$name: the app is now more sensitive to it in this home'
            : '$name: already caught at the usual setting; nothing changed'),
      ));
      Navigator.of(context).pop(true);
    } on StateError catch (e) {
      setState(() => _message = e.message);
    }
  }

  @override
  void dispose() {
    if (!_done) _api.cancelCalibration();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final s = _state;
    final name = SoundClass.of(widget.label).name;
    return Scaffold(
      appBar: AppBar(title: Text('Calibrate: $name')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Text(
            '${_how[widget.label] ?? 'Make the sound'}, then tap Record (5 seconds). '
            'Leave the phone where it usually sits. Listening pauses while you record. Do this ${s.needed} times.',
          ),
          const SizedBox(height: 16),
          LinearProgressIndicator(value: s.samples / s.needed),
          const SizedBox(height: 8),
          Text('${s.samples} of ${s.needed} recordings', key: const Key('calibration-count')),
          if (_message != null)
            Padding(
              padding: const EdgeInsets.only(top: 8),
              child: Text(_message!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
            ),
          const SizedBox(height: 24),
          FilledButton.icon(
            onPressed: _recording || s.canFinish ? null : () => _record(),
            icon: Icon(_recording ? Icons.hourglass_top : Icons.mic),
            label: Text(_recording ? 'Recording...' : 'Record $name'),
          ),
          const SizedBox(height: 16),
          const Text('Optional: record 5 seconds of ordinary sound in the room (TV, talking, kitchen), '
              'so the app does not become too sensitive to it.'),
          const SizedBox(height: 8),
          OutlinedButton.icon(
            onPressed: _recording || s.background ? null : () => _record(background: true),
            icon: Icon(s.background ? Icons.check : Icons.home_outlined),
            label: Text(s.background ? 'Home sound recorded' : 'Record home sound'),
          ),
          const SizedBox(height: 24),
          FilledButton(onPressed: s.canFinish && !_recording ? _save : null, child: const Text('Save')),
        ],
      ),
    );
  }
}
