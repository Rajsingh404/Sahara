import 'package:flutter/material.dart';

import '../audio/sahara_service.dart';
import '../ui/app_scope.dart';

/// Custom sounds (Phase 7): list, delete and enrol user-chosen sounds.
///
/// Enrolment is a one-time setup. Each sample is recorded and analysed in native memory and the
/// audio is discarded at once; the phone keeps only a short list of numbers describing the sound.
class CustomSoundsScreen extends StatefulWidget {
  const CustomSoundsScreen({super.key});

  @override
  State<CustomSoundsScreen> createState() => _CustomSoundsScreenState();
}

class _CustomSoundsScreenState extends State<CustomSoundsScreen> {
  List<String>? _sounds;

  SaharaApi get _api => AppScope.of(context).api;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    _sounds ??= const [];
    _reload();
  }

  Future<void> _reload() async {
    final sounds = await _api.listCustomSounds();
    if (mounted) setState(() => _sounds = sounds);
  }

  Future<void> _delete(String name) async {
    final sounds = await _api.deleteCustomSound(name);
    if (mounted) setState(() => _sounds = sounds);
  }

  Future<void> _add() async {
    final saved = await Navigator.of(context).push<bool>(MaterialPageRoute(builder: (_) => const EnrolScreen()));
    if (saved == true) _reload();
  }

  @override
  Widget build(BuildContext context) {
    final sounds = _sounds ?? const [];
    return Scaffold(
      appBar: AppBar(title: const Text('Custom sounds')),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: _add,
        icon: const Icon(Icons.add),
        label: const Text('Add a sound'),
      ),
      body: ListView(
        padding: const EdgeInsets.only(bottom: 96),
        children: [
          const _Explainer(),
          if (sounds.isEmpty)
            const ListTile(leading: Icon(Icons.info_outline), title: Text('No custom sounds yet'))
          else
            for (final name in sounds)
              ListTile(
                leading: const Icon(Icons.graphic_eq),
                title: Text(name),
                subtitle: const Text('Alerts on the phone; the wristband uses the doorbell pattern'),
                trailing: IconButton(
                  tooltip: 'Delete $name',
                  icon: const Icon(Icons.delete_outline),
                  onPressed: () => _delete(name),
                ),
              ),
        ],
      ),
    );
  }
}

class _Explainer extends StatelessWidget {
  const _Explainer();

  @override
  Widget build(BuildContext context) => const Card(
    margin: EdgeInsets.all(16),
    child: Padding(
      padding: EdgeInsets.all(16),
      child: Text(
        'Experimental. Works best for a distinctive sound that repeats the same way in your home, '
        'such as a pressure cooker whistle or a particular bell. It cannot tell words or names apart.\n\n'
        'You record the sound 5 to 10 times. Each recording is analysed on this phone and deleted '
        'immediately; only a short description of the sound is kept, and nothing is uploaded.',
      ),
    ),
  );
}

/// Records 5-10 samples of one sound and saves it under a name.
class EnrolScreen extends StatefulWidget {
  const EnrolScreen({super.key});

  @override
  State<EnrolScreen> createState() => _EnrolScreenState();
}

class _EnrolScreenState extends State<EnrolScreen> {
  final _name = TextEditingController();
  EnrolmentState _state = const EnrolmentState();
  bool _recording = false;
  bool _done = false;
  String? _message;

  SaharaApi get _api => AppScope.of(context).api;

  Future<void> _record() async {
    setState(() {
      _recording = true;
      _message = null;
    });
    final s = await _api.recordEnrolmentSample();
    if (!mounted) return;
    setState(() {
      _recording = false;
      _state = s;
      _message = s.error;
    });
  }

  Future<void> _save() async {
    final error = await _api.finishEnrolment(_name.text);
    if (!mounted) return;
    if (error == null) {
      _done = true;
      Navigator.of(context).pop(true);
    } else {
      setState(() => _message = error);
    }
  }

  @override
  void dispose() {
    if (!_done) _api.cancelEnrolment();
    _name.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final s = _state;
    return Scaffold(
      appBar: AppBar(title: const Text('Add a custom sound')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          TextField(
            controller: _name,
            maxLength: 40,
            decoration: const InputDecoration(labelText: 'Name', hintText: 'e.g. Pressure cooker'),
          ),
          const SizedBox(height: 8),
          Text(
            'Make the sound, then tap Record (3 seconds). Listening pauses while you record. '
            'Record it ${s.minSamples} to ${s.maxSamples} times, from where you usually are.',
          ),
          const SizedBox(height: 16),
          LinearProgressIndicator(value: s.samples / s.maxSamples),
          const SizedBox(height: 8),
          Text('${s.samples} of ${s.minSamples}-${s.maxSamples} samples', key: const Key('enrol-count')),
          if (_message != null)
            Padding(
              padding: const EdgeInsets.only(top: 8),
              child: Text(_message!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
            ),
          const SizedBox(height: 24),
          FilledButton.icon(
            onPressed: _recording || s.full ? null : _record,
            icon: Icon(_recording ? Icons.hourglass_top : Icons.mic),
            label: Text(_recording ? 'Recording...' : 'Record sample'),
          ),
          const SizedBox(height: 8),
          OutlinedButton(onPressed: s.canFinish && !_recording ? _save : null, child: const Text('Save sound')),
        ],
      ),
    );
  }
}
