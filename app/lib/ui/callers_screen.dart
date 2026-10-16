import 'package:flutter/material.dart';

import 'app_scope.dart';
import 'permissions.dart';

/// Wristband / native class id for a name call (ModelConfig.NAME_CALL_ID, docs/ble_protocol.md).
const nameCallClassId = 0x20;

/// Enrol the people who call the user, so a call shows "Raj is calling you" instead of "Someone".
class CallersScreen extends StatefulWidget {
  const CallersScreen({super.key});

  @override
  State<CallersScreen> createState() => _CallersScreenState();
}

class _CallersScreenState extends State<CallersScreen> {
  final _name = TextEditingController();
  String? _recordingFor;

  @override
  void dispose() {
    _name.dispose();
    super.dispose();
  }

  Future<void> _record(String name) async {
    final state = AppScope.of(context);
    final messenger = ScaffoldMessenger.of(context);
    if (name.trim().isEmpty) return;
    if (!await Permissions.ensureListening()) {
      messenger.showSnackBar(const SnackBar(content: Text('The microphone is needed to learn a voice.')));
      return;
    }
    setState(() => _recordingFor = name.trim());
    final error = await state.enrollCaller(name.trim());
    if (!mounted) return;
    setState(() => _recordingFor = null);
    _name.clear();
    messenger.showSnackBar(SnackBar(content: Text(error ?? 'Saved. Two more recordings make it more reliable.')));
  }

  @override
  Widget build(BuildContext context) {
    final state = AppScope.of(context);
    final s = state.status;
    final user = s.personal.name ?? 'your name';
    final recording = _recordingFor != null;
    return Scaffold(
      appBar: AppBar(title: const Text('People who call you')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Text(
            'Anyone who calls "$user" is detected, enrolled or not. Enrol family and friends so the alert says who it is. '
            'Each person holds the phone and says "$user" the way they normally call you. Three recordings each work best.',
          ),
          if (!s.personal.callerIdReady)
            Card(
              color: Theme.of(context).colorScheme.errorContainer,
              child: const ListTile(
                leading: Icon(Icons.warning_amber),
                title: Text('Voice model not installed, so callers cannot be named yet. Calls are still detected.'),
              ),
            ),
          if (recording)
            Card(
              color: Theme.of(context).colorScheme.primaryContainer,
              child: ListTile(
                leading: const Icon(Icons.mic, size: 36),
                title: Text('Recording: $_recordingFor, say "$user" now'),
                subtitle: const LinearProgressIndicator(),
              ),
            ),
          const SizedBox(height: 16),
          for (final c in s.callers)
            Card(
              child: ListTile(
                leading: const CircleAvatar(child: Icon(Icons.person)),
                title: Text(c.name),
                subtitle: Text('${c.samples} recording${c.samples == 1 ? '' : 's'}'),
                trailing: Wrap(
                  children: [
                    IconButton(
                      tooltip: 'Record again',
                      icon: const Icon(Icons.mic_none),
                      onPressed: recording || !s.personal.callerIdReady ? null : () => _record(c.name),
                    ),
                    IconButton(
                      tooltip: 'Remove',
                      icon: const Icon(Icons.delete_outline),
                      onPressed: recording ? null : () => state.removeCaller(c.name),
                    ),
                  ],
                ),
              ),
            ),
          const SizedBox(height: 8),
          TextField(
            controller: _name,
            enabled: !recording && s.personal.callerIdReady,
            textCapitalization: TextCapitalization.words,
            decoration: const InputDecoration(labelText: 'Name of the person calling', border: OutlineInputBorder()),
            onSubmitted: _record,
          ),
          const SizedBox(height: 8),
          SizedBox(
            height: 56,
            child: FilledButton.icon(
              icon: const Icon(Icons.mic),
              label: const Text('Record their voice', style: TextStyle(fontSize: 18)),
              onPressed: recording || !s.personal.callerIdReady ? null : () => _record(_name.text),
            ),
          ),
          const SizedBox(height: 16),
          Text(
            'Only a short list of numbers describing each voice is kept on this phone. The recording itself is discarded.',
            style: Theme.of(context).textTheme.bodySmall,
          ),
        ],
      ),
    );
  }
}
