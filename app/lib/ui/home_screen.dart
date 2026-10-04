import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../detection/sound_class.dart';
import 'app_scope.dart';
import 'history_screen.dart';
import 'permissions.dart';
import 'settings_screen.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final state = AppScope.of(context);
    final s = state.status;
    return Scaffold(
      appBar: AppBar(
        title: const Text('SAHARA'),
        actions: [
          IconButton(
            tooltip: 'History',
            icon: const Icon(Icons.history),
            onPressed: () => Navigator.of(context).push(MaterialPageRoute(builder: (_) => const HistoryScreen())),
          ),
          IconButton(
            tooltip: 'Settings',
            icon: const Icon(Icons.settings),
            onPressed: () => Navigator.of(context).push(MaterialPageRoute(builder: (_) => const SettingsScreen())),
          ),
        ],
      ),
      body: state.loading
          ? const Center(child: CircularProgressIndicator())
          : RefreshIndicator(
              onRefresh: state.refresh,
              child: ListView(
                padding: const EdgeInsets.all(16),
                children: [
                  const _ListenCard(),
                  if (!s.modelReady)
                    _Banner(icon: Icons.warning_amber, text: 'Model not installed: ${s.modelReason ?? 'unknown'}'),
                  if (s.error != null) _Banner(icon: Icons.error_outline, text: s.error!),
                  const SizedBox(height: 8),
                  const _WristbandTile(),
                  const SizedBox(height: 16),
                  Text('Sounds', style: Theme.of(context).textTheme.titleMedium),
                  const SizedBox(height: 8),
                  for (var i = 0; i < s.labels.length; i++) _SoundRow(index: i),
                  const SizedBox(height: 16),
                  Text(
                    'Sounds are analysed on this phone in one-second pieces that are immediately discarded. '
                    'Nothing is recorded or uploaded.',
                    style: Theme.of(context).textTheme.bodySmall,
                  ),
                ],
              ),
            ),
    );
  }
}

class _ListenCard extends StatelessWidget {
  const _ListenCard();

  @override
  Widget build(BuildContext context) {
    final state = AppScope.of(context);
    final listening = state.status.listening;
    final scheme = Theme.of(context).colorScheme;
    return Card(
      color: listening ? scheme.primaryContainer : scheme.surfaceContainerHighest,
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          children: [
            Icon(listening ? Icons.hearing : Icons.hearing_disabled, size: 64),
            const SizedBox(height: 8),
            Text(
              listening ? 'Listening for sounds' : 'Not listening',
              style: Theme.of(context).textTheme.headlineSmall,
            ),
            const SizedBox(height: 8),
            if (listening) _LevelMeter(energy: state.liveEnergy),
            const SizedBox(height: 12),
            SizedBox(
              width: double.infinity,
              height: 56,
              child: FilledButton.icon(
                icon: Icon(listening ? Icons.stop : Icons.play_arrow),
                label: Text(listening ? 'Stop' : 'Start listening', style: const TextStyle(fontSize: 18)),
                onPressed: state.status.modelReady || listening ? () => _toggle(context, !listening) : null,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Future<void> _toggle(BuildContext context, bool on) async {
    final state = AppScope.of(context);
    final messenger = ScaffoldMessenger.of(context);
    if (on && !await Permissions.ensureListening()) {
      messenger.showSnackBar(
        SnackBar(
          content: const Text('SAHARA needs the microphone to hear sounds around you.'),
          action: SnackBarAction(label: 'Settings', onPressed: Permissions.openSettings),
        ),
      );
      return;
    }
    final error = await state.setListening(on);
    if (error != null) messenger.showSnackBar(SnackBar(content: Text(error)));
  }
}

class _LevelMeter extends StatelessWidget {
  const _LevelMeter({required this.energy});
  final double energy;

  @override
  Widget build(BuildContext context) {
    // Mean-square energy spans roughly 1e-6 (quiet room) to 1e-1 (very loud); show it on a log scale.
    final db = 10 * math.log(energy.clamp(1e-6, 1.0)) / math.ln10;
    final level = ((db + 60) / 60).clamp(0.0, 1.0);
    return Semantics(
      label: 'Sound level',
      value: '${(level * 100).round()} percent',
      child: LinearProgressIndicator(value: level, minHeight: 8, borderRadius: BorderRadius.circular(4)),
    );
  }
}

class _Banner extends StatelessWidget {
  const _Banner({required this.icon, required this.text});
  final IconData icon;
  final String text;

  @override
  Widget build(BuildContext context) => Card(
    color: Theme.of(context).colorScheme.errorContainer,
    child: ListTile(leading: Icon(icon), title: Text(text)),
  );
}

class _WristbandTile extends StatelessWidget {
  const _WristbandTile();

  @override
  Widget build(BuildContext context) {
    final band = AppScope.of(context).status.wristband;
    final subtitle = !band.paired
        ? 'No wristband paired'
        : '${band.name ?? 'SAHARA band'} · ${band.state}'
              '${band.batteryPct != null ? ' · ${band.batteryPct}% battery' : ''}';
    return Card(
      child: ListTile(
        leading: Icon(band.connected ? Icons.watch : Icons.watch_off_outlined),
        title: const Text('Wristband'),
        subtitle: Text(subtitle),
        trailing: const Icon(Icons.chevron_right),
        onTap: () => Navigator.of(context).push(MaterialPageRoute(builder: (_) => const SettingsScreen())),
      ),
    );
  }
}

class _SoundRow extends StatelessWidget {
  const _SoundRow({required this.index});
  final int index;

  @override
  Widget build(BuildContext context) {
    final state = AppScope.of(context);
    final label = state.status.labels[index];
    final sound = SoundClass.of(label);
    final enabled = state.status.isEnabled(label);
    final score = index < state.liveScores.length ? state.liveScores[index] : 0.0;
    return Card(
      child: ListTile(
        leading: CircleAvatar(backgroundColor: sound.color, foregroundColor: Colors.white, child: Icon(sound.icon)),
        title: Text(sound.name),
        subtitle: state.status.listening && enabled
            ? LinearProgressIndicator(value: score.clamp(0.0, 1.0), color: sound.color)
            : Text(enabled ? (sound.safetyCritical ? 'Safety alert' : 'Alert on') : 'Off'),
        trailing: Switch(value: enabled, onChanged: (v) => state.setClassEnabled(label, v)),
      ),
    );
  }
}
