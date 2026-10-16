import 'package:flutter/material.dart';

import '../audio/sahara_service.dart';
import '../detection/sound_class.dart';
import 'app_scope.dart';
import 'permissions.dart';

class SettingsScreen extends StatelessWidget {
  const SettingsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final state = AppScope.of(context);
    final s = state.status;
    return Scaffold(
      appBar: AppBar(title: const Text('Settings')),
      body: ListView(
        padding: const EdgeInsets.symmetric(vertical: 8),
        children: [
          const _SectionTitle('Detection'),
          ListTile(title: const Text('Sensitivity'), subtitle: Text(_sensitivityLabel(s.sensitivity))),
          Slider(
            value: s.sensitivity,
            min: -1,
            max: 1,
            divisions: 4,
            label: _sensitivityLabel(s.sensitivity),
            onChanged: state.setSensitivity,
          ),
          SwitchListTile(title: const Text('Vibrate phone on alert'), value: s.vibrate, onChanged: state.setVibrate),
          const Divider(),
          const _SectionTitle('Wristband'),
          const _WristbandSection(),
          const Divider(),
          const _SectionTitle('Test alerts'),
          const ListTile(subtitle: Text('Fires the full alert path (screen, vibration, wristband) without any sound.')),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              for (var i = 0; i < s.labels.length; i++)
                ActionChip(
                  avatar: Icon(SoundClass.of(s.labels[i]).icon, size: 18),
                  label: Text(SoundClass.of(s.labels[i]).name),
                  onPressed: () => state.api.simulateAlert(i),
                ),
            ],
          ),
          const Divider(),
          const _SectionTitle('Privacy'),
          const ListTile(
            leading: Icon(Icons.lock_outline),
            title: Text('Everything stays on this phone'),
            subtitle: Text(
              'Audio is held in memory for about one second, analysed, and overwritten. '
              'The app has no internet permission in release builds. The wristband only receives which sound was detected.',
            ),
          ),
        ],
      ),
    );
  }

  static String _sensitivityLabel(double v) => switch (v) {
    <= -0.75 => 'Fewest alerts',
    < -0.25 => 'Fewer alerts',
    < 0.25 => 'Balanced',
    < 0.75 => 'More alerts',
    _ => 'Most alerts',
  };
}

class _SectionTitle extends StatelessWidget {
  const _SectionTitle(this.text);
  final String text;

  @override
  Widget build(BuildContext context) => Padding(
    padding: const EdgeInsets.fromLTRB(16, 12, 16, 4),
    child: Text(
      text,
      style: Theme.of(context).textTheme.titleSmall?.copyWith(color: Theme.of(context).colorScheme.primary),
    ),
  );
}

class _WristbandSection extends StatefulWidget {
  const _WristbandSection();

  @override
  State<_WristbandSection> createState() => _WristbandSectionState();
}

class _WristbandSectionState extends State<_WristbandSection> {
  bool _scanning = false;
  List<ScannedBand>? _found;

  Future<void> _scan() async {
    final api = AppScope.of(context).api;
    final messenger = ScaffoldMessenger.of(context);
    if (!await Permissions.ensureBluetooth()) {
      messenger.showSnackBar(const SnackBar(content: Text('Bluetooth permission is needed to find the wristband.')));
      return;
    }
    setState(() => _scanning = true);
    final found = await api.scanWristbands();
    if (!mounted) return;
    setState(() {
      _scanning = false;
      _found = found;
    });
  }

  @override
  Widget build(BuildContext context) {
    final state = AppScope.of(context);
    final band = state.status.wristband;
    return Column(
      children: [
        if (band.paired)
          ListTile(
            leading: Icon(band.connected ? Icons.bluetooth_connected : Icons.bluetooth_searching),
            title: Text(band.name ?? 'SAHARA band'),
            subtitle: Text(
              [
                band.state,
                if (band.batteryPct != null) '${band.batteryPct}% battery',
                if (band.charging == true) 'charging',
              ].join(' · '),
            ),
            trailing: PopupMenuButton<String>(
              onSelected: (v) async {
                final messenger = ScaffoldMessenger.of(context);
                if (v == 'test') {
                  final ok = await state.api.testWristband();
                  messenger.showSnackBar(SnackBar(content: Text(ok ? 'Sent a test buzz' : 'Wristband not connected')));
                } else if (v == 'forget') {
                  await state.api.forgetWristband();
                  await state.refresh();
                }
              },
              itemBuilder: (_) => const [
                PopupMenuItem(value: 'test', child: Text('Test buzz')),
                PopupMenuItem(value: 'forget', child: Text('Forget')),
              ],
            ),
          ),
        ListTile(
          leading: _scanning
              ? const SizedBox(width: 24, height: 24, child: CircularProgressIndicator(strokeWidth: 2))
              : const Icon(Icons.search),
          title: Text(band.paired ? 'Pair a different wristband' : 'Find wristband'),
          onTap: _scanning ? null : _scan,
        ),
        if (_found != null && _found!.isEmpty)
          const ListTile(subtitle: Text('No SAHARA wristband found. Check that it is on and nearby.')),
        for (final b in _found ?? const <ScannedBand>[])
          ListTile(
            leading: const Icon(Icons.watch),
            title: Text(b.name),
            subtitle: Text('${b.address} · ${b.rssi} dBm'),
            onTap: () async {
              await state.api.connectWristband(b);
              setState(() => _found = null);
              await state.refresh();
            },
          ),
      ],
    );
  }
}
