import 'package:flutter/material.dart';

import 'app_scope.dart';

class HistoryScreen extends StatelessWidget {
  const HistoryScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final state = AppScope.of(context);
    final items = state.history;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Recent alerts'),
        actions: [
          if (items.isNotEmpty)
            IconButton(tooltip: 'Clear history', icon: const Icon(Icons.delete_outline), onPressed: state.clearHistory),
        ],
      ),
      body: items.isEmpty
          ? const Center(child: Text('No alerts yet'))
          : ListView.builder(
              itemCount: items.length,
              itemBuilder: (context, i) {
                final e = items[i];
                final s = e.sound;
                return ListTile(
                  leading: CircleAvatar(backgroundColor: s.color, foregroundColor: Colors.white, child: Icon(s.icon)),
                  title: Text(s.name),
                  subtitle: Text('${_format(e.time)} · ${(e.confidence * 100).round()}%'),
                  trailing: e.safetyCritical ? const Icon(Icons.warning_amber, color: Colors.red) : null,
                );
              },
            ),
    );
  }

  static String _format(DateTime t) {
    String two(int n) => n.toString().padLeft(2, '0');
    return '${t.year}-${two(t.month)}-${two(t.day)} ${two(t.hour)}:${two(t.minute)}:${two(t.second)}';
  }
}
