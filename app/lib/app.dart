import 'dart:async';

import 'package:flutter/material.dart';

import 'alerts/alert_screen.dart';
import 'app_state.dart';
import 'detection/detection_event.dart';
import 'ui/app_scope.dart';
import 'ui/home_screen.dart';

class SaharaApp extends StatefulWidget {
  const SaharaApp({super.key, required this.state});
  final AppState state;

  @override
  State<SaharaApp> createState() => _SaharaAppState();
}

class _SaharaAppState extends State<SaharaApp> with WidgetsBindingObserver {
  final _navigator = GlobalKey<NavigatorState>();
  StreamSubscription<DetectionEvent>? _alerts;
  bool _alertOpen = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _alerts = widget.state.alerts.listen(_showAlert);
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState s) {
    if (s == AppLifecycleState.resumed) widget.state.refresh();
  }

  void _showAlert(DetectionEvent e) {
    final nav = _navigator.currentState;
    if (nav == null) return;
    if (_alertOpen) nav.pop();
    _alertOpen = true;
    nav
        .push(
          MaterialPageRoute(
            fullscreenDialog: true,
            builder: (_) => AlertScreen(
              event: e,
              onDismiss: () {
                widget.state.api.dismissAlert();
                nav.pop();
              },
            ),
          ),
        )
        .then((_) => _alertOpen = false);
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _alerts?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AppScope(
      state: widget.state,
      child: MaterialApp(
        title: 'SAHARA',
        navigatorKey: _navigator,
        theme: ThemeData(colorSchemeSeed: const Color(0xFFE65100), useMaterial3: true),
        darkTheme: ThemeData(colorSchemeSeed: const Color(0xFFE65100), brightness: Brightness.dark, useMaterial3: true),
        home: const HomeScreen(),
      ),
    );
  }
}
