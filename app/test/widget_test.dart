import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:sahara/alerts/alert_screen.dart';
import 'package:sahara/app.dart';
import 'package:sahara/app_state.dart';
import 'package:sahara/audio/sahara_service.dart';
import 'package:sahara/detection/detection_event.dart';

import 'fake_api.dart';

Future<(FakeSaharaApi, AppState)> pumpApp(WidgetTester tester, {bool modelReady = true}) async {
  final api = FakeSaharaApi(modelReady: modelReady);
  final state = AppState(api);
  await tester.pumpWidget(SaharaApp(state: state));
  await state.init();
  await tester.pumpAndSettle();
  return (api, state);
}

/// Native events hop through two async streams before the UI sees them.
Future<void> settleEvents(WidgetTester tester) async {
  for (var i = 0; i < 3; i++) {
    await tester.pump();
  }
  await tester.pump(const Duration(milliseconds: 400));
}

void main() {
  testWidgets('home lists all eight sounds', (tester) async {
    await pumpApp(tester);
    expect(find.text('Not listening'), findsOneWidget);
    await tester.scrollUntilVisible(find.text('Appliance beep'), 200);
    expect(find.text('Appliance beep'), findsOneWidget);
    expect(find.text('Smoke alarm', skipOffstage: false), findsOneWidget);
  });

  testWidgets('missing model shows a banner and disables Start', (tester) async {
    await pumpApp(tester, modelReady: false);
    expect(find.textContaining('Model not installed'), findsOneWidget);
    final button = tester.widget<FilledButton>(
      find.ancestor(of: find.text('Start listening'), matching: find.byType(FilledButton)),
    );
    expect(button.onPressed, isNull);
  });

  testWidgets('a detection opens the full-screen alert and dismiss stops it', (tester) async {
    final (api, state) = await pumpApp(tester);
    api.controller.add(
      DetectionArrived(
        DetectionEvent(
          classId: 0,
          label: 'smoke_alarm',
          confidence: 0.93,
          time: DateTime(2026, 10, 4, 9, 30),
          safetyCritical: true,
        ),
        opened: false,
      ),
    );
    await settleEvents(tester);
    expect(find.byType(AlertScreen), findsOneWidget);
    expect(find.text('SMOKE ALARM'), findsOneWidget);
    expect(state.history, hasLength(1));

    await tester.tap(find.text('Dismiss'));
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 500));
    expect(find.byType(AlertScreen), findsNothing);
    expect(api.calls, contains('dismiss'));
  });

  testWidgets('switching a sound off updates native settings', (tester) async {
    final (api, _) = await pumpApp(tester);
    await tester.tap(find.byType(Switch).first);
    await tester.pump();
    expect(api.calls, contains('settings'));
    expect(api.lastSettings['enabledClasses'], isNot(contains('smoke_alarm')));
  });

  testWidgets('live scores and state events update the home screen', (tester) async {
    final (api, state) = await pumpApp(tester);
    api.controller.add(const StateChanged(true, null));
    api.controller.add(const ScoresArrived([0.9, 0, 0, 0, 0, 0, 0, 0], 0.01));
    await settleEvents(tester);
    expect(state.status.listening, isTrue);
    expect(find.text('Listening for sounds'), findsOneWidget);
    expect(find.byType(LinearProgressIndicator), findsWidgets);
    api.controller.add(const StateChanged(false, 'Microphone unavailable'));
    await settleEvents(tester);
    expect(find.text('Microphone unavailable'), findsOneWidget);
  });
}
