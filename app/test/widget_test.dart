import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:sahara/alerts/alert_screen.dart';
import 'package:sahara/app.dart';
import 'package:sahara/app_state.dart';
import 'package:sahara/audio/sahara_service.dart';
import 'package:sahara/detection/detection_event.dart';
import 'package:sahara/detection/sound_class.dart';
import 'package:sahara/personalization/custom_sounds_screen.dart';
import 'package:sahara/personalization/home_calibration_screen.dart';

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

  testWidgets('custom sound enrolment needs 5 samples and a name, then lists the sound', (tester) async {
    final (api, _) = await pumpApp(tester);
    tester
        .state<NavigatorState>(find.byType(Navigator).first)
        .push(MaterialPageRoute<void>(builder: (_) => const CustomSoundsScreen()));
    await tester.pumpAndSettle();
    expect(find.text('No custom sounds yet'), findsOneWidget);
    await tester.tap(find.text('Add a sound'));
    await tester.pumpAndSettle();

    await tester.enterText(find.byType(TextField), 'Pressure cooker');
    FilledButton recordButton() => tester.widget<FilledButton>(find.byType(FilledButton));
    OutlinedButton saveButton() => tester.widget<OutlinedButton>(find.widgetWithText(OutlinedButton, 'Save sound'));
    for (var i = 0; i < 5; i++) {
      expect(saveButton().onPressed, isNull, reason: 'save stays off below 5 samples');
      await tester.tap(find.text('Record sample'));
      await tester.pumpAndSettle();
    }
    expect(recordButton().onPressed, isNotNull);
    expect(find.text('5 of 5-10 samples'), findsOneWidget);
    await tester.tap(find.text('Save sound'));
    await tester.pumpAndSettle();

    expect(find.text('Pressure cooker'), findsOneWidget);
    expect(api.calls.where((c) => c == 'record'), hasLength(5));
    expect(api.calls, isNot(contains('cancelEnrolment')));
  });

  testWidgets('home calibration takes 3 recordings of a sound, then saves', (tester) async {
    final (api, _) = await pumpApp(tester);
    tester
        .state<NavigatorState>(find.byType(Navigator).first)
        .push(MaterialPageRoute<void>(builder: (_) => const HomeCalibrationScreen()));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Doorbell'));
    await tester.pumpAndSettle();
    FilledButton saveButton() => tester.widget<FilledButton>(find.widgetWithText(FilledButton, 'Save'));
    for (var i = 0; i < 3; i++) {
      expect(saveButton().onPressed, isNull, reason: 'save stays off below 3 recordings');
      await tester.tap(find.text('Record Doorbell'));
      await tester.pumpAndSettle();
    }
    await tester.tap(find.text('Record home sound'));
    await tester.pumpAndSettle();
    expect(find.text('3 of 3 recordings'), findsOneWidget);
    await tester.tap(find.widgetWithText(FilledButton, 'Save'));
    await tester.pumpAndSettle();
    expect(api.calls.where((c) => c == 'calibrate:doorbell'), hasLength(3));
    expect(api.calls, contains('calibrate:doorbell:background'));
    expect(api.calls, contains('finishCalibration'));
    expect(api.calls, isNot(contains('cancelCalibration')));
  });

  test('custom alert labels get their own display name', () {
    final s = SoundClass.of('custom:Pressure cooker');
    expect(s.name, 'Pressure cooker');
    expect(s.safetyCritical, isFalse);
  });

  testWidgets('a name call says who is calling, or Someone', (tester) async {
    final (api, _) = await pumpApp(tester);
    api.controller.add(
      DetectionArrived(
        DetectionEvent(
          classId: 0x20,
          label: 'name_call',
          confidence: 0.8,
          time: DateTime(2026, 10, 16),
          safetyCritical: false,
          detail: 'Raj',
        ),
        opened: false,
      ),
    );
    await settleEvents(tester);
    expect(find.text('RAJ IS CALLING YOU'), findsOneWidget);
    await tester.tap(find.text('Dismiss'));
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 500));

    api.controller.add(
      DetectionArrived(
        DetectionEvent(
          classId: 0x20,
          label: 'name_call',
          confidence: 0.8,
          time: DateTime(2026, 10, 16),
          safetyCritical: false,
        ),
        opened: false,
      ),
    );
    await settleEvents(tester);
    expect(find.text('SOMEONE IS CALLING YOU'), findsOneWidget);
  });

  testWidgets('other sounds are notices, not full-screen alerts', (tester) async {
    final (api, state) = await pumpApp(tester);
    api.controller.add(
      DetectionArrived(
        DetectionEvent(
          classId: 0x22,
          label: 'other_sound',
          confidence: 0.5,
          time: DateTime(2026, 10, 16),
          safetyCritical: false,
          detail: 'Cat',
        ),
        opened: false,
      ),
    );
    await settleEvents(tester);
    expect(find.byType(AlertScreen), findsNothing);
    expect(state.history.single.title, 'Cat');
    await tester.scrollUntilVisible(find.text('Last heard: Cat'), 200);
    expect(find.text('Last heard: Cat'), findsOneWidget);
  });

  testWidgets('enrolling a caller adds them to the list', (tester) async {
    // permission_handler: report every requested permission as granted (1).
    tester.binding.defaultBinaryMessenger.setMockMethodCallHandler(
      const MethodChannel('flutter.baseflow.com/permissions/methods'),
      (call) async => call.method == 'requestPermissions' ? {for (final p in (call.arguments as List)) p: 1} : 1,
    );
    final (api, state) = await pumpApp(tester);
    await tester.scrollUntilVisible(find.text('When someone calls Aryan'), 200);
    await tester.tap(find.text('When someone calls Aryan'));
    await tester.pumpAndSettle();
    await tester.enterText(find.byType(TextField), 'Prachi');
    await tester.tap(find.text('Record their voice'));
    await tester.pumpAndSettle();
    expect(api.calls, contains('enroll:Prachi'));
    expect(state.status.callers.single.name, 'Prachi');
    expect(find.text('1 recording'), findsOneWidget);
  });
}
