import 'package:flutter/material.dart';

import 'app.dart';
import 'app_state.dart';
import 'audio/sahara_service.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(SaharaApp(state: AppState(ChannelSaharaApi())..init()));
}
