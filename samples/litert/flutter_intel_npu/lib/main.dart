// Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import 'dart:io';

import 'package:flutter/widgets.dart';
import 'package:flutter_edge_ai/flutter_edge_ai.dart';
import 'package:flutter_edge_ai_litertlm/flutter_edge_ai_litertlm.dart';

import 'app/demo_app.dart';
import 'app/demo_config.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Core registers no engine on its own; .litertlm is the only desktop engine.
  await FlutterEdgeAi.initialize(
    inferenceEngines: const [LiteRtLmEngine()],
  );
  FlutterEdgeAi.logLevel = EdgeAiLogLevel.info;

  final config = DemoConfig.fromEnvironment();
  for (final e in config.errors) {
    stderr.writeln('CONFIG_ERROR $e');
  }
  // Nobody is watching a headless run: a bad config must fail it, not leave
  // the app waiting on a screen.
  if (config.errors.isNotEmpty && config.headless) exit(2);
  runApp(DemoApp(config: config));
}
