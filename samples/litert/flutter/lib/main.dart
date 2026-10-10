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

import 'dart:io' show Platform;

import 'package:flutter/material.dart';

import 'app.dart';
import 'config/dependencies.dart';
import 'config/licenses.dart';
import 'selftest/self_test_mode.dart';
import 'selftest/self_test_options.dart';

/// [args]: the Linux runner passes argv after the binary name
/// (`my_application.cc`, `fl_dart_project_set_dart_entrypoint_arguments`);
/// macOS passes `NSProcessInfo.arguments` by default
/// (`FlutterDartProject.dartEntrypointArguments`). `--selftest` (or
/// `SELFTEST=1`) runs the headless self-test instead of the app.
Future<void> main(List<String> args) async {
  if (SelfTestOptions.parse(args, Platform.environment) case final selfTest?) {
    await runSelfTestMode(selfTest);
  }
  WidgetsFlutterBinding.ensureInitialized();
  registerModelLicenses();
  final AppDependencies dependencies;
  try {
    dependencies = await AppDependencies.create();
  } catch (e, st) {
    debugPrint('Startup failed: $e\n$st');
    runApp(_StartupErrorApp(error: e));
    return;
  }
  runApp(App(dependencies: dependencies));
}

/// Shown when `AppDependencies.create` throws (a bad define such as
/// `VOICE_GATE_DBFS`, or the inference engine failing to initialize). No
/// fallback.
class _StartupErrorApp extends StatelessWidget {
  const _StartupErrorApp({required this.error});

  final Object error;

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        body: Center(
          child: Padding(
            padding: const EdgeInsets.all(24),
            child: Text('Startup failed: $error'),
          ),
        ),
      ),
    );
  }
}
