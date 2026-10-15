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

import 'package:flutter/material.dart';

import '../core/engines/engine_pool.dart';
import '../core/power/power_source.dart';
import 'act_navigator.dart';
import 'demo_config.dart';

/// Root widget. Owns the app-lifetime services (config, engine pool, power
/// source) and hands them to the act navigator.
class DemoApp extends StatefulWidget {
  const DemoApp({super.key, required this.config});

  final DemoConfig config;

  @override
  State<DemoApp> createState() => _DemoAppState();
}

class _DemoAppState extends State<DemoApp> {
  late final EnginePool _pool = EnginePool(
    policy: widget.config.policy,
    modelPath: widget.config.modelPath,
  );
  late final PowerSource _power = powerSourceFor(file: widget.config.powerFile)..start();

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Intel NPU Demo',
      theme: ThemeData(colorSchemeSeed: Colors.indigo, useMaterial3: true),
      darkTheme: ThemeData(
        colorSchemeSeed: Colors.indigo,
        brightness: Brightness.dark,
        useMaterial3: true,
      ),
      home: ActNavigator(config: widget.config, pool: _pool, power: _power),
    );
  }

  @override
  void dispose() {
    _power.dispose();
    _pool.releaseAll();
    super.dispose();
  }
}
