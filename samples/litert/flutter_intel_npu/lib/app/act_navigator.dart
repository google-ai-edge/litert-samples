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
import '../features/act1_inference/act1_screen.dart';
import '../features/act2_long_context/act2_screen.dart';
import '../features/act3_scene/act3_screen.dart';
import '../features/act3_scene/runner/runner_screen.dart';
import '../features/act4_agent/act4_screen.dart';
import 'demo_config.dart';

/// Four acts, one running app. No routing package: the presenter goes
/// home → act → home. Act 1's engines stay resident across acts (EnginePool);
/// Acts 2 and 4 release their large models when they close.
class ActNavigator extends StatelessWidget {
  const ActNavigator({super.key, required this.config, required this.pool, required this.power});

  final DemoConfig config;
  final EnginePool pool;
  final PowerSource power;

  static const acts = [
    (n: 1, title: 'Inference', subtitle: 'Same model, same prompt: CPU · GPU · NPU'),
    (n: 2, title: 'Long context', subtitle: 'A 45-minute meeting in one piece'),
    (n: 3, title: "The NPU's own argument", subtitle: 'Heavy 3D on the GPU, live commentary on the NPU'),
    (n: 4, title: 'Meeting agent', subtitle: 'Gemma 4 does the follow-up of a meeting with tools'),
  ];

  Widget screenFor(int act) => switch (act) {
        1 => Act1Screen(config: config, pool: pool, power: power),
        2 => Act2Screen(config: config, pool: pool),
        3 => config.act3.isRunner
            ? RunnerScreen(config: config, power: power)
            : Act3Screen(config: config, power: power),
        4 => Act4Screen(config: config, pool: pool),
        _ => _NoSuchAct(act: act),
      };

  @override
  Widget build(BuildContext context) {
    if (config.errors.isNotEmpty) return _ConfigErrors(errors: config.errors);
    if (config.act != 0) return screenFor(config.act);
    return Scaffold(
      appBar: AppBar(title: const Text('Intel NPU demo')),
      body: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 640),
          child: ListView(
            shrinkWrap: true,
            padding: const EdgeInsets.all(24),
            children: [
              for (final a in acts)
                Card(
                  child: ListTile(
                    leading: CircleAvatar(child: Text('${a.n}')),
                    title: Text(a.title),
                    subtitle: Text(a.subtitle),
                    trailing: const Icon(Icons.chevron_right),
                    onTap: () => Navigator.of(context).push(
                      MaterialPageRoute<void>(builder: (_) => screenFor(a.n)),
                    ),
                  ),
                ),
              const SizedBox(height: 16),
              Text(
                config.modelPath.isEmpty
                    ? 'MODEL_PATH is not set: the scene runs, the model rows cannot (see README).'
                    : '${config.modelFileName} · policy ${pool.policy.name} · act 2: ${config.act2.modelPathOr(config.modelPath).split(RegExp(r'[\\/]')).last}',
                style: Theme.of(context).textTheme.bodySmall,
                textAlign: TextAlign.center,
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _NoSuchAct extends StatelessWidget {
  const _NoSuchAct({required this.act});
  final int act;

  @override
  Widget build(BuildContext context) => Scaffold(
        appBar: AppBar(title: Text('Act $act')),
        body: const Center(child: Text('There are four acts: ACT=1, 2, 3 or 4 (0 = home).')),
      );
}

/// Shown instead of the acts when the configuration has problems: running
/// on defaults the user did not ask for would measure the wrong thing.
class _ConfigErrors extends StatelessWidget {
  const _ConfigErrors({required this.errors});
  final List<String> errors;

  @override
  Widget build(BuildContext context) => Scaffold(
        appBar: AppBar(title: const Text('Configuration problems')),
        body: ListView(
          padding: const EdgeInsets.all(24),
          children: [
            for (final e in errors) ListTile(leading: const Icon(Icons.error_outline), title: Text(e)),
            const SizedBox(height: 16),
            const Text('Fix the dart-defines or the DEMO_CONFIG file and start again (see README).'),
          ],
        ),
      );
}
