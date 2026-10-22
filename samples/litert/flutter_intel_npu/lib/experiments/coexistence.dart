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

// Experiment: can several LiteRT-LM engines be
// alive at the same time in one process, each on its own backend?
//
// Each engine is its own FlutterEdgeAi.openModel model. Run:
//   flutter run -d macos -t lib/experiments/coexistence.dart \
//     --dart-define=ORDER=cpu,gpu --dart-define=RESULTS_PATH=/tmp/coex.json
//
// ignore_for_file: avoid_print

import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/widgets.dart';
import 'package:flutter_edge_ai/flutter_edge_ai.dart';
import 'package:flutter_edge_ai_litertlm/flutter_edge_ai_litertlm.dart';

import '../app/demo_config.dart';

const _order = String.fromEnvironment('ORDER', defaultValue: 'cpu,gpu');
const _maxTokens = int.fromEnvironment('MAX_TOKENS', defaultValue: 2048);
const _resultsPath = String.fromEnvironment('RESULTS_PATH');
const _prompt = 'In two sentences, what is a neural processing unit?';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  // No fallback: an engine that does not load on its backend is a failed step.
  await FlutterEdgeAi.initialize(inferenceEngines: const [LiteRtLmEngine(backendFallback: false)]);
  runApp(const Center(child: Text('coexistence experiment — see log', textDirection: TextDirection.ltr)));
  unawaited(_run());
}

Future<void> _run() async {
  final config = DemoConfig.fromEnvironment();
  final order = _order.split(',').map((s) => s.trim()).toList();
  final report = <String, Object?>{
    'model': config.modelFileName,
    'order': order,
    'max_tokens': _maxTokens,
    'os': Platform.operatingSystem,
    'steps': <Map<String, Object?>>[],
  };
  final steps = report['steps'] as List<Map<String, Object?>>;
  final models = <String, InferenceModel>{};
  // The NPU needs its own compiled bundle (NPU_MODEL_PATH); a generic one
  // does not load there without a fallback.
  String modelFor(String backend) => backend == 'npu' ? config.npuModelPath ?? config.modelPath : config.modelPath;
  Future<InferenceModel> open(String backend) async {
    final installed = await FlutterEdgeAi.installModel(modelType: ModelType.gemma4, fileType: ModelFileType.litertlm)
        .fromFile(modelFor(backend))
        .install(activate: false);
    return FlutterEdgeAi.openModel(
      installed.spec,
      maxTokens: _maxTokens,
      preferredBackend: PreferredBackend.values.byName(backend),
    );
  }

  var exitCode = 0;

  Future<void> step(String name, Future<Map<String, Object?>> Function() body) async {
    final sw = Stopwatch()..start();
    try {
      final extra = await body();
      steps.add({'step': name, 'ok': true, 'ms': sw.elapsedMilliseconds, ...extra});
      print('COEX_STEP ok   $name ${sw.elapsedMilliseconds}ms $extra');
    } catch (e, st) {
      steps.add({'step': name, 'ok': false, 'ms': sw.elapsedMilliseconds, 'error': '$e'});
      print('COEX_STEP FAIL $name: $e\n$st');
      exitCode = 1;
    }
  }

  Future<Map<String, Object?>> generate(String backend) async {
    final session = await models[backend]!.createSession(temperature: 0.8, topK: 1, topP: 0.95, randomSeed: 1);
    final buf = StringBuffer();
    var chunks = 0;
    final SessionMetrics m;
    try {
      await session.addQueryChunk(Message.text(text: _prompt, isUser: true));
      await for (final chunk in session.getResponseAsync()) {
        buf.write(chunk);
        chunks++;
      }
      m = session.getSessionMetrics();
    } finally {
      await session.close();
    }
    return {
      'active': models[backend]!.activeBackend?.name,
      'chunks': chunks,
      'ttft_ms': m.timeToFirstTokenMs,
      'tps': m.tokensPerSecond,
      'preview': buf.toString().replaceAll('\n', ' ').substring(0, buf.length.clamp(0, 80)),
    };
  }

  // 1. Create every engine, in the requested order, keeping all alive.
  for (final backend in order) {
    await step('create:$backend', () async {
      models[backend] = await open(backend);
      return {'alive': models.keys.toList()};
    });
  }

  // 2. Generate on each while all are alive, then again on the first one to
  //    prove creating the later engines did not break the earlier one.
  for (final backend in order) {
    if (models.containsKey(backend)) await step('generate:$backend', () => generate(backend));
  }
  if (models.containsKey(order.first)) {
    await step('generate-again:${order.first}', () => generate(order.first));
  }

  // 3. Tear down in reverse order, then re-create the first backend to check
  //    delete → create still works after coexistence.
  for (final backend in order.reversed) {
    final m = models.remove(backend);
    if (m != null) await step('close:$backend', () async { await m.close(); return {}; });
  }
  await step('reopen:${order.first}', () async {
    final m = await open(order.first);
    models[order.first] = m;
    try {
      return await generate(order.first);
    } finally {
      await models.remove(order.first)?.close();
    }
  });
  // A step that failed after its open left its model here.
  for (final m in models.values) {
    await m.close();
  }

  report['verdict'] = exitCode == 0 ? 'coexistence OK' : 'FAILED (see steps)';
  final json = const JsonEncoder.withIndent('  ').convert(report);
  print('COEX_DONE $json');
  if (_resultsPath.isNotEmpty) await File(_resultsPath).writeAsString(json);
  exit(exitCode);
}
