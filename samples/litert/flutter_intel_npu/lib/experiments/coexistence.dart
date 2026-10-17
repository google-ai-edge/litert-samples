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
// Bypasses flutter_gemma's single-active-model policy by talking to the FFI
// client directly. Run:
//   flutter run -d macos -t lib/experiments/coexistence.dart \
//     --dart-define=ORDER=cpu,gpu --dart-define=RESULTS_PATH=/tmp/coex.json
//
// ignore_for_file: implementation_imports, avoid_print

import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/widgets.dart';
import 'package:flutter_gemma_litertlm/src/ffi/litert_lm_client.dart';

import '../app/demo_config.dart';

const _order = String.fromEnvironment('ORDER', defaultValue: 'cpu,gpu');
const _maxTokens = int.fromEnvironment('MAX_TOKENS', defaultValue: 2048);
const _resultsPath = String.fromEnvironment('RESULTS_PATH');
const _prompt = 'In two sentences, what is a neural processing unit?';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
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
  final clients = <String, LiteRtLmFfiClient>{};
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
    final c = clients[backend]!;
    await c.createConversation(topK: 1);
    final buf = StringBuffer();
    var chunks = 0;
    await for (final chunk in c.chat(_prompt)) {
      buf.write(chunk);
      chunks++;
    }
    final m = c.getSessionMetrics();
    c.closeConversation();
    return {
      'chunks': chunks,
      'ttft_ms': m.timeToFirstTokenMs,
      'tps': m.tokensPerSecond,
      'preview': buf.toString().replaceAll('\n', ' ').substring(0, buf.length.clamp(0, 80)),
    };
  }

  // 1. Create every engine, in the requested order, keeping all alive.
  for (final backend in order) {
    await step('create:$backend', () async {
      final c = LiteRtLmFfiClient();
      await c.initialize(modelPath: config.modelPath, backend: backend, maxTokens: _maxTokens);
      clients[backend] = c;
      return {'alive': clients.keys.toList()};
    });
  }

  // 2. Generate on each while all are alive, then again on the first one to
  //    prove creating the later engines did not break the earlier one.
  for (final backend in order) {
    if (clients.containsKey(backend)) await step('generate:$backend', () => generate(backend));
  }
  if (clients.containsKey(order.first)) {
    await step('generate-again:${order.first}', () => generate(order.first));
  }

  // 3. Tear down in reverse order, then re-create the first backend to check
  //    delete → create still works after coexistence.
  for (final backend in order.reversed) {
    final c = clients.remove(backend);
    if (c != null) await step('shutdown:$backend', () async { await c.shutdown(); return {}; });
  }
  await step('recreate:${order.first}', () async {
    final c = LiteRtLmFfiClient();
    await c.initialize(modelPath: config.modelPath, backend: order.first, maxTokens: _maxTokens);
    clients[order.first] = c;
    final r = await generate(order.first);
    await c.shutdown();
    clients.remove(order.first);
    return r;
  });

  report['verdict'] = exitCode == 0 ? 'coexistence OK' : 'FAILED (see steps)';
  final json = const JsonEncoder.withIndent('  ').convert(report);
  print('COEX_DONE $json');
  if (_resultsPath.isNotEmpty) await File(_resultsPath).writeAsString(json);
  exit(exitCode);
}
