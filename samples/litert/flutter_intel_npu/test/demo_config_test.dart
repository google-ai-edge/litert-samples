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

import 'package:flutter_edge_ai/flutter_edge_ai.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:intel_npu_demo/app/demo_config.dart';
import 'package:intel_npu_demo/core/power/power_source.dart';

DemoConfig parse(Map<String, Object?> o, [String? readError]) => DemoConfig.fromOverrides((o, readError));

void main() {
  group('DemoConfig', () {
    test('an empty config is valid and leaves MODEL_PATH unset', () {
      final c = parse({});
      expect(c.errors, isEmpty);
      expect(c.modelPath, '');
      expect(c.act3.targetFps, 60);
    });

    test('accepts strings for numbers and booleans', () {
      final c = parse({'ACT': '3', 'AUTORUN': 'true', 'ACT3_RENDER_SCALE': '0.75', 'MAX_TOKENS': 2048.0});
      expect(c.errors, isEmpty);
      expect(c.act, 3);
      expect(c.autorun, isTrue);
      expect(c.act3.renderScale, 0.75);
      expect(c.maxTokens, 2048);
    });

    test('reports bad values instead of ignoring or throwing', () {
      final c = parse({
        'ACT': 'three',
        'AUTORUN': 'yes',
        'ACT3_NARRATOR_PLAN': 'gpu,cpu:30,tpu:5',
        'POWER_MODE': 'Watts',
        'POLICY': 'lazy',
        'BACKENDS': 'cpu,xpu',
        'ACT3_TIME_OF_DAY': 'noon',
        'PROMPT_FILE': '/nonexistent/prompt.txt',
        'MODEL_PATH': '/nonexistent/model.litertlm',
      });
      final keys = c.errors.map((e) => e.split(':').first).toList();
      expect(
        keys,
        containsAll(['ACT', 'AUTORUN', 'ACT3_NARRATOR_PLAN', 'POWER_MODE', 'POLICY', 'BACKENDS', 'ACT3_TIME_OF_DAY', 'PROMPT_FILE', 'MODEL_PATH']),
      );
      // The good parts survive.
      expect(c.act3.narratorPlan, [('cpu', 30)]);
      expect(c.backends, [PreferredBackend.cpu, PreferredBackend.cpu]);
      expect(c.powerMode, PowerMode.auto);
    });

    test('an unreadable DEMO_CONFIG is an error', () {
      final c = parse({}, 'DEMO_CONFIG: cannot read /x.json');
      expect(c.errors, ['DEMO_CONFIG: cannot read /x.json']);
    });

    test('Act 4: the model menu and run settings', () {
      final model = _anyFile('m.litertlm');
      final c = parse({
        'ACT4_MODELS': [
          {'name': '26B NPU', 'path': model, 'backend': 'npu', 'maxTokens': 32768, 'think': false},
          {'path': model, 'backend': 'GPU', 'maxTokens': '16384', 'think': 'true'},
        ],
        'ACT4_MODEL': 1,
        'ACT4_SAMPLER': 'seeded',
        'ACT4_SEED': '3',
      });
      expect(c.errors, isEmpty);
      final a = c.act4;
      expect(a.models.map((m) => (m.name, m.backend, m.maxTokens, m.think)), [
        ('26B NPU', PreferredBackend.npu, 32768, false),
        ('m.litertlm', PreferredBackend.gpu, 16384, true),
      ]);
      expect((a.modelIndex, a.think, a.greedy, a.temp, a.seed, a.maxCalls), (1, null, false, 0.3, 3, 40));
      expect((a.maxOutputTokensFor(think: true), a.maxOutputTokensFor(think: false)), (6144, 4096));
    });

    test('Act 4: the menu can come as a dart-define string', () {
      final c = parse({'ACT4_MODELS': '[{"path": "/nonexistent/m.litertlm", "backend": "tpu"}]', 'ACT4_MODEL': 2});
      expect(c.errors, [
        'ACT4_MODELS[0].backend: expected cpu|gpu|npu, got "tpu"',
        'ACT4_MODELS[0].path: no file at /nonexistent/m.litertlm',
        'ACT4_MODEL: expected 0..0 (an index into ACT4_MODELS), got 2',
      ]);
    });

    test('Act 4: bad values are reported', () {
      final c = parse({'ACT4_MODELS': '[{', 'ACT4_THINK': 'maybe', 'ACT4_SAMPLER': 'hot', 'ACT4_AUTORUN': true});
      expect(c.errors.map((e) => e.split(':').first).toSet(), {'ACT4_MODELS', 'ACT4_THINK', 'ACT4_SAMPLER', 'ACT4_AUTORUN'});
      expect(c.act4.greedy, isTrue);
    });

    test('Act 4: meeting, workspace and limits', () {
      final file = _anyFile('meeting.txt');
      final ok = parse({
        'ACT4_MODELS': [
          {'path': file, 'backend': 'cpu'},
        ],
        'ACT4_AUTORUN': true,
        'ACT4_MEETING': file,
        'ACT4_WORKSPACE': file,
      });
      expect(ok.errors, isEmpty);
      expect((ok.act4.meetingPath, ok.act4.workspacePath, ok.act4.autorun), (file, file, true));
      final bad = parse({
        'ACT4_MODELS': [
          {'path': file, 'max_tokens': 4096, 'maxTokens': 0},
        ],
        'ACT4_MEETING': '/nonexistent/m.txt',
        'ACT4_WORKSPACE': '/nonexistent/w.json',
        'ACT4_MAX_CALLS': 0,
        'ACT4_MAX_OUTPUT_TOKENS': -1,
        'ACT4_TEMP': 0,
      });
      expect(bad.errors.map((e) => e.split(':').first).toSet(), {
        'ACT4_MODELS[0].max_tokens',
        'ACT4_MODELS[0].backend',
        'ACT4_MODELS[0].maxTokens',
        'ACT4_MEETING',
        'ACT4_WORKSPACE',
        'ACT4_MAX_CALLS',
        'ACT4_MAX_OUTPUT_TOKENS',
        'ACT4_TEMP',
      });
    });

    test('a run that exits by itself is headless', () {
      expect(parse({'AUTORUN_EXIT': true}).headless, isTrue);
      expect(parse({}).headless, isFalse);
    });
  });
}

/// A real file for a config key that must point at one; deleted with the
/// temporary folder.
String _anyFile(String name) {
  final dir = Directory.systemTemp.createTempSync('act4_cfg');
  addTearDown(() => dir.deleteSync(recursive: true));
  return (File('${dir.path}${Platform.pathSeparator}$name')..writeAsStringSync('x')).path;
}
