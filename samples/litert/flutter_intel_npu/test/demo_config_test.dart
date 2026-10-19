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

    test('a run that exits by itself is headless', () {
      expect(parse({'AUTORUN_EXIT': true}).headless, isTrue);
      expect(parse({}).headless, isFalse);
    });
  });
}
