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
import 'package:intel_npu_demo/core/narration/scene_narrator.dart';
import 'package:intel_npu_demo/core/scene/scene_events.dart';

void main() {
  test('a phase reports only its own turns on its own backend', () {
    final events = SceneEventLog();
    final n = SceneNarrator(events: events, modelPath: 'unused.litertlm');
    NarrationTurn Function(int) turn(PreferredBackend b, double tps) => (seq) => NarrationTurn(
          seq: seq,
          at: seq.toDouble(),
          backend: b,
          text: '$b $seq',
          eventsIn: 1,
          ttft: const Duration(milliseconds: 100),
          tokensPerSecond: tps,
          outputTokens: 20,
          inputTokens: 200,
        );

    n.backend = PreferredBackend.gpu;
    n.markPhase();
    for (var i = 0; i < 6; i++) {
      n.debugRecordTurn(turn(PreferredBackend.gpu, 83));
    }
    expect(n.toJson(phase: true)['avg_tps'], 83.0);

    // The committed results once reported a "cpu" phase at 63 tok/s: six of
    // its eleven turns were the GPU's.
    n.backend = PreferredBackend.cpu;
    n.markPhase();
    for (var i = 0; i < 5; i++) {
      n.debugRecordTurn(turn(PreferredBackend.cpu, 39));
    }
    final phase = n.toJson(phase: true);
    expect(phase['turns'], 5);
    expect(phase['avg_tps'], 39.0);
    expect((phase['transcript']! as List).every((l) => (l as String).startsWith('[CPU]')), isTrue);
    expect(n.toJson()['turns'], 11, reason: 'the whole-run summary keeps every turn');

    n.dispose();
    events.dispose();
  });
}
