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

import 'package:flutter/foundation.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/repositories/diagnostics_repository.dart';
import 'package:litert_edge_demos/domain/models/camera_side_event.dart';
import 'package:litert_edge_demos/domain/models/diagnostics_snapshot.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/domain/models/route_decision.dart';
import 'package:litert_edge_demos/domain/models/visual_memory.dart';
import 'package:litert_edge_demos/ui/core/debug_overlay.dart';

/// The overlay's visual-memory lines: its size, the last frame embedding,
/// the last question's best cosine and the failures; and a memory
/// question's search in the camera block.
void main() {
  List<String> memoryLines(DiagnosticsSnapshot s) =>
      debugOverlayLines(s).where((l) => l.contains('memory')).toList();

  test('none while the camera demo is closed or no memory is wired', () {
    expect(memoryLines(const DiagnosticsSnapshot()), isEmpty);
    expect(
      memoryLines(
        const DiagnosticsSnapshot(memory: VisualMemoryStatus(frames: 3)),
      ),
      isEmpty,
    );
  });

  test('frames, span, bytes, the last embed, the best cosine, the failures; '
      'the reason when it is off', () {
    const status = VisualMemoryStatus(
      active: true,
      frames: 37,
      bytes: 4 << 20,
      span: Duration(minutes: 2),
      lastEmbed: Duration(milliseconds: 260),
      failures: 2,
      failuresInARow: 1,
      lastBestCosine: 0.712,
    );
    expect(memoryLines(const DiagnosticsSnapshot(memory: status)), [
      'memory 37 frames · 120 s · 4.00 MB · embed 260 ms · best 0.71 · '
          'failed 2 (1 in a row)',
    ]);
    final off = status.copyWith(
      unavailable: (reason: 'EmbeddingGemma is still loading'),
    );
    expect(
      memoryLines(DiagnosticsSnapshot(memory: off)).last,
      'memory off: EmbeddingGemma is still loading',
    );
    expect(
      memoryLines(
        DiagnosticsSnapshot(memory: status.copyWith(preparing: true)),
      ),
      ['memory preparing (image encoder)'],
    );
    expect(
      memoryLines(DiagnosticsSnapshot(memory: status.copyWith(enabled: false))),
      ['memory switched off (camera settings)'],
    );
  });

  test('a memory question\'s search in the camera block', () {
    final s = DiagnosticsSnapshot(
      lastCameraTurn: CameraTurnMetrics(
        route: const MemoryRoute('yellow mug', 'memory:where-was'),
        routeTime: const Duration(microseconds: 120),
        memory: MemoryRecalled(
          recall: MemoryHit(
            entry: MemoryEntry(
              id: 1,
              frameId: 44,
              atMicros: 0,
              png: Uint8List(4),
              width: 320,
              height: 180,
              counts: const {},
              vector: Float32List(3),
              grey: Uint8List(1024),
            ),
            cosine: 0.712,
            rival: 0.65,
            age: const Duration(seconds: 12),
          ),
          query: 'yellow mug',
          elapsed: const Duration(milliseconds: 38),
        ),
      ),
    );
    final lines = debugOverlayLines(s);
    expect(lines, contains(startsWith('camq memory · rule memory:where-was')));
    expect(
      lines,
      contains(
        'camq memory "yellow mug" 38 ms · hit 0.71 (rival 0.65) · #44 12 s '
        'ago',
      ),
    );
  });

  test('the diagnostics repository follows the memory\'s status', () async {
    final models = ValueNotifier<Map<ModelId, ModelState>>(const {});
    final memory = ValueNotifier(const VisualMemoryStatus());
    final diagnostics = DiagnosticsRepository(
      models: models,
      memory: memory,
      minInterval: Duration.zero,
      sampleRss: false,
    );
    memory.value = const VisualMemoryStatus(active: true, frames: 5);
    expect(diagnostics.latest.memory?.frames, 5);
    diagnostics.dispose();
    memory.dispose();
    models.dispose();
  });
}
