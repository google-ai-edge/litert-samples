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
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/repositories/diagnostics_repository.dart';
import 'package:litert_edge_demos/domain/models/assistant_event.dart';
import 'package:litert_edge_demos/domain/models/camera_side_event.dart';
import 'package:litert_edge_demos/domain/models/diagnostics_snapshot.dart';
import 'package:litert_edge_demos/domain/models/live_state.dart';
import 'package:litert_edge_demos/domain/models/llm_image.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/domain/models/route_decision.dart';
import 'package:litert_edge_demos/domain/models/voice.dart';
import 'package:litert_edge_demos/ui/core/debug_overlay.dart';

GenerationMetrics _metrics(int chunks) => GenerationMetrics(
  timeToFirstToken: const Duration(milliseconds: 350),
  chunks: chunks,
  tokensPerSecond: 21.5,
  tokensPerSecondSource: TokenRateSource.chunks,
  total: const Duration(seconds: 2),
  stopped: false,
);

void main() {
  // testWidgets for its fake clock: pump(duration) fires timers.
  testWidgets('coalesces bursts into one snapshot per interval', (
    tester,
  ) async {
    final models = ValueNotifier<Map<ModelId, ModelState>>(const {});
    final repo = DiagnosticsRepository(models: models, sampleRss: false);
    var updates = 0;
    repo.snapshot.addListener(() => updates++);

    await tester.pump(Duration.zero); // the initial snapshot
    expect(updates, 1);

    for (var i = 1; i <= 10; i++) {
      repo.recordGeneration(_metrics(i));
    }
    await tester.pump(const Duration(milliseconds: 100));
    expect(updates, 1, reason: 'held back within the 250 ms window');

    await tester.pump(const Duration(milliseconds: 200));
    expect(updates, 2);
    expect(repo.snapshot.value.lastGeneration?.chunks, 10);

    repo.dispose();
    models.dispose();
  });

  testWidgets('tracks loaded models from model states', (tester) async {
    final models = ValueNotifier<Map<ModelId, ModelState>>(const {
      ModelId.chat: ModelLoading(),
    });
    final repo = DiagnosticsRepository(models: models, sampleRss: false);
    await tester.pump(Duration.zero);
    expect(repo.snapshot.value.models, isEmpty);

    models.value = const {
      ModelId.chat: ModelReady(
        LoadedModelInfo(
          modelId: 'gemma-4-E2B-it',
          backend: 'gpu',
          loadTime: Duration(milliseconds: 4200),
          warmUpTime: Duration(milliseconds: 800),
        ),
      ),
    };
    repo.recordGeneration(_metrics(42));
    await tester.pump(const Duration(milliseconds: 300));

    final lines = debugOverlayLines(repo.snapshot.value);
    expect(lines.first, 'gemma-4-E2B-it · GPU · load 4200 ms · warm 800 ms');
    expect(lines[1], 'Whisper base (STT): not loaded');
    expect(lines[2], 'Inflect-nano-v2 (TTS): not loaded');
    expect(lines[3], 'YOLO26n detector: not loaded');
    expect(lines[4], 'moonshine-tiny (STT, Live camera): not loaded');
    expect(lines[5], 'TTFT 350 ms · 21.5 tok/s (chunks) · chunks 42');
    expect(lines[6], 'STT active: none (switching…)');
    expect(lines[7], 'voice idle · STT – · first audio –');
    expect(lines.last, startsWith('RSS'));

    repo.dispose();
    models.dispose();
  });

  testWidgets('shows the live detector: label, fps and p50 pre/run/post '
      'while running, "paused (Gemma)" while paused', (tester) async {
    final models = ValueNotifier<Map<ModelId, ModelState>>(const {
      ModelId.yolo26n: ModelReady(
        LoadedModelInfo(
          modelId: 'yolo26n_fp16_rawhead',
          backend: 'gpu',
          loadTime: Duration(milliseconds: 350),
          warmUpTime: Duration(milliseconds: 5),
          detail: 'GPU fp32 full',
        ),
      ),
    });
    final liveState = ValueNotifier<LiveState>(
      const LiveRunning('fixture (35 images)'),
    );
    final liveStats = ValueNotifier(
      const LiveStats(
        fps: 15.04,
        processed: 100,
        preMs: 1.31,
        runMs: 4.22,
        postMs: 0.86,
        latencyMs: 7.9,
        droppedRate: 12,
      ),
    );
    final repo = DiagnosticsRepository(
      models: models,
      liveState: liveState,
      liveStats: liveStats,
      sampleRss: false,
    );
    await tester.pump(const Duration(milliseconds: 300));

    var lines = debugOverlayLines(repo.snapshot.value);
    expect(
      lines,
      contains(
        'yolo26n_fp16_rawhead · GPU fp32 full · load 350 ms · warm 5 ms',
      ),
    );
    expect(
      lines,
      contains(
        'det GPU fp32 full · 15.0 fps · p50 pre/run/post 1.3/4.2/0.9 ms',
      ),
    );
    expect(
      lines,
      contains(
        'det src fixture (35 images) · lat 7.9 ms · drop busy 0 rate 12',
      ),
    );
    // A network camera's own rate and size (the frames before the gate).
    liveState.value = const LiveRunning('Network camera · 192.168.1.23:8080');
    liveStats.value = const LiveStats(
      fps: 15,
      sourceFps: 29.7,
      sourceWidth: 1280,
      sourceHeight: 720,
      latencyMs: 9.1,
    );
    await tester.pump(const Duration(milliseconds: 300));
    expect(
      debugOverlayLines(repo.snapshot.value),
      contains(
        'det src Network camera · 192.168.1.23:8080 1280×720 @ 29.7 fps · '
        'lat 9.1 ms · drop busy 0 rate 0',
      ),
    );
    liveState.value = const LiveRunning('fixture (35 images)');

    final since = DateTime(2026, 10, 2, 12);
    liveState.value = LivePaused(
      source: 'fixture (35 images)',
      reason: 'Gemma',
      since: since,
    );
    await tester.pump(const Duration(milliseconds: 300));
    lines = debugOverlayLines(
      repo.snapshot.value,
      now: since.add(const Duration(milliseconds: 2400)),
    );
    expect(lines, contains('det paused (Gemma) 2.4 s'));
    expect(lines.where((l) => l.contains('fps')), isEmpty);

    liveState.value = const LiveStopped();
    await tester.pump(const Duration(milliseconds: 300));
    expect(
      debugOverlayLines(repo.snapshot.value).where((l) => l.startsWith('det')),
      isEmpty,
    );

    repo.dispose();
    liveState.dispose();
    liveStats.dispose();
    models.dispose();
  });

  testWidgets('the voice line shows a mic that is still opening', (
    tester,
  ) async {
    final models = ValueNotifier<Map<ModelId, ModelState>>(const {});
    final repo = DiagnosticsRepository(models: models, sampleRss: false);
    repo.recordVoicePhase(TurnPhase.openingMic);
    await tester.pump(const Duration(milliseconds: 300));

    expect(
      debugOverlayLines(repo.snapshot.value),
      contains('voice openingMic · STT – · first audio –'),
    );

    repo.dispose();
    models.dispose();
  });

  testWidgets('voice lines: phase, STT, first audio, TTS per clause, '
      'barge-in', (tester) async {
    final models = ValueNotifier<Map<ModelId, ModelState>>(const {});
    final repo = DiagnosticsRepository(models: models, sampleRss: false);
    repo
      ..recordVoicePhase(TurnPhase.speaking)
      ..recordVoiceTurn(
        const VoiceTurnMetrics(
          typed: false,
          stt: Duration(milliseconds: 412),
          firstAudio: Duration(milliseconds: 1380),
          sampleRate: 24000,
          ttsClauses: [Duration(milliseconds: 38), Duration(milliseconds: 41)],
        ),
      )
      ..recordBargeIn(
        const BargeInMetrics(
          wasPlaying: true,
          silenced: Duration(milliseconds: 1),
          interruptDone: Duration(milliseconds: 240),
        ),
      );
    await tester.pump(const Duration(milliseconds: 300));

    final lines = debugOverlayLines(repo.snapshot.value);
    expect(
      lines,
      contains('voice speaking · STT 412 ms · first audio 1380 ms'),
    );
    expect(lines, contains('TTS/clause 38/41 ms (2)'));
    expect(lines, contains('barge-in silence 1 ms · drain 240 ms'));

    repo.dispose();
    models.dispose();
  });

  testWidgets('llm turns count the chat\'s generations; the camq line shows '
      'route, rule, router and snapshot times', (tester) async {
    final models = ValueNotifier<Map<ModelId, ModelState>>(const {});
    final busy = ValueNotifier(false);
    final repo = DiagnosticsRepository(
      models: models,
      llmBusy: busy,
      sampleRss: false,
    );
    busy.value = true;
    busy.value = false;
    busy.value = true;
    busy.value = true; // no rising edge
    busy.value = false;
    repo.recordCameraTurn(
      const CameraTurnMetrics(
        route: FastRoute(FastIntent.count, 'count', cls: 15),
        routeTime: Duration(microseconds: 120),
        snapshotLatency: Duration(milliseconds: 14),
        basis: 'cat ×2 · remote',
      ),
    );
    await tester.pump(const Duration(milliseconds: 300));

    expect(repo.latest.llmTurns, 2);
    expect(
      debugOverlayLines(repo.snapshot.value),
      contains(
        'camq fast/count · rule count · route 0.12 ms · snap 14 ms · '
        'llm turns 2',
      ),
    );
    repo.dispose();
    busy.dispose();
    models.dispose();
  });

  test('overlay: the attachment, sent / re-sent / kept, image tokens, '
      'context use and a budget reset', () {
    final image = LlmImage(
      png: Uint8List(1656000),
      width: 1024,
      height: 768,
      sourceWidth: 4032,
      sourceHeight: 3024,
      sourceBytes: 2400000,
      normalizeTime: const Duration(milliseconds: 61),
    );
    GenerationMetrics turn({
      bool sent = true,
      ImageLoss? resent,
      bool reset = false,
    }) => GenerationMetrics(
      timeToFirstToken: const Duration(milliseconds: 980),
      chunks: 12,
      tokensPerSecond: 30,
      tokensPerSecondSource: TokenRateSource.native,
      total: const Duration(seconds: 2),
      stopped: false,
      imageAttached: true,
      imageSent: sent,
      imageResent: resent,
      contextReset: reset,
      contextTokens: 1234,
      prefillTokens: sent ? 290 : 12,
      promptTokens: 7,
    );
    List<String> lines(GenerationMetrics g) => debugOverlayLines(
      DiagnosticsSnapshot(lastGeneration: g, attachment: image),
    );

    expect(
      lines(turn()),
      containsAllInOrder([
        'img 1024×768 PNG 1.58 MB · from 4032×3024 2.29 MB · norm 61 ms',
        'img sent · ≈283 tok',
        'ctx 1234/${kLlmConfig.maxTokens} tok · prefill 290',
      ]),
    );
    expect(
      lines(turn(resent: ImageLoss.stop)),
      contains('img resent (lost: stop) · ≈283 tok'),
    );
    expect(lines(turn(sent: false)), contains('img in context (not re-sent)'));
    expect(
      lines(turn(resent: ImageLoss.budget, reset: true)),
      contains(
        'ctx 1234/${kLlmConfig.maxTokens} tok · prefill 290 · reset (budget)',
      ),
    );
    expect(
      debugOverlayLines(const DiagnosticsSnapshot())
          .where((l) => l.startsWith('img') || l.startsWith('ctx')),
      isEmpty,
      reason: 'nothing about images before the first one',
    );
  });

  testWidgets('recordAttachment sets and clears the attachment', (
    tester,
  ) async {
    final models = ValueNotifier<Map<ModelId, ModelState>>(const {});
    final repo = DiagnosticsRepository(models: models, sampleRss: false);
    final image = LlmImage(
      png: Uint8List(8),
      width: 2,
      height: 2,
      sourceWidth: 2,
      sourceHeight: 2,
      sourceBytes: 8,
      normalizeTime: Duration.zero,
    );

    repo.recordAttachment(image);
    expect(repo.latest.attachment, same(image));
    repo.recordGeneration(_metrics(3));
    expect(repo.latest.attachment, same(image), reason: 'kept by copyWith');
    repo.recordAttachment(null);
    expect(repo.latest.attachment, isNull);

    await tester.pump(const Duration(milliseconds: 300));
    repo.dispose();
    models.dispose();
  });
}
