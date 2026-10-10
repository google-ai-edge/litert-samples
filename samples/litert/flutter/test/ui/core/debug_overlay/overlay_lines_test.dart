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

// debugOverlayLines on its own: the order of the blocks and the lines the
// repository and layout tests do not reach (model labels, a stopped
// generation, the chat's own context window, Demo 3's camera lines, memory
// sizes, a typed turn and the voice gate).
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/domain/models/assistant_event.dart';
import 'package:litert_edge_demos/domain/models/camera_side_event.dart';
import 'package:litert_edge_demos/domain/models/diagnostics_snapshot.dart';
import 'package:litert_edge_demos/domain/models/knowledge.dart';
import 'package:litert_edge_demos/domain/models/live_state.dart';
import 'package:litert_edge_demos/domain/models/llm_image.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/domain/models/route_decision.dart';
import 'package:litert_edge_demos/domain/models/scene_snapshot.dart';
import 'package:litert_edge_demos/domain/models/skill_catalog.dart';
import 'package:litert_edge_demos/domain/models/voice.dart';
import 'package:litert_edge_demos/ui/core/debug_overlay.dart';

final _now = DateTime(2026, 10, 7, 12);

Duration _ms(int ms) => Duration(milliseconds: ms);

LoadedModelInfo _model(
  String id,
  String backend, {
  String? detail,
  ChatModelFacts? chat,
  int load = 120,
  int warm = 30,
}) => LoadedModelInfo(
  modelId: id,
  backend: backend,
  loadTime: _ms(load),
  warmUpTime: _ms(warm),
  detail: detail,
  chat: chat,
);

ChatModelFacts _facts({String name = 'My Gemma', int context = 4096}) =>
    ChatModelFacts(
      name: name,
      custom: true,
      source: 'models folder',
      requestedBackend: 'gpu',
      requestedContext: context,
      contextTokens: context,
      images: true,
      tools: true,
      modelType: 'gemma4',
    );

GenerationMetrics _generation({
  Duration? ttft = const Duration(milliseconds: 350),
  Duration? stopLatency,
  bool stopped = false,
  int? contextTokens,
  int? prefillTokens,
  bool imageAttached = false,
}) => GenerationMetrics(
  timeToFirstToken: ttft,
  chunks: ttft == null ? 0 : 42,
  tokensPerSecond: ttft == null ? null : 21.5,
  tokensPerSecondSource: TokenRateSource.chunks,
  total: _ms(2000),
  stopped: stopped,
  stopLatency: stopLatency,
  imageAttached: imageAttached,
  imageSent: imageAttached,
  contextTokens: contextTokens,
  prefillTokens: prefillTokens,
);

void main() {
  test('the blocks in order: the models (the embedder heads the knowledge '
      'base), detection, the generation and its stop, the image, the '
      'voice, the skills, the knowledge base, the camera; RSS last', () {
    final lines = debugOverlayLines(
      DiagnosticsSnapshot(
        models: {
          ModelId.chat: _model('gemma-4-E2B-it', 'gpu'),
          ModelId.embeddingGemma: _model('embeddinggemma-300M', 'cpu'),
        },
        liveState: const LiveStarting(),
        lastGeneration: _generation(
          stopped: true,
          stopLatency: _ms(42),
          imageAttached: true,
        ),
        attachment: LlmImage(
          png: Uint8List(4096),
          width: 640,
          height: 480,
          sourceWidth: 640,
          sourceHeight: 480,
          sourceBytes: 4096,
          normalizeTime: _ms(9),
        ),
        skills: const SkillCatalog(
          directory: '/skills',
          skills: [],
          errors: [],
          fingerprint: 'f',
        ),
        knowledge: const KnowledgeWaiting(),
        lastCameraTurn: const CameraTurnMetrics(
          route: DetailedRoute('describe'),
          routeTime: Duration(microseconds: 500),
        ),
        rssBytes: 1 << 29,
      ),
      now: _now,
    );
    int at(String prefix) => lines.indexWhere((l) => l.startsWith(prefix));

    final order = [
      at('gemma-4-E2B-it'),
      at('det '),
      at('TTFT'),
      at('stop '),
      at('img '),
      at('voice '),
      at('skills '),
      at('embeddinggemma-300M'),
      at('KB '),
      at('camq '),
      at('RSS '),
    ];
    expect(order, isNot(contains(-1)), reason: lines.join('\n'));
    expect(order, [...order]..sort(), reason: lines.join('\n'));
    expect(at('stop '), at('TTFT') + 1, reason: 'the stop follows its turn');
    expect(lines.last, startsWith('RSS '));
    expect(
      lines.where((l) => l.startsWith('embeddinggemma-300M')),
      hasLength(1),
      reason: 'the embedder is not in the model block too',
    );
  });

  test("model lines: the chat model's own name, the detail over the "
      'backend, ACTIVE on the recognizer in use only', () {
    final lines = debugOverlayLines(
      DiagnosticsSnapshot(
        models: {
          ModelId.chat: _model(
            'gemma-4-E2B-it',
            'gpu',
            chat: _facts(),
            load: 4200,
            warm: 800,
          ),
          ModelId.whisperBase: _model('whisper_base_int8', 'cpu'),
          ModelId.yolo26n: _model(
            'yolo26n_fp16_rawhead',
            'gpu',
            detail: 'GPU fp32 full',
            load: 350,
            warm: 5,
          ),
          ModelId.moonshineTiny: _model('moonshine_tiny', 'cpu', load: 80),
        },
        activeStt: const ActiveSttInfo(
          id: ModelId.moonshineTiny,
          modelId: 'moonshine_tiny',
          switchTime: Duration(milliseconds: 38),
        ),
      ),
      now: _now,
    );

    expect(lines.take(5), [
      'My Gemma · GPU · load 4200 ms · warm 800 ms',
      'whisper_base_int8 · CPU · load 120 ms · warm 30 ms',
      'Inflect-nano-v2 (TTS): not loaded',
      'yolo26n_fp16_rawhead · GPU fp32 full · load 350 ms · warm 5 ms',
      'moonshine_tiny · CPU · load 80 ms · warm 30 ms · ACTIVE',
    ]);
    expect(lines.where((l) => l.contains('ACTIVE')), hasLength(1));
    expect(lines, contains('STT active: moonshine_tiny · switch 38 ms'));
  });

  test('a turn stopped before its first token: dashes for TTFT and the '
      'rate, then the stop latency', () {
    final lines = debugOverlayLines(
      DiagnosticsSnapshot(
        lastGeneration: _generation(
          ttft: null,
          stopped: true,
          stopLatency: _ms(42),
        ),
      ),
      now: _now,
    );

    final generation = lines.indexOf('TTFT – · – tok/s (chunks) · chunks 0');
    expect(generation, isNot(-1), reason: lines.join('\n'));
    expect(lines[generation + 1], 'stop 42 ms');
  });

  test("the context use is out of the chat model's own window when it "
      'reports one', () {
    final lines = debugOverlayLines(
      DiagnosticsSnapshot(
        models: {
          ModelId.chat: _model('gemma', 'gpu', chat: _facts(context: 2048)),
        },
        lastGeneration: _generation(contextTokens: 900, prefillTokens: 120),
      ),
      now: _now,
    );

    expect(lines, contains('ctx 900/2048 tok · prefill 120'));
  });

  test('Demo 3: a detailed question on a frozen frame, its PNG, the '
      'reset that failed, and why there was no frame or no chat', () {
    final snapshot = DiagnosticsSnapshot(
      llmTurns: 3,
      liveState: LivePaused(
        source: 'camera',
        reason: 'gemma',
        since: _now.subtract(_ms(1500)),
      ),
      lastCameraTurn: CameraTurnMetrics(
        route: const DetailedRoute('describe'),
        routeTime: const Duration(microseconds: 1250),
        image: EncodedSnapshot(
          png: Uint8List(2048),
          frameId: 77,
          width: 896,
          height: 504,
          unmirrored: true,
          encodeTime: _ms(41),
        ),
        snapshotError: 'no frame yet',
        chatError: 'the chat is busy',
      ),
      lastCameraReset: _ms(120),
      cameraResetError: 'close failed',
      frozenFrameId: 77,
    );

    expect(
      debugOverlayLines(snapshot, now: _now),
      containsAllInOrder([
        'det paused (gemma) 1.5 s',
        'camq detailed · rule describe · route 1.25 ms · snap – · '
            'llm turns 3',
        'camq frame #77 896×504 PNG 2 KB · enc 41 ms · unmirrored · TTFT –',
        'camq reset 120 ms FAILED: close failed',
        'view frozen on #77 · det paused',
        'camq no frame: no frame yet',
        'camq chat: the chat is busy',
      ]),
    );

    // Live again after a reset that worked.
    final live = debugOverlayLines(
      snapshot.copyWith(
        liveState: const LiveRunning('camera'),
        cameraReset: (elapsed: _ms(95), error: null),
      ),
      now: _now,
    );
    expect(live, contains('camq reset 95 ms'));
    expect(live, contains('view frozen on #77 · det live'));
  });

  test('detection without a backend label, before its first frame: '
      '"detector", no source size, dashes for the latency', () {
    final lines = debugOverlayLines(
      DiagnosticsSnapshot(
        models: {ModelId.yolo26n: _model('yolo26n', 'gpu')},
        liveState: const LiveRunning('fixture (3 images)'),
        liveStats: const LiveStats(fps: 15),
      ),
      now: _now,
    );

    expect(
      lines,
      containsAllInOrder([
        'det detector · 15.0 fps · p50 pre/run/post –/–/– ms',
        'det src fixture (3 images) · lat – ms · drop busy 0 rate 0',
      ]),
    );
  });

  test('memory: MB below a gigabyte, GB with two decimals above', () {
    final lines = debugOverlayLines(
      const DiagnosticsSnapshot(
        rssBytes: 700 << 20,
        peakRssBytes: 3 << 29, // 1.5 GB
      ),
      now: _now,
    );

    expect(lines.last, 'RSS 700 MB · peak 1.50 GB');
  });

  test('voice: a typed turn says so with its outcome; a spoken one shows '
      'the gate it passed', () {
    List<String> lines(VoiceTurnMetrics turn) =>
        debugOverlayLines(DiagnosticsSnapshot(lastVoiceTurn: turn), now: _now);

    expect(
      lines(
        VoiceTurnMetrics(
          typed: true,
          firstAudio: _ms(900),
          outcome: TurnOutcome.interrupted,
        ),
      ),
      contains('voice idle · typed · first audio 900 ms · interrupted'),
    );
    expect(
      lines(
        VoiceTurnMetrics(
          typed: false,
          stt: _ms(65),
          peakDbfs: -18.24,
          gateDbfs: -45,
          voiced: _ms(1820),
        ),
      ),
      containsAllInOrder([
        'voice idle · STT 65 ms · first audio –',
        'gate peak -18.2 dBFS · voiced 1820 ms over -45.0 dBFS',
      ]),
    );
  });
}
