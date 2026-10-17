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

import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/repositories/diagnostics_repository.dart';
import 'package:litert_edge_demos/domain/models/camera_side_event.dart';
import 'package:litert_edge_demos/domain/models/chat_capabilities.dart';
import 'package:litert_edge_demos/domain/models/detection.dart';
import 'package:litert_edge_demos/domain/models/detection_summary.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/domain/models/route_decision.dart';
import 'package:litert_edge_demos/domain/models/scene_snapshot.dart';
import 'package:litert_edge_demos/domain/models/visual_memory.dart';
import 'package:litert_edge_demos/domain/models/voice.dart';
import 'package:litert_edge_demos/domain/use_cases/camera_turn_responder.dart';
import 'package:litert_edge_demos/domain/use_cases/memory_answers.dart';
import 'package:litert_edge_demos/domain/use_cases/voice_assistant.dart';
import 'package:litert_edge_demos/domain/vision/coco_vocabulary.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../fakes/fake_audio_repository.dart';
import '../../fakes/fake_conversation_repository.dart';
import '../../fakes/fake_speech.dart';

Future<void> settle() => pumpEventQueue();

MemoryHit hitOf({
  Map<int, int> counts = const {},
  bool ambiguous = false,
  bool unmirrored = false,
}) => MemoryHit(
  entry: MemoryEntry(
    id: 3,
    frameId: 77,
    atMicros: 0,
    png: Uint8List.fromList([0x89, 0x50, 0x4E, 0x47, 77]),
    width: 320,
    height: 180,
    unmirrored: unmirrored,
    counts: counts,
    vector: Float32List(3),
    grey: Uint8List(1024),
  ),
  cosine: 0.76,
  rival: 0.67,
  age: const Duration(seconds: 12),
  ambiguous: ambiguous,
);

/// The frame at the release (id 9), for a question the frame now answers.
SceneSnapshot nowSnapshot() {
  final frame = DetectionFrame(
    frameId: 9,
    width: 640,
    height: 480,
    boxes: Float32List(0),
    preMicros: 1000,
    runMicros: 4000,
    postMicros: 1000,
    backend: DetectorBackend.gpu,
  );
  return SceneSnapshot(
    frameId: 9,
    detections: frame,
    summary: DetectionSummary.fromWindow([frame], minScore: 0.4),
    pixels: RgbaPixels(width: 2, height: 1, bytes: Uint8List(8)),
    mirrored: false,
    latency: const Duration(milliseconds: 14),
  );
}

void main() {
  late List<String> log;
  late FakeAudioRepository audio;
  late FakeRecognizer recognizer;
  late RecordingSynth synth;
  late ValueNotifier<Map<ModelId, ModelState>> models;
  late DiagnosticsRepository diagnostics;
  late Completer<Result<SceneSnapshot>> snapshot;
  late FakeConversationRepository conversation;
  late List<String> recalled;
  late Result<MemoryRecall> recall;
  late int latestCalls;
  late Result<MemoryRecall> latest;
  late VoiceAssistant<CameraSideEvent> assistant;
  late List<VoiceAssistantEvent<CameraSideEvent>> events;
  late StreamSubscription<VoiceAssistantEvent<CameraSideEvent>> sub;
  late int encodes;

  /// Set by a test whose question the frame now answers (the detailed
  /// path); false: encoding a snapshot fails the turn.
  late bool frameNow;

  Future<VoiceAssistant<CameraSideEvent>> build({
    bool withMemory = true,
  }) async {
    final speech = await loadedSpeech(
      recognizer: recognizer,
      synthesizer: synth,
    );
    return VoiceAssistant(
      speech: speech,
      audio: audio,
      responders: CameraTurnResponder(
        capture: () {
          log.add('capture');
          return snapshot.future;
        },
        conversation: conversation,
        encode: (s) async {
          encodes++;
          if (!frameNow) {
            throw StateError('a memory turn never encodes a snapshot');
          }
          return Result.ok(
            EncodedSnapshot(
              png: Uint8List.fromList([0x89, 0x50, 0x4E, 0x47, s.frameId]),
              frameId: s.frameId,
              width: 640,
              height: 480,
              unmirrored: false,
              encodeTime: const Duration(milliseconds: 9),
            ),
          );
        },
        recall: withMemory
            ? (phrase) async {
                recalled.add(phrase);
                return recall;
              }
            : null,
        latest: withMemory
            ? () async {
                latestCalls++;
                return latest;
              }
            : null,
        idleWait: const Duration(milliseconds: 200),
        resetWait: const Duration(milliseconds: 500),
      ),
      diagnostics: diagnostics,
    );
  }

  setUp(() async {
    log = [];
    audio = FakeAudioRepository(log: log)..autoDrain = true;
    recognizer = FakeRecognizer('Where was the sign about the wifi password?');
    synth = RecordingSynth();
    models = ValueNotifier(const {});
    diagnostics = DiagnosticsRepository(
      models: models,
      minInterval: Duration.zero,
      sampleRss: false,
    );
    // The release's snapshot never lands: a memory turn must not wait for it.
    snapshot = Completer();
    conversation = FakeConversationRepository();
    await conversation.open(kCameraProfile);
    recalled = [];
    recall = Result.ok(hitOf());
    latestCalls = 0;
    latest = Result.ok(
      MemoryHit(
        entry: hitOf().entry,
        cosine: null,
        age: const Duration(seconds: 12),
      ),
    );
    encodes = 0;
    frameNow = false;
    assistant = await build();
    events = [];
    sub = assistant.events.listen(events.add);
  });

  tearDown(() async {
    await sub.cancel();
    await assistant.dispose();
    await conversation.close();
    diagnostics.dispose();
    models.dispose();
  });

  List<CameraSideEvent> sides() => [
    for (final e in events)
      if (e case SideEvent(:final event)) event,
  ];

  String? answer() => events
      .whereType<AssistantSaid<CameraSideEvent>>()
      .map((e) => e.text)
      .firstOrNull;

  Future<void> untilAsked(int n) async {
    for (var i = 0; i < 200 && conversation.prompts.length < n; i++) {
      await settle();
    }
    expect(conversation.prompts, hasLength(n), reason: 'Gemma was asked');
  }

  test('a hit with images on: the phrase searches the memory, the frame goes '
      'to Gemma with the question, the chat is reset after; the release\'s '
      'snapshot is never awaited or encoded', () async {
    await assistant.micDown();
    final turn = assistant.micUp();
    await untilAsked(1);

    expect(recalled, ['sign about the wifi password']);
    final route = sides().whereType<RouteChosen>().single.route;
    expect(route, isA<MemoryRoute>());
    expect(route.rule, 'memory:where-was');
    final found = sides().whereType<MemoryRecalled>().single;
    expect(found.query, 'sign about the wifi password');
    expect(found.recall, isA<MemoryHit>());
    expect(conversation.images.single?.last, 77, reason: 'the memory PNG');
    expect(
      conversation.prompts.single,
      allOf(
        startsWith('This frame is from the camera 12 seconds ago'),
        endsWith('Question: Where was the sign about the wifi password?'),
      ),
    );
    conversation.emit('On the desk, ');
    conversation.emit('next to the mug.');
    await conversation.finish();
    final result = await turn;
    await settle();

    expect(result.outcome, TurnOutcome.completed);
    expect(answer(), 'On the desk, next to the mug.');
    expect(sides().whereType<SnapshotTaken>(), isEmpty);
    // The overlay's frame is the recalled one, not the previous turn's:
    // sent after the recall, the very PNG Gemma got.
    final sent = sides().whereType<FrameSentToGemma>().single.image;
    expect(sent.frameId, 77);
    expect(conversation.images.single, same(sent.png));
    expect((sent.width, sent.height), (320, 180));
    expect(sent.encodeTime, Duration.zero, reason: 'encoded when stored');
    expect(
      sides().indexOf(sides().whereType<MemoryRecalled>().single),
      lessThan(sides().indexOf(sides().whereType<FrameSentToGemma>().single)),
    );
    expect(encodes, 0);
    expect(sides().whereType<DetailedAnswered>(), hasLength(1));
    expect(conversation.resetCalls, 1);
  });

  test('a remembered frame mirrored back says so in FrameSentToGemma (the '
      'overlay\'s "unmirrored")', () async {
    recall = Result.ok(hitOf(unmirrored: true));
    await assistant.micDown();
    final turn = assistant.micUp();
    await untilAsked(1);
    expect(
      sides().whereType<FrameSentToGemma>().single.image.unmirrored,
      isTrue,
    );
    conversation.emit('On the desk.');
    await conversation.finish();
    expect((await turn).outcome, TurnOutcome.completed);
  });

  test(
    'nothing to search for ("What did you see?"): the newest stored frame '
    'goes to Gemma, said as the most recent one; nothing is embedded',
    () async {
      recognizer.text = 'What did you see?';
      await assistant.micDown();
      final turn = assistant.micUp();
      await untilAsked(1);

      expect(recalled, isEmpty, reason: 'no phrase to embed');
      expect(latestCalls, 1);
      final route = sides().whereType<RouteChosen>().single.route;
      expect(route, isA<MemoryRoute>().having((r) => r.latest, 'latest', true));
      expect(conversation.images.single?.last, 77, reason: 'the newest PNG');
      expect(
        conversation.prompts.single,
        allOf(
          startsWith(
            'This frame is from the camera 12 seconds ago: the most recent '
            'frame in its visual memory.',
          ),
          endsWith('Question: What did you see?'),
        ),
      );
      conversation.emit('A desk with a mug.');
      await conversation.finish();
      await turn;
      expect(
        sides().whereType<SnapshotTaken>(),
        isEmpty,
        reason: 'not awaited',
      );
      expect(encodes, 0);
    },
  );

  test('nothing to search for and nothing stored: the frame now answers it '
      '(the detailed path)', () async {
    recognizer.text = 'What have you seen?';
    latest = const Result.ok(MemoryEmpty());
    conversation.capabilities = const ChatCapabilities(
      modelName: 'Gemma 3 NPU',
      images: false,
      tools: false,
    );
    await assistant.micDown();
    final turn = assistant.micUp();
    await settle();
    final route = sides().whereType<RouteChosen>().single.route;
    expect(route, isA<DetailedRoute>());
    expect(route.rule, 'memory-empty');
    snapshot.completeError(StateError('the test ends at the snapshot'));
    await turn;
  });

  group('nothing to search for with the memory switched off or off: the '
      'frame now answers it (the detailed path), the rule says why', () {
    /// "What did you see?" with [newest] from the memory (null: an app
    /// without one) goes to Gemma with the frame at the release.
    Future<void> answeredFromTheFrameNow(
      String rule, {
      Result<MemoryRecall>? newest,
    }) async {
      frameNow = true;
      if (newest == null) {
        await sub.cancel();
        await assistant.dispose();
        assistant = await build(withMemory: false);
        events = [];
        sub = assistant.events.listen(events.add);
      } else {
        latest = newest;
      }
      recognizer.text = 'What did you see?';
      snapshot.complete(Result.ok(nowSnapshot()));
      await assistant.micDown();
      final turn = assistant.micUp();
      await untilAsked(1);

      final route = sides().whereType<RouteChosen>().single.route;
      expect(route, isA<DetailedRoute>());
      expect(route.rule, rule);
      expect(sides().whereType<MemoryRecalled>(), isEmpty);
      expect(recalled, isEmpty);
      expect(encodes, 1);
      expect(sides().whereType<SnapshotTaken>().single.snapshot.frameId, 9);
      expect(sides().whereType<FrameSentToGemma>().single.image.frameId, 9);
      expect(conversation.images.single?.last, 9, reason: 'the frame now');
      expect(conversation.prompts.single, endsWith('What did you see?'));
      conversation.emit('A desk with a mug.');
      await conversation.finish();
      final result = await turn;
      expect(result.outcome, TurnOutcome.completed);
      expect(answer(), 'A desk with a mug.');
    }

    test('switched off (the default below 6.5 GiB of RAM)', () async {
      await answeredFromTheFrameNow(
        'memory-disabled',
        newest: const Result.ok(MemoryDisabled()),
      );
      expect(latestCalls, 1);
    });

    test('off (the embedder failed)', () async {
      await answeredFromTheFrameNow(
        'memory-off',
        newest: const Result.ok(MemoryOff('EmbeddingGemma failed to load')),
      );
    });

    test('an app without a visual memory', () async {
      await answeredFromTheFrameNow('memory-off');
    });

    test('a question with something to search for is still told the memory '
        'is switched off, without the LLM', () async {
      recall = const Result.ok(MemoryDisabled());
      await assistant.micDown();
      await assistant.micUp();
      expect(answer(), MemoryAnswers.disabled);
      expect(sides().whereType<RouteChosen>().single.route, isA<MemoryRoute>());
      expect(conversation.prompts, isEmpty);
    });
  });

  test('a hit with images off: a template with the time and that frame\'s '
      'detections, no LLM', () async {
    conversation.capabilities = const ChatCapabilities(
      modelName: 'Gemma 3 NPU',
      images: false,
      tools: false,
    );
    recall = Result.ok(hitOf(counts: {kCocoNames.indexOf('cup'): 1}));
    await assistant.micDown();
    final result = await assistant.micUp();

    expect(result.outcome, TurnOutcome.completed);
    expect(
      answer(),
      'I saw it 12 seconds ago. In that frame the detector saw a cup.',
    );
    expect(conversation.prompts, isEmpty);
    expect(conversation.resetCalls, 0);
  });

  test(
    'not accepted: "I haven\'t seen that in the last N minutes", no LLM',
    () async {
      recognizer.text = 'Have you seen a giraffe?';
      recall = const Result.ok(
        MemoryMiss(
          cosine: 0.58,
          span: Duration(minutes: 2),
          frames: 37,
          reason: 'below the floor 0.62',
        ),
      );
      await assistant.micDown();
      await assistant.micUp();

      expect(recalled, ['giraffe']);
      expect(answer(), "I haven't seen that in the last 2 minutes.");
      expect(conversation.prompts, isEmpty);
      expect(
        sides().whereType<MemoryRecalled>().single.recall,
        isA<MemoryMiss>(),
      );
    },
  );

  test('an ambiguous hit with images on: said first that it may be in more '
      'than one place, then Gemma on the newest frame', () async {
    recall = Result.ok(hitOf(ambiguous: true));
    await assistant.micDown();
    final turn = assistant.micUp();
    await untilAsked(1);
    expect(conversation.images.single?.last, 77);
    expect(
      conversation.prompts.single,
      contains('the most recent of several frames from different places'),
    );
    conversation.emit('On the desk.');
    await conversation.finish();
    await turn;
    await settle();
    expect(
      answer(),
      'I may have seen that in more than one place; the latest was 12 '
      'seconds ago. On the desk.',
    );
    expect(sides().whereType<MemoryRecalled>().single.recall, isA<MemoryHit>());
  });

  test('an ambiguous hit with images off: the same lead, then the detector\'s '
      'list for that frame, no LLM', () async {
    conversation.capabilities = const ChatCapabilities(
      modelName: 'Gemma 3 NPU',
      images: false,
      tools: false,
    );
    recall = Result.ok(
      hitOf(ambiguous: true, counts: {kCocoNames.indexOf('cup'): 1}),
    );
    await assistant.micDown();
    await assistant.micUp();
    expect(
      answer(),
      'I may have seen that in more than one place; the latest was 12 '
      'seconds ago. In that frame the detector saw a cup.',
    );
    expect(conversation.prompts, isEmpty);
  });

  test('switched off in the camera settings: said so, no LLM', () async {
    recall = const Result.ok(MemoryDisabled());
    await assistant.micDown();
    await assistant.micUp();
    expect(answer(), 'Visual memory is off (camera settings).');
    expect(conversation.prompts, isEmpty);
  });

  test('nothing remembered yet, and the memory off, each say so', () async {
    recall = const Result.ok(MemoryEmpty());
    await assistant.micDown();
    await assistant.micUp();
    expect(answer(), MemoryAnswers.empty);

    events.clear();
    recall = const Result.ok(
      MemoryOff('embedding a frame failed 3 times in a row (engine)'),
    );
    await assistant.micDown();
    await assistant.micUp();
    expect(
      answer(),
      'My visual memory is off: embedding a frame failed 3 times in a row '
      '(engine).',
    );
    expect(conversation.prompts, isEmpty);
  });

  test(
    'a phrase that cannot be embedded is a turn error, never "not seen"',
    () async {
      recall = Result.error(Exception('embedder closed'));
      await assistant.micDown();
      final result = await assistant.micUp();

      expect(result.outcome, TurnOutcome.failed);
      expect(answer(), isNull);
      expect(sides().whereType<MemoryRecalled>(), isEmpty);
    },
  );

  test('without a visual memory the question is told so', () async {
    await sub.cancel();
    await assistant.dispose();
    assistant = await build(withMemory: false);
    events = [];
    sub = assistant.events.listen(events.add);

    await assistant.micDown();
    await assistant.micUp();
    expect(answer(), 'My visual memory is off: this app has no visual memory.');
  });

  test(
    'the camera chat not open: the hit is still told, from the template',
    () async {
      conversation.leaveOpen(kVoiceChatProfile);
      await assistant.micDown();
      await assistant.micUp();

      expect(sides().whereType<CameraChatUnavailable>(), hasLength(1));
      expect(
        answer(),
        "I saw it 12 seconds ago. The detector didn't recognize anything in "
        'that frame.',
      );
      expect(conversation.prompts, isEmpty);
    },
  );
}
