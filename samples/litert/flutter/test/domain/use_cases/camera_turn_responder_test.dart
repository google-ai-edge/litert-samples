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
import 'package:litert_edge_demos/data/repositories/conversation_repository.dart';
import 'package:litert_edge_demos/data/repositories/diagnostics_repository.dart';
import 'package:litert_edge_demos/domain/models/camera_side_event.dart';
import 'package:litert_edge_demos/domain/models/chat_capabilities.dart';
import 'package:litert_edge_demos/domain/models/detection.dart';
import 'package:litert_edge_demos/domain/models/detection_summary.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/domain/models/route_decision.dart';
import 'package:litert_edge_demos/domain/models/scene_snapshot.dart';
import 'package:litert_edge_demos/domain/models/voice.dart';
import 'package:litert_edge_demos/domain/use_cases/camera_turn_responder.dart';
import 'package:litert_edge_demos/domain/use_cases/voice_assistant.dart';
import 'package:litert_edge_demos/domain/vision/coco_vocabulary.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../fakes/fake_audio_repository.dart';
import '../../fakes/fake_conversation_repository.dart';
import '../../fakes/fake_speech.dart';
import '../../support/pcm.dart';

Future<void> settle() => pumpEventQueue();

int cls(String name) => kCocoNames.indexOf(name);

/// The cats fixture's frame (test_assets/yolo26n/cats_golden.json).
DetectionFrame catsFrame() => DetectionFrame(
  frameId: 42,
  width: 640,
  height: 480,
  boxes: Float32List.fromList([
    344, 24.5, 640, 374.8, 0.912, cls('cat').toDouble(), //
    6.9, 55.1, 317.3, 466.1, 0.899, cls('cat').toDouble(),
    40.4, 74, 175.9, 118.6, 0.857, cls('remote').toDouble(),
    0.8, 0.6, 640, 480, 0.297, cls('sofa').toDouble(),
  ]),
  preMicros: 1000,
  runMicros: 9000,
  postMicros: 600,
  backend: DetectorBackend.gpu,
);

SceneSnapshot catsSnapshot({bool mirrored = false}) => SceneSnapshot(
  frameId: 42,
  detections: catsFrame(),
  summary: DetectionSummary.fromWindow([
    for (var i = 0; i < 5; i++) catsFrame(),
  ], minScore: 0.4),
  pixels: RgbaPixels(width: 2, height: 1, bytes: Uint8List(8)),
  mirrored: mirrored,
  latency: const Duration(milliseconds: 14),
);

/// What the fake encoder returns for [s]: a fresh PNG object per call.
EncodedSnapshot encodedOf(SceneSnapshot s) => EncodedSnapshot(
  png: Uint8List.fromList([0x89, 0x50, 0x4E, 0x47, s.frameId]),
  frameId: s.frameId,
  width: 640,
  height: 480,
  unmirrored: s.mirrored,
  encodeTime: const Duration(milliseconds: 9),
);

void main() {
  late List<String> log;
  late FakeAudioRepository audio;
  late FakeRecognizer recognizer;
  late RecordingSynth synth;
  late ValueNotifier<Map<ModelId, ModelState>> models;
  late DiagnosticsRepository diagnostics;
  late Completer<Result<SceneSnapshot>> snapshot;
  late VoiceAssistant<CameraSideEvent> assistant;
  late List<VoiceAssistantEvent<CameraSideEvent>> events;
  late StreamSubscription<VoiceAssistantEvent<CameraSideEvent>> sub;
  late FakeConversationRepository conversation;
  late List<SceneSnapshot> encoded;
  Result<EncodedSnapshot>? encodeResult;

  setUp(() async {
    log = [];
    audio = FakeAudioRepository(log: log)..autoDrain = true;
    recognizer = FakeRecognizer('How many cats do you see?');
    synth = RecordingSynth();
    models = ValueNotifier(const {});
    diagnostics = DiagnosticsRepository(
      models: models,
      minInterval: Duration.zero,
      sampleRss: false,
    );
    snapshot = Completer();
    conversation = FakeConversationRepository();
    await conversation.open(kCameraProfile);
    encoded = [];
    encodeResult = null;
    final speech = await loadedSpeech(
      recognizer: recognizer,
      synthesizer: synth,
    );
    assistant = VoiceAssistant(
      speech: speech,
      audio: audio,
      responders: CameraTurnResponder(
        capture: () {
          log.add('capture');
          return snapshot.future;
        },
        conversation: conversation,
        encode: (s) async {
          encoded.add(s);
          return encodeResult ?? Result.ok(encodedOf(s));
        },
        idleWait: const Duration(milliseconds: 200),
        resetWait: const Duration(milliseconds: 500),
      ),
      diagnostics: diagnostics,
    );
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

  test('"How many cats do you see?" on the cats frame speaks "I count '
      'two cats." with no LLM', () async {
    await assistant.micDown();
    final turn = assistant.micUp();
    expect(log.last, 'capture', reason: 'the snapshot starts at release');
    await settle();
    expect(log.indexOf('capture'), lessThan(log.indexOf('capture.stop')));

    snapshot.complete(Result.ok(catsSnapshot()));
    final result = await turn;

    expect(result.outcome, TurnOutcome.completed);
    expect(answer(), 'I count two cats.');
    expect(synth.synthesized, ['I count two cats.']);
    expect(log, contains('beginPlayback(24000)'));
    final route = sides().whereType<RouteChosen>().single.route;
    expect(route, isA<FastRoute>());
    expect((route as FastRoute).intent, FastIntent.count);
    expect(route.cls, cls('cat'));
    expect(route.rule, 'count');
    expect(sides().whereType<SnapshotTaken>().single.snapshot.frameId, 42);
    expect(sides().whereType<FastAnswered>().single.basis, 'cat ×2 · remote');
  });

  test('a presence question and an inventory question', () async {
    snapshot.complete(Result.ok(catsSnapshot()));
    recognizer.text = 'Is there a giraffe?';
    await assistant.micDown();
    await assistant.micUp();
    expect(answer(), "I don't see a giraffe right now.");

    events.clear();
    snapshot = Completer()..complete(Result.ok(catsSnapshot()));
    recognizer.text = 'What do you see?';
    await assistant.micDown();
    await assistant.micUp();
    expect(answer(), 'I see two cats and a remote.');
  });

  test('a moonshine homophone is corrected before routing: "Is there '
      'a cop?" is the fast cup presence question, and the correction is '
      'reported', () async {
    snapshot.complete(Result.ok(catsSnapshot()));
    recognizer.text = 'Is there a cop?';
    await assistant.micDown();
    await assistant.micUp();

    final route = sides().whereType<RouteChosen>().single.route;
    expect(route, isA<FastRoute>());
    expect((route as FastRoute).cls, cls('cup'));
    expect(answer(), "I don't see a cup right now.");
    final corrected = sides().whereType<TranscriptCorrected>().single;
    expect(corrected.corrections.single.heard, 'cop');
    expect(corrected.corrections.single.meant, 'cup');
    expect(corrected.text, 'Is there a cup?');
    expect(conversation.prompts, isEmpty, reason: 'no Gemma turn');
  });

  /// Waits until Gemma has been asked (the detailed turn reached `ask`).
  Future<void> untilAsked(int n) async {
    for (var i = 0; i < 200 && conversation.prompts.length < n; i++) {
      await settle();
    }
    expect(conversation.prompts, hasLength(n), reason: 'Gemma was asked');
  }

  test(
    'images off (a custom chat model without a vision encoder): a '
    'detailed question never sends the frame; the detections answer',
    () async {
      conversation.capabilities = const ChatCapabilities(
        modelName: 'Gemma 3 NPU',
        images: false,
        tools: false,
      );
      snapshot.complete(Result.ok(catsSnapshot()));
      recognizer.text = 'What color is the cat?';
      await assistant.micDown();
      final result = await assistant.micUp();

      expect(result.outcome, TurnOutcome.completed);
      expect(
        conversation.prompts,
        isEmpty,
        reason: 'nothing to the chat model',
      );
      expect(encoded, isEmpty, reason: 'the frame is not even encoded');
      expect(
        answer(),
        '${CameraTurnResponder.imagesOff} I see two cats and a remote.',
      );
      final off = sides().whereType<DetailedUnavailable>().single;
      expect(off.reason, contains('Gemma 3 NPU'));
      expect(off.reason, contains('without images'));
      expect(off.basis, 'cat ×2 · remote');
      expect(sides().whereType<FrameSentToGemma>(), isEmpty);
      expect(conversation.resetCalls, 0);
    },
  );

  group('detailed path', () {
    test(
      'the frame goes to Gemma as a PNG with the detector hint; the reply '
      'is spoken; the chat is reset afterwards, off the critical path',
      () async {
        snapshot.complete(Result.ok(catsSnapshot()));
        recognizer.text = 'What color is the cat?';
        await assistant.micDown();
        final turn = assistant.micUp();
        await untilAsked(1);

        expect(encoded.single.frameId, 42);
        final sent = sides().whereType<FrameSentToGemma>().single.image;
        expect(sent.frameId, 42, reason: 'the frozen frame is the one sent');
        expect(conversation.images.single, same(sent.png));
        expect(
          conversation.prompts.single,
          allOf(
            contains('cat (right), cat (left), remote (top left)'),
            endsWith('Question: What color is the cat?'),
          ),
        );
        expect(conversation.resetCalls, 0, reason: 'not before the answer');

        conversation.emit('The cats are grey ');
        conversation.emit('and white.');
        await conversation.finish();
        final result = await turn;
        await settle();

        expect(result.outcome, TurnOutcome.completed);
        expect(answer(), 'The cats are grey and white.');
        final route = sides().whereType<RouteChosen>().single.route;
        expect(route, isA<DetailedRoute>());
        expect(route.rule, 'detail:color');
        expect(sides().whereType<DetailedAnswered>(), hasLength(1));
        expect(conversation.resetCalls, 1);
        expect(conversation.openedProfiles.last, kCameraProfile);
        expect(sides().whereType<CameraChatReset>().single.error, isNull);
        expect(conversation.stopCalls, 0);
      },
    );

    test('a mirroring source: the hint positions are mirrored too', () async {
      snapshot.complete(Result.ok(catsSnapshot(mirrored: true)));
      recognizer.text = 'Where is the remote?';
      await assistant.micDown();
      final turn = assistant.micUp();
      await untilAsked(1);
      expect(conversation.prompts.single, contains('remote (top right)'));
      conversation.emit('Top right.');
      await conversation.finish();
      expect((await turn).outcome, TurnOutcome.completed);
    });

    test(
      'the camera chat not open: a clear spoken notice, no ask, no reset',
      () async {
        conversation.leaveOpen(kVoiceChatProfile); // e.g. the open still runs
        snapshot.complete(Result.ok(catsSnapshot()));
        recognizer.text = 'Describe the scene.';
        await assistant.micDown();
        await assistant.micUp();

        expect(answer(), CameraTurnResponder.chatNotReady);
        expect(sides().whereType<CameraChatUnavailable>(), hasLength(1));
        expect(conversation.prompts, isEmpty);
        expect(encoded, isEmpty);
        expect(conversation.resetCalls, 0);
      },
    );

    test('barge-in during the answer stops only its own ask; the reset still '
        'runs; a late stop afterwards does not stop the next turn', () async {
      snapshot.complete(Result.ok(catsSnapshot()));
      recognizer.text = 'Describe the scene.';
      await assistant.micDown();
      final first = assistant.micUp();
      await untilAsked(1);
      conversation.emit('Two cats ');
      await settle();

      // Barge-in: the mic goes down during the answer.
      await assistant.micDown();
      expect((await first).outcome, TurnOutcome.superseded);
      for (var i = 0; i < 20; i++) {
        await settle();
      }
      expect(conversation.stopCalls, 1, reason: 'its own ask was running');
      expect(conversation.resetCalls, 1, reason: 'stateless after a stop too');

      // The next question waits for that reset, then asks.
      snapshot = Completer()..complete(Result.ok(catsSnapshot()));
      final second = assistant.micUp();
      await untilAsked(2);
      final stopsBefore = conversation.stopCalls;
      conversation.emit('A remote.');
      await conversation.finish();
      await second;
      expect(conversation.stopCalls, stopsBefore);
    });

    test('a failed answer is a turn error (rethrown, never an empty reply); '
        'the chat is still reset', () async {
      snapshot.complete(Result.ok(catsSnapshot()));
      recognizer.text = 'Describe the scene.';
      await assistant.micDown();
      final turn = assistant.micUp();
      await untilAsked(1);
      await conversation.fail(
        const ConversationImageUnsupportedException('no vision'),
      );
      final result = await turn;
      await settle();

      expect(result.outcome, TurnOutcome.failed);
      expect(
        events.whereType<TurnFailed<CameraSideEvent>>().single.error,
        isA<ConversationImageUnsupportedException>(),
      );
      expect(conversation.resetCalls, 1);
    });

    test(
      'an encoding failure is a turn error and Gemma is not asked',
      () async {
        encodeResult = Result.error(Exception('codec'));
        snapshot.complete(Result.ok(catsSnapshot()));
        recognizer.text = 'Describe the scene.';
        await assistant.micDown();
        final result = await assistant.micUp();

        expect(result.outcome, TurnOutcome.failed);
        expect(conversation.prompts, isEmpty);
        expect(conversation.resetCalls, 0);
      },
    );

    test('the next detailed question waits for the previous reset before it '
        'checks the chat', () async {
      snapshot.complete(Result.ok(catsSnapshot()));
      recognizer.text = 'Describe the scene.';
      await assistant.micDown();
      final first = assistant.micUp();
      await untilAsked(1);
      final gate = conversation.openGate = Completer<void>();
      conversation.emit('Two cats.');
      await conversation.finish();
      await first;
      expect(conversation.resetCalls, 1);

      snapshot = Completer()..complete(Result.ok(catsSnapshot()));
      await assistant.micDown();
      final second = assistant.micUp();
      for (var i = 0; i < 20; i++) {
        await settle();
      }
      expect(conversation.prompts, hasLength(1), reason: 'reset still runs');
      gate.complete();
      await untilAsked(2);
      conversation.emit('A remote.');
      await conversation.finish();
      expect((await second).outcome, TurnOutcome.completed);
    });

    test('a stop that landed before generation began is sent again when text '
        'still arrives (scoped to this ask)', () async {
      snapshot.complete(Result.ok(catsSnapshot()));
      recognizer.text = 'Describe the scene.';
      await assistant.micDown();
      final turn = assistant.micUp();
      await untilAsked(1);
      conversation.stopGate = Completer<void>(); // the stop does not land yet
      final stopping = assistant.stop();
      await settle();
      conversation.emit('late text');
      await settle();
      expect(conversation.stopCalls, 2, reason: 'stop, then the re-send');
      conversation.stopGate!.complete();
      await stopping;
      await turn;
    });
  });

  test('no frame: the turn says the camera is not running and the UI gets '
      'why', () async {
    snapshot.complete(
      const Result.error(
        CaptureUnavailableException("The camera isn't running"),
      ),
    );
    await assistant.micDown();
    await assistant.micUp();

    expect(answer(), CameraTurnResponder.cameraNotRunning);
    expect(
      sides().whereType<SnapshotFailed>().single.message,
      "The camera isn't running",
    );
  });

  test(
    'a capture timeout says so, in other words than "not running"',
    () async {
      snapshot.complete(
        const Result.error(
          CaptureUnavailableException(
            'No camera frame was detected within 1500 ms',
            kind: CaptureFailure.timedOut,
          ),
        ),
      );
      await assistant.micDown();
      await assistant.micUp();

      expect(answer(), CameraTurnResponder.cameraTimedOut);
      expect(answer(), isNot(CameraTurnResponder.cameraNotRunning));
      expect(
        sides().whereType<SnapshotFailed>().single.message,
        contains('1500 ms'),
      );
    },
  );

  test('a failed pipeline says the camera stopped', () async {
    snapshot.complete(
      const Result.error(
        CaptureUnavailableException(
          'Detector failed: GPU lost',
          kind: CaptureFailure.failed,
        ),
      ),
    );
    await assistant.micDown();
    await assistant.micUp();

    expect(answer(), CameraTurnResponder.cameraFailed);
  });

  test(
    'a not-heard press discards the snapshot: no side events, no STT',
    () async {
      audio.nextUtterance = Utterance(
        pcm: quiet(const Duration(seconds: 1)),
        held: const Duration(seconds: 1),
      );
      await assistant.micDown();
      final result = await assistant.micUp();
      snapshot.complete(Result.ok(catsSnapshot()));
      await settle();

      expect(result.outcome, TurnOutcome.notHeard);
      expect(log, contains('capture'), reason: 'started at release');
      expect(sides(), isEmpty);
      expect(recognizer.calls, 0);
    },
  );
}
