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
import 'package:litert_edge_demos/config/live_camera_config.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/config/voice_config.dart';
import 'package:litert_edge_demos/data/repositories/audio_repository.dart';
import 'package:litert_edge_demos/data/repositories/diagnostics_repository.dart';
import 'package:litert_edge_demos/data/repositories/live_detection_repository.dart';
import 'package:litert_edge_demos/data/repositories/speech_repository.dart';
import 'package:litert_edge_demos/domain/models/camera_side_event.dart';
import 'package:litert_edge_demos/domain/models/chat_capabilities.dart';
import 'package:litert_edge_demos/domain/models/frame_source_info.dart';
import 'package:litert_edge_demos/domain/models/frame_source_spec.dart';
import 'package:litert_edge_demos/domain/models/live_state.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/domain/models/route_decision.dart';
import 'package:litert_edge_demos/domain/models/voice.dart';
import 'package:litert_edge_demos/domain/use_cases/camera_turn_responder.dart';
import 'package:litert_edge_demos/domain/use_cases/voice_assistant.dart';
import 'package:litert_edge_demos/ui/features/live_camera/view_models/live_camera_view_model.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../../../fakes/fake_audio_repository.dart';
import '../../../../fakes/fake_conversation_repository.dart';
import '../../../../fakes/fake_detector.dart';
import '../../../../fakes/fake_frame_source.dart';
import '../../../../fakes/fake_speech.dart';

/// Polls [condition] (turns run on real timers): bounded by real time,
/// generously, not by a count of event-loop turns.
Future<void> until(bool Function() condition, {String? reason}) async {
  final watch = Stopwatch()..start();
  while (!condition() && watch.elapsed < const Duration(seconds: 15)) {
    await Future<void>.delayed(const Duration(milliseconds: 2));
  }
  expect(condition(), isTrue, reason: reason);
}

const _fixture = Result<FrameSourceSpec>.ok(FixtureSourceSpec(['/fixtures']));

/// One period of the live rate gate ([kLiveDetectFps]).
const _gatePeriodMicros = 1000000 ~/ kLiveDetectFps;

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  late FakeConversationRepository conversation;
  late List<FakeFrameSource> sources;
  late LiveDetectionRepository live;
  late SpeechRepository speech;
  late ValueNotifier<Map<ModelId, ModelState>> models;
  late DiagnosticsRepository diagnostics;
  late FakeRecognizer recognizer;
  late List<ModelId> sttActivations;
  Result<void> sttResult = const Result.ok(null);
  late List<String> log;
  late FakeAudioRepository audio;

  /// The live repository's clock (its rate gate's): the tests move it.
  late int nowMicros;

  /// Captures the turns asked for (a snapshot is the next frame after one).
  late int captures;

  Future<Result<void>> activateStt(ModelId id) async {
    sttActivations.add(id);
    return sttResult;
  }

  VoiceAssistant<CameraSideEvent> assistantFor(LiveDetectionRepository live) =>
      VoiceAssistant(
        speech: speech,
        audio: audio,
        responders: CameraTurnResponder(
          capture: () {
            captures++;
            return live.capture();
          },
          conversation: conversation,
          encode: live.encodeForLlm,
        ),
        diagnostics: diagnostics,
      );

  setUp(() async {
    log = [];
    audio = FakeAudioRepository(log: log)..autoDrain = true;
    recognizer = FakeRecognizer('How many cats do you see?');
    sttActivations = [];
    sttResult = const Result.ok(null);
    speech = await loadedSpeech(
      recognizer: recognizer,
      synthesizer: RecordingSynth(),
    );
    models = ValueNotifier(const {});
    diagnostics = DiagnosticsRepository(
      models: models,
      minInterval: Duration.zero,
      sampleRss: false,
    );
    conversation = FakeConversationRepository();
    sources = [];
    nowMicros = 0;
    captures = 0;
    live = LiveDetectionRepository(
      detector: FakeDetector(autoComplete: true),
      createSource: (_) {
        final s = FakeFrameSource();
        sources.add(s);
        return Result.ok(s);
      },
      clockMicros: () => nowMicros,
    );
  });

  tearDown(() async {
    await live.close();
    await conversation.close();
    diagnostics.dispose();
    models.dispose();
  });

  LiveCameraViewModel create({Result<FrameSourceSpec> source = _fixture}) =>
      LiveCameraViewModel(
        conversation: conversation,
        live: live,
        assistant: assistantFor(live),
        diagnostics: diagnostics,
        activateStt: activateStt,
        source: source,
      );

  /// One live frame past the rate gate: the clock moves one gate period on
  /// (no real sleep), and the frame is detected and published.
  Future<void> liveFrame() async {
    nowMicros += _gatePeriodMicros;
    final before = live.frames.value;
    sources.single.emit();
    await until(() => !identical(live.frames.value, before));
  }

  /// Releases the mic and, once the turn asked for its capture, delivers
  /// the snapshot frame (the next frame after release); returns the turn.
  Future<void> releaseWithSnapshot(LiveCameraViewModel viewModel) {
    final asked = captures;
    final releasing = viewModel.releaseMic();
    return until(() => captures > asked).then((_) {
      sources.single.emit();
      return releasing;
    });
  }

  test(
    'entering Demo 3 opens the camera chat and starts live detection',
    () async {
      final viewModel = create();
      await until(() => live.state.value is LiveRunning && viewModel.chatReady);

      expect(conversation.openedProfiles, [kCameraProfile]);
      expect(viewModel.chatReady, isTrue);
      expect(viewModel.chatError, isNull);
      expect(live.state.value, isA<LiveRunning>());
      expect(viewModel.frames, same(live.frames));
      expect(viewModel.blackFrames, same(live.blackFrames));
      expect(viewModel.blackFramesWarning, startsWith('Camera delivers black'));
      expect(viewModel.detectorLabel, 'GPU fp32 full');
      expect(viewModel.mirrorBoxes, isFalse);

      viewModel.dispose();
    },
  );

  test('entering Demo 3 asks for the microphone once the '
      'camera has started (one OS dialog at a time); a denial is a lasting '
      'error with Retry', () async {
    audio.micAccess = Result.error(MicAccessException(kMicAccessMessage));
    final viewModel = create();
    await until(() => viewModel.micAccess.result != null);

    expect(live.state.value, isA<LiveRunning>());
    expect(audio.micAccessRequests, 1);
    expect(viewModel.micAccessError, contains(kMicAccessMessage));

    audio.micAccess = const Result.ok(null);
    await viewModel.micAccess.execute();
    expect(viewModel.micAccessError, isNull);
    expect(audio.micAccessRequests, 2);
    viewModel.dispose();
  });

  test('a corrected transcript shows in the caption with what was '
      'heard', () async {
    recognizer.text = 'Is there a cop?';
    final viewModel = create();
    await until(() => live.state.value is LiveRunning);
    await viewModel.pressMic();
    await releaseWithSnapshot(viewModel);
    await until(() => viewModel.exchange.question != null);
    expect(viewModel.exchange.question, "Is there a cup? (heard 'cop')");
    viewModel.dispose();
  });

  test('mirrorBoxes follows the source: preview XOR frames', () async {
    final mirroredPreviewOnly = LiveDetectionRepository(
      detector: FakeDetector(autoComplete: true),
      createSource: (_) => Result.ok(
        FakeFrameSource()
          ..startResult = const Result.ok(
            FrameSourceInfo(
              label: 'windows-like',
              width: 1280,
              height: 720,
              format: FramePixelFormat.bgra8888,
              mirrored: false,
              previewMirrored: true,
            ),
          ),
      ),
    );
    addTearDown(mirroredPreviewOnly.close);
    final viewModel = LiveCameraViewModel(
      conversation: conversation,
      live: mirroredPreviewOnly,
      assistant: assistantFor(mirroredPreviewOnly),
      diagnostics: diagnostics,
      activateStt: activateStt,
      source: _fixture,
    );
    expect(viewModel.mirrorBoxes, isFalse, reason: 'nothing started yet');
    await until(() => mirroredPreviewOnly.state.value is LiveRunning);

    expect(viewModel.mirrorBoxes, isTrue);
    viewModel.dispose();
  });

  test("the chat is not ready while its open waits for the previous demo's "
      'turn to drain', () async {
    conversation
      ..leaveOpen(kVoiceChatProfile)
      ..openGate = Completer<void>();
    final viewModel = create();
    await pumpEventQueue();

    expect(viewModel.open.running, isTrue);
    expect(viewModel.chatReady, isFalse);

    conversation.openGate!.complete();
    await until(() => viewModel.chatReady);
    viewModel.dispose();
  });

  test('a failed chat open is shown and Retry reopens it', () async {
    conversation.openResult = Result.error(Exception('no model'));
    final viewModel = create();
    await until(() => viewModel.open.result != null);
    expect(viewModel.chatError, contains('no model'));
    expect(viewModel.chatReady, isFalse);

    conversation.openResult = const Result.ok(null);
    await viewModel.open.execute();

    expect(viewModel.chatError, isNull);
    expect(viewModel.chatReady, isTrue);
    viewModel.dispose();
  });

  test(
    'a bad frame-source configuration is shown and nothing starts',
    () async {
      final viewModel = create(
        source: const Result.error(
          FrameSourceUnavailableException('FIXTURE_DIR not set'),
        ),
      );
      await until(() => viewModel.startLive.result != null);

      expect(viewModel.startError, 'FIXTURE_DIR not set');
      expect(sources, isEmpty);
      viewModel.dispose();
    },
  );

  test('Retry after a live failure starts a new source', () async {
    final viewModel = create();
    await until(() => live.state.value is LiveRunning);
    await live.stop(owner: viewModel); // e.g. after a pipeline failure

    await viewModel.retryStart();

    expect(sources, hasLength(2));
    expect(live.state.value, isA<LiveRunning>());
    viewModel.dispose();
  });

  test('leaving stops live detection, but not a newer owner\'s', () async {
    final first = create();
    await until(() => live.state.value is LiveRunning);
    first.dispose();
    await until(() => live.state.value is LiveStopped);
    expect(sources.single.stopped, isTrue);

    final second = create();
    await until(() => live.state.value is LiveRunning);
    final third = create(); // takes over, then the second leaves late
    await until(() => sources.length == 3 && live.state.value is LiveRunning);
    second.dispose();
    await pumpEventQueue();
    expect(live.state.value, isA<LiveRunning>(), reason: 'owner token');
    third.dispose();
  });

  test('a voice question about the live frame: question, spoken answer, '
      'route chip and overlay metrics; no LLM turn', () async {
    final viewModel = create();
    await until(() => live.state.value is LiveRunning);
    // Five live frames for the summary (the fake detector sees one cat).
    for (var i = 0; i < 5; i++) {
      await liveFrame();
    }
    expect(live.recent, hasLength(5), reason: 'none dropped by the gate');

    await viewModel.pressMic();
    await releaseWithSnapshot(viewModel);

    expect(viewModel.exchange.question, 'How many cats do you see?');
    expect(viewModel.exchange.answer, 'I count one cat.');
    expect(viewModel.exchange.route, 'cat — detector, no LLM');
    expect(viewModel.exchange.detailed, isFalse);
    final metrics = diagnostics.latest.lastCameraTurn!;
    expect(metrics.route.rule, 'count');
    expect(metrics.snapshotLatency, isNotNull);
    expect(metrics.basis, 'cat');
    expect(conversation.prompts, isEmpty, reason: 'no LLM');
    viewModel.dispose();
  });

  test('without a running camera the answer says so and the notice shows '
      'why', () async {
    final viewModel = create(
      source: const Result.error(
        FrameSourceUnavailableException('FIXTURE_DIR not set'),
      ),
    );
    await until(() => viewModel.startLive.result != null);

    await viewModel.pressMic();
    await viewModel.releaseMic();

    expect(viewModel.exchange.answer, CameraTurnResponder.cameraNotRunning);
    expect(viewModel.exchange.notice, "The camera isn't running");
    viewModel.dispose();
  });

  test('a detailed question without a frame keeps its route chip and is '
      'recorded for the overlay', () async {
    recognizer.text = 'What color is the cat?';
    final viewModel = create(
      source: const Result.error(
        FrameSourceUnavailableException('FIXTURE_DIR not set'),
      ),
    );
    await until(() => viewModel.startLive.result != null);

    await viewModel.pressMic();
    await viewModel.releaseMic();

    expect(viewModel.exchange.question, 'What color is the cat?');
    expect(viewModel.exchange.detailed, isTrue);
    expect(viewModel.exchange.route, contains('detailed (detail:color)'));
    expect(viewModel.exchange.notice, "The camera isn't running");
    expect(viewModel.exchange.answer, CameraTurnResponder.cameraNotRunning);
    final metrics = diagnostics.latest.lastCameraTurn;
    expect(metrics, isNotNull, reason: 'the overlay records the turn');
    expect(metrics!.route, isA<DetailedRoute>());
    expect(metrics.snapshotError, "The camera isn't running");
    viewModel.dispose();
  });

  group('detailed path', () {
    Future<LiveCameraViewModel> running() async {
      final viewModel = create();
      await until(() => live.state.value is LiveRunning);
      return viewModel;
    }

    /// Presses and releases the mic and delivers the snapshot frame; the
    /// turn goes on.
    Future<void> ask(LiveCameraViewModel viewModel) async {
      await viewModel.pressMic();
      final asked = captures;
      unawaited(viewModel.releaseMic());
      await until(() => captures > asked);
      sources.single.emit(); // the frame after release is the snapshot
    }

    test('entering makes moonshine the active recognizer, in parallel with '
        'the chat', () async {
      final viewModel = await running();
      expect(sttActivations, [ModelId.moonshineTiny]);
      expect(viewModel.sttError, isNull);
      viewModel.dispose();
    });

    test(
      'a failed recognizer switch is shown and Retry runs it again',
      () async {
        sttResult = Result.error(Exception('moonshine missing'));
        final viewModel = await running();
        expect(viewModel.sttError, contains('moonshine missing'));
        sttResult = const Result.ok(null);
        await viewModel.selectStt.execute();
        expect(viewModel.sttError, isNull);
        expect(sttActivations, hasLength(2));
        viewModel.dispose();
      },
    );

    test('a detailed question freezes the view on its own snapshot (the '
        'frame sent to Gemma) until the answer has been spoken', () async {
      recognizer.text = 'What color is the cat?';
      final viewModel = await running();
      await ask(viewModel);
      await until(() => viewModel.frozen.value != null);

      final frozen = viewModel.frozen.value!;
      await until(() => conversation.prompts.isNotEmpty);
      expect(viewModel.lastSentImage?.frameId, frozen.frameId);
      expect(frozen.detections.frameId, frozen.frameId);
      expect((frozen.image.width, frozen.image.height), (640, 480));
      expect(frozen.flip, isFalse);
      expect(viewModel.frozenBoxes.value, same(frozen.detections));
      expect(viewModel.exchange.detailed, isTrue);
      expect(viewModel.exchange.route, contains('frame #${frozen.frameId}'));
      expect(diagnostics.latest.frozenFrameId, frozen.frameId);

      conversation.emit('The cat is grey.');
      await pumpEventQueue();
      expect(viewModel.frozen.value, isNotNull, reason: 'still answering');
      await conversation.finish();
      await until(() => viewModel.phase == TurnPhase.idle);

      expect(viewModel.frozen.value, isNull, reason: 'spoken: live again');
      expect(viewModel.exchange.answer, 'The cat is grey.');
      expect(diagnostics.latest.frozenFrameId, isNull);
      final metrics = diagnostics.latest.lastCameraTurn!;
      expect(metrics.image?.frameId, frozen.frameId);
      expect(metrics.timeToFirstToken, isNotNull);
      await until(() => diagnostics.latest.lastCameraReset != null);
      expect(conversation.resetCalls, 1);
      viewModel.dispose();
    });

    test(
      'pressing the mic during the answer (barge-in) unfreezes at once',
      () async {
        recognizer.text = 'Describe the scene.';
        final viewModel = await running();
        await ask(viewModel);
        await until(() => viewModel.frozen.value != null);
        await until(() => conversation.prompts.isNotEmpty);
        conversation.emit('Two cats ');

        final pressing = viewModel.pressMic();
        expect(viewModel.frozen.value, isNull, reason: 'before any await');
        await pressing;
        viewModel.dispose();
      },
    );

    test('a tap unfreezes; the answer goes on', () async {
      recognizer.text = 'Describe the scene.';
      final viewModel = await running();
      await ask(viewModel);
      await until(() => viewModel.frozen.value != null);
      await until(() => conversation.prompts.isNotEmpty);

      viewModel.unfreeze();
      expect(viewModel.frozen.value, isNull);
      conversation.emit('Two cats on a sofa.');
      await conversation.finish();
      await until(() => viewModel.phase == TurnPhase.idle);
      expect(viewModel.exchange.answer, 'Two cats on a sofa.');
      viewModel.dispose();
    });

    test('images off: the label says detailed answers '
        'are off; a detailed question answers from the detections, never '
        'freezes, never sends the frame', () async {
      conversation.capabilities = const ChatCapabilities(
        modelName: 'Gemma 3 NPU',
        images: false,
        tools: false,
      );
      recognizer.text = 'What color is the cat?';
      final viewModel = await running();
      expect(viewModel.detailedEnabled, isFalse);
      expect(viewModel.detailedOffReason, contains('Gemma 3 NPU'));
      expect(viewModel.detailedOffReason, contains('detections only'));

      await ask(viewModel);
      await until(() => viewModel.phase == TurnPhase.idle);

      expect(viewModel.frozen.value, isNull);
      expect(conversation.prompts, isEmpty);
      expect(viewModel.lastSentImage, isNull);
      expect(
        viewModel.exchange.answer,
        startsWith(CameraTurnResponder.imagesOff),
      );
      expect(viewModel.exchange.route, contains('detailed answers off'));
      expect(viewModel.exchange.detailed, isFalse);
      viewModel.dispose();
    });

    test('a fast question never freezes', () async {
      final viewModel = await running();
      for (var i = 0; i < 3; i++) {
        await liveFrame();
      }
      await ask(viewModel);
      await until(() => viewModel.phase == TurnPhase.idle);
      expect(viewModel.exchange.answer, startsWith('I count'));
      expect(viewModel.frozen.value, isNull);
      expect(conversation.prompts, isEmpty);
      viewModel.dispose();
    });
  });

  test('black frames: only macOS blames the terminal', () {
    expect(blackFramesHint(TargetPlatform.macOS), contains('Privacy'));
    for (final p in [TargetPlatform.iOS, TargetPlatform.android]) {
      expect(blackFramesHint(p), isNot(contains('Privacy')));
    }
  });
}
