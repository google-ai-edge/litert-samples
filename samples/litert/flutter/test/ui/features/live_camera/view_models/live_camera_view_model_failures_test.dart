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

// Demo 3's failure and barge-in paths the main view model suite does not
// reach: "The answer failed" notice (also on a second turn, where a failure
// before the question must not show the first turn's exchange with it), and
// the late side events of a turn a barge-in already replaced (its metrics
// and its chat reset still count; it never touches the screen). Past
// reviews found notices lost in exactly this barge-in case.
import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter_edge_ai_speech/flutter_edge_ai_speech.dart'
    show VoiceResponder;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/repositories/audio_repository.dart';
import 'package:litert_edge_demos/data/repositories/diagnostics_repository.dart';
import 'package:litert_edge_demos/data/repositories/live_detection_repository.dart';
import 'package:litert_edge_demos/data/repositories/speech_repository.dart';
import 'package:litert_edge_demos/domain/models/camera_side_event.dart';
import 'package:litert_edge_demos/domain/models/detection.dart';
import 'package:litert_edge_demos/domain/models/detection_summary.dart';
import 'package:litert_edge_demos/domain/models/frame_source_spec.dart';
import 'package:litert_edge_demos/domain/models/live_state.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/domain/models/route_decision.dart';
import 'package:litert_edge_demos/domain/models/scene_snapshot.dart';
import 'package:litert_edge_demos/domain/models/voice.dart';
import 'package:litert_edge_demos/domain/use_cases/camera_turn_responder.dart';
import 'package:litert_edge_demos/domain/use_cases/turn_responder_factory.dart';
import 'package:litert_edge_demos/domain/use_cases/voice_assistant.dart';
import 'package:litert_edge_demos/ui/features/live_camera/view_models/live_camera_view_model.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../../../fakes/fake_audio_repository.dart';
import '../../../../fakes/fake_conversation_repository.dart';
import '../../../../fakes/fake_detector.dart';
import '../../../../fakes/fake_frame_source.dart';
import '../../../../fakes/fake_speech.dart';
import '../../../../support/pcm.dart';

Future<void> settle() => pumpEventQueue();

const _fixture = Result<FrameSourceSpec>.ok(FixtureSourceSpec(['/fixtures']));

/// A responder factory whose turns the test plays by hand: each turn's
/// reply is [_ScriptedTurn.reply], and its side channel is kept so the test
/// can send that turn's facts at any time (also after a barge-in replaced
/// it).
class _ScriptedTurns implements TurnResponderFactory<CameraSideEvent> {
  final List<_ScriptedTurn> turns = [];

  @override
  TurnPreparation<CameraSideEvent> prepare(TurnRequest request) {
    final turn = _ScriptedTurn();
    turns.add(turn);
    return turn;
  }
}

class _ScriptedTurn implements TurnPreparation<CameraSideEvent> {
  final StreamController<String> reply = StreamController();
  void Function(CameraSideEvent event)? onSide;
  bool stopped = false;

  void side(CameraSideEvent event) => onSide!(event);

  @override
  VoiceResponder responder(void Function(CameraSideEvent event) onSide) {
    this.onSide = onSide;
    return VoiceResponder(
      respond: (_) => reply.stream,
      stop: () async {
        stopped = true;
        await reply.close();
      },
    );
  }

  @override
  void discard() {}
}

/// The real responders, except that every turn after the first cannot be
/// prepared: that turn fails before its question is known.
class _FailingAfterFirst implements TurnResponderFactory<CameraSideEvent> {
  _FailingAfterFirst(this._inner);

  final TurnResponderFactory<CameraSideEvent> _inner;
  int _turns = 0;

  @override
  TurnPreparation<CameraSideEvent> prepare(TurnRequest request) {
    if (++_turns > 1) throw StateError('no responder');
    return _inner.prepare(request);
  }
}

SceneSnapshot _snapshot(int frameId) {
  final frame = DetectionFrame(
    frameId: frameId,
    width: 640,
    height: 480,
    boxes: Float32List.fromList([10, 20, 110, 220, 0.9, 15]),
    preMicros: 1000,
    runMicros: 4000,
    postMicros: 1000,
    backend: DetectorBackend.gpu,
  );
  return SceneSnapshot(
    frameId: frameId,
    detections: frame,
    summary: DetectionSummary.fromWindow([frame], minScore: 0.4),
    pixels: RgbaPixels(width: 2, height: 1, bytes: Uint8List(8)),
    mirrored: false,
    latency: const Duration(milliseconds: 14),
  );
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  late FakeConversationRepository conversation;
  late List<FakeFrameSource> sources;
  late LiveDetectionRepository live;
  late SpeechRepository speech;
  late ValueNotifier<Map<ModelId, ModelState>> models;
  late DiagnosticsRepository diagnostics;
  late FakeRecognizer recognizer;
  late FakeAudioRepository audio;

  setUp(() async {
    audio = FakeAudioRepository()..autoDrain = true;
    recognizer = FakeRecognizer('How many cats do you see?');
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
    live = LiveDetectionRepository(
      detector: FakeDetector(autoComplete: true),
      createSource: (_) {
        final s = FakeFrameSource();
        sources.add(s);
        return Result.ok(s);
      },
    );
  });

  tearDown(() async {
    await live.close();
    await conversation.close();
    diagnostics.dispose();
    models.dispose();
  });

  /// Demo 3 over the real assistant; [responders] replaces the real
  /// CameraTurnResponder.
  Future<LiveCameraViewModel> running({
    TurnResponderFactory<CameraSideEvent>? responders,
  }) async {
    final viewModel = LiveCameraViewModel(
      conversation: conversation,
      live: live,
      assistant: VoiceAssistant(
        speech: speech,
        audio: audio,
        responders:
            responders ??
            CameraTurnResponder(
              capture: live.capture,
              conversation: conversation,
              encode: live.encodeForLlm,
            ),
        diagnostics: diagnostics,
      ),
      diagnostics: diagnostics,
      activateStt: (_) async => const Result.ok(null),
      source: _fixture,
    );
    await settle();
    expect(live.state.value, isA<LiveRunning>());
    return viewModel;
  }

  /// Presses and releases the mic and delivers the snapshot frame.
  Future<void> ask(LiveCameraViewModel viewModel) async {
    await viewModel.pressMic();
    final releasing = viewModel.releaseMic();
    await settle();
    sources.single.emit(); // the frame after release is the snapshot
    unawaited(releasing);
  }

  Future<void> until(bool Function() condition) async {
    for (var i = 0; i < 50 && !condition(); i++) {
      await settle();
    }
    expect(condition(), isTrue);
  }

  group('"The answer failed"', () {
    test('a detailed answer that fails mid-reply: the notice says so, the '
        'question, the route chip and the partial answer stay, the view '
        'unfreezes, the error phase; the chat is still reset', () async {
      recognizer.text = 'What color is the cat?';
      final viewModel = await running();
      await ask(viewModel);
      await until(() => viewModel.frozen.value != null);
      await until(() => conversation.prompts.isNotEmpty);
      final frameId = viewModel.frozen.value!.frameId;
      conversation.emit('The cat is ');
      await settle();

      await conversation.fail(Exception('GPU lost'));
      await until(() => viewModel.phase == TurnPhase.error);

      final exchange = viewModel.exchange;
      expect(exchange.notice, 'The answer failed: Exception: GPU lost');
      expect(exchange.question, 'What color is the cat?');
      expect(exchange.route, contains('frame #$frameId'));
      expect(exchange.detailed, isTrue);
      expect(exchange.answer, 'The cat is', reason: 'what the user saw');
      expect(viewModel.frozen.value, isNull, reason: 'live again');
      expect(diagnostics.latest.frozenFrameId, isNull);
      await until(() => diagnostics.latest.lastCameraReset != null);
      expect(conversation.resetCalls, 1);
      expect(viewModel.chatError, isNull);
      viewModel.dispose();
    });

    test(
      'a recognizer failure (no question yet) shows only the notice',
      () async {
        recognizer.error = StateError('stt boom');
        final viewModel = await running();

        await viewModel.pressMic();
        await viewModel.releaseMic();

        expect(viewModel.phase, TurnPhase.error);
        final exchange = viewModel.exchange;
        expect(exchange.notice, 'The answer failed: Bad state: stt boom');
        expect(exchange.question, isNull);
        expect(exchange.answer, isNull);
        expect(conversation.prompts, isEmpty);
        viewModel.dispose();
      },
    );
  });

  group('"The answer failed" on a second turn (the first one answered)', () {
    /// The real responders (what [running] builds without `responders`).
    CameraTurnResponder realResponders() => CameraTurnResponder(
      capture: live.capture,
      conversation: conversation,
      encode: live.encodeForLlm,
    );

    /// A first turn answered from the detections, to the end: its question,
    /// route chip and answer are on screen when the next one starts.
    Future<void> answerFirstTurn(LiveCameraViewModel viewModel) async {
      await viewModel.pressMic();
      final releasing = viewModel.releaseMic();
      await settle();
      sources.single.emit(); // the frame after release is the snapshot
      await releasing;
      expect(viewModel.phase, TurnPhase.idle);
      expect(viewModel.exchange.question, 'How many cats do you see?');
      expect(viewModel.exchange.route, contains('detector, no LLM'));
      expect(viewModel.exchange.answer, isNotNull);
    }

    void expectOnlyTheNotice(LiveCameraViewModel viewModel, String notice) {
      expect(viewModel.phase, TurnPhase.error);
      final exchange = viewModel.exchange;
      expect(exchange.notice, notice);
      expect(exchange.question, isNull, reason: "the first turn's question");
      expect(exchange.route, isNull, reason: "the first turn's route chip");
      expect(exchange.detailed, isFalse);
      expect(exchange.answer, isNull, reason: "the first turn's answer");
    }

    test('a recognizer failure (no question yet) shows only the notice, not '
        "the first turn's question, route chip or answer", () async {
      final viewModel = await running();
      await answerFirstTurn(viewModel);

      recognizer.error = StateError('stt boom');
      await viewModel.pressMic();
      await viewModel.releaseMic();

      expectOnlyTheNotice(viewModel, 'The answer failed: Bad state: stt boom');
      viewModel.dispose();
    });

    test('a turn that cannot start (its responder) shows only the notice, '
        "not the first turn's", () async {
      final viewModel = await running(
        responders: _FailingAfterFirst(realResponders()),
      );
      await answerFirstTurn(viewModel);

      await viewModel.pressMic();
      await viewModel.releaseMic();

      expectOnlyTheNotice(
        viewModel,
        'The answer failed: Bad state: no responder',
      );
      viewModel.dispose();
    });

    test('a submitted utterance that fails in the recognizer shows only the '
        'notice', () async {
      final viewModel = await running();
      await answerFirstTurn(viewModel);

      recognizer.error = StateError('stt boom');
      await viewModel.submitUtterance(speechUtterance());

      expectOnlyTheNotice(viewModel, 'The answer failed: Bad state: stt boom');
      viewModel.dispose();
    });

    /// The second turn asks a detailed question; returns once its frame
    /// went to Gemma, with that frame's id.
    Future<int> askDetailed(LiveCameraViewModel viewModel) async {
      recognizer.text = 'What color is the cat?';
      await ask(viewModel);
      await until(() => viewModel.frozen.value != null);
      await until(() => conversation.prompts.isNotEmpty);
      return viewModel.frozen.value!.frameId;
    }

    test('a reply that fails partway keeps its own question, route chip and '
        "partial answer with the notice, never the first turn's", () async {
      final viewModel = await running();
      await answerFirstTurn(viewModel);
      final firstAnswer = viewModel.exchange.answer;

      final frameId = await askDetailed(viewModel);
      conversation.emit('The cat is ');
      await settle();
      await conversation.fail(Exception('GPU lost'));
      await until(() => viewModel.phase == TurnPhase.error);

      final exchange = viewModel.exchange;
      expect(exchange.notice, 'The answer failed: Exception: GPU lost');
      expect(exchange.question, 'What color is the cat?');
      expect(exchange.route, contains('frame #$frameId'));
      expect(exchange.detailed, isTrue);
      expect(exchange.answer, 'The cat is');
      expect(exchange.answer, isNot(firstAnswer));
      viewModel.dispose();
    });

    test('a reply that fails before its first word keeps its question and '
        "route chip, without an answer (not the first turn's)", () async {
      final viewModel = await running();
      await answerFirstTurn(viewModel);

      final frameId = await askDetailed(viewModel);
      await conversation.fail(Exception('GPU lost'));
      await until(() => viewModel.phase == TurnPhase.error);

      final exchange = viewModel.exchange;
      expect(exchange.notice, 'The answer failed: Exception: GPU lost');
      expect(exchange.question, 'What color is the cat?');
      expect(exchange.route, contains('frame #$frameId'));
      expect(exchange.answer, isNull);
      viewModel.dispose();
    });
  });

  group('a barged-in turn (its late side events are detached)', () {
    /// A detailed answer streaming with the view frozen, then the barge-in.
    /// [beforeBargeIn] runs once the answer streams.
    Future<LiveCameraViewModel> bargeInDuringDetailed({
      void Function()? beforeBargeIn,
    }) async {
      recognizer.text = 'Describe the scene.';
      final viewModel = await running();
      await ask(viewModel);
      await until(() => viewModel.frozen.value != null);
      await until(() => conversation.prompts.isNotEmpty);
      conversation.emit('Two cats ');
      await settle();
      beforeBargeIn?.call();
      await viewModel.pressMic(); // the barge-in
      return viewModel;
    }

    test("the stopped generation's metrics and its chat reset still reach "
        'the overlay; the screen keeps the interrupted answer and the new '
        'press', () async {
      final viewModel = await bargeInDuringDetailed();
      final cameraTurn = diagnostics.latest.lastCameraTurn;

      await until(() => diagnostics.latest.lastCameraReset != null);

      final generation = diagnostics.latest.lastGeneration;
      expect(generation, isNotNull, reason: 'recorded although detached');
      expect(generation!.stopped, isTrue);
      expect(diagnostics.latest.cameraResetError, isNull);
      expect(conversation.resetCalls, 1);
      expect(viewModel.chatError, isNull);
      expect(
        diagnostics.latest.lastCameraTurn,
        same(cameraTurn),
        reason: "a detached turn never rewrites the overlay's camera turn",
      );
      expect(viewModel.isListening, isTrue);
      expect(viewModel.frozen.value, isNull);
      expect(viewModel.exchange.question, 'Describe the scene.');
      expect(viewModel.exchange.answer, 'Two cats');
      await viewModel.stop.execute();
      viewModel.dispose();
    });

    test("its failed chat reset is not lost: the chat error shows (and "
        'notifies) while the new press listens; Retry reopens', () async {
      // The detached turn's reset reopens the chat; that reopen fails.
      final viewModel = await bargeInDuringDetailed(
        beforeBargeIn: () => conversation
          ..openGate = Completer<void>()
          ..openResult = Result.error(Exception('no memory')),
      );
      await until(() => conversation.resetCalls == 1);
      expect(viewModel.chatError, isNull, reason: 'the reset still runs');
      final exchange = viewModel.exchange;
      var notified = 0;
      viewModel.addListener(() => notified++);

      conversation.openGate!.complete();
      await until(() => diagnostics.latest.cameraResetError != null);

      expect(
        viewModel.chatError,
        'Resetting the camera chat failed: Exception: no memory',
      );
      expect(diagnostics.latest.cameraResetError, contains('no memory'));
      expect(notified, greaterThanOrEqualTo(1), reason: 'the view rebuilds');
      expect(viewModel.isListening, isTrue, reason: 'the new press goes on');
      expect(viewModel.exchange, same(exchange), reason: 'no screen change');

      conversation
        ..openGate = null
        ..openResult = const Result.ok(null);
      await viewModel.open.execute();
      expect(viewModel.chatError, isNull);
      expect(viewModel.chatReady, isTrue);
      await viewModel.stop.execute();
      viewModel.dispose();
    });

    test(
      'every late fact of a replaced turn: never a freeze, a sent image, '
      'an exchange or camera-turn change, or a rebuild; its generation '
      'metrics are recorded; its reset sets and clears the chat error',
      () async {
        final turns = _ScriptedTurns();
        final viewModel = await running(responders: turns);
        await viewModel.pressMic();
        unawaited(viewModel.releaseMic());
        await until(() => viewModel.phase == TurnPhase.thinking);
        final replaced = turns.turns.single;
        expect(viewModel.exchange.question, 'How many cats do you see?');

        await viewModel.pressMic(); // the barge-in
        await settle();
        expect(replaced.stopped, isTrue);
        expect(viewModel.isListening, isTrue);
        final exchange = viewModel.exchange;
        expect(exchange.answer, '(stopped)');
        final cameraTurn = diagnostics.latest.lastCameraTurn;
        var notified = 0;
        viewModel.addListener(() => notified++);

        final snapshot = _snapshot(7);
        replaced
          ..side(
            const RouteChosen(
              DetailedRoute('detail:scene'),
              Duration(milliseconds: 1),
            ),
          )
          ..side(SnapshotTaken(snapshot))
          ..side(
            FrameSentToGemma(
              EncodedSnapshot(
                png: Uint8List.fromList([0x89, 0x50, 0x4E, 0x47]),
                frameId: 7,
                width: 640,
                height: 480,
                unmirrored: false,
                encodeTime: const Duration(milliseconds: 9),
              ),
            ),
          )
          ..side(const SnapshotFailed("The camera isn't running"))
          ..side(const FastAnswered(answer: 'I count one cat.', basis: 'cat'))
          ..side(
            const DetailedUnavailable(
              reason: 'off',
              answer: 'I see a cat.',
              basis: 'cat',
            ),
          )
          ..side(const CameraChatUnavailable('The camera chat is not open'))
          ..side(const TranscriptCorrected('Is there a cup?', []));
        await settle();

        expect(viewModel.frozen.value, isNull, reason: 'never frozen');
        expect(diagnostics.latest.frozenFrameId, isNull);
        expect(viewModel.lastSentImage, isNull);
        expect(viewModel.exchange, same(exchange));
        expect(diagnostics.latest.lastCameraTurn, same(cameraTurn));
        expect(notified, 0);

        final metrics = FakeConversationRepository.metrics(stopped: true);
        replaced.side(DetailedAnswered(metrics));
        expect(diagnostics.latest.lastGeneration, same(metrics));
        expect(diagnostics.latest.lastCameraTurn, same(cameraTurn));
        expect(notified, 0, reason: 'metrics only');

        replaced.side(
          const CameraChatReset(
            elapsed: Duration(milliseconds: 30),
            error: 'reset boom',
          ),
        );
        expect(
          viewModel.chatError,
          'Resetting the camera chat failed: reset boom',
        );
        expect(diagnostics.latest.cameraResetError, 'reset boom');
        expect(
          diagnostics.latest.lastCameraReset,
          const Duration(milliseconds: 30),
        );
        expect(notified, 1);

        replaced.side(
          const CameraChatReset(elapsed: Duration(milliseconds: 20)),
        );
        expect(viewModel.chatError, isNull);
        expect(notified, 2);
        expect(viewModel.exchange, same(exchange));

        await viewModel.stop.execute();
        viewModel.dispose();
      },
    );
  });

  // The live (not detached) side events and freeze branches the groups above
  // do not reach, pinned before the projection moved into
  // CameraExchangeReducer.
  group('a live turn played by hand', () {
    const detailed = RouteChosen(
      DetailedRoute('detail:scene'),
      Duration(milliseconds: 1),
    );

    /// Presses and releases the mic over scripted responders; returns the
    /// turn once its question is on screen.
    Future<(LiveCameraViewModel, _ScriptedTurn)> started() async {
      final turns = _ScriptedTurns();
      final viewModel = await running(responders: turns);
      await viewModel.pressMic();
      unawaited(viewModel.releaseMic());
      await until(() => viewModel.phase == TurnPhase.thinking);
      expect(viewModel.exchange.question, 'How many cats do you see?');
      return (viewModel, turns.turns.single);
    }

    SceneSnapshot mirroredPreview(int frameId) {
      final snapshot = _snapshot(frameId);
      return SceneSnapshot(
        frameId: frameId,
        detections: snapshot.detections,
        summary: snapshot.summary,
        pixels: snapshot.pixels,
        mirrored: false,
        previewMirrored: true,
        latency: snapshot.latency,
      );
    }

    test('a camera chat that is not open: the notice joins the question and '
        'the detailed route chip; the overlay records the chat error; the '
        'committed answer keeps the notice', () async {
      final (viewModel, turn) = await started();
      turn.side(detailed);
      expect(
        viewModel.exchange.route,
        'detailed (detail:scene) — ${conversation.capabilities.modelName}',
      );
      expect(viewModel.exchange.detailed, isTrue);

      turn.side(const CameraChatUnavailable('The camera chat is not open'));
      var exchange = viewModel.exchange;
      expect(exchange.question, 'How many cats do you see?');
      expect(exchange.route, startsWith('detailed (detail:scene) — '));
      expect(exchange.detailed, isTrue);
      expect(exchange.notice, 'The camera chat is not open');
      expect(exchange.answer, isNull);
      final metrics = diagnostics.latest.lastCameraTurn!;
      expect(metrics.chatError, 'The camera chat is not open');
      expect(metrics.route, isA<DetailedRoute>());
      expect(metrics.routeTime, const Duration(milliseconds: 1));

      turn.reply.add('The chat is not ready yet.');
      await turn.reply.close();
      await until(() => viewModel.phase == TurnPhase.idle);
      exchange = viewModel.exchange;
      expect(exchange.answer, 'The chat is not ready yet.');
      expect(exchange.notice, 'The camera chat is not open');
      expect(exchange.route, startsWith('detailed (detail:scene) — '));
      expect(exchange.detailed, isTrue);
      viewModel.dispose();
    });

    test('a fast route alone changes nothing on screen; its answer sets the '
        'fast route chip', () async {
      final (viewModel, turn) = await started();
      final before = viewModel.exchange;
      turn.side(
        const RouteChosen(
          FastRoute(FastIntent.count, 'count', cls: 15),
          Duration(milliseconds: 2),
        ),
      );
      expect(viewModel.exchange, same(before), reason: 'a fast route alone');
      turn.side(const FastAnswered(answer: 'I count one cat.', basis: 'cat'));
      expect(viewModel.exchange.route, 'cat — detector, no LLM');
      expect(viewModel.exchange.detailed, isFalse);
      expect(diagnostics.latest.lastCameraTurn?.basis, 'cat');
      await viewModel.stop.execute();
      viewModel.dispose();
    });

    test("a detailed turn's snapshot freezes the view once decoded (flipped "
        'when only the preview is mirrored); a newer snapshot replaces it; '
        'Stop unfreezes', () async {
      final (viewModel, turn) = await started();
      turn
        ..side(detailed)
        ..side(SnapshotTaken(_snapshot(7)));
      expect(viewModel.frozen.value, isNull, reason: 'still decoding');
      expect(
        diagnostics.latest.lastCameraTurn?.snapshotLatency,
        const Duration(milliseconds: 14),
      );
      await until(() => viewModel.frozen.value != null);
      expect(viewModel.frozen.value!.frameId, 7);
      expect(viewModel.frozen.value!.flip, isFalse);
      expect(viewModel.frozenBoxes.value?.frameId, 7);
      expect(diagnostics.latest.frozenFrameId, 7);

      turn.side(SnapshotTaken(mirroredPreview(8)));
      await until(() => viewModel.frozen.value?.frameId == 8);
      expect(viewModel.frozen.value!.flip, isTrue);
      expect(viewModel.frozenBoxes.value?.frameId, 8);
      expect(diagnostics.latest.frozenFrameId, 8);

      await viewModel.stop.execute();
      expect(viewModel.frozen.value, isNull);
      expect(viewModel.frozenBoxes.value, isNull);
      expect(diagnostics.latest.frozenFrameId, isNull);
      viewModel.dispose();
    });

    test('a snapshot still decoding when the view unfreezes (a tap) is '
        'dropped', () async {
      final (viewModel, turn) = await started();
      turn
        ..side(detailed)
        ..side(SnapshotTaken(_snapshot(7)));
      viewModel.unfreeze();
      for (var i = 0; i < 20; i++) {
        await settle();
      }
      expect(viewModel.frozen.value, isNull);
      expect(viewModel.frozenBoxes.value, isNull);
      expect(diagnostics.latest.frozenFrameId, isNull);
      await viewModel.stop.execute();
      viewModel.dispose();
    });

    test('a mic that cannot open after an answered turn: only its message '
        'shows', () async {
      final viewModel = await running();
      await viewModel.pressMic();
      final releasing = viewModel.releaseMic();
      await settle();
      sources.single.emit(); // the frame after release is the snapshot
      await releasing;
      expect(viewModel.exchange.answer, isNotNull);

      audio.startError = const MicAccessException('Microphone access is off');
      await viewModel.pressMic();
      await until(() => viewModel.exchange.notice != null);
      final exchange = viewModel.exchange;
      expect(exchange.notice, 'Microphone access is off');
      expect(exchange.question, isNull);
      expect(exchange.answer, isNull);
      expect(exchange.route, isNull);
      viewModel.dispose();
    });
  });
}
