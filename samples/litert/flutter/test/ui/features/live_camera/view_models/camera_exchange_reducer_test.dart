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

import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/domain/models/assistant_event.dart';
import 'package:litert_edge_demos/domain/models/camera_side_event.dart';
import 'package:litert_edge_demos/domain/models/chat_capabilities.dart';
import 'package:litert_edge_demos/domain/models/detection.dart';
import 'package:litert_edge_demos/domain/models/detection_summary.dart';
import 'package:litert_edge_demos/domain/models/route_decision.dart';
import 'package:litert_edge_demos/domain/models/scene_snapshot.dart';
import 'package:litert_edge_demos/domain/models/voice.dart';
import 'package:litert_edge_demos/domain/use_cases/voice_assistant.dart';
import 'package:litert_edge_demos/domain/vision/stt_corrections.dart';
import 'package:litert_edge_demos/ui/features/live_camera/view_models/camera_exchange_reducer.dart';

typedef _Event = VoiceAssistantEvent<CameraSideEvent>;

const _reducer = CameraExchangeReducer();

const _gemma = ChatCapabilities(
  modelName: 'Gemma 4 E2B',
  images: true,
  tools: true,
);

const _imagesOff = ChatCapabilities(
  modelName: 'Gemma 3 NPU',
  images: false,
  tools: false,
);

const _detailed = RouteChosen(
  DetailedRoute('detail:color'),
  Duration(milliseconds: 3),
);

const _fast = RouteChosen(
  FastRoute(FastIntent.count, 'count', cls: 15),
  Duration(milliseconds: 2),
);

_Event _side(CameraSideEvent event) => SideEvent(event, detached: false);

_Event _detachedSide(CameraSideEvent event) => SideEvent(event, detached: true);

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

EncodedSnapshot _png(int frameId) => EncodedSnapshot(
  png: Uint8List.fromList([0x89, 0x50, 0x4E, 0x47]),
  frameId: frameId,
  width: 1024,
  height: 576,
  unmirrored: false,
  encodeTime: const Duration(milliseconds: 9),
);

GenerationMetrics _metrics({Duration? ttft}) => GenerationMetrics(
  timeToFirstToken: ttft,
  chunks: 2,
  tokensPerSecond: 20,
  tokensPerSecondSource: TokenRateSource.chunks,
  total: const Duration(milliseconds: 300),
  stopped: false,
);

/// Runs [events] from [state] with [chat]; returns every update.
List<CameraExchangeUpdate> _run(
  List<_Event> events, {
  CameraExchangeState state = const CameraExchangeState(),
  ChatCapabilities chat = _gemma,
}) {
  final updates = <CameraExchangeUpdate>[];
  var current = state;
  for (final event in events) {
    final update = _reducer.reduce(current, event, chat);
    updates.add(update);
    current = update.state;
  }
  return updates;
}

/// The state after [events].
CameraExchangeState _after(
  List<_Event> events, {
  CameraExchangeState state = const CameraExchangeState(),
  ChatCapabilities chat = _gemma,
}) => _run(
  events,
  state: state,
  chat: chat,
).fold(state, (_, update) => update.state);

/// A turn whose question arrived: pending, then the question.
CameraExchangeState _asked(
  String question, {
  CameraExchangeState state = const CameraExchangeState(),
}) => _after([
  UserSaid(question, typed: false),
], state: _reducer.turnStarted(state));

void _expectNoEffects(CameraExchangeUpdate update) {
  expect(update.unfreeze, isFalse);
  expect(update.generation, isNull);
  expect(update.turn, isNull);
  expect(update.freeze, isNull);
  expect(update.chatReset, isNull);
}

void main() {
  group('turn start and question', () {
    test('a started turn has its question pending; the screen keeps the '
        'previous exchange', () {
      final before = _asked('How many cats?');
      final started = _reducer.turnStarted(before);
      expect(started.questionPending, isTrue);
      expect(started.exchange, same(before.exchange));
    });

    test('the question replaces the exchange and starts the turn facts over; '
        'the last frame sent to Gemma stays', () {
      final previous = _after([
        _side(_detailed),
        _side(SnapshotTaken(_snapshot(4))),
        _side(FrameSentToGemma(_png(4))),
        _side(DetailedAnswered(_metrics(ttft: const Duration(seconds: 1)))),
        const AssistantSaid('A grey cat.', interrupted: false),
      ], state: _asked('What color is the cat?'));

      final update = _reducer.reduce(
        _reducer.turnStarted(previous),
        const UserSaid('How many cats?', typed: false),
        _gemma,
      );
      final state = update.state;
      expect(state.exchange.question, 'How many cats?');
      expect(state.exchange.answer, isNull);
      expect(state.exchange.route, isNull);
      expect(state.exchange.detailed, isFalse);
      expect(state.exchange.notice, isNull);
      expect(state.questionPending, isFalse);
      expect(state.route, isNull);
      expect(state.snapshotLatency, isNull);
      expect(state.turnImage, isNull);
      expect(state.turnTtft, isNull);
      expect(state.sentImage?.frameId, 4);
      expect(update.notify, isTrue);
      _expectNoEffects(update);
    });
  });

  group('the answer', () {
    test('committed: joins the question, route chip and notice; leaves the '
        'frozen frame', () {
      final asked = _after([
        _side(_detailed),
        _side(const SnapshotFailed("The camera isn't running")),
      ], state: _asked('What color is the cat?'));
      final update = _reducer.reduce(
        asked,
        const AssistantSaid('I cannot see it.', interrupted: false),
        _gemma,
      );
      final exchange = update.state.exchange;
      expect(exchange.question, 'What color is the cat?');
      expect(exchange.answer, 'I cannot see it.');
      expect(exchange.route, 'detailed (detail:color) — Gemma 4 E2B');
      expect(exchange.detailed, isTrue);
      expect(exchange.notice, "The camera isn't running");
      expect(update.unfreeze, isTrue);
      expect(update.notify, isTrue);
    });

    test('stopped before any text: "(stopped)"; stopped partway: the partial '
        'text', () {
      final asked = _asked('Describe the scene.');
      expect(
        _after([
          const AssistantSaid('', interrupted: true),
        ], state: asked).exchange.answer,
        '(stopped)',
      );
      expect(
        _after([
          const AssistantSaid('Two cats', interrupted: true),
        ], state: asked).exchange.answer,
        'Two cats',
      );
      expect(
        _after([
          const AssistantSaid('', interrupted: false),
        ], state: asked).exchange.answer,
        '',
      );
    });
  });

  group('no LLM call', () {
    test('not heard: only the notice, for every reason; nothing unfreezes', () {
      final answered = _after([
        const AssistantSaid('I count one cat.', interrupted: false),
      ], state: _asked('How many cats?'));
      for (final (reason, text) in [
        (
          NotHeardReason.tooShort,
          "Didn't catch that — hold the mic button while you speak.",
        ),
        (
          NotHeardReason.releasedBeforeListening,
          'The mic was still opening — wait for Listening… before you speak.',
        ),
        (NotHeardReason.silent, "Didn't catch that — it was too quiet."),
        (
          NotHeardReason.emptyTranscript,
          "Didn't catch that — no words were recognized.",
        ),
      ]) {
        final update = _reducer.reduce(
          _reducer.turnStarted(answered),
          NotHeard(reason),
          _gemma,
        );
        final exchange = update.state.exchange;
        expect(exchange.notice, text, reason: reason.name);
        expect(exchange.question, isNull);
        expect(exchange.answer, isNull);
        expect(exchange.route, isNull);
        expect(update.notify, isTrue);
        _expectNoEffects(update);
      }
    });

    test('a mic that gives no usable audio: only its message', () {
      final update = _reducer.reduce(
        _asked('How many cats?'),
        const MicUnavailable('Microphone access is off'),
        _gemma,
      );
      expect(update.state.exchange.notice, 'Microphone access is off');
      expect(update.state.exchange.question, isNull);
      _expectNoEffects(update);
    });
  });

  group('"The answer failed"', () {
    test('before its question (pending): only the notice, never the previous '
        "turn's exchange", () {
      final answered = _after([
        _side(_fast),
        _side(const FastAnswered(answer: 'I count one cat.', basis: 'cat')),
        const AssistantSaid('I count one cat.', interrupted: false),
      ], state: _asked('How many cats?'));

      final update = _reducer.reduce(
        _reducer.turnStarted(answered),
        TurnFailed(StateError('stt boom')),
        _gemma,
      );
      final exchange = update.state.exchange;
      expect(exchange.notice, 'The answer failed: Bad state: stt boom');
      expect(exchange.question, isNull);
      expect(exchange.answer, isNull);
      expect(exchange.route, isNull);
      expect(exchange.detailed, isFalse);
      expect(update.unfreeze, isTrue);
      expect(update.notify, isTrue);
    });

    test("mid-reply: this turn's question, route chip and partial answer stay "
        'with the notice', () {
      final update = _run([
        _side(_detailed),
        _side(SnapshotTaken(_snapshot(7))),
        _side(FrameSentToGemma(_png(7))),
        const AssistantSaid('The cat is', interrupted: false),
        TurnFailed(Exception('GPU lost')),
      ], state: _asked('What color is the cat?')).last;
      final exchange = update.state.exchange;
      expect(exchange.notice, 'The answer failed: Exception: GPU lost');
      expect(exchange.question, 'What color is the cat?');
      expect(
        exchange.route,
        'detailed (detail:color) — frame #7 1024×576 → Gemma',
      );
      expect(exchange.detailed, isTrue);
      expect(exchange.answer, 'The cat is');
      expect(update.unfreeze, isTrue);
    });
  });

  group('a live turn', () {
    test('a corrected transcript shows what was heard', () {
      final update = _reducer.reduce(
        _asked('Is there a cop?'),
        _side(
          const TranscriptCorrected('Is there a cup?', [
            SttCorrection('cop', 'cup'),
          ]),
        ),
        _gemma,
      );
      expect(update.state.exchange.question, "Is there a cup? (heard 'cop')");
      _expectNoEffects(update);
    });

    test('a detailed route shows its chip with the chat model; a fast one, '
        'or a detailed one with images off, changes nothing on screen', () {
      final asked = _asked('What color is the cat?');
      final detailed = _reducer.reduce(asked, _side(_detailed), _gemma);
      expect(detailed.state.route, same(_detailed));
      expect(
        detailed.state.exchange.route,
        'detailed (detail:color) — Gemma 4 E2B',
      );
      expect(detailed.state.exchange.detailed, isTrue);
      expect(detailed.state.exchange.question, 'What color is the cat?');
      _expectNoEffects(detailed);

      final fast = _reducer.reduce(asked, _side(_fast), _gemma);
      expect(fast.state.route, same(_fast));
      expect(fast.state.exchange, same(asked.exchange));
      expect(fast.notify, isTrue);

      final off = _reducer.reduce(asked, _side(_detailed), _imagesOff);
      expect(off.state.route, same(_detailed));
      expect(off.state.exchange, same(asked.exchange));
    });

    test("only a detailed turn with images freezes on its snapshot, and "
        'records the turn with the latency', () {
      final snapshot = _snapshot(7);
      final detailed = _run([
        _side(_detailed),
        _side(SnapshotTaken(snapshot)),
      ], state: _asked('What color is the cat?')).last;
      expect(detailed.freeze, same(snapshot));
      expect(detailed.turn?.route, _detailed.route);
      expect(detailed.turn?.routeTime, const Duration(milliseconds: 3));
      expect(detailed.turn?.snapshotLatency, const Duration(milliseconds: 14));
      expect(detailed.unfreeze, isFalse);

      for (final (name, events, chat) in [
        ('fast', [_side(_fast)], _gemma),
        ('images off', [_side(_detailed)], _imagesOff),
        ('no route', <_Event>[], _gemma),
      ]) {
        final update = _run(
          [...events, _side(SnapshotTaken(snapshot))],
          state: _asked('Question?'),
          chat: chat,
        ).last;
        expect(update.freeze, isNull, reason: name);
        expect(update.turn, isNull, reason: name);
        expect(
          update.state.snapshotLatency,
          const Duration(milliseconds: 14),
          reason: name,
        );
      }
    });

    test('a frame that could not be captured keeps the route chip with the '
        'notice and is recorded', () {
      final update = _run([
        _side(_detailed),
        _side(const SnapshotFailed("The camera isn't running")),
      ], state: _asked('What color is the cat?')).last;
      final exchange = update.state.exchange;
      expect(exchange.route, 'detailed (detail:color) — Gemma 4 E2B');
      expect(exchange.detailed, isTrue);
      expect(exchange.notice, "The camera isn't running");
      expect(update.turn?.snapshotError, "The camera isn't running");
      expect(update.freeze, isNull);
    });

    test('a fast answer and a detailed one with images off: the chip names '
        'the basis; the turn records it', () {
      final fast = _run([
        _side(_fast),
        _side(const FastAnswered(answer: 'I count two cats.', basis: 'cat ×2')),
      ], state: _asked('How many cats?')).last;
      expect(fast.state.exchange.route, 'cat ×2 — detector, no LLM');
      expect(fast.state.exchange.detailed, isFalse);
      expect(fast.state.exchange.question, 'How many cats?');
      expect(fast.turn?.basis, 'cat ×2');

      final off = _run(
        [
          _side(_detailed),
          _side(
            const DetailedUnavailable(
              reason: 'off',
              answer: 'I see a cat.',
              basis: 'cat',
            ),
          ),
        ],
        state: _asked('What color is the cat?'),
        chat: _imagesOff,
      ).last;
      expect(
        off.state.exchange.route,
        'cat — detector only (detailed answers off)',
      );
      expect(off.state.exchange.detailed, isFalse);
      expect(off.turn?.basis, 'cat');
    });

    test('a camera chat that is not open keeps the chip with the notice and '
        'records the chat error', () {
      final update = _run([
        _side(_detailed),
        _side(const CameraChatUnavailable('The camera chat is not open')),
      ], state: _asked('What color is the cat?')).last;
      final exchange = update.state.exchange;
      expect(exchange.route, 'detailed (detail:color) — Gemma 4 E2B');
      expect(exchange.detailed, isTrue);
      expect(exchange.notice, 'The camera chat is not open');
      expect(update.turn?.chatError, 'The camera chat is not open');
    });

    test('the frame sent to Gemma: the chip names it, the notice stays, it is '
        'the sent image and the turn records it', () {
      final update = _run([
        _side(_detailed),
        _side(const SnapshotFailed('slow')),
        _side(FrameSentToGemma(_png(9))),
      ], state: _asked('What color is the cat?')).last;
      final state = update.state;
      expect(
        state.exchange.route,
        'detailed (detail:color) — frame #9 1024×576 → Gemma',
      );
      expect(state.exchange.detailed, isTrue);
      expect(state.exchange.notice, 'slow');
      expect(state.sentImage?.frameId, 9);
      expect(state.turnImage?.frameId, 9);
      expect(update.turn?.image?.frameId, 9);
    });

    test('a frame sent before any route: rule "?", nothing recorded', () {
      final update = _reducer.reduce(
        _asked('Question?'),
        _side(FrameSentToGemma(_png(2))),
        _gemma,
      );
      expect(
        update.state.exchange.route,
        startsWith('detailed (?) — frame #2'),
      );
      expect(update.turn, isNull);
    });

    test("Gemma's figures: recorded, with the time to first token on the "
        'turn; a later answer without one clears it', () {
      final first = _metrics(ttft: const Duration(milliseconds: 120));
      final updates = _run([
        _side(_detailed),
        _side(DetailedAnswered(first)),
        _side(DetailedAnswered(_metrics())),
      ], state: _asked('What color is the cat?'));
      expect(updates[1].generation, same(first));
      expect(
        updates[1].turn?.timeToFirstToken,
        const Duration(milliseconds: 120),
      );
      expect(updates[1].state.exchange, same(updates[0].state.exchange));
      expect(updates[2].turn?.timeToFirstToken, isNull);
      expect(updates[2].state.turnTtft, isNull);
    });

    test('the chat reset: recorded, the screen keeps its exchange', () {
      final asked = _asked('Question?');
      const reset = CameraChatReset(
        elapsed: Duration(milliseconds: 30),
        error: 'reset boom',
      );
      final update = _reducer.reduce(asked, _side(reset), _gemma);
      expect(update.chatReset, same(reset));
      expect(update.state, same(asked));
      expect(update.notify, isTrue);
    });
  });

  group('a turn a barge-in replaced (detached)', () {
    test('its late facts never touch the screen, freeze, record a turn or '
        'rebuild', () {
      final asked = _asked('Describe the scene.');
      for (final event in <CameraSideEvent>[
        const TranscriptCorrected('Is there a cup?', []),
        _detailed,
        SnapshotTaken(_snapshot(7)),
        const SnapshotFailed("The camera isn't running"),
        const FastAnswered(answer: 'I count one cat.', basis: 'cat'),
        const DetailedUnavailable(reason: 'off', answer: 'x', basis: 'cat'),
        const CameraChatUnavailable('The camera chat is not open'),
        FrameSentToGemma(_png(7)),
      ]) {
        final update = _reducer.reduce(asked, _detachedSide(event), _gemma);
        expect(update.state, same(asked), reason: '$event');
        expect(update.notify, isFalse, reason: '$event');
        _expectNoEffects(update);
      }
    });

    test('its generation figures are recorded without a rebuild', () {
      final asked = _asked('Describe the scene.');
      final metrics = _metrics();
      final update = _reducer.reduce(
        asked,
        _detachedSide(DetailedAnswered(metrics)),
        _gemma,
      );
      expect(update.generation, same(metrics));
      expect(update.turn, isNull);
      expect(update.state, same(asked));
      expect(update.notify, isFalse);
    });

    test('its chat reset is recorded and rebuilds (the chat error)', () {
      final asked = _asked('Describe the scene.');
      const reset = CameraChatReset(elapsed: Duration(milliseconds: 20));
      final update = _reducer.reduce(asked, _detachedSide(reset), _gemma);
      expect(update.chatReset, same(reset));
      expect(update.state, same(asked));
      expect(update.notify, isTrue);
    });
  });

  test('the frozen frame is left when the turn ends (idle, error), not while '
      'the mic opens, listens or the answer runs', () {
    for (final phase in TurnPhase.values) {
      expect(
        _reducer.unfreezesAt(phase),
        phase == TurnPhase.idle || phase == TurnPhase.error,
        reason: phase.name,
      );
    }
  });
}
