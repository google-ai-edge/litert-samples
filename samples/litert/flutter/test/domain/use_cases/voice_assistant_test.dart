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
import 'dart:typed_data';

import 'package:fake_async/fake_async.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_edge_ai_speech/flutter_edge_ai_speech.dart'
    show VoiceResponder;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/config/voice_config.dart';
import 'package:litert_edge_demos/data/repositories/audio_repository.dart';
import 'package:litert_edge_demos/data/repositories/diagnostics_repository.dart';
import 'package:litert_edge_demos/domain/models/chat_side_event.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/domain/models/voice.dart';
import 'package:litert_edge_demos/domain/use_cases/chat_turn_responder.dart';
import 'package:litert_edge_demos/domain/use_cases/turn_responder_factory.dart';
import 'package:litert_edge_demos/domain/use_cases/voice_assistant.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../fakes/fake_audio_repository.dart';
import '../../fakes/fake_conversation_repository.dart';
import '../../fakes/fake_speech.dart';
import '../../support/pcm.dart';

/// Logs `conversation.stop` into the shared log, so tests can check that
/// playback stops before the model is asked to.
class _LoggingConversation extends FakeConversationRepository {
  _LoggingConversation(this.log);

  final List<String> log;

  @override
  Future<void> stop() {
    log.add('conversation.stop');
    return super.stop();
  }
}

/// Wraps a factory and logs the release hook into the shared log.
class _RecordingFactory<S> implements TurnResponderFactory<S> {
  _RecordingFactory(this._inner, this.log);

  final TurnResponderFactory<S> _inner;
  final List<String> log;

  final List<TurnRequest> requests = [];

  @override
  TurnPreparation<S> prepare(TurnRequest request) {
    requests.add(request);
    log.add('prepare(typed: ${request.typed})');
    final inner = _inner.prepare(request);
    return _RecordingPreparation(inner, log);
  }
}

class _RecordingPreparation<S> implements TurnPreparation<S> {
  _RecordingPreparation(this._inner, this.log);

  final TurnPreparation<S> _inner;
  final List<String> log;

  @override
  VoiceResponder responder(void Function(S event) onSide) {
    log.add('responder');
    return _inner.responder(onSide);
  }

  @override
  void discard() {
    log.add('discard');
    _inner.discard();
  }
}

/// The real assistant, the real SpeechRepository and VoiceSession, the real
/// ChatTurnResponder; fakes only at the edges (models, chat, audio I/O).
class _Harness {
  final List<String> log = [];
  late final _LoggingConversation conversation = _LoggingConversation(log);
  late final FakeAudioRepository audio = FakeAudioRepository(log: log);
  final FakeRecognizer recognizer = FakeRecognizer();
  final RecordingSynth synth = RecordingSynth();
  final ValueNotifier<Map<ModelId, ModelState>> models = ValueNotifier(
    const {},
  );
  late final DiagnosticsRepository diagnostics = DiagnosticsRepository(
    models: models,
    minInterval: Duration.zero,
    sampleRss: false,
  );
  late VoiceAssistant<ChatSideEvent> assistant;
  late _RecordingFactory<ChatSideEvent> factory;
  final List<VoiceAssistantEvent<ChatSideEvent>> events = [];
  final List<TurnPhase> phases = [];
  StreamSubscription<VoiceAssistantEvent<ChatSideEvent>>? _sub;

  Future<void> init() async {
    // The demo opens the chat before any turn; the turns here ask it.
    conversation.leaveOpen(kVoiceChatProfile);
    final speech = await loadedSpeech(
      recognizer: recognizer,
      synthesizer: synth,
    );
    assistant = VoiceAssistant(
      speech: speech,
      audio: audio,
      responders: factory = _RecordingFactory(
        ChatTurnResponder(conversation: conversation),
        log,
      ),
      diagnostics: diagnostics,
    );
    _sub = assistant.events.listen(events.add);
    assistant.phase.addListener(() => phases.add(assistant.phase.value));
  }

  TurnPhase get phase => assistant.phase.value;

  List<String> get beginPlaybacks =>
      log.where((e) => e.startsWith('beginPlayback')).toList();

  Future<void> dispose() async {
    await _sub?.cancel();
    await assistant.dispose();
    diagnostics.dispose();
    models.dispose();
    await conversation.close();
  }
}

/// Lets VoiceSession's chain of awaits run.
Future<void> settle() => pumpEventQueue();

Matcher _said(String text, {bool interrupted = false}) =>
    isA<AssistantSaid<ChatSideEvent>>()
        .having((e) => e.text, 'text', text)
        .having((e) => e.interrupted, 'interrupted', interrupted);

void main() {
  late _Harness h;

  setUp(() async {
    h = _Harness();
    await h.init();
  });

  tearDown(() => h.dispose());

  test('voice turn: phases in order, audio from the first clause, one '
      'beginPlayback at 24 kHz, idle only after the audio drained', () async {
    await h.assistant.micDown();
    expect(h.phase, TurnPhase.listening);

    var done = false;
    final turn = h.assistant.micUp();
    unawaited(turn.then((_) => done = true));
    await settle();
    expect(h.recognizer.calls, 1);
    expect(h.conversation.prompts, ['What is the capital of France?']);
    expect(h.phase, TurnPhase.thinking);

    h.conversation.emit('Paris is the capital of France. ');
    await settle();
    expect(h.synth.synthesized, ['Paris is the capital of France.']);
    expect(h.beginPlaybacks, ['beginPlayback(24000)']);
    expect(h.phase, TurnPhase.speaking);
    expect(h.assistant.partialReply.value, 'Paris is the capital of France. ');

    h.conversation.emit('It is lovely.');
    await h.conversation.finish();
    await settle();
    expect(h.synth.synthesized.last, 'It is lovely.');
    expect(h.audio.lastPlayback!.ended, isTrue);
    expect(h.audio.lastPlayback!.chunks, hasLength(2));
    expect(done, isFalse, reason: 'waits for the audio to drain');
    expect(h.phase, TurnPhase.speaking);

    h.audio.lastPlayback!.completeDrain();
    final result = await turn;

    expect(result.outcome, TurnOutcome.completed);
    expect(h.phase, TurnPhase.idle);
    expect(h.beginPlaybacks, hasLength(1));
    expect(h.phases, [
      TurnPhase.openingMic,
      TurnPhase.listening,
      TurnPhase.transcribing,
      TurnPhase.thinking,
      TurnPhase.speaking,
      TurnPhase.idle,
    ]);
    expect(
      h.events.whereType<UserSaid<ChatSideEvent>>().single.text,
      'What is the capital of France?',
    );
    expect(
      h.events.whereType<AssistantSaid<ChatSideEvent>>().single,
      _said('Paris is the capital of France. It is lovely.'),
    );
    expect(h.events.whereType<SideEvent<ChatSideEvent>>(), hasLength(1));
    final metrics = h.diagnostics.latest.lastVoiceTurn!;
    expect(metrics.typed, isFalse);
    expect(metrics.stt, isNotNull);
    expect(metrics.firstAudio, isNotNull);
    expect(metrics.sampleRate, 24000);
    expect(metrics.ttsClauses, hasLength(2));
    expect(metrics.outcome, TurnOutcome.completed);
  });

  group('a slow mic start (the audio warm-up still running)', () {
    /// The fake's hold clock, moved by hand: a capture's `held` counts from
    /// its start, like the real repository's.
    var now = Duration.zero;

    setUp(() {
      now = Duration.zero;
      h.audio
        ..clock = (() => now)
        ..startGate = Completer<void>();
    });

    test('a press shows openingMic until the capture runs, then listening; '
        'the overlay follows', () async {
      final down = h.assistant.micDown();
      await settle();
      expect(
        h.phase,
        TurnPhase.openingMic,
        reason: 'nothing is recorded yet: not listening',
      );
      expect(h.diagnostics.latest.voicePhase, TurnPhase.openingMic);
      expect(h.assistant.isBusy, isTrue);

      now += const Duration(milliseconds: 2600); // a cold macOS output start
      h.audio.startGate!.complete();
      expect((await down), isA<Ok<void>>());

      expect(h.phase, TurnPhase.listening);
      expect(h.diagnostics.latest.voicePhase, TurnPhase.listening);
      expect(h.phases, [TurnPhase.openingMic, TurnPhase.listening]);
      await h.assistant.stop();
    });

    test('a release while the mic opens records nothing: the capture is '
        'cancelled once it starts, no STT, and the user is told to wait for '
        'Listening', () async {
      final down = h.assistant.micDown();
      await settle();
      now += const Duration(milliseconds: 2600); // held through the start
      final up = h.assistant.micUp();
      await settle();
      expect(h.phase, TurnPhase.openingMic, reason: 'still starting');

      h.audio.startGate!.complete();
      await down;
      final result = await up;

      expect(result.outcome, TurnOutcome.notHeard);
      expect(
        h.events.single,
        isA<NotHeard<ChatSideEvent>>().having(
          (e) => e.reason,
          'reason',
          NotHeardReason.releasedBeforeListening,
        ),
      );
      final capture = h.audio.captures.single;
      expect(capture.cancelled, isTrue, reason: 'not left open');
      expect(capture.stopped, isFalse, reason: 'nothing to transcribe');
      expect(h.log, isNot(contains('prepare(typed: false)')));
      expect(h.recognizer.calls, 0);
      expect(h.conversation.prompts, isEmpty);
      expect(h.phases, [TurnPhase.openingMic, TurnPhase.idle]);
      expect(h.diagnostics.latest.lastVoiceTurn!.outcome, TurnOutcome.notHeard);
    });

    test('two slips before the start: one wait-for-Listening notice, and the '
        'capture is cancelled', () async {
      final firstDown = h.assistant.micDown();
      await settle();
      final firstUp = h.assistant.micUp();
      await settle();
      unawaited(h.assistant.micDown());
      await settle();
      final secondUp = h.assistant.micUp();
      await settle();

      h.audio.startGate!.complete();
      await firstDown;
      expect((await firstUp).outcome, TurnOutcome.superseded);
      expect((await secondUp).outcome, TurnOutcome.notHeard);
      expect(
        h.events.single,
        isA<NotHeard<ChatSideEvent>>().having(
          (e) => e.reason,
          'reason',
          NotHeardReason.releasedBeforeListening,
        ),
      );
      expect(h.audio.captures.single.cancelled, isTrue);
      expect(h.phases, [TurnPhase.openingMic, TurnPhase.idle]);
    });

    test('a press again while the slipped press still opens takes the start '
        'over: no notice for the slip, one capture, Listening, and its '
        'release runs the turn', () async {
      h.audio.autoDrain = true;
      final firstDown = h.assistant.micDown();
      await settle();
      final slip = h.assistant.micUp(); // the finger slips…
      await settle();
      final secondDown = h.assistant.micDown(); // …and presses again
      await settle();
      expect(h.phase, TurnPhase.openingMic);

      now += const Duration(milliseconds: 2600);
      h.audio.startGate!.complete();
      await firstDown;
      await secondDown;
      expect((await slip).outcome, TurnOutcome.superseded);
      expect(h.phase, TurnPhase.listening, reason: 'the press is held');
      expect(h.events, isEmpty, reason: 'no notice for the slip');

      now += const Duration(seconds: 1);
      final turn = h.assistant.micUp();
      await settle();
      expect(h.recognizer.calls, 1);
      h.conversation.emit('Paris.');
      await h.conversation.finish();
      expect((await turn).outcome, TurnOutcome.completed);

      expect(h.log.where((e) => e == 'startCapture'), hasLength(1));
      expect(h.audio.captures.single.stopped, isTrue);
      expect(h.events.whereType<NotHeard<ChatSideEvent>>(), isEmpty);
      expect(h.phases.take(3), [
        TurnPhase.openingMic,
        TurnPhase.listening,
        TurnPhase.transcribing,
      ]);
    });

    test('the hold counts from the capture start, not from the press: '
        '200 ms after a 2.6 s start is too short', () async {
      final down = h.assistant.micDown();
      await settle();
      now += const Duration(milliseconds: 2600);
      h.audio.startGate!.complete();
      await down;
      now += const Duration(milliseconds: 200); // < minUtterance (300 ms)

      final result = await h.assistant.micUp();

      expect(result.outcome, TurnOutcome.notHeard);
      expect(
        h.events.single,
        isA<NotHeard<ChatSideEvent>>().having(
          (e) => e.reason,
          'reason',
          NotHeardReason.tooShort,
        ),
      );
      expect(h.recognizer.calls, 0);
    });

    test('a barge-in silences the reply at the press, before the mic '
        'opens', () async {
      h.audio.startGate = null;
      final first = h.assistant.sendText('Tell me a story');
      await settle();
      h.conversation.emit('The first sentence is here. ');
      await settle();
      expect(h.phase, TurnPhase.speaking);
      final playback = h.audio.lastPlayback!;

      h.audio.startGate = Completer<void>();
      final down = h.assistant.micDown();

      // Synchronously at the press: no await has run yet.
      expect(playback.stopped, isTrue);
      expect(h.phase, TurnPhase.openingMic);
      expect(
        h.log.indexOf('playback.stop'),
        lessThan(h.log.indexOf('startCapture')),
      );
      final bargeIn = h.diagnostics.latest.lastBargeIn!;
      expect(bargeIn.wasPlaying, isTrue);
      expect(bargeIn.silenced, isNotNull);
      expect((await first).outcome, TurnOutcome.superseded);
      expect(
        h.events.whereType<AssistantSaid<ChatSideEvent>>().single,
        _said('The first sentence is here.', interrupted: true),
      );
      await settle();
      expect(h.phase, TurnPhase.openingMic, reason: 'the mic still opens');

      h.audio.startGate!.complete();
      await down;
      expect(h.phase, TurnPhase.listening);
      await h.assistant.stop();
    });

    test('a mic that fails after the slow start: MicUnavailable and the '
        'error phase, never listening', () async {
      h.audio.startError = const MicAccessException('Microphone access is off');
      final down = h.assistant.micDown();
      await settle();
      expect(h.phase, TurnPhase.openingMic);

      h.audio.startGate!.complete();
      expect(await down, isA<Error<void>>());

      expect(h.events.single, isA<MicUnavailable<ChatSideEvent>>());
      expect(h.phases, [TurnPhase.openingMic, TurnPhase.error]);
      expect(
        h.diagnostics.latest.lastVoiceTurn!.outcome,
        TurnOutcome.micUnavailable,
      );
      expect((await h.assistant.micUp()).outcome, TurnOutcome.ignored);
    });

    test('Stop while the mic opens: idle at once, and the capture that '
        'starts afterwards is cancelled', () async {
      final down = h.assistant.micDown();
      await settle();

      await h.assistant.stop();
      expect(h.phase, TurnPhase.idle);

      h.audio.startGate!.complete();
      await down;
      expect(h.audio.captures.single.cancelled, isTrue);
      expect(h.phases, [TurnPhase.openingMic, TurnPhase.idle]);
    });

    test('a slip, then the start fails: one MicUnavailable, the release is superseded', () async {
      h.audio.startError = const MicAccessException('off');
      final down = h.assistant.micDown();
      await settle();
      final up = h.assistant.micUp();
      await settle();
      h.audio.startGate!.complete();
      expect(await down, isA<Error<void>>());
      expect((await up).outcome, TurnOutcome.superseded);
      expect(h.events.single, isA<MicUnavailable<ChatSideEvent>>());
      expect(h.phases, [TurnPhase.openingMic, TurnPhase.error]);
      expect(
        h.diagnostics.latest.lastVoiceTurn!.outcome,
        TurnOutcome.micUnavailable,
      );
    });

    test('a re-press, then the start fails: the held press gets MicUnavailable and the error phase', () async {
      h.audio.startError = const MicAccessException('off');
      final down = h.assistant.micDown();
      await settle();
      final up = h.assistant.micUp();
      await settle();
      final down2 = h.assistant.micDown();
      await settle();
      h.audio.startGate!.complete();
      expect(await down, isA<Error<void>>());
      expect(await down2, isA<Ok<void>>());
      expect((await up).outcome, TurnOutcome.superseded);
      expect(h.events.single, isA<MicUnavailable<ChatSideEvent>>());
      expect(h.phase, TurnPhase.error);
      expect((await h.assistant.micUp()).outcome, TurnOutcome.ignored);
      expect(h.events, hasLength(1));
    });

    test('two slips, then the start fails: one notice', () async {
      h.audio.startError = const MicAccessException('off');
      final down = h.assistant.micDown();
      await settle();
      final up = h.assistant.micUp();
      await settle();
      unawaited(h.assistant.micDown());
      await settle();
      final up2 = h.assistant.micUp();
      await settle();
      h.audio.startGate!.complete();
      await down;
      expect((await up).outcome, TurnOutcome.superseded);
      expect((await up2).outcome, TurnOutcome.superseded);
      expect(h.events.single, isA<MicUnavailable<ChatSideEvent>>());
      expect(h.phase, TurnPhase.error);
    });

    test('a slip, then Stop: nothing left open, no notice', () async {
      final down = h.assistant.micDown();
      await settle();
      final up = h.assistant.micUp();
      await settle();
      await h.assistant.stop();
      h.audio.startGate!.complete();
      await down;
      expect((await up).outcome, TurnOutcome.superseded);
      expect(h.events, isEmpty);
      expect(h.audio.captures.single.cancelled, isTrue);
      expect(h.phases, [TurnPhase.openingMic, TurnPhase.idle]);
    });

    test(
      'a re-press, then Stop: nothing left open, the late release is ignored',
      () async {
        final down = h.assistant.micDown();
        await settle();
        final up = h.assistant.micUp();
        await settle();
        unawaited(h.assistant.micDown());
        await settle();
        await h.assistant.stop();
        h.audio.startGate!.complete();
        await down;
        expect((await up).outcome, TurnOutcome.superseded);
        expect((await h.assistant.micUp()).outcome, TurnOutcome.ignored);
        expect(h.events, isEmpty);
        expect(h.audio.captures.single.cancelled, isTrue);
        expect(h.phase, TurnPhase.idle);
      },
    );

    test('a slip, then dispose: the capture is cancelled, no event', () async {
      final down = h.assistant.micDown();
      await settle();
      final up = h.assistant.micUp();
      await settle();
      final disposing = h.assistant.dispose();
      h.audio.startGate!.complete();
      await down;
      await disposing;
      expect((await up).outcome, TurnOutcome.superseded);
      expect(h.events, isEmpty);
      expect(h.audio.captures.single.cancelled, isTrue);
    });

    test('two slips and a held press, then the start succeeds: listening, no notice', () async {
      final down = h.assistant.micDown();
      await settle();
      final up1 = h.assistant.micUp();
      await settle();
      unawaited(h.assistant.micDown());
      await settle();
      final up2 = h.assistant.micUp();
      await settle();
      unawaited(h.assistant.micDown());
      await settle();
      h.audio.startGate!.complete();
      await down;
      expect((await up1).outcome, TurnOutcome.superseded);
      expect((await up2).outcome, TurnOutcome.superseded);
      expect(h.phase, TurnPhase.listening);
      expect(h.events, isEmpty);
      expect(h.audio.captures.single.isOpen, isTrue);
      await h.assistant.stop();
    });
  });

  test('a zero-byte first clause does not start playback or the speaking '
      'phase; the first real chunk does, once', () async {
    h.synth.emptyOn = '(music)';
    final turn = h.assistant.sendText('Play something');
    await settle();
    h.conversation.emit('(music) plays softly. ');
    await settle();
    expect(h.beginPlaybacks, isEmpty);
    expect(h.phase, TurnPhase.thinking);

    h.conversation.emit('Here is a real sentence. ');
    await settle();
    expect(h.beginPlaybacks, ['beginPlayback(24000)']);
    expect(h.phase, TurnPhase.speaking);

    h.audio.autoDrain = true;
    await h.conversation.finish();
    expect((await turn).outcome, TurnOutcome.completed);
    expect(h.beginPlaybacks, hasLength(1));
  });

  group('gate: no STT and no LLM call', () {
    test('a silent capture is "not heard"', () async {
      h.audio.nextUtterance = Utterance(
        pcm: quiet(const Duration(seconds: 1)),
        held: const Duration(seconds: 1),
      );
      await h.assistant.micDown();
      final result = await h.assistant.micUp();

      expect(result.outcome, TurnOutcome.notHeard);
      expect(
        h.events.single,
        isA<NotHeard<ChatSideEvent>>().having(
          (e) => e.reason,
          'reason',
          NotHeardReason.silent,
        ),
      );
      expect(h.recognizer.calls, 0);
      expect(h.conversation.prompts, isEmpty);
      expect(h.phase, TurnPhase.idle);
    });

    test('a single click is not speech', () async {
      h.audio.nextUtterance = Utterance(
        pcm: click(const Duration(seconds: 1)),
        held: const Duration(seconds: 1),
      );
      await h.assistant.micDown();
      final result = await h.assistant.micUp();

      expect(result.outcome, TurnOutcome.notHeard);
      expect(h.recognizer.calls, 0);
      expect(h.conversation.prompts, isEmpty);
    });

    test('constant room noise above the gate is not speech', () async {
      h.audio.nextUtterance = Utterance(
        pcm: noise(const Duration(seconds: 2)),
        held: const Duration(seconds: 2),
      );
      await h.assistant.micDown();
      final result = await h.assistant.micUp();

      expect(result.outcome, TurnOutcome.notHeard);
      expect(h.recognizer.calls, 0);
    });

    test('speech over room noise passes, and the gate figures '
        'reach the overlay', () async {
      final pcm = BytesBuilder()
        ..add(noise(const Duration(milliseconds: 500)))
        ..add(tone(const Duration(seconds: 1)))
        ..add(noise(const Duration(milliseconds: 500)));
      h.audio
        ..nextUtterance = Utterance(
          pcm: pcm.takeBytes(),
          held: const Duration(seconds: 2),
        )
        ..autoDrain = true;
      await h.assistant.micDown();
      final turn = h.assistant.micUp();
      await settle();
      expect(h.recognizer.calls, 1);
      h.conversation.emit('Paris.');
      await h.conversation.finish();
      await turn;
      final m = h.diagnostics.latest.lastVoiceTurn!;
      expect(m.peakDbfs, closeTo(-13.5, 1));
      expect(m.gateDbfs, isNotNull);
      expect(m.voiced!.inMilliseconds, closeTo(1000, 40));
    });

    test('a too-short press is "not heard", even when loud', () async {
      h.audio.nextUtterance = Utterance(
        pcm: tone(const Duration(milliseconds: 200)),
        held: const Duration(milliseconds: 200),
      );
      await h.assistant.micDown();
      final result = await h.assistant.micUp();

      expect(result.outcome, TurnOutcome.notHeard);
      expect(
        (h.events.single as NotHeard<ChatSideEvent>).reason,
        NotHeardReason.tooShort,
      );
      expect(h.recognizer.calls, 0);
      expect(h.conversation.prompts, isEmpty);
    });

    test('an all-zero capture is a mic-access error, not "Didn\'t catch '
        'that" (macOS TCC hands a blocked app digital silence)', () async {
      h.audio.nextUtterance = Utterance(
        pcm: zeros(const Duration(seconds: 2)),
        held: const Duration(seconds: 2),
      );
      await h.assistant.micDown();
      final result = await h.assistant.micUp();

      expect(result.outcome, TurnOutcome.micUnavailable);
      expect(result.error, isA<MicAccessException>());
      final event = h.events.single;
      expect(event, isA<MicUnavailable<ChatSideEvent>>());
      expect(
        (event as MicUnavailable<ChatSideEvent>).message,
        allOf(
          contains('Microphone access is off for this app'),
          // The settings path of the platform under test (Android has its
          // own; test/config/voice_config_test.dart).
          contains(kMicAccessMessage),
          contains('digital silence'),
        ),
      );
      expect(h.events.whereType<NotHeard<ChatSideEvent>>(), isEmpty);
      expect(h.recognizer.calls, 0);
      expect(h.conversation.prompts, isEmpty);
      expect(h.phase, TurnPhase.error);
    });

    test(
      'a press that delivered no bytes at all is a mic-access error too',
      () async {
        h.audio.nextUtterance = Utterance(
          pcm: zeros(Duration.zero),
          held: const Duration(seconds: 1),
        );
        await h.assistant.micDown();
        final result = await h.assistant.micUp();

        expect(result.outcome, TurnOutcome.micUnavailable);
        expect(
          (h.events.single as MicUnavailable<ChatSideEvent>).message,
          contains('no audio arrived'),
        );
      },
    );

    test('an empty transcript is "not heard" with no LLM call', () async {
      h.recognizer.text = '  ';
      await h.assistant.micDown();
      final result = await h.assistant.micUp();

      expect(result.outcome, TurnOutcome.notHeard);
      expect(h.recognizer.calls, 1);
      expect(
        (h.events.single as NotHeard<ChatSideEvent>).reason,
        NotHeardReason.emptyTranscript,
      );
      expect(h.conversation.prompts, isEmpty);
      expect(h.phase, TurnPhase.idle);
    });

    test('permission denied at mic-down is a mic-access error', () async {
      h.audio.startError = const MicAccessException('Microphone access is off');
      final result = await h.assistant.micDown();

      expect(result, isA<Error<void>>());
      expect(h.events.single, isA<MicUnavailable<ChatSideEvent>>());
      expect(h.phase, TurnPhase.error);
      expect((await h.assistant.micUp()).outcome, TurnOutcome.ignored);
      expect(h.conversation.prompts, isEmpty);
    });
  });

  test('barge-in while thinking: the partial reply is committed as '
      'interrupted, the mic opens at once, and the next ask waits for the '
      'stopped turn to end', () async {
    await h.assistant.micDown();
    final first = h.assistant.micUp();
    await settle();
    h.conversation.emit('Once upon a time');
    await settle();
    expect(h.phase, TurnPhase.thinking);
    expect(h.beginPlaybacks, isEmpty, reason: 'no clause finished yet');

    // The stopped turn takes a while to drain.
    h.conversation.stopGate = Completer<void>();
    h.recognizer.text = 'Second question?';
    await h.assistant.micDown();

    expect(h.phase, TurnPhase.listening);
    expect((await first).outcome, TurnOutcome.superseded);
    expect(
      h.events.whereType<AssistantSaid<ChatSideEvent>>().single,
      _said('Once upon a time', interrupted: true),
    );
    expect(h.assistant.partialReply.value, isEmpty);
    expect(h.log, contains('conversation.stop'));

    final second = h.assistant.micUp();
    await settle();
    expect(
      h.conversation.prompts,
      hasLength(1),
      reason: 'the next turn waits for the interrupted one',
    );
    expect(h.recognizer.calls, 1);

    h.conversation.stopGate!.complete();
    await settle();
    expect(h.conversation.prompts, [
      'What is the capital of France?',
      'Second question?',
    ]);
    expect(h.conversation.isGenerating.value, isTrue);

    h.audio.autoDrain = true;
    h.conversation.emit('Answer two.');
    await h.conversation.finish();
    expect((await second).outcome, TurnOutcome.completed);
    expect(h.phase, TurnPhase.idle);
    // The detached turn's stop metrics still reach the side channel.
    expect(
      h.events.whereType<SideEvent<ChatSideEvent>>().map((e) => e.detached),
      [true, false],
    );
  });

  test('barge-in while speaking: playback stops before the model is '
      'stopped, and stale chunks never reach the player', () async {
    final first = h.assistant.sendText('Tell me a story');
    await settle();
    h.conversation.emit('The first sentence is here. ');
    await settle();
    expect(h.phase, TurnPhase.speaking);
    final playback = h.audio.lastPlayback!;

    // The second clause is synthesizing when the user presses the mic.
    h.synth
      ..gate = Completer<void>()
      ..gateFrom = 2;
    h.conversation.emit('The second sentence is here. ');
    await settle();
    expect(h.synth.synthesized, hasLength(2));

    await h.assistant.micDown();
    expect(playback.stopped, isTrue);
    expect(h.phase, TurnPhase.listening);
    final stopAt = h.log.indexOf('playback.stop');
    expect(stopAt, isNonNegative);
    expect(h.log.indexOf('conversation.stop'), greaterThan(stopAt));

    h.synth.gate!.complete();
    await settle();
    expect(h.log.skip(stopAt), isNot(contains('enqueue')));
    expect(h.log, isNot(contains('enqueue-after-stop')));
    expect(playback.chunks, hasLength(1));
    expect((await first).outcome, TurnOutcome.superseded);
    expect(
      h.events.whereType<AssistantSaid<ChatSideEvent>>().single,
      _said(
        'The first sentence is here. The second sentence is here.',
        interrupted: true,
      ),
    );
    final bargeIn = h.diagnostics.latest.lastBargeIn!;
    expect(bargeIn.wasPlaying, isTrue);
    expect(bargeIn.silenced, isNotNull);

    await h.assistant.stop(); // closes the barge-in's capture
    expect(h.phase, TurnPhase.idle);
  });

  test('Stop during speaking: playback stops first, the partial reply is '
      'kept as interrupted, then idle', () async {
    final turn = h.assistant.sendText('Tell me a story');
    await settle();
    h.conversation.emit('The first sentence is here. ');
    await settle();
    final playback = h.audio.lastPlayback!;

    await h.assistant.stop();

    expect(playback.stopped, isTrue);
    expect(
      h.log.indexOf('playback.stop'),
      lessThan(h.log.indexOf('conversation.stop')),
    );
    expect((await turn).outcome, TurnOutcome.interrupted);
    expect(
      h.events.whereType<AssistantSaid<ChatSideEvent>>().single,
      _said('The first sentence is here.', interrupted: true),
    );
    expect(h.phase, TurnPhase.idle);
  });

  group('errors end in the error phase, and the next turn works', () {
    test('the model fails mid-reply', () async {
      final turn = h.assistant.sendText('Hi');
      await settle();
      h.conversation.emit('Par');
      await settle();
      await h.conversation.fail(Exception('GPU lost'));
      final result = await turn;

      expect(result.outcome, TurnOutcome.failed);
      expect(result.error.toString(), contains('GPU lost'));
      expect(h.phase, TurnPhase.error);
      expect(
        h.events.whereType<AssistantSaid<ChatSideEvent>>().single.text,
        'Par',
      );
      expect(
        h.events.last,
        isA<TurnFailed<ChatSideEvent>>().having(
          (e) => '${e.error}',
          'error',
          contains('GPU lost'),
        ),
      );

      h.audio.autoDrain = true;
      final next = h.assistant.sendText('Again');
      await settle();
      h.conversation.emit('Sure thing.');
      await h.conversation.finish();
      expect((await next).outcome, TurnOutcome.completed);
      expect(h.phase, TurnPhase.idle);
    });

    test('the recognizer fails', () async {
      h.recognizer.error = StateError('stt boom');
      await h.assistant.micDown();
      final result = await h.assistant.micUp();

      expect(result.outcome, TurnOutcome.failed);
      expect(h.phase, TurnPhase.error);
      expect(h.conversation.prompts, isEmpty);
      expect(h.events.single, isA<TurnFailed<ChatSideEvent>>());
    });

    test(
      'the synthesizer fails: the model is stopped and the turn fails',
      () async {
        h.synth.throwOn = 'boom';
        final turn = h.assistant.sendText('Hi');
        await settle();
        h.conversation.emit('This clause goes boom now. ');
        final result = await turn;

        expect(result.outcome, TurnOutcome.failed);
        expect(result.error.toString(), contains('synth boom'));
        expect(h.conversation.stopCalls, greaterThanOrEqualTo(1));
        expect(h.phase, TurnPhase.error);
      },
    );

    test('an empty reply is a failure, not an empty bubble', () async {
      final turn = h.assistant.sendText('Hi');
      await settle();
      await h.conversation.finish();
      final result = await turn;

      expect(result.outcome, TurnOutcome.failed);
      expect(result.error, isA<EmptyReplyException>());
      expect(h.phase, TurnPhase.error);
    });
  });

  test('typed turn: no STT, the text is echoed as the user entry', () async {
    h.audio.autoDrain = true;
    final turn = h.assistant.sendText('  Hello there  ');
    await settle();
    expect(h.recognizer.calls, 0);
    expect(h.conversation.prompts, ['Hello there']);
    final said = h.events.whereType<UserSaid<ChatSideEvent>>().single;
    expect(said.text, 'Hello there');
    expect(said.typed, isTrue);

    h.conversation.emit('Hi! How can I help you today?');
    await h.conversation.finish();
    expect((await turn).outcome, TurnOutcome.completed);
    expect(h.phases, [TurnPhase.thinking, TurnPhase.speaking, TurnPhase.idle]);
    expect(h.diagnostics.latest.lastVoiceTurn!.typed, isTrue);
  });

  test('speech off: no TTS, no playback, idle when the text ends', () async {
    h.assistant.speakReplies = false;
    final turn = h.assistant.sendText('Hello');
    await settle();
    h.conversation.emit('Hi there. How are you? ');
    await h.conversation.finish();

    expect((await turn).outcome, TurnOutcome.completed);
    expect(h.synth.synthesized, isEmpty);
    expect(h.beginPlaybacks, isEmpty);
    expect(h.phases, [TurnPhase.thinking, TurnPhase.idle]);
  });

  test('reaching the STT window ends the press and runs the turn', () async {
    h.audio.autoDrain = true;
    await h.assistant.micDown();
    h.audio.lastOnLimit!();
    await settle();

    expect(h.log, contains('capture.stop'));
    expect(h.conversation.prompts, hasLength(1));
    expect(
      (await h.assistant.micUp()).outcome,
      TurnOutcome.ignored,
      reason: 'the release after the auto-stop has nothing to do',
    );
    h.conversation.emit('Paris.');
    await h.conversation.finish();
    await settle();
    expect(h.phase, TurnPhase.idle);
  });

  test('a stop that never ends the reply stream: Stop still resolves within '
      '2 × drainTimeout and the turn ends interrupted', () {
    fakeAsync((async) {
      final f = _Harness();
      unawaited(f.init());
      async.flushMicrotasks();

      unawaited(f.assistant.sendText('Tell me a long story'));
      async.flushMicrotasks();
      f.conversation.emit('Once upon');
      // The chat never ends the stopped turn.
      f.conversation.stopGate = Completer<void>();
      async.flushMicrotasks();

      var stopped = false;
      unawaited(f.assistant.stop().then((_) => stopped = true));
      async.elapse(const Duration(seconds: 4));
      expect(stopped, isFalse);
      async.elapse(const Duration(seconds: 7)); // 11 s > 2 × 5 s
      expect(stopped, isTrue);
      expect(f.phase, TurnPhase.idle);
      expect(
        f.events.whereType<AssistantSaid<ChatSideEvent>>().single,
        _said('Once upon', interrupted: true),
      );
      unawaited(f.assistant.dispose());
      async.flushMicrotasks();
    });
  });

  test('the turn after a forced drain waits for the stuck generation, then '
      'asks and completes', () {
    fakeAsync((async) {
      final f = _Harness();
      unawaited(f.init());
      async.flushMicrotasks();
      f.audio.autoDrain = true;

      unawaited(f.assistant.sendText('Tell me a long story'));
      async.flushMicrotasks();
      f.conversation.emit('Once upon');
      f.conversation.stopGate = Completer<void>(); // the stop never lands
      async.flushMicrotasks();
      unawaited(f.assistant.stop());
      async.elapse(const Duration(seconds: 11)); // forced drain
      expect(f.phase, TurnPhase.idle);
      expect(f.conversation.isGenerating.value, isTrue, reason: 'still stuck');

      TurnResult? next;
      unawaited(f.assistant.sendText('Again').then((r) => next = r));
      async.elapse(const Duration(seconds: 2));
      expect(f.conversation.prompts, hasLength(1), reason: 'waits, not busy');

      // The stuck generation ends on its own; the waiting turn asks now.
      f.conversation.stopGate!.complete();
      async.flushMicrotasks();
      async.elapse(const Duration(milliseconds: 10));
      expect(f.conversation.prompts, ['Tell me a long story', 'Again']);
      f.conversation.emit('Sure.');
      async.elapse(const Duration(milliseconds: 10));
      expect(f.assistant.partialReply.value, 'Sure.');
      expect(f.phase, TurnPhase.thinking);
      // Its natural end is not observable under fake_async: VoiceSession
      // awaits the cancel of the finished reply stream on the root zone.
      // The real-async tests cover completion.
      expect(next, isNull);
      unawaited(f.assistant.dispose());
      async.flushMicrotasks();
    });
  });

  test('leaving mid-turn leaves the overlay at idle', () async {
    unawaited(h.assistant.sendText('Tell me a story'));
    await settle();
    h.conversation.emit('The first sentence is here. ');
    await settle();
    expect(h.diagnostics.latest.voicePhase, TurnPhase.speaking);

    await h.assistant.dispose();

    expect(h.diagnostics.latest.voicePhase, TurnPhase.idle);
  });

  test('dispose during a reply silences it and ends the turn', () async {
    final turn = h.assistant.sendText('Tell me a story');
    await settle();
    h.conversation.emit('The first sentence is here. ');
    await settle();
    final playback = h.audio.lastPlayback!;

    final disposing = h.assistant.dispose();
    expect(playback.stopped, isTrue);
    expect((await turn).outcome, TurnOutcome.superseded);
    await disposing;
    expect(h.conversation.stopCalls, 1);
  });

  group('release hook (prepare at release, discard when the turn does not '
      'run)', () {
    test('mic up: prepare runs before the capture closes; the turn builds '
        'its responder and never discards', () async {
      h.audio.autoDrain = true;
      await h.assistant.micDown();
      final turn = h.assistant.micUp();
      expect(
        h.log.last,
        'prepare(typed: false)',
        reason: 'synchronously at release',
      );
      await settle();
      expect(
        h.log.indexOf('prepare(typed: false)'),
        lessThan(h.log.indexOf('capture.stop')),
      );
      expect(h.log, contains('responder'));
      h.conversation.emit('Paris.');
      await h.conversation.finish();
      expect((await turn).outcome, TurnOutcome.completed);
      expect(h.log, isNot(contains('discard')));
    });

    test('a not-heard press discards its preparation', () async {
      h.audio.nextUtterance = Utterance(
        pcm: quiet(const Duration(seconds: 1)),
        held: const Duration(seconds: 1),
      );
      await h.assistant.micDown();
      await h.assistant.micUp();
      expect(h.log.where((e) => e == 'prepare(typed: false)'), hasLength(1));
      expect(h.log, contains('discard'));
      expect(h.log, isNot(contains('responder')));
    });

    test('an all-zero press discards too', () async {
      h.audio.nextUtterance = Utterance(
        pcm: zeros(const Duration(seconds: 1)),
        held: const Duration(seconds: 1),
      );
      await h.assistant.micDown();
      await h.assistant.micUp();
      expect(h.log, contains('discard'));
      expect(h.log, isNot(contains('responder')));
    });

    test('a release superseded while its capture closes discards', () async {
      await h.assistant.micDown();
      final release = h.assistant.micUp();
      await h.assistant.stop(); // supersedes the release in flight
      expect((await release).outcome, TurnOutcome.superseded);
      expect(h.log, contains('discard'));
      expect(h.log, isNot(contains('responder')));
    });

    test('a per-turn image reaches prepare with its turn', () async {
      h.audio.autoDrain = true;
      final image = Uint8List.fromList([1, 2, 3]);
      unawaited(h.assistant.sendText('What is this?', image: image));
      await settle();
      expect(h.factory.requests.single.image, same(image));
      h.conversation.emit('A cat.');
      await h.conversation.finish();
      await settle();

      await h.assistant.micDown();
      unawaited(h.assistant.micUp(image: image));
      expect(h.factory.requests.last.image, same(image));
      expect(h.factory.requests.last.typed, isFalse);
    });

    test(
      'the turn image reaches the chat through ChatTurnResponder '
      "and comes back on the user's entry; a turn without one has none",
      () async {
        h.audio.autoDrain = true;
        final image = Uint8List.fromList([0x89, 0x50, 0x4E, 0x47]);
        final turn = h.assistant.sendText('What animal is this?', image: image);
        await settle();
        expect(h.conversation.images.single, same(image));
        final said = h.events.whereType<UserSaid<ChatSideEvent>>().single;
        expect(said.image, same(image));
        h.conversation.emit('A cat.');
        await h.conversation.finish();
        await turn;

        await h.assistant.micDown();
        final voice = h.assistant.micUp();
        await settle();
        expect(h.conversation.images.last, isNull);
        expect(
          h.events.whereType<UserSaid<ChatSideEvent>>().last.image,
          isNull,
        );
        h.conversation.emit('Paris.');
        await h.conversation.finish();
        await voice;
      },
    );

    test('typed turns prepare with typed: true', () async {
      h.audio.autoDrain = true;
      final turn = h.assistant.sendText('Hi');
      expect(h.log.first, 'prepare(typed: true)');
      await settle();
      h.conversation.emit('Hello.');
      await h.conversation.finish();
      await turn;
      expect(h.log, contains('responder'));
    });
  });
}
