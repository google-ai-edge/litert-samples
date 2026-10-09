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

// Push-to-talk paths of the voice turn machine that the main suites
// (voice_assistant_test.dart, voice_assistant_failures_test.dart) do not
// reach: an action while the capture closes or while an early release
// cancels it, a typed or fed turn during a press, a stale STT-window limit,
// a start that fails for a reason other than access, and calls after
// dispose.
import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter_edge_ai_speech/flutter_edge_ai_speech.dart'
    show VoiceResponder;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/repositories/audio_repository.dart';
import 'package:litert_edge_demos/data/repositories/diagnostics_repository.dart';
import 'package:litert_edge_demos/domain/models/audio_devices.dart';
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

/// [FakeAudioRepository] whose captures can hold their `stop` and `cancel`
/// on a gate (a recorder that takes a while to close).
class _GatedAudio implements AudioRepository {
  _GatedAudio(this.inner);

  final FakeAudioRepository inner;

  /// When set, a capture's `stop` waits for it before it closes.
  Completer<void>? stopGate;

  /// When set, a capture's `cancel` waits for it before it cancels.
  Completer<void>? cancelGate;

  @override
  ValueListenable<double> get inputLevel => inner.inputLevel;

  @override
  ValueListenable<AudioDeviceStatus> get devices => inner.devices;

  @override
  Future<Result<void>> prepare() => inner.prepare();

  @override
  Future<Result<void>> requestMicAccess() => inner.requestMicAccess();

  @override
  Future<Result<CaptureHandle>> startCapture({
    required Duration maxLength,
    required void Function() onLimit,
  }) async {
    final started = await inner.startCapture(
      maxLength: maxLength,
      onLimit: onLimit,
    );
    return switch (started) {
      Ok(:final value) => Result.ok(_GatedCapture(this, value)),
      Error(:final error) => Result.error(error),
    };
  }

  @override
  Result<PlaybackHandle> beginPlayback(int sampleRate) =>
      inner.beginPlayback(sampleRate);

  @override
  Future<void> close() => inner.close();
}

class _GatedCapture implements CaptureHandle {
  _GatedCapture(this._audio, this._inner);

  final _GatedAudio _audio;
  final CaptureHandle _inner;

  @override
  Future<Result<Utterance>> stop() async {
    await _audio.stopGate?.future;
    return _inner.stop();
  }

  @override
  Future<void> cancel() async {
    await _audio.cancelGate?.future;
    return _inner.cancel();
  }
}

/// Logs the release hook (prepare, responder, discard) into the shared log.
class _LoggingFactory implements TurnResponderFactory<ChatSideEvent> {
  _LoggingFactory(this._inner, this.log);

  final TurnResponderFactory<ChatSideEvent> _inner;
  final List<String> log;

  @override
  TurnPreparation<ChatSideEvent> prepare(TurnRequest request) {
    log.add('prepare(typed: ${request.typed})');
    return _LoggingPreparation(_inner.prepare(request), log);
  }
}

class _LoggingPreparation implements TurnPreparation<ChatSideEvent> {
  _LoggingPreparation(this._inner, this.log);

  final TurnPreparation<ChatSideEvent> _inner;
  final List<String> log;

  @override
  VoiceResponder responder(void Function(ChatSideEvent event) onSide) {
    log.add('responder');
    return _inner.responder(onSide);
  }

  @override
  void discard() {
    log.add('discard');
    _inner.discard();
  }
}

/// The real assistant over the real SpeechRepository, VoiceSession and
/// ChatTurnResponder; fakes at the edges.
class _Harness {
  final List<String> log = [];
  final FakeConversationRepository conversation = FakeConversationRepository();
  late final FakeAudioRepository fakeAudio = FakeAudioRepository(log: log);
  late final _GatedAudio audio = _GatedAudio(fakeAudio);
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
  final List<VoiceAssistantEvent<ChatSideEvent>> events = [];
  final List<TurnPhase> phases = [];
  StreamSubscription<VoiceAssistantEvent<ChatSideEvent>>? _sub;
  bool _disposed = false;

  Future<void> init() async {
    conversation.leaveOpen(kVoiceChatProfile);
    assistant = VoiceAssistant(
      speech: await loadedSpeech(recognizer: recognizer, synthesizer: synth),
      audio: audio,
      responders: _LoggingFactory(
        ChatTurnResponder(conversation: conversation),
        log,
      ),
      diagnostics: diagnostics,
    );
    _sub = assistant.events.listen(events.add);
    assistant.phase.addListener(() => phases.add(assistant.phase.value));
  }

  TurnPhase get phase => assistant.phase.value;

  VoiceTurnMetrics? get lastTurn => diagnostics.latest.lastVoiceTurn;

  Future<void> disposeAssistant() async {
    if (_disposed) return;
    _disposed = true;
    await _sub?.cancel();
    await assistant.dispose();
  }

  Future<void> dispose() async {
    await disposeAssistant();
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

  group('an action while the released capture closes', () {
    setUp(() => h.audio.stopGate = Completer<void>());

    test('Stop: idle at once, the release is superseded with its '
        'preparation dropped, no STT, no event', () async {
      await h.assistant.micDown();
      final release = h.assistant.micUp();
      await settle();
      expect(h.phase, TurnPhase.transcribing);

      await h.assistant.stop();
      expect(h.phase, TurnPhase.idle);

      h.audio.stopGate!.complete();
      expect((await release).outcome, TurnOutcome.superseded);
      expect(h.log, containsAllInOrder(['prepare(typed: false)', 'discard']));
      expect(h.log, isNot(contains('responder')));
      expect(h.recognizer.calls, 0);
      expect(h.events, isEmpty);
      expect(h.phases, [
        TurnPhase.openingMic,
        TurnPhase.listening,
        TurnPhase.transcribing,
        TurnPhase.idle,
      ]);
    });

    test('a new press: the release is superseded and the new capture '
        'listens', () async {
      final gate = h.audio.stopGate!;
      await h.assistant.micDown();
      final release = h.assistant.micUp();
      await settle();

      h.audio.stopGate = null;
      await h.assistant.micDown();
      expect(h.phase, TurnPhase.listening);
      expect(h.fakeAudio.captures, hasLength(2));

      gate.complete();
      expect((await release).outcome, TurnOutcome.superseded);
      expect(h.log, containsAllInOrder(['prepare(typed: false)', 'discard']));
      expect(h.recognizer.calls, 0);
      expect(h.phase, TurnPhase.listening, reason: 'the new press is held');
      expect(h.fakeAudio.captures.last.isOpen, isTrue);
      await h.assistant.stop();
      await settle();
      expect(h.fakeAudio.captures.last.cancelled, isTrue);
      expect(h.events, isEmpty);
    });

    test('a typed send: the release is superseded and the typed turn '
        'runs', () async {
      h.fakeAudio.autoDrain = true;
      await h.assistant.micDown();
      final release = h.assistant.micUp();
      await settle();

      final typed = h.assistant.sendText('Hello');
      await settle();
      h.audio.stopGate!.complete();
      expect((await release).outcome, TurnOutcome.superseded);
      h.conversation.emit('Hi there.');
      await h.conversation.finish();
      expect((await typed).outcome, TurnOutcome.completed);

      expect(h.recognizer.calls, 0);
      expect(h.conversation.prompts, ['Hello']);
      expect(
        h.events.whereType<UserSaid<ChatSideEvent>>().single.typed,
        isTrue,
      );
      expect(h.phase, TurnPhase.idle);
    });
  });

  group('a release before the mic opened, superseded while its capture '
      'cancels', () {
    setUp(() {
      h.fakeAudio.startGate = Completer<void>();
      h.audio.cancelGate = Completer<void>();
    });

    test('by Stop: no notice, idle', () async {
      final down = h.assistant.micDown();
      await settle();
      final up = h.assistant.micUp();
      await settle();
      h.fakeAudio.startGate!.complete();
      await down;
      await settle();
      expect(h.phase, TurnPhase.openingMic, reason: 'the cancel still runs');

      await h.assistant.stop();
      h.audio.cancelGate!.complete();

      expect((await up).outcome, TurnOutcome.superseded);
      expect(h.events, isEmpty);
      expect(h.fakeAudio.captures.single.cancelled, isTrue);
      expect(h.phases, [TurnPhase.openingMic, TurnPhase.idle]);
    });

    test('by a typed send: no notice, the typed turn runs', () async {
      h.fakeAudio.autoDrain = true;
      final down = h.assistant.micDown();
      await settle();
      final up = h.assistant.micUp();
      await settle();
      h.fakeAudio.startGate!.complete();
      await down;
      await settle();

      final typed = h.assistant.sendText('Hello');
      h.audio.cancelGate!.complete();
      expect((await up).outcome, TurnOutcome.superseded);
      await settle();
      h.conversation.emit('Hi there.');
      await h.conversation.finish();
      expect((await typed).outcome, TurnOutcome.completed);

      expect(h.events.whereType<NotHeard<ChatSideEvent>>(), isEmpty);
      expect(h.fakeAudio.captures.single.cancelled, isTrue);
    });
  });

  group('a typed or fed turn during a press drops the capture', () {
    test('a typed send while listening: the capture is cancelled, no STT, '
        'the typed turn runs, a late release is ignored', () async {
      h.fakeAudio.autoDrain = true;
      await h.assistant.micDown();
      expect(h.phase, TurnPhase.listening);

      final typed = h.assistant.sendText('Hello');
      await settle();
      expect(h.fakeAudio.captures.single.cancelled, isTrue);
      h.conversation.emit('Hi there.');
      await h.conversation.finish();
      expect((await typed).outcome, TurnOutcome.completed);

      expect((await h.assistant.micUp()).outcome, TurnOutcome.ignored);
      expect(h.recognizer.calls, 0);
      expect(h.log.where((e) => e.startsWith('prepare')), [
        'prepare(typed: true)',
      ]);
      expect(h.phases, [
        TurnPhase.openingMic,
        TurnPhase.listening,
        TurnPhase.thinking,
        TurnPhase.speaking,
        TurnPhase.idle,
      ]);
    });

    test('a typed send while the mic opens: the capture that starts '
        'afterwards is cancelled and never listens', () async {
      h.fakeAudio
        ..autoDrain = true
        ..startGate = Completer<void>();
      final down = h.assistant.micDown();
      await settle();

      final typed = h.assistant.sendText('Hello');
      await settle();
      h.fakeAudio.startGate!.complete();
      expect(await down, isA<Ok<void>>());
      expect(h.fakeAudio.captures.single.cancelled, isTrue);

      h.conversation.emit('Hi there.');
      await h.conversation.finish();
      expect((await typed).outcome, TurnOutcome.completed);
      expect(h.phases, isNot(contains(TurnPhase.listening)));
      expect(h.events.whereType<MicUnavailable<ChatSideEvent>>(), isEmpty);
    });

    test('a fed utterance while listening: the capture is cancelled and the '
        'fed audio is transcribed', () async {
      h.fakeAudio.autoDrain = true;
      await h.assistant.micDown();

      final fed = h.assistant.submitUtterance(speechUtterance());
      await settle();
      expect(h.fakeAudio.captures.single.cancelled, isTrue);
      expect(h.recognizer.calls, 1);
      h.conversation.emit('Paris.');
      await h.conversation.finish();
      expect((await fed).outcome, TurnOutcome.completed);
      expect(h.fakeAudio.captures.single.stopped, isFalse);
    });
  });

  group('a typed or fed turn during a reply supersedes it', () {
    test('a typed send while speaking: the reply is committed as '
        'interrupted and the new turn runs after it drained', () async {
      final first = h.assistant.sendText('Tell me a story');
      await settle();
      h.conversation.emit('The first sentence is here. ');
      await settle();
      expect(h.phase, TurnPhase.speaking);
      final playback = h.fakeAudio.lastPlayback!;

      h.fakeAudio.autoDrain = true;
      final second = h.assistant.sendText('Something else');
      expect(playback.stopped, isTrue);
      expect((await first).outcome, TurnOutcome.superseded);
      await settle();
      h.conversation.emit('Sure.');
      await h.conversation.finish();
      expect((await second).outcome, TurnOutcome.completed);

      expect(h.events.whereType<AssistantSaid<ChatSideEvent>>(), [
        _said('The first sentence is here.', interrupted: true),
        _said('Sure.'),
      ]);
      expect(h.diagnostics.latest.lastBargeIn!.wasPlaying, isTrue);
    });

    test('a fed utterance while thinking: the partial reply is committed as '
        'interrupted and the fed turn runs', () async {
      final first = h.assistant.sendText('Tell me a story');
      await settle();
      h.conversation.emit('Once upon');
      await settle();
      expect(h.phase, TurnPhase.thinking);

      h.fakeAudio.autoDrain = true;
      h.recognizer.text = 'Second question?';
      final fed = h.assistant.submitUtterance(speechUtterance());
      expect((await first).outcome, TurnOutcome.superseded);
      await settle();
      h.conversation.emit('Answer.');
      await h.conversation.finish();
      expect((await fed).outcome, TurnOutcome.completed);

      expect(h.conversation.prompts, ['Tell me a story', 'Second question?']);
      expect(h.events.whereType<AssistantSaid<ChatSideEvent>>(), [
        _said('Once upon', interrupted: true),
        _said('Answer.'),
      ]);
      expect(h.diagnostics.latest.lastBargeIn!.wasPlaying, isFalse);
    });
  });

  test('the STT-window limit of a capture already released does not end '
      'the next press', () async {
    h.fakeAudio.autoDrain = true;
    await h.assistant.micDown();
    final staleLimit = h.fakeAudio.lastOnLimit!;
    final first = h.assistant.micUp();
    await settle();
    h.conversation.emit('Paris.');
    await h.conversation.finish();
    expect((await first).outcome, TurnOutcome.completed);

    await h.assistant.micDown();
    expect(h.phase, TurnPhase.listening);
    staleLimit();
    await settle();

    expect(h.phase, TurnPhase.listening);
    expect(h.fakeAudio.captures.last.isOpen, isTrue);
    expect(h.recognizer.calls, 1);
    await h.assistant.stop();
  });

  test('a start that fails for a reason other than access says the '
      'microphone did not start', () async {
    h.fakeAudio.startError = Exception('the recorder is busy');

    final result = await h.assistant.micDown();

    expect(result, isA<Error<void>>());
    expect(
      h.events.single,
      isA<MicUnavailable<ChatSideEvent>>().having(
        (e) => e.message,
        'message',
        'The microphone did not start: Exception: the recorder is busy',
      ),
    );
    expect(h.phases, [TurnPhase.openingMic, TurnPhase.error]);
    expect(h.lastTurn?.outcome, TurnOutcome.micUnavailable);
    expect(h.lastTurn?.typed, isFalse);
    expect(h.lastTurn?.peakDbfs, isNull);
  });

  test('a press while listening is a no-op: one capture, still '
      'listening', () async {
    await h.assistant.micDown();
    expect(await h.assistant.micDown(), isA<Ok<void>>());

    expect(h.log.where((e) => e == 'startCapture'), hasLength(1));
    expect(h.phases, [TurnPhase.openingMic, TurnPhase.listening]);
    await h.assistant.stop();
  });

  test('Stop while listening cancels the capture and goes idle', () async {
    await h.assistant.micDown();

    await h.assistant.stop();

    expect(h.fakeAudio.captures.single.cancelled, isTrue);
    expect(h.phases, [
      TurnPhase.openingMic,
      TurnPhase.listening,
      TurnPhase.idle,
    ]);
    expect(h.events, isEmpty);
  });

  test('dispose while listening cancels the capture', () async {
    await h.assistant.micDown();

    await h.disposeAssistant();

    expect(h.fakeAudio.captures.single.cancelled, isTrue);
    expect(h.diagnostics.latest.voicePhase, TurnPhase.idle);
  });

  group('the app leaves the foreground (cancelCapture)', () {
    test('with the button held: the capture is cancelled, idle, no turn; '
        'the release that follows (the pointer cancel) is ignored', () async {
      await h.assistant.micDown();

      expect(await h.assistant.cancelCapture(), isTrue);

      expect(h.fakeAudio.captures.single.cancelled, isTrue);
      expect(h.fakeAudio.captures.single.stopped, isFalse);
      expect((await h.assistant.micUp()).outcome, TurnOutcome.ignored);
      expect(h.recognizer.calls, 0);
      expect(h.log, isNot(contains('prepare(typed: false)')));
      expect(h.events, isEmpty);
      expect(h.phases, [
        TurnPhase.openingMic,
        TurnPhase.listening,
        TurnPhase.idle,
      ]);
    });

    test('while the mic still opens: the capture is cancelled once it has '
        'started and never listens', () async {
      h.fakeAudio.startGate = Completer<void>();
      final down = h.assistant.micDown();
      await settle();

      expect(await h.assistant.cancelCapture(), isTrue);
      h.fakeAudio.startGate!.complete();

      expect(await down, isA<Ok<void>>());
      expect(h.fakeAudio.captures.single.cancelled, isTrue);
      expect(h.phases, [TurnPhase.openingMic, TurnPhase.idle]);
      expect((await h.assistant.micUp()).outcome, TurnOutcome.ignored);
      expect(h.events, isEmpty);
    });

    test('a release still waiting for its capture is superseded, not '
        'transcribed', () async {
      await h.assistant.micDown();
      final release = h.assistant.micUp();

      expect(await h.assistant.cancelCapture(), isTrue);

      expect((await release).outcome, TurnOutcome.superseded);
      expect(h.fakeAudio.captures.single.cancelled, isTrue);
      expect(h.recognizer.calls, 0);
      expect(h.phase, TurnPhase.idle);
    });

    test('a turn past its capture goes on', () async {
      h.fakeAudio.autoDrain = true;
      h.audio.stopGate = Completer<void>();
      await h.assistant.micDown();
      final release = h.assistant.micUp();
      await settle();
      expect(h.phase, TurnPhase.transcribing);

      expect(await h.assistant.cancelCapture(), isFalse);
      expect(h.phase, TurnPhase.transcribing);

      h.audio.stopGate!.complete();
      await settle();
      h.conversation.emit('Hi there.');
      await h.conversation.finish();
      expect((await release).outcome, TurnOutcome.completed);
    });

    test('without a press, or after dispose, it does nothing', () async {
      expect(await h.assistant.cancelCapture(), isFalse);
      await h.disposeAssistant();
      expect(await h.assistant.cancelCapture(), isFalse);
      expect(h.phases, isEmpty);
    });
  });

  group('after dispose', () {
    setUp(() => h.disposeAssistant());

    test('a press is an error and opens nothing', () async {
      final result = await h.assistant.micDown();

      expect(result, isA<Error<void>>());
      expect(
        '${(result as Error<void>).error}',
        contains('VoiceAssistant disposed'),
      );
      expect(h.log, isNot(contains('startCapture')));
    });

    test('a release, a send and a fed utterance are ignored; Stop and '
        'dispose do nothing', () async {
      expect((await h.assistant.micUp()).outcome, TurnOutcome.ignored);
      expect((await h.assistant.sendText('Hi')).outcome, TurnOutcome.ignored);
      expect(
        (await h.assistant.submitUtterance(speechUtterance())).outcome,
        TurnOutcome.ignored,
      );
      await h.assistant.stop();
      await h.assistant.dispose();
      expect(h.log, isEmpty);
    });
  });

  test('an empty typed text is ignored: nothing is prepared', () async {
    expect((await h.assistant.sendText('   ')).outcome, TurnOutcome.ignored);
    expect(h.log, isEmpty);
    expect(h.phases, isEmpty);
  });

  test('Stop while idle does nothing', () async {
    await h.assistant.stop();
    expect(h.phases, isEmpty);
    expect(h.events, isEmpty);
  });

  group('the overlay figures', () {
    test('a late figure of a turn a barge-in replaced never overwrites the '
        "newer turn's record", () async {
      final first = h.assistant.sendText('Tell me a story');
      await settle();
      h.conversation.emit('The first sentence is here. ');
      await settle();

      h.fakeAudio.autoDrain = true;
      final second = h.assistant.sendText('Again');
      expect((await first).outcome, TurnOutcome.superseded);
      final superseded = h.lastTurn;
      expect(superseded?.outcome, TurnOutcome.superseded);
      expect(superseded?.typed, isTrue);
      expect(superseded?.firstAudio, isNotNull);
      expect(superseded?.total, isNotNull);

      await settle();
      h.conversation.emit('Sure.');
      await h.conversation.finish();
      expect((await second).outcome, TurnOutcome.completed);
      expect(h.lastTurn?.outcome, TurnOutcome.completed);
    });

    test('a voice turn records the capture close, STT, the first text and '
        'audio, the clauses and the total', () async {
      h.fakeAudio.autoDrain = true;
      await h.assistant.micDown();
      final turn = h.assistant.micUp();
      await settle();
      h.conversation.emit('Paris is the capital. It is lovely.');
      await h.conversation.finish();
      expect((await turn).outcome, TurnOutcome.completed);

      final m = h.lastTurn!;
      expect(m.typed, isFalse);
      expect(m.captureClose, isNotNull);
      expect(m.stt, isNotNull);
      expect(m.firstText, isNotNull);
      expect(m.firstAudio, isNotNull);
      expect(m.sampleRate, 24000);
      expect(m.ttsClauses, hasLength(2));
      expect(m.total, isNotNull);
      expect(m.outcome, TurnOutcome.completed);
      expect(m.peakDbfs, isNotNull);
      expect(m.gateDbfs, isNotNull);
      expect(m.voiced, isNotNull);
    });

    test('a silent press records the gate figures with "not heard"', () async {
      h.fakeAudio.nextUtterance = Utterance(
        pcm: quiet(const Duration(seconds: 1)),
        held: const Duration(seconds: 1),
      );
      await h.assistant.micDown();
      expect((await h.assistant.micUp()).outcome, TurnOutcome.notHeard);

      final m = h.lastTurn!;
      expect(m.outcome, TurnOutcome.notHeard);
      expect(m.typed, isFalse);
      expect(m.peakDbfs, isNotNull);
      expect(m.gateDbfs, isNotNull);
      expect(m.voiced, Duration.zero);
      expect(m.stt, isNull);
      expect(m.total, isNull);
    });
  });
}
