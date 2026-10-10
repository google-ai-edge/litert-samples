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

// User-visible failure paths of the voice turn machine that the main suite
// (voice_assistant_test.dart) does not reach: a capture that fails to stop,
// every way a turn can fail before it starts, a playback that cannot begin,
// and "Speak replies" switched off while a reply plays.
import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter_edge_ai_speech/flutter_edge_ai_speech.dart'
    show VoiceResponder;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/repositories/audio_repository.dart';
import 'package:litert_edge_demos/data/repositories/diagnostics_repository.dart';
import 'package:litert_edge_demos/data/repositories/speech_repository.dart';
import 'package:litert_edge_demos/data/services/speech/stt_service.dart'
    show SpeechNotReadyException;
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

/// The real chat responder, with switches that make `prepare` or
/// `responder` throw, and a log of what the turn machine called.
class _SwitchableFactory implements TurnResponderFactory<ChatSideEvent> {
  _SwitchableFactory(this._inner, this.log);

  final TurnResponderFactory<ChatSideEvent> _inner;
  final List<String> log;

  /// Thrown by [prepare] (at release).
  StateError? prepareThrows;

  /// Thrown by the preparation's `responder` (when the turn starts).
  StateError? responderThrows;

  @override
  TurnPreparation<ChatSideEvent> prepare(TurnRequest request) {
    log.add('prepare');
    if (prepareThrows case final error?) throw error;
    return _SwitchablePreparation(this, _inner.prepare(request));
  }
}

class _SwitchablePreparation implements TurnPreparation<ChatSideEvent> {
  _SwitchablePreparation(this._owner, this._inner);

  final _SwitchableFactory _owner;
  final TurnPreparation<ChatSideEvent> _inner;

  @override
  VoiceResponder responder(void Function(ChatSideEvent event) onSide) {
    _owner.log.add('responder');
    if (_owner.responderThrows case final error?) throw error;
    return _inner.responder(onSide);
  }

  @override
  void discard() {
    _owner.log.add('discard');
    _inner.discard();
  }
}

/// An audio repository whose playback cannot begin (the output device went
/// away after it was prepared).
class _NoPlaybackAudio extends FakeAudioRepository {
  _NoPlaybackAudio({super.log});

  @override
  Result<PlaybackHandle> beginPlayback(int sampleRate) {
    log.add('beginPlayback($sampleRate)');
    return const Result.error(PlaybackException('the output device is gone'));
  }
}

/// The real assistant over the real SpeechRepository and VoiceSession and
/// the real ChatTurnResponder; fakes at the edges. [speech] replaces the
/// loaded speech models (to play models that are not loaded).
class _Harness {
  _Harness({FakeAudioRepository Function(List<String> log)? audio})
    : _makeAudio = audio;

  final FakeAudioRepository Function(List<String> log)? _makeAudio;
  final List<String> log = [];
  final FakeConversationRepository conversation = FakeConversationRepository();
  late final FakeAudioRepository audio =
      _makeAudio?.call(log) ?? FakeAudioRepository(log: log);
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
  late final _SwitchableFactory factory = _SwitchableFactory(
    ChatTurnResponder(conversation: conversation),
    log,
  );
  late VoiceAssistant<ChatSideEvent> assistant;
  final List<VoiceAssistantEvent<ChatSideEvent>> events = [];
  final List<TurnPhase> phases = [];
  StreamSubscription<VoiceAssistantEvent<ChatSideEvent>>? _sub;

  Future<void> init({SpeechRepository? speech}) async {
    conversation.leaveOpen(kVoiceChatProfile);
    assistant = VoiceAssistant(
      speech:
          speech ??
          await loadedSpeech(recognizer: recognizer, synthesizer: synth),
      audio: audio,
      responders: factory,
      diagnostics: diagnostics,
    );
    _sub = assistant.events.listen(events.add);
    assistant.phase.addListener(() => phases.add(assistant.phase.value));
  }

  TurnPhase get phase => assistant.phase.value;

  VoiceTurnMetrics? get lastTurn => diagnostics.latest.lastVoiceTurn;

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

/// Speech models that were never installed or loaded.
SpeechRepository _unloadedSpeech() =>
    SpeechRepository(stt: fakeSttService(), tts: fakeTtsService());

/// Lets VoiceSession's chain of awaits run.
Future<void> settle() => pumpEventQueue();

Matcher _failedWith(Object matcher) => isA<TurnFailed<ChatSideEvent>>().having(
  (e) => '${e.error}',
  'error',
  matcher,
);

Matcher _said(String text, {bool interrupted = false}) =>
    isA<AssistantSaid<ChatSideEvent>>()
        .having((e) => e.text, 'text', text)
        .having((e) => e.interrupted, 'interrupted', interrupted);

/// The next turn after a failure runs normally (the failure left nothing
/// stuck).
Future<void> _expectNextTypedTurnCompletes(_Harness h) async {
  h.audio.autoDrain = true;
  final next = h.assistant.sendText('Again');
  await settle();
  h.conversation.emit('Sure thing.');
  await h.conversation.finish();
  expect((await next).outcome, TurnOutcome.completed);
  expect(h.phase, TurnPhase.idle);
}

void main() {
  late _Harness h;

  tearDown(() => h.dispose());

  group('the capture fails to stop', () {
    setUp(() async {
      h = _Harness();
      await h.init();
    });

    test('MicUnavailable with its reason, the error phase and a gate record; '
        'no STT, no model call, the preparation is dropped; the next press '
        'works', () async {
      h.audio.stopResult = Result.error(Exception('the recorder died'));
      await h.assistant.micDown();

      final result = await h.assistant.micUp();

      expect(result.outcome, TurnOutcome.micUnavailable);
      expect('${result.error}', contains('the recorder died'));
      expect(
        h.events.single,
        isA<MicUnavailable<ChatSideEvent>>().having(
          (e) => e.message,
          'message',
          contains('the recorder died'),
        ),
      );
      expect(h.phase, TurnPhase.error);
      expect(h.phases, [
        TurnPhase.openingMic,
        TurnPhase.listening,
        TurnPhase.transcribing,
        TurnPhase.error,
      ]);
      expect(h.recognizer.calls, 0);
      expect(h.conversation.prompts, isEmpty);
      expect(h.log, containsAllInOrder(['prepare', 'capture.stop', 'discard']));
      expect(h.log, isNot(contains('responder')));
      expect(h.lastTurn?.outcome, TurnOutcome.micUnavailable);
      expect(h.lastTurn?.peakDbfs, isNull, reason: 'no audio was measured');

      h.audio.stopResult = null;
      h.audio.autoDrain = true;
      await h.assistant.micDown();
      expect(h.phase, TurnPhase.listening);
      final next = h.assistant.micUp();
      await settle();
      h.conversation.emit('Paris.');
      await h.conversation.finish();
      expect((await next).outcome, TurnOutcome.completed);
      expect(h.phase, TurnPhase.idle);
    });
  });

  group('the turn cannot start: TurnFailed, the error phase, a failed '
      'metrics record, and the next turn works', () {
    test('the audio output fails to prepare (typed turn): the model is never '
        'asked and the preparation is dropped', () async {
      h = _Harness();
      await h.init();
      h.audio.prepareResult = const Result.error(
        PlaybackException('no output device'),
      );

      final result = await h.assistant.sendText('Hello');

      expect(result.outcome, TurnOutcome.failed);
      expect(result.error, isA<PlaybackException>());
      expect(h.events.single, _failedWith(contains('no output device')));
      expect(h.phase, TurnPhase.error);
      expect(h.phases, [TurnPhase.thinking, TurnPhase.error]);
      expect(h.conversation.prompts, isEmpty);
      expect(h.log, ['prepare', 'discard']);
      expect(h.lastTurn?.outcome, TurnOutcome.failed);
      expect(h.lastTurn?.typed, isTrue);

      // The fake (like the real one) prepares again on the next turn.
      h.audio.prepareResult = const Result.ok(null);
      await _expectNextTypedTurnCompletes(h);
      expect(h.audio.prepareCalls, 2);
    });

    test('the audio output fails to prepare (voice turn): no STT ran, and '
        'the gate figures stay on the record', () async {
      h = _Harness();
      await h.init();
      h.audio.prepareResult = const Result.error(
        PlaybackException('no output device'),
      );

      final result = await h.assistant.submitUtterance(speechUtterance());

      expect(result.outcome, TurnOutcome.failed);
      expect(h.events.single, _failedWith(contains('no output device')));
      expect(h.phase, TurnPhase.error);
      expect(h.recognizer.calls, 0);
      expect(h.conversation.prompts, isEmpty);
      expect(h.log, ['prepare', 'discard']);
      expect(h.lastTurn?.outcome, TurnOutcome.failed);
      expect(h.lastTurn?.typed, isFalse);
      expect(h.lastTurn?.peakDbfs, isNotNull, reason: 'the gate passed');
    });

    test('with speech off the audio output is not needed: no prepare, the '
        'turn completes', () async {
      h = _Harness();
      await h.init();
      h.audio.prepareResult = const Result.error(
        PlaybackException('no output device'),
      );
      h.assistant.speakReplies = false;

      final turn = h.assistant.sendText('Hello');
      await settle();
      h.conversation.emit('Hi there.');
      await h.conversation.finish();

      expect((await turn).outcome, TurnOutcome.completed);
      expect(h.audio.prepareCalls, 0);
      expect(h.events.whereType<TurnFailed<ChatSideEvent>>(), isEmpty);
    });

    test('a failed warm-up (prepareAudio) shows nothing; the first turn '
        'prepares again and fails visibly', () async {
      h = _Harness();
      await h.init();
      h.audio.prepareResult = const Result.error(
        PlaybackException('no output device'),
      );

      await h.assistant.prepareAudio();
      expect(h.events, isEmpty);
      expect(h.phase, TurnPhase.idle);
      expect(h.audio.prepareCalls, 1);

      final result = await h.assistant.sendText('Hello');
      expect(result.outcome, TurnOutcome.failed);
      expect(h.audio.prepareCalls, 2);
      expect(h.events.single, _failedWith(contains('no output device')));
    });

    test('building the responder throws: that error, no STT and no model '
        'call', () async {
      h = _Harness();
      await h.init();
      h.factory.responderThrows = StateError('responder boom');

      final result = await h.assistant.submitUtterance(speechUtterance());

      expect(result.outcome, TurnOutcome.failed);
      expect('${result.error}', contains('responder boom'));
      expect(h.events.single, _failedWith(contains('responder boom')));
      expect(h.phase, TurnPhase.error);
      expect(h.recognizer.calls, 0);
      expect(h.conversation.prompts, isEmpty);
      expect(h.log, ['prepare', 'responder']);
      expect(h.lastTurn?.outcome, TurnOutcome.failed);

      h.factory.responderThrows = null;
      await _expectNextTypedTurnCompletes(h);
    });

    test('preparing the turn throws at release: the turn fails with that '
        'error when it would start; never asked', () async {
      h = _Harness();
      await h.init();
      h.factory.prepareThrows = StateError('prepare boom');

      final result = await h.assistant.sendText('Hello');

      expect(result.outcome, TurnOutcome.failed);
      expect('${result.error}', contains('prepare boom'));
      expect(h.events.single, _failedWith(contains('prepare boom')));
      expect(h.phase, TurnPhase.error);
      expect(h.conversation.prompts, isEmpty);
      expect(h.log, ['prepare'], reason: 'no responder, nothing to discard');

      h.factory.prepareThrows = null;
      await _expectNextTypedTurnCompletes(h);
    });

    test('a prepare that throws on a press the gate rejects is not a '
        'failure: "not heard"', () async {
      h = _Harness();
      await h.init();
      h.factory.prepareThrows = StateError('prepare boom');
      h.audio.nextUtterance = Utterance(
        pcm: quiet(const Duration(seconds: 1)),
        held: const Duration(seconds: 1),
      );
      await h.assistant.micDown();

      final result = await h.assistant.micUp();

      expect(result.outcome, TurnOutcome.notHeard);
      expect(h.events.single, isA<NotHeard<ChatSideEvent>>());
      expect(h.phase, TurnPhase.idle);
    });

    test('the speech recognizer is not loaded: a voice turn fails before it '
        'starts', () async {
      h = _Harness();
      await h.init(speech: _unloadedSpeech());

      final result = await h.assistant.submitUtterance(speechUtterance());

      expect(result.outcome, TurnOutcome.failed);
      expect(result.error, isA<SpeechNotReadyException>());
      expect(h.events.single, _failedWith(contains('speech recognizer')));
      expect(h.phase, TurnPhase.error);
      expect(h.conversation.prompts, isEmpty);
      expect(h.log, ['prepare', 'responder']);
      expect(h.lastTurn?.outcome, TurnOutcome.failed);
    });

    test('the speech synthesizer is not loaded: a spoken typed turn fails '
        'before it starts', () async {
      h = _Harness();
      await h.init(speech: _unloadedSpeech());

      final result = await h.assistant.sendText('Hello');

      expect(result.outcome, TurnOutcome.failed);
      expect(result.error, isA<SpeechNotReadyException>());
      expect(h.events.single, _failedWith(contains('speech synthesizer')));
      expect(h.phase, TurnPhase.error);
      expect(h.conversation.prompts, isEmpty);
    });
  });

  group('playback cannot begin', () {
    setUp(() async {
      h = _Harness(audio: (log) => _NoPlaybackAudio(log: log));
      await h.init();
    });

    test('the turn fails with the playback error, the text shown so far is '
        'kept, and the model is stopped', () async {
      final turn = h.assistant.sendText('Tell me a story');
      await settle();
      h.conversation.emit('The first sentence is here. ');

      final result = await turn;

      expect(result.outcome, TurnOutcome.failed);
      expect(result.error, isA<PlaybackException>());
      expect(h.beginPlaybacks, ['beginPlayback(24000)']);
      expect(h.phase, TurnPhase.error);
      expect(
        h.events.whereType<AssistantSaid<ChatSideEvent>>().single,
        _said('The first sentence is here.'),
      );
      expect(h.events.last, _failedWith(contains('output device is gone')));
      await settle();
      expect(h.conversation.stopCalls, greaterThanOrEqualTo(1));
      expect(h.conversation.isGenerating.value, isFalse);
      expect(h.lastTurn?.outcome, TurnOutcome.failed);
    });
  });

  group('"Speak replies" switched off during a turn', () {
    setUp(() async {
      h = _Harness();
      await h.init();
    });

    test('while a reply plays: the playback stops at once, the phase goes '
        'back to thinking, later audio is dropped, and the reply completes as '
        'text', () async {
      final turn = h.assistant.sendText('Tell me a story');
      await settle();
      h.conversation.emit('The first sentence is here. ');
      await settle();
      expect(h.phase, TurnPhase.speaking);
      final playback = h.audio.lastPlayback!;
      expect(playback.chunks, hasLength(1));

      h.assistant.speakReplies = false;

      expect(h.assistant.speakReplies, isFalse);
      expect(playback.stopped, isTrue, reason: 'silenced at once');
      expect(h.phase, TurnPhase.thinking);

      h.conversation.emit('The second sentence is here. ');
      await settle();
      h.conversation.emit('The end.');
      await h.conversation.finish();
      final result = await turn;

      expect(result.outcome, TurnOutcome.completed);
      expect(playback.chunks, hasLength(1), reason: 'nothing after the mute');
      expect(h.log, isNot(contains('enqueue-after-stop')));
      expect(h.beginPlaybacks, hasLength(1), reason: 'no new playback');
      expect(
        h.events.whereType<AssistantSaid<ChatSideEvent>>().single,
        _said(
          'The first sentence is here. The second sentence is here. The end.',
        ),
      );
      expect(h.phases, [
        TurnPhase.thinking,
        TurnPhase.speaking,
        TurnPhase.thinking,
        TurnPhase.idle,
      ]);
      expect(h.lastTurn?.outcome, TurnOutcome.completed);
    });

    test('before the first audio: no playback starts for that turn, and '
        'switching back on does not unmute it; the next turn speaks', () async {
      final turn = h.assistant.sendText('Hello');
      await settle();
      expect(h.phase, TurnPhase.thinking);

      h.assistant.speakReplies = false;
      expect(h.phase, TurnPhase.thinking);
      h.assistant.speakReplies = true;

      h.conversation.emit('Hi there. How are you? ');
      await h.conversation.finish();
      expect((await turn).outcome, TurnOutcome.completed);
      expect(h.beginPlaybacks, isEmpty);
      expect(h.phases, [TurnPhase.thinking, TurnPhase.idle]);

      h.audio.autoDrain = true;
      final next = h.assistant.sendText('Again');
      await settle();
      h.conversation.emit('Sure thing.');
      await h.conversation.finish();
      expect((await next).outcome, TurnOutcome.completed);
      expect(h.beginPlaybacks, ['beginPlayback(24000)']);
    });

    test('while idle it only changes the setting', () async {
      h.assistant.speakReplies = false;
      expect(h.phases, isEmpty);
      h.assistant.speakReplies = false; // the same value: a no-op
      expect(h.assistant.speakReplies, isFalse);
    });
  });
}
