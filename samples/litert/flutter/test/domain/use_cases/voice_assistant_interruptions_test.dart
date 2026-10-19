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

// The voice turn machine and what ends a turn from outside: an interruption
// of the audio session (a call, Siri, an alarm; the headphones unplugged) and
// the app hidden on iOS. And the audio taken per turn: activated by a press
// or a spoken reply, handed back at the end of every press or turn.
import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/repositories/audio_repository.dart';
import 'package:litert_edge_demos/data/repositories/diagnostics_repository.dart';
import 'package:litert_edge_demos/domain/models/audio_interruption.dart';
import 'package:litert_edge_demos/domain/models/chat_side_event.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/domain/models/voice.dart';
import 'package:litert_edge_demos/domain/use_cases/chat_turn_responder.dart';
import 'package:litert_edge_demos/domain/use_cases/voice_assistant.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../fakes/fake_audio_repository.dart';
import '../../fakes/fake_conversation_repository.dart';
import '../../fakes/fake_speech.dart';

/// Logs `conversation.stop` into the shared log.
class _LoggingConversation extends FakeConversationRepository {
  _LoggingConversation(this.log);

  final List<String> log;

  @override
  Future<void> stop() {
    log.add('conversation.stop');
    return super.stop();
  }
}

/// The real assistant, SpeechRepository, VoiceSession and ChatTurnResponder;
/// fakes at the edges.
class _Harness {
  final List<String> log = [];
  late final _LoggingConversation conversation = _LoggingConversation(log);
  late final FakeAudioRepository audio = FakeAudioRepository(log: log);
  final FakeRecognizer recognizer = FakeRecognizer();
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
  StreamSubscription<VoiceAssistantEvent<ChatSideEvent>>? _sub;

  Future<void> init() async {
    conversation.leaveOpen(kVoiceChatProfile);
    assistant = VoiceAssistant(
      speech: await loadedSpeech(
        recognizer: recognizer,
        synthesizer: RecordingSynth(),
      ),
      audio: audio,
      responders: ChatTurnResponder(conversation: conversation),
      diagnostics: diagnostics,
    );
    _sub = assistant.events.listen(events.add);
  }

  TurnPhase get phase => assistant.phase.value;

  Future<void> dispose() async {
    await _sub?.cancel();
    await assistant.dispose();
    diagnostics.dispose();
    models.dispose();
    await conversation.close();
  }
}

Future<void> settle() => pumpEventQueue();

Matcher _said(String text, {bool interrupted = false}) =>
    isA<AssistantSaid<ChatSideEvent>>()
        .having((e) => e.text, 'text', text)
        .having((e) => e.interrupted, 'interrupted', interrupted);

Matcher _interruptedBy(InterruptionCause cause) =>
    isA<TurnInterrupted<ChatSideEvent>>().having(
      (e) => e.cause,
      'cause',
      cause,
    );

void main() {
  late _Harness h;

  setUp(() async {
    h = _Harness();
    await h.init();
  });

  tearDown(() => h.dispose());

  /// A typed turn whose first clause plays; returns its result.
  Future<Future<TurnResult>> speaking() async {
    final turn = h.assistant.sendText('Tell me a story');
    await settle();
    h.conversation.emit('The first sentence is here. ');
    await settle();
    expect(h.phase, TurnPhase.speaking);
    return turn;
  }

  group('an interruption of the audio session', () {
    test('during the reply: the playback was stopped, the model is stopped, '
        'the partial reply is committed as interrupted and then the notice; '
        'idle, the audio handed back; the next turn prepares the audio again '
        'and completes', () async {
      final turn = await speaking();
      final playback = h.audio.lastPlayback!;

      h.audio.interrupt(AudioInterruption.taken);
      await settle();

      expect(playback.stopped, isTrue);
      expect(h.log, contains('conversation.stop'));
      expect((await turn).outcome, TurnOutcome.interrupted);
      final said = h.events.indexWhere((e) => e is AssistantSaid);
      final notice = h.events.indexWhere((e) => e is TurnInterrupted);
      expect(
        h.events[said],
        _said('The first sentence is here.', interrupted: true),
      );
      expect(h.events[notice], _interruptedBy(InterruptionCause.audioTaken));
      expect(notice, greaterThan(said));
      expect(h.phase, TurnPhase.idle);
      expect(h.audio.sessionLog, [
        'activate',
        'interrupt(taken)',
        'deactivate',
      ]);

      h.audio.autoDrain = true;
      final next = h.assistant.sendText('Once more');
      await settle();
      h.conversation.emit('Again.');
      await h.conversation.finish();
      expect((await next).outcome, TurnOutcome.completed);
      expect(h.audio.activateCalls, 2);
      expect(h.audio.playbacks, hasLength(2), reason: 'it played again');
    });

    test('while listening: the capture is cancelled, the notice shows, idle; '
        'the release that follows runs no turn', () async {
      await h.assistant.micDown();
      expect(h.phase, TurnPhase.listening);

      h.audio.interrupt(AudioInterruption.taken);
      await settle();

      expect(h.audio.captures.single.cancelled, isTrue);
      expect(h.phase, TurnPhase.idle);
      expect(h.events.single, _interruptedBy(InterruptionCause.audioTaken));
      expect(h.audio.deactivateCalls, 1);

      expect((await h.assistant.micUp()).outcome, TurnOutcome.ignored);
      expect(h.recognizer.calls, 0);
      expect(h.conversation.prompts, isEmpty);
    });

    test('the headphones unplugged during the reply end it the same way, '
        'with their own notice', () async {
      final turn = await speaking();

      h.audio.interrupt(AudioInterruption.outputRemoved);
      await settle();

      expect((await turn).outcome, TurnOutcome.interrupted);
      expect(h.events.last, _interruptedBy(InterruptionCause.outputRemoved));
    });

    test('a turn still waiting for the barged-in one to drain is dropped '
        'before it asks the model', () async {
      final first = h.assistant.sendText('Tell me a story');
      await settle();
      h.conversation
        ..emit('Once upon a time')
        ..stopGate = Completer<void>();
      await settle();
      await h.assistant.micDown(); // the barge-in
      final second = h.assistant.micUp();
      await settle();
      expect(h.phase, TurnPhase.transcribing);

      h.audio.interrupt(AudioInterruption.taken);
      await settle();
      expect(h.phase, TurnPhase.idle);
      expect(h.events.last, _interruptedBy(InterruptionCause.audioTaken));

      h.conversation.stopGate!.complete();
      expect((await second).outcome, TurnOutcome.superseded);
      expect((await first).outcome, TurnOutcome.superseded);
      expect(h.conversation.prompts, ['Tell me a story']);
    });

    test('with nothing running nothing happens', () async {
      h.audio.interrupt(AudioInterruption.taken);
      await settle();
      expect(h.events, isEmpty);
      expect(h.phase, TurnPhase.idle);
      expect(h.audio.deactivateCalls, 0);
    });
  });

  test('the app hidden on iOS (interrupt with background) while the model '
      'generates: the model is stopped, the partial reply kept as '
      'interrupted, the notice says why', () async {
    await h.assistant.micDown();
    final turn = h.assistant.micUp();
    await settle();
    h.conversation.emit('Once upon a time');
    await settle();
    expect(h.phase, TurnPhase.thinking);

    await h.assistant.interrupt(InterruptionCause.background);

    expect(h.log, contains('conversation.stop'));
    expect((await turn).outcome, TurnOutcome.interrupted);
    expect(
      h.events.whereType<AssistantSaid<ChatSideEvent>>().single,
      _said('Once upon a time', interrupted: true),
    );
    expect(h.events.last, _interruptedBy(InterruptionCause.background));
    expect(h.phase, TurnPhase.idle);
  });

  group('the audio per turn', () {
    test('a spoken reply activates it, the end of each turn hands it back '
        'once; with speech off nothing is activated', () async {
      h.audio.autoDrain = true;
      final spoken = h.assistant.sendText('One');
      await settle();
      h.conversation.emit('Reply one.');
      await h.conversation.finish();
      await spoken;
      expect(h.audio.sessionLog, ['activate', 'deactivate']);

      h.assistant.speakReplies = false;
      final silent = h.assistant.sendText('Two');
      await settle();
      h.conversation.emit('Reply two.');
      await h.conversation.finish();
      await silent;
      expect(h.audio.sessionLog, ['activate', 'deactivate', 'deactivate']);
    });

    test('a barge-in keeps it between the two turns; the end of the press '
        'hands it back', () async {
      await speaking();
      await h.assistant.micDown();
      expect(h.phase, TurnPhase.listening);
      expect(h.audio.deactivateCalls, 0);

      await h.assistant.stop();
      expect(h.phase, TurnPhase.idle);
      expect(h.audio.deactivateCalls, 1);
    });

    test('a press the OS keeps the audio from (a call): its message, not '
        '"the microphone did not start"; the error phase hands the audio '
        'back', () async {
      h.audio.activateResult = const Result.error(
        AudioFocusException(kAudioFocusMessage),
      );
      await h.assistant.micDown();
      expect(
        h.events.single,
        isA<MicUnavailable<ChatSideEvent>>().having(
          (e) => e.message,
          'message',
          kAudioFocusMessage,
        ),
      );
      expect(h.phase, TurnPhase.error);
      expect(h.audio.deactivateCalls, 1);
    });
  });
}
