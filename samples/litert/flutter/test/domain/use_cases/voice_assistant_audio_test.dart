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
import 'package:litert_edge_demos/data/repositories/audio_repository.dart'
    show MicAccessException;
import 'package:litert_edge_demos/data/repositories/diagnostics_repository.dart';
import 'package:litert_edge_demos/domain/audio/transcript_corrections.dart';
import 'package:litert_edge_demos/domain/models/audio_interruption.dart';
import 'package:litert_edge_demos/domain/models/chat_capabilities.dart';
import 'package:litert_edge_demos/domain/models/chat_side_event.dart';
import 'package:litert_edge_demos/domain/models/conversation_history.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/domain/models/voice.dart';
import 'package:litert_edge_demos/domain/use_cases/chat_turn_responder.dart';
import 'package:litert_edge_demos/domain/use_cases/prompt_builder.dart';
import 'package:litert_edge_demos/domain/use_cases/voice_assistant.dart';
import 'package:litert_edge_demos/ui/features/voice_chat/view_models/voice_chat_reducer.dart';
import 'package:litert_edge_demos/utils/result.dart';
import 'package:litert_edge_demos/utils/wav.dart';

import '../../fakes/fake_audio_repository.dart';
import '../../fakes/fake_conversation_repository.dart';
import '../../fakes/fake_speech.dart';
import '../../support/pcm.dart';

const _audioGemma = ChatCapabilities(
  modelName: 'Gemma 4 E2B',
  images: true,
  tools: true,
  audio: true,
);

/// Demo 1's assistant in audio mode: the real VoiceAssistant, SpeechRepository
/// and VoiceSession, the real ChatTurnResponder; fakes at the edges.
class _Harness {
  _Harness({required this.audioMode});

  final bool audioMode;
  final FakeConversationRepository conversation = FakeConversationRepository();
  final FakeAudioRepository audio = FakeAudioRepository()..autoDrain = true;
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
  final List<(int, String)> transcripts = [];
  late ChatTurnResponder responders;

  Future<void> init() async {
    if (audioMode) conversation.capabilities = _audioGemma;
    conversation.leaveOpen(kVoiceChatProfile);
    final speech = await loadedSpeech(
      recognizer: recognizer,
      synthesizer: synth,
    );
    responders = ChatTurnResponder(conversation: conversation);
    assistant = VoiceAssistant(
      speech: speech,
      audio: audio,
      responders: responders,
      diagnostics: diagnostics,
      correctTranscript: correctVoiceTranscript,
      onTranscript: (turn, text) {
        transcripts.add((turn, text));
        responders.transcribed(turn, text);
      },
    );
    assistant.events.listen(events.add);
    assistant.phase.addListener(() => phases.add(assistant.phase.value));
  }

  /// A spoken question through the PCM entry point; the model answers
  /// [reply] and the audio drains by itself.
  Future<TurnResult> ask({String reply = 'Paris.'}) async {
    final turn = assistant.submitUtterance(speechUtterance());
    await settle();
    conversation.emit(reply);
    await conversation.finish();
    return turn;
  }

  Future<void> dispose() async {
    await assistant.dispose();
    diagnostics.dispose();
    models.dispose();
    await conversation.close();
  }

  /// The spoken questions' entries as Demo 1's chat shows them (its reducer
  /// over every event so far): each one's turn and transcript.
  List<(int, String?)> get spokenEntries {
    const reducer = VoiceChatReducer();
    var state = VoiceChatState.empty;
    for (final event in events) {
      state = reducer.reduce(state, event).state;
    }
    return [
      for (final entry in state.entries)
        if (entry.spoken case final spoken?) (spoken.turn, spoken.transcript),
    ];
  }

  /// The turn numbers of the spoken questions so far, in order.
  List<int> get spokenTurns => [
    for (final e in events)
      if (e case UserSpoke(:final turn)) turn,
  ];
}

Future<void> settle() => pumpEventQueue();

void main() {
  group('a chat model that hears audio', () {
    late _Harness h;

    setUp(() async {
      h = _Harness(audioMode: true);
      await h.init();
    });

    tearDown(() => h.dispose());

    test('a spoken turn is an audio turn: no STT before the reply, the '
        'recording goes to the chat as a WAV with the spoken question\'s '
        'prompt, and the user entry is created at once', () async {
      final turn = h.assistant.submitUtterance(speechUtterance());
      await settle();

      expect(h.recognizer.calls, 0, reason: 'no transcript before the reply');
      expect(h.phase, TurnPhase.thinking);
      final spoke = h.events.whereType<UserSpoke<ChatSideEvent>>().single;
      expect(spoke.duration, const Duration(seconds: 1));
      expect(h.events.whereType<UserSaid<ChatSideEvent>>(), isEmpty);
      final wav = h.conversation.audios.single!;
      expect(wavDuration(wav), const Duration(seconds: 1));
      expect(pcm16FromWav(wav), speechUtterance().pcm);
      expect(h.conversation.prompts.single, PromptBuilder.spokenQuestion);
      expect(h.conversation.historyTexts.single, kSpokenQuestionPlaceholder);
      expect(h.conversation.askedTurns.single, spoke.turn);

      h.conversation.emit('Paris.');
      await h.conversation.finish();
      expect((await turn).outcome, TurnOutcome.completed);
      expect(h.phases, [
        TurnPhase.transcribing,
        TurnPhase.thinking,
        TurnPhase.speaking,
        TurnPhase.idle,
      ]);
      final metrics = h.diagnostics.latest.lastVoiceTurn!;
      expect(metrics.audio, isTrue);
      expect(metrics.stt, isNull);
      expect(metrics.firstAudio, isNotNull);
    });

    test('after the reply has played and the turn is idle, Whisper '
        'transcribes the recording: the words reach the entry and the text '
        'history, corrected', () async {
      h.recognizer.text = ' What is littered?';
      await h.ask();
      // The turn ended (drained): the job starts now, and only now.
      await settle();
      expect(h.recognizer.calls, 1);
      expect(h.recognizer.pcms.single, speechUtterance().pcm);
      final spoke = h.events.whereType<UserSpoke<ChatSideEvent>>().single;
      final arrived = h.events.whereType<TranscriptArrived<ChatSideEvent>>();
      expect(arrived.single.turn, spoke.turn);
      expect(arrived.single.text, 'What is LiteRT?');
      expect(h.transcripts, [(spoke.turn, 'What is LiteRT?')]);
      expect(h.conversation.historyUpdates, [(spoke.turn, 'What is LiteRT?')]);
      expect(h.diagnostics.latest.lastVoiceTurn!.transcript, isNotNull);
      // The arrival comes after the reply was committed.
      expect(
        h.events.indexOf(arrived.single),
        greaterThan(
          h.events.indexWhere((e) => e is AssistantSaid<ChatSideEvent>),
        ),
      );
    });

    test('the transcript job waits for the playback to drain', () async {
      h.audio.autoDrain = false;
      final turn = h.ask();
      await settle();
      expect(h.audio.lastPlayback, isNotNull);
      expect(h.recognizer.calls, 0, reason: 'still playing');
      h.audio.lastPlayback!.completeDrain();
      await turn;
      await settle();
      expect(h.recognizer.calls, 1);
    });

    test('a completed audio turn whose recording has no words is still a '
        'completed turn; the entry says why there is no transcript', () async {
      h.recognizer.text = '';
      expect((await h.ask()).outcome, TurnOutcome.completed);
      await settle();
      expect(h.events.whereType<NotHeard<ChatSideEvent>>(), isEmpty);
      final failed = h.events.whereType<TranscriptFailed<ChatSideEvent>>();
      expect(failed.single.reason, 'no words recognized');
      expect(h.transcripts, isEmpty);
      expect(
        h.diagnostics.latest.lastVoiceTurn!.transcriptError,
        'no words recognized',
      );
    });

    test('a failing recognizer: TranscriptFailed with its reason', () async {
      h.recognizer.error = StateError('stt worker died');
      await h.ask();
      await settle();
      expect(
        h.events.whereType<TranscriptFailed<ChatSideEvent>>().single.reason,
        contains('stt worker died'),
      );
    });

    test('a mic press while a transcript is pending does not drop it: the '
        'earlier question\'s entry and the text history still get its '
        'words', () async {
      final gate = h.recognizer.gate = Completer<void>();
      await h.ask();
      await settle();
      expect(h.recognizer.calls, 1, reason: 'the job is pending');
      final turn = h.spokenTurns.single;

      expect(await h.assistant.micDown(), isA<Ok<void>>());
      expect(h.phase, TurnPhase.listening);
      gate.complete();
      await settle();

      const words = 'What is the capital of France?';
      expect(
        h.events.whereType<TranscriptArrived<ChatSideEvent>>().map(
          (e) => (e.turn, e.text),
        ),
        [(turn, words)],
      );
      expect(h.conversation.historyUpdates, [(turn, words)]);
      expect(h.spokenEntries, [(turn, words)]);
      expect(h.phase, TurnPhase.listening, reason: 'the press goes on');
      await h.assistant.stop();
    });

    test('Stop, then a press at once: the stopped question still gets its '
        'words', () async {
      final gate = h.recognizer.gate = Completer<void>();
      final turn = h.assistant.submitUtterance(speechUtterance());
      await settle();
      h.conversation.emit('Paris');
      await settle();
      await h.assistant.stop();
      expect((await turn).outcome, TurnOutcome.interrupted);
      await settle();
      expect(h.recognizer.calls, 1);

      await h.assistant.micDown();
      gate.complete();
      await settle();

      expect(h.spokenEntries, [
        (h.spokenTurns.single, 'What is the capital of France?'),
      ]);
      await h.assistant.stop();
    });

    test(
      'a press that hears nothing (too short) keeps the pending job',
      () async {
        final gate = h.recognizer.gate = Completer<void>();
        await h.ask();
        await settle();
        h.audio.nextUtterance = speechUtterance(
          length: const Duration(milliseconds: 100),
        );

        await h.assistant.micDown();
        expect((await h.assistant.micUp()).outcome, TurnOutcome.notHeard);
        gate.complete();
        await settle();

        expect(h.events.whereType<NotHeard<ChatSideEvent>>(), hasLength(1));
        expect(h.spokenEntries, [
          (h.spokenTurns.single, 'What is the capital of France?'),
        ]);
      },
    );

    test('a mic that fails to start keeps the pending job', () async {
      final gate = h.recognizer.gate = Completer<void>();
      await h.ask();
      await settle();
      h.audio.startError = const MicAccessException('no microphone');

      expect(await h.assistant.micDown(), isA<Error<void>>());
      expect(h.phase, TurnPhase.error);
      gate.complete();
      await settle();

      expect(h.spokenEntries, [
        (h.spokenTurns.single, 'What is the capital of France?'),
      ]);
    });

    test('leaving the demo (dispose) drops it', () async {
      final gate = h.recognizer.gate = Completer<void>();
      await h.ask();
      await settle();
      final before = h.events.length;
      await h.assistant.dispose();
      gate.complete();
      await settle();
      expect(h.events.length, before);
      expect(h.transcripts, isEmpty);
    });

    test('two quick audio turns: the second does not drop the first\'s '
        'transcript; each lands in its own entry and history line', () async {
      final gate = h.recognizer.gate = Completer<void>();
      h.recognizer.texts.addAll([
        'What is the capital of France?',
        'And of Germany?',
      ]);
      await h.ask();
      await settle();
      await h.ask(reply: 'Berlin.');
      await settle();
      expect(h.recognizer.calls, 1, reason: 'both pending, one at a time');

      gate.complete();
      await settle();
      expect(h.recognizer.calls, 2);

      final [first, second] = h.spokenTurns;
      expect(first, isNot(second));
      expect(h.spokenEntries, [
        (first, 'What is the capital of France?'),
        (second, 'And of Germany?'),
      ]);
      expect(h.conversation.historyUpdates, [
        (first, 'What is the capital of France?'),
        (second, 'And of Germany?'),
      ]);
      expect(h.transcripts, h.conversation.historyUpdates);
    });

    test('a press during the reply\'s generation (a barge-in) gets the '
        'question no transcript (its recording is dropped) and says so: '
        'TranscriptFailed(interrupted) for its number; the partial reply is '
        'committed as interrupted', () async {
      h.audio.autoDrain = false;
      unawaited(h.assistant.submitUtterance(speechUtterance()));
      await settle();
      final spoke = h.events.whereType<UserSpoke<ChatSideEvent>>().single;
      h.conversation.emit('Paris is the capital. It is ');
      await settle();
      expect(h.phase, TurnPhase.speaking);

      await h.assistant.micDown();
      await settle();
      await h.assistant.stop();
      await settle();

      expect(h.recognizer.calls, 0);
      final failed = h.events.whereType<TranscriptFailed<ChatSideEvent>>();
      expect(failed, hasLength(1));
      expect(failed.single.turn, spoke.turn);
      expect(failed.single.interrupted, isTrue);
      expect(failed.single.reason, kInterruptedTranscriptReason);
      expect(
        h.events.whereType<AssistantSaid<ChatSideEvent>>().single.interrupted,
        isTrue,
      );
    });

    test('a press during the playback tail (the reply generated in full, '
        'only its speech still playing) transcribes the question as a '
        'normal end does, once the press has ended: the words reach its '
        'entry and the text history', () async {
      h.audio.autoDrain = false;
      h.recognizer.text = 'What is the capital of France?';
      unawaited(h.assistant.submitUtterance(speechUtterance()));
      await settle();
      final spoke = h.events.whereType<UserSpoke<ChatSideEvent>>().single;
      h.conversation.emit('Paris is the capital of France.');
      await h.conversation.finish();
      await settle();
      expect(h.phase, TurnPhase.speaking, reason: 'only the tail plays');
      expect(h.recognizer.calls, 0);

      await h.assistant.micDown();
      await settle();
      expect(h.phase, TurnPhase.listening);
      expect(
        h.recognizer.calls,
        0,
        reason:
            'queued until nothing runs: Whisper must not compete with '
            'the next turn',
      );

      // The press ends without a turn: idle, so the job runs now.
      await h.assistant.stop();
      await settle();

      expect(h.recognizer.calls, 1);
      expect(h.recognizer.pcms.single, speechUtterance().pcm);
      expect(
        h.events.whereType<TranscriptFailed<ChatSideEvent>>(),
        isEmpty,
        reason: 'not an interrupted question',
      );
      final arrived = h.events.whereType<TranscriptArrived<ChatSideEvent>>();
      expect(arrived.single.turn, spoke.turn);
      expect(h.conversation.historyUpdates, [
        (spoke.turn, 'What is the capital of France?'),
      ]);
      expect(h.spokenEntries, [(spoke.turn, 'What is the capital of France?')]);
    });

    test('a barge-in on the playback tail that asks the next question: the '
        'first question\'s transcript waits while that turn generates, runs '
        'once its reply is generated (beside its playback only), and both '
        'land in their own entries, in order', () async {
      h.audio.autoDrain = false;
      h.recognizer.texts.addAll([
        'What is the capital of France?',
        'And of Germany?',
      ]);
      unawaited(h.assistant.submitUtterance(speechUtterance()));
      await settle();
      h.conversation.emit('Paris is the capital of France.');
      await h.conversation.finish();
      await settle();
      expect(h.phase, TurnPhase.speaking, reason: 'only the tail plays');

      await h.assistant.micDown();
      final next = h.assistant.micUp();
      await settle();
      expect(h.phase, TurnPhase.thinking, reason: 'the next audio turn runs');
      h.conversation.emit('Berlin');
      await settle();
      expect(h.recognizer.calls, 0, reason: 'nothing transcribes beside it');

      h.conversation.emit(' is the capital of Germany.');
      await h.conversation.finish();
      await settle();
      expect(h.phase, TurnPhase.speaking);
      expect(
        h.recognizer.calls,
        1,
        reason: 'its reply is generated: the first question goes now',
      );

      h.audio.lastPlayback!.completeDrain();
      expect((await next).outcome, TurnOutcome.completed);
      await settle();

      expect(h.recognizer.calls, 2);
      final [first, second] = h.spokenTurns;
      expect(h.spokenEntries, [
        (first, 'What is the capital of France?'),
        (second, 'And of Germany?'),
      ]);
      expect(h.conversation.historyUpdates, [
        (first, 'What is the capital of France?'),
        (second, 'And of Germany?'),
      ]);
    });

    test('a chain of presses: on the first reply\'s tail, then during the '
        'second reply\'s generation: the first question\'s words reach the '
        'text history as soon as that generation is cut, before the third '
        'question is asked (its chat is rebuilt from that history)', () async {
      h.audio.autoDrain = false;
      h.recognizer.text = 'What is the capital of France?';
      unawaited(h.assistant.submitUtterance(speechUtterance()));
      await settle();
      h.conversation.emit('Paris is the capital of France.');
      await h.conversation.finish();
      await settle();
      expect(h.phase, TurnPhase.speaking, reason: 'only the tail plays');

      // The second question, pressed on the first reply's tail.
      await h.assistant.micDown();
      unawaited(h.assistant.micUp());
      await settle();
      h.conversation.emit('Berlin is');
      await settle();
      expect(h.recognizer.calls, 0, reason: 'its reply is generating');

      // The third, pressed during the second reply's generation.
      await h.assistant.micDown();
      await settle();
      final [first, second] = h.spokenTurns;
      expect(h.recognizer.calls, 1);
      expect(h.conversation.historyUpdates, [
        (first, 'What is the capital of France?'),
      ]);
      expect(h.conversation.prompts, hasLength(2), reason: 'not asked yet');
      expect(
        h.events.whereType<TranscriptFailed<ChatSideEvent>>().single.turn,
        second,
        reason: 'the cut question goes without words, as before',
      );

      h.audio.autoDrain = true;
      final third = h.assistant.micUp();
      await settle();
      expect(h.conversation.prompts, hasLength(3));
      h.conversation.emit('Rome.');
      await h.conversation.finish();
      expect((await third).outcome, TurnOutcome.completed);
    });

    test('leaving the demo during the playback tail starts no transcription '
        '(the next demo\'s recognizer switch waits for none)', () async {
      h.audio.autoDrain = false;
      unawaited(h.assistant.submitUtterance(speechUtterance()));
      await settle();
      h.conversation.emit('Paris is the capital of France.');
      await h.conversation.finish();
      await settle();
      expect(h.phase, TurnPhase.speaking);

      await h.assistant.dispose();
      await settle();

      expect(h.recognizer.calls, 0);
    });

    test(
      'Stop or a finished reply never reports an interrupted transcript',
      () async {
        await h.ask(reply: 'Paris.');
        await settle();
        final turn = h.assistant.submitUtterance(speechUtterance());
        await settle();
        h.conversation.emit('Berlin');
        await settle();
        await h.assistant.stop();
        await turn;
        await settle();

        expect(
          h.events.whereType<TranscriptFailed<ChatSideEvent>>().where(
            (e) => e.interrupted,
          ),
          isEmpty,
        );
        expect(
          h.events.whereType<TranscriptArrived<ChatSideEvent>>(),
          hasLength(2),
        );
      },
    );

    test('Stop (not a press) still transcribes the question', () async {
      final turn = h.assistant.submitUtterance(speechUtterance());
      await settle();
      h.conversation.emit('Paris');
      await settle();
      await h.assistant.stop();
      expect((await turn).outcome, TurnOutcome.interrupted);
      await settle();
      expect(h.recognizer.calls, 1);
      expect(
        h.events.whereType<TranscriptArrived<ChatSideEvent>>(),
        hasLength(1),
      );
    });

    for (final (what, cause) in [
      ('a call takes the audio', InterruptionCause.audioTaken),
      ('the app is hidden on iOS', InterruptionCause.background),
    ]) {
      test('$what during the reply: like Stop, the question still gets its '
          'transcript', () async {
        final turn = h.assistant.submitUtterance(speechUtterance());
        await settle();
        h.conversation.emit('Paris');
        await settle();

        if (cause == InterruptionCause.audioTaken) {
          h.audio.interrupt(AudioInterruption.taken);
        } else {
          await h.assistant.interrupt(cause);
        }

        expect((await turn).outcome, TurnOutcome.interrupted);
        await settle();
        expect(h.recognizer.calls, 1);
        expect(
          h.events.whereType<TranscriptArrived<ChatSideEvent>>().single.turn,
          h.spokenTurns.single,
        );
        expect(
          h.events.whereType<TurnInterrupted<ChatSideEvent>>().single.cause,
          cause,
        );
        expect(h.spokenEntries.single.$2, isNotNull);
      });
    }

    test('a typed turn never goes as audio', () async {
      final turn = h.assistant.sendText('What is LiteRT?');
      await settle();
      h.conversation.emit('A runtime.');
      await h.conversation.finish();
      await turn;
      expect(h.conversation.audios.single, isNull);
      expect(h.conversation.historyTexts.single, 'What is LiteRT?');
      expect(h.events.whereType<UserSaid<ChatSideEvent>>(), hasLength(1));
      expect(h.events.whereType<UserSpoke<ChatSideEvent>>(), isEmpty);
      expect(h.recognizer.calls, 0);
    });
  });

  group('a chat model without audio: the Whisper path', () {
    late _Harness h;

    setUp(() async {
      h = _Harness(audioMode: false);
      await h.init();
    });

    tearDown(() => h.dispose());

    test('a spoken turn is transcribed first, corrected, and asked as text; '
        'no transcript job follows', () async {
      h.recognizer.text = ' How do I open a littered and file?';
      await h.ask(reply: 'With the inspector.');
      await settle();

      expect(h.recognizer.calls, 1);
      final said = h.events.whereType<UserSaid<ChatSideEvent>>().single;
      expect(said.text, 'How do I open a LiteRT-LM file?');
      expect(h.conversation.prompts.single.trim(), said.text);
      expect(h.conversation.historyTexts.single, said.text);
      expect(h.conversation.audios.single, isNull);
      expect(h.events.whereType<UserSpoke<ChatSideEvent>>(), isEmpty);
      expect(h.events.whereType<TranscriptArrived<ChatSideEvent>>(), isEmpty);
      expect(h.diagnostics.latest.lastVoiceTurn!.audio, isFalse);
    });

    test('an empty transcript still ends the turn as not heard', () async {
      h.recognizer.text = '  ';
      final result = await h.assistant.submitUtterance(speechUtterance());
      expect(result.outcome, TurnOutcome.notHeard);
      expect(
        h.events.whereType<NotHeard<ChatSideEvent>>().single.reason,
        NotHeardReason.emptyTranscript,
      );
      expect(h.conversation.prompts, isEmpty);
    });
  });

  test(
    'TurnRequest numbers: every turn gets its own, typed ones too',
    () async {
      final h = _Harness(audioMode: true);
      await h.init();
      addTearDown(h.dispose);
      await h.ask();
      final typed = h.assistant.sendText('Hi');
      await settle();
      h.conversation.emit('Hello.');
      await h.conversation.finish();
      await typed;
      expect(h.conversation.askedTurns, hasLength(2));
      expect(h.conversation.askedTurns.toSet(), hasLength(2));
      expect(h.conversation.askedTurns.first, isNotNull);
    },
  );
}

extension on _Harness {
  TurnPhase get phase => assistant.phase.value;
}
