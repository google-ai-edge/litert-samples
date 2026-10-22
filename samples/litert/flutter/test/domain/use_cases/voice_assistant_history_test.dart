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
import 'package:flutter_edge_ai/flutter_edge_ai.dart'
    show ModelType, PreferredBackend;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/repositories/conversation/history_mirror.dart';
import 'package:litert_edge_demos/data/repositories/conversation_repository.dart';
import 'package:litert_edge_demos/data/repositories/conversation_repository_edge_ai.dart';
import 'package:litert_edge_demos/data/repositories/diagnostics_repository.dart';
import 'package:litert_edge_demos/data/repositories/speech_repository.dart';
import 'package:litert_edge_demos/data/services/llm/llm_service.dart';
import 'package:litert_edge_demos/domain/models/assistant_event.dart';
import 'package:litert_edge_demos/domain/models/chat_model_config.dart';
import 'package:litert_edge_demos/domain/models/chat_side_event.dart';
import 'package:litert_edge_demos/domain/models/conversation_history.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/domain/models/voice.dart';
import 'package:litert_edge_demos/domain/use_cases/chat_turn_responder.dart';
import 'package:litert_edge_demos/domain/use_cases/voice_assistant.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../fakes/fake_audio_repository.dart';
import '../../fakes/fake_inference_model.dart';
import '../../fakes/fake_speech.dart';
import '../../support/pcm.dart';

/// Gemma 4 E2B with its audio encoder: spoken questions go as audio, and
/// their transcripts arrive after the reply.
const _audioGemma = ChatModelConfig(
  name: 'Gemma 4 E2B',
  modelType: ModelType.gemma4,
  llm: LlmConfig(
    maxTokens: 8192,
    backend: PreferredBackend.gpu,
    maxNumImages: 1,
    supportImage: true,
    supportAudio: true,
  ),
  tools: false,
);

/// One visit of Demo 1: its own responder and assistant (the screen's view
/// model makes both on entry), on the app's one conversation.
final class _Visit {
  _Visit({
    required ConversationRepository conversation,
    required SpeechRepository speech,
    required FakeAudioRepository audio,
    required DiagnosticsRepository diagnostics,
  }) {
    final responders = ChatTurnResponder(
      conversation: conversation,
      // A busy chat is asked at once, so the precheck refuses it.
      idleWait: const Duration(milliseconds: 20),
    );
    assistant = VoiceAssistant(
      speech: speech,
      audio: audio,
      responders: responders,
      diagnostics: diagnostics,
      onTranscript: responders.transcribed,
    );
    assistant.events.listen(events.add);
  }

  late final VoiceAssistant<ChatSideEvent> assistant;
  final List<VoiceAssistantEvent<ChatSideEvent>> events = [];

  /// The turn numbers of its spoken questions, in order.
  List<int> get spokenTurns => [
    for (final e in events)
      if (e case UserSpoke(:final turn)) turn,
  ];
}

/// Lets queued microtasks and zero-length timers run.
Future<void> settle([int rounds = 20]) async {
  for (var i = 0; i < rounds; i++) {
    await Future<void>.delayed(Duration.zero);
  }
}

/// The conversation's text mirror belongs to the chat, which outlives a
/// visit of Demo 1: a re-entered demo whose open failed keeps talking on
/// the previous visit's chat. Over the real repository, responder and
/// assistant (the fake engine and speech at the edges).
void main() {
  late FakeInferenceModel model;
  late EdgeAiConversationRepository repo;
  late SpeechRepository speech;
  late FakeRecognizer recognizer;
  late FakeAudioRepository audio;
  final models = ValueNotifier<Map<ModelId, ModelState>>(const {});
  late DiagnosticsRepository diagnostics;
  late DebugPrintCallback originalDebugPrint;

  /// Waits until the chat's reply stream is being listened to.
  Future<void> untilStreaming() async {
    for (var i = 0; i < 200; i++) {
      if (model.created.isNotEmpty && model.lastSession.streaming) return;
      await settle(1);
    }
    fail('the turn never reached the engine');
  }

  /// The conversation over an engine on [backend], the voice chat open.
  Future<void> openOn(PreferredBackend backend) async {
    model = FakeInferenceModel(activeBackend: backend);
    final llm = LlmService(loadModel: (_) async => model);
    expect(await llm.load(_audioGemma), isA<Ok<LlmInfo>>());
    repo = EdgeAiConversationRepository(
      llm: llm,
      stopTimeout: const Duration(milliseconds: 50),
    );
    expect(await repo.open(kVoiceChatProfile), isA<Ok<void>>());
  }

  setUp(() async {
    originalDebugPrint = debugPrint;
    debugPrint = (message, {wrapWidth}) {};
    await openOn(PreferredBackend.gpu);
    recognizer = FakeRecognizer();
    audio = FakeAudioRepository()..autoDrain = true;
    speech = await loadedSpeech(
      recognizer: recognizer,
      synthesizer: RecordingSynth(),
    );
    diagnostics = DiagnosticsRepository(
      models: models,
      minInterval: Duration.zero,
      sampleRss: false,
    );
  });

  tearDown(() async {
    await repo.close();
    diagnostics.dispose();
    debugPrint = originalDebugPrint;
  });

  _Visit visit() => _Visit(
    conversation: repo,
    speech: speech,
    audio: audio,
    diagnostics: diagnostics,
  );

  test('re-entered while the previous reply still stops (the open fails, '
      'the old chat stays open with its history): a turn of the new visit '
      'that fails its precheck still gets a transcript, and that transcript '
      'never overwrites the earlier visit\'s entry', () async {
    // Visit 1: an audio question, answered and transcribed into the mirror.
    final first = visit();
    recognizer.text = 'What is the capital of France?';
    final asked = first.assistant.submitUtterance(speechUtterance());
    await untilStreaming();
    model.lastSession
      ..emit('Paris.')
      ..finish();
    expect((await asked).outcome, TurnOutcome.completed);
    await settle();
    expect(repo.history.single.user, 'What is the capital of France?');
    expect(repo.history.single.turn, first.spokenTurns.single);
    await first.assistant.dispose();

    // The visit's last reply ignores its stop past the stop timeout (native
    // keeps generating), so entering the demo again cannot open a new chat:
    // the old one stays, with the same profile, so the screen counts as
    // ready.
    model.lastSession.closeOnCancel = false;
    final stuck = repo.ask('Tell me a long story.').toList();
    await untilStreaming();
    final reopened = await repo.open(kVoiceChatProfile);
    expect(reopened, isA<Error<void>>());
    expect(repo.profile, kVoiceChatProfile);

    // Visit 2: its first question fails ("already being generated"), but the
    // question was asked all the same, so its recording is transcribed.
    final second = visit();
    recognizer.text = 'Something else entirely';
    final failed = await second.assistant.submitUtterance(speechUtterance());
    expect(failed.outcome, TurnOutcome.failed);
    expect('${failed.error}', contains('already being generated'));
    await settle();
    expect(recognizer.calls, 2, reason: 'the failed turn was transcribed');

    expect(
      repo.history.first.user,
      'What is the capital of France?',
      reason: 'the earlier visit\'s entry keeps its own words',
    );
    expect(
      second.spokenTurns.single,
      isNot(first.spokenTurns.single),
      reason: 'turn numbers never repeat while the chat lives',
    );
    expect(
      repo.history.map((t) => t.user),
      isNot(contains('Something else entirely')),
      reason: 'the failed turn never reached the model',
    );

    model.lastSession.finish();
    final ended = await stuck;
    expect(ended.last, isA<AssistantDone>());
    await second.assistant.dispose();
  });

  test('on the GPU a press during a reply\'s generation cuts it, and the '
      'next question runs on the same conversation (rewound in place, its '
      'recordings kept): nothing is replayed', () async {
    recognizer.text = 'What is the capital of France?';
    final v = visit();
    unawaited(v.assistant.submitUtterance(speechUtterance()));
    await untilStreaming();
    model.lastSession
      ..emit('Paris is the capital of France.')
      ..finish();
    await settle();
    audio.autoDrain = false;
    await v.assistant.micDown();
    final second = v.assistant.micUp();
    await untilStreaming();
    model.lastSession.emit('Berlin is');
    await settle();
    final session = model.lastSession;
    await v.assistant.micDown();
    await settle();

    audio.autoDrain = true;
    final third = v.assistant.micUp();
    await untilStreaming();
    expect(model.lastSession, same(session), reason: 'no clearHistory');
    final asked = session.queries.last;
    expect(asked.text, isNot(startsWith(HistoryMirror.replayHeader)));
    expect(asked.hasAudio, isTrue);
    expect(model.chat!.fullHistory.where((m) => m.hasAudio), hasLength(3));
    session
      ..emit('Rome.')
      ..finish();
    expect((await third).outcome, TurnOutcome.completed);
    expect((await second).outcome, TurnOutcome.superseded);
    await v.assistant.dispose();
  });

  test('on the NPU a press on a reply\'s playback tail, then one during the '
      'next reply\'s generation: the first question\'s words reach the text '
      'history when that generation is cut, so the chat rebuilt for the '
      'third question replays them (not the placeholder)', () async {
    await repo.close();
    await openOn(PreferredBackend.npu);
    audio.autoDrain = false;
    recognizer.text = 'What is the capital of France?';
    final v = visit();
    // The first question, answered in full: only its playback tail runs.
    unawaited(v.assistant.submitUtterance(speechUtterance()));
    await untilStreaming();
    model.lastSession
      ..emit('Paris is the capital of France.')
      ..finish();
    await settle();
    expect(v.assistant.phase.value, TurnPhase.speaking);

    // The second, pressed on that tail; its reply is still generating
    // when the third press cuts it (the chat is then rebuilt from the text
    // history for the third question).
    await v.assistant.micDown();
    final second = v.assistant.micUp();
    await untilStreaming();
    model.lastSession.emit('Berlin is');
    await settle();
    expect(recognizer.calls, 0, reason: 'its reply is generating');
    await v.assistant.micDown();
    await settle();
    expect(recognizer.calls, 1);
    expect(repo.history.first.user, 'What is the capital of France?');

    audio.autoDrain = true;
    final third = v.assistant.micUp();
    await untilStreaming();
    final replay = model.lastSession.queries.first.text;
    expect(replay, startsWith(HistoryMirror.replayHeader));
    expect(
      replay,
      contains(
        'User: What is the capital of France?\n'
        'Assistant: Paris is the capital of France.\n',
      ),
    );
    expect(
      replay,
      contains('User: $kSpokenQuestionPlaceholder\nAssistant: Berlin is\n'),
      reason: 'the cut question has no words',
    );
    model.lastSession
      ..emit('Rome.')
      ..finish();
    expect((await third).outcome, TurnOutcome.completed);
    expect((await second).outcome, TurnOutcome.superseded);
    await v.assistant.dispose();
  });

  test('two assistants never hand out the same turn number', () async {
    final a = visit();
    final b = visit();
    final turns = <int>[];
    for (final v in [a, b, a]) {
      final asked = v.assistant.submitUtterance(speechUtterance());
      await untilStreaming();
      model.lastSession
        ..emit('Ok.')
        ..finish();
      await asked;
      await settle();
      turns.add(v.spokenTurns.last);
    }
    expect(turns.toSet(), hasLength(3));
    await a.assistant.dispose();
    await b.assistant.dispose();
  });
}
