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

import 'package:flutter_edge_ai_agent/flutter_edge_ai_agent.dart'
    show Skill, SkillType;
import 'package:flutter_edge_ai_speech/flutter_edge_ai_speech.dart'
    show VoiceResponder;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/repositories/conversation_repository.dart';
import 'package:litert_edge_demos/domain/models/chat_capabilities.dart';
import 'package:litert_edge_demos/domain/models/chat_side_event.dart';
import 'package:litert_edge_demos/domain/models/conversation_history.dart';
import 'package:litert_edge_demos/domain/models/knowledge.dart';
import 'package:litert_edge_demos/domain/models/knowledge_query.dart';
import 'package:litert_edge_demos/domain/use_cases/chat_turn_responder.dart';
import 'package:litert_edge_demos/domain/use_cases/prompt_builder.dart';
import 'package:litert_edge_demos/domain/use_cases/turn_responder_factory.dart';
import 'package:litert_edge_demos/utils/result.dart';
import 'package:litert_edge_demos/utils/wav.dart';

import '../../fakes/fake_conversation_repository.dart';
import '../../fakes/fake_knowledge.dart';

const _time = Skill(
  name: 'current-time',
  description: 'The time.',
  instructions: 'Call the `run_intent` tool with intent `current_time`.',
  type: SkillType.intent,
);

const _audioGemma = ChatCapabilities(
  modelName: 'Gemma 4 E2B',
  images: true,
  tools: true,
  audio: true,
);

/// Demo 1's responder in audio mode: the recording searched as audio, the
/// spoken question's prompt, the recording to the chat.
void main() {
  late FakeConversationRepository conversation;
  late List<ChatSideEvent> sides;
  final wav = wavFromPcm16(Uint8List(32000), sampleRate: 16000);

  setUp(() {
    conversation = FakeConversationRepository()..capabilities = _audioGemma;
    conversation.leaveOpen(kVoiceChatProfile);
    sides = [];
  });

  tearDown(() => conversation.close());

  ChatTurnResponder responders(FakeRetriever? retriever) =>
      ChatTurnResponder(conversation: conversation, retriever: retriever);

  VoiceResponder responderFor(
    ChatTurnResponder r, {
    bool typed = false,
    int turn = 5,
    Uint8List? image,
  }) => r
      .prepare(TurnRequest(typed: typed, turn: turn, image: image))
      .responder(sides.add);

  /// Runs one audio turn; the model answers [reply].
  Future<String> audioTurn(
    VoiceResponder responder, {
    String reply = 'It runs on device [1].',
  }) async {
    final text = StringBuffer();
    final done = responder.respondToAudio!(wav)
        .listen(text.write)
        .asFuture<void>();
    await pumpEventQueue();
    conversation.emit(reply);
    await conversation.finish();
    await done;
    return text.toString();
  }

  Retrieval used(List<Passage> passages) => Retrieval(
    outcome: RetrievalOutcome.used,
    passages: passages,
    candidates: passages,
    gate: 0.69,
    latency: const Duration(milliseconds: 170),
    queryKind: KnowledgeQueryKind.audio,
  );

  group('which turns are audio turns', () {
    test('a spoken turn on a chat model that hears audio gets '
        'respondToAudio', () {
      expect(responderFor(responders(null)).respondToAudio, isNotNull);
    });

    test('a typed turn never does', () {
      expect(
        responderFor(responders(null), typed: true).respondToAudio,
        isNull,
      );
    });

    test('a chat model without audio: the Whisper path', () {
      conversation.capabilities = const ChatCapabilities(
        modelName: 'Gemma 4 E2B',
        images: true,
        tools: true,
      );
      expect(responderFor(responders(null)).respondToAudio, isNull);
    });
  });

  test('the recording is searched as audio (the retriever gates it at the '
      'audio threshold); the excerpts go into the spoken prompt with the '
      'recording; the retrieval goes out first and its citations follow the '
      'reply', () async {
    final passages = [
      passage(1, similarity: 0.74),
      passage(2, similarity: 0.7),
    ];
    final retriever = FakeRetriever(Result.ok(used(passages)));
    final reply = await audioTurn(responderFor(responders(retriever)));

    final query = retriever.queries.single as KnowledgeAudioQuery;
    expect(query.kind, KnowledgeQueryKind.audio);
    expect(query.wav, same(wav));
    expect(conversation.prompts.single, PromptBuilder.buildSpoken(passages));
    expect(conversation.audios.single, same(wav));
    expect(conversation.historyTexts.single, kSpokenQuestionPlaceholder);
    expect(conversation.askedTurns.single, 5);
    expect(reply, 'It runs on device [1].');
    expect(
      sides.first,
      isA<ChatRetrieval>()
          .having((e) => e.retrieval.passages, 'passages', passages)
          .having(
            (e) => e.retrieval.queryKind,
            'kind',
            KnowledgeQueryKind.audio,
          ),
    );
    expect(sides.last, isA<ChatGenerationDone>());
  });

  test('below the gate: the minimal spoken prompt, never empty', () async {
    final retriever = FakeRetriever(
      Result.ok(
        Retrieval(
          outcome: RetrievalOutcome.belowGate,
          candidates: [passage(1, similarity: 0.55)],
          gate: 0.69,
          queryKind: KnowledgeQueryKind.audio,
        ),
      ),
    );
    await audioTurn(responderFor(responders(retriever)));
    expect(conversation.prompts.single, PromptBuilder.spokenQuestion);
    expect(conversation.audios.single, same(wav));
  });

  test('a failed search goes on without excerpts and says so', () async {
    final retriever = FakeRetriever(
      const Result.error(
        KnowledgeQueryTooLongException(
          kind: KnowledgeQueryKind.audio,
          length: '34.0 s',
          limit: '30 s',
        ),
      ),
    );
    await audioTurn(responderFor(responders(retriever)));
    expect(conversation.prompts.single, PromptBuilder.spokenQuestion);
    final retrieval = sides.whereType<ChatRetrieval>().single.retrieval;
    expect(retrieval.outcome, RetrievalOutcome.failed);
    expect(retrieval.queryKind, KnowledgeQueryKind.audio);
    expect(retrieval.detail, contains('too long'));
  });

  test('on an agent chat the excerpts leave the skills open, and a picture '
      'gets the hint that does too', () async {
    await conversation.open(kVoiceChatProfile, skills: const [_time]);
    expect(conversation.hasSkills, isTrue);
    final cats = Uint8List.fromList([0x89, 0x50, 0x4E, 0x47]);
    final retriever = FakeRetriever(Result.ok(used([passage(1)])));
    await audioTurn(responderFor(responders(retriever), image: cats));
    final prompt = conversation.prompts.single;
    expect(
      prompt,
      PromptBuilder.buildSpoken(
        [passage(1)],
        skills: const ['current-time'],
        image: true,
      ),
    );
    expect(
      prompt,
      contains(PromptBuilder.excerptSkills(const ['current-time'])),
      reason: 'the chat\'s own skills, by name',
    );
    expect(prompt, endsWith(PromptBuilder.spokenImageHint));
    expect(
      prompt,
      isNot(contains(PromptBuilder.directAnswerHint(const ['current-time']))),
    );
    expect(conversation.images.single, same(cats));
  });

  test('a chat that cannot take the recording fails the turn (a stream '
      'error for VoiceSession), never a text-only turn', () async {
    // The chat was opened before the model was reloaded without audio.
    conversation.leaveOpen(kVoiceChatProfile);
    conversation.capabilities = const ChatCapabilities(
      modelName: 'Gemma 4 E2B',
      images: true,
      tools: true,
    );
    final responder = ChatTurnResponder(conversation: conversation)
        .prepare(const TurnRequest(typed: false, turn: 1))
        .responder(sides.add);
    // The responder was built for a model without audio: no audio turn.
    expect(responder.respondToAudio, isNull);

    // Built while the model heard audio, asked after the chat lost it.
    conversation.capabilities = _audioGemma;
    final built = ChatTurnResponder(conversation: conversation)
        .prepare(const TurnRequest(typed: false, turn: 2))
        .responder(sides.add);
    conversation
      ..capabilities = const ChatCapabilities(
        modelName: 'Gemma 4 E2B',
        images: true,
        tools: true,
      )
      ..leaveOpen(kVoiceChatProfile);
    await expectLater(
      built.respondToAudio!(wav),
      emitsError(isA<ConversationAudioUnsupportedException>()),
    );
    expect(conversation.prompts.single, PromptBuilder.spokenQuestion);
  });

  test('a stop before the ask: nothing is asked', () async {
    final retriever = FakeRetriever(Result.ok(used([passage(1)])))
      ..gate = Completer<void>();
    final responder = responderFor(responders(retriever));
    final text = <String>[];
    final done = responder.respondToAudio!(wav)
        .listen(text.add)
        .asFuture<void>();
    await pumpEventQueue();
    await responder.stop();
    retriever.gate!.complete();
    await done;
    expect(conversation.prompts, isEmpty);
    expect(sides, isEmpty);
  });

  test('transcribed puts the words into the conversation\'s text history', () {
    responders(null).transcribed(5, 'What is LiteRT?');
    expect(conversation.historyUpdates, [(5, 'What is LiteRT?')]);
  });

  test('a typed or transcribed turn records its question (trimmed) as the '
      'history text', () async {
    final responder = responderFor(responders(null), typed: true, turn: 9);
    final done = responder.respond(' What is LiteRT? ').drain<void>();
    await pumpEventQueue();
    conversation.emit('A runtime.');
    await conversation.finish();
    await done;
    expect(conversation.historyTexts.single, 'What is LiteRT?');
    expect(conversation.askedTurns.single, 9);
    expect(conversation.audios.single, isNull);
  });
}
