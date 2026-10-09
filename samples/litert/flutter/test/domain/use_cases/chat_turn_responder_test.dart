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

import 'package:flutter_edge_ai_speech/flutter_edge_ai_speech.dart'
    show VoiceResponder;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/domain/models/chat_side_event.dart';
import 'package:litert_edge_demos/domain/models/knowledge.dart';
import 'package:litert_edge_demos/domain/use_cases/chat_turn_responder.dart';
import 'package:litert_edge_demos/domain/use_cases/prompt_builder.dart';
import 'package:litert_edge_demos/domain/use_cases/turn_responder_factory.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../fakes/fake_conversation_repository.dart';
import '../../fakes/fake_knowledge.dart';

/// The per-turn retrieval in Demo 1's responder.
void main() {
  late FakeConversationRepository conversation;
  late List<ChatSideEvent> sides;

  setUp(() {
    // The demo opens the chat before any turn; the turns here ask it.
    conversation = FakeConversationRepository()..leaveOpen(kVoiceChatProfile);
    sides = [];
  });

  tearDown(() => conversation.close());

  VoiceResponder responderWith(FakeRetriever? retriever) => ChatTurnResponder(
    conversation: conversation,
    retriever: retriever,
  ).prepare(const TurnRequest(typed: true)).responder(sides.add);

  /// Runs one turn: [question] in, the model answers [reply].
  Future<String> turn(
    VoiceResponder responder,
    String question, {
    String reply = 'It is a runtime [1].',
  }) async {
    final text = StringBuffer();
    final done = responder
        .respond(question)
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
    gate: 0.4,
    latency: const Duration(milliseconds: 130),
  );

  test('excerpts that cleared the gate land in the prompt; the retrieval '
      'goes out first as a side event', () async {
    final passages = [passage(1, similarity: 0.7), passage(2, similarity: 0.5)];
    final retriever = FakeRetriever(Result.ok(used(passages)));

    final reply = await turn(responderWith(retriever), 'What is LiteRT?');

    expect(retriever.questions, ['What is LiteRT?']);
    expect(
      conversation.prompts.single,
      PromptBuilder.build('What is LiteRT?', passages),
    );
    expect(
      conversation.prompts.single,
      contains('[1] LiteRT overview › Section 1'),
    );
    expect(conversation.prompts.single, endsWith('Question: What is LiteRT?'));
    expect(reply, 'It is a runtime [1].', reason: 'the UI keeps the markers');
    expect(sides, hasLength(2));
    expect(
      sides.first,
      isA<ChatRetrieval>().having(
        (e) => e.retrieval.passages,
        'passages',
        passages,
      ),
    );
    expect(sides.last, isA<ChatGenerationDone>());
  });

  test('below the gate the plain question goes out', () async {
    final retriever = FakeRetriever(
      Result.ok(
        Retrieval(
          outcome: RetrievalOutcome.belowGate,
          candidates: [passage(1, similarity: 0.2)],
          gate: 0.4,
          latency: const Duration(milliseconds: 120),
        ),
      ),
    );

    await turn(responderWith(retriever), 'How do I boil an egg?');

    expect(conversation.prompts.single, 'How do I boil an egg?');
    expect(
      (sides.first as ChatRetrieval).retrieval.outcome,
      RetrievalOutcome.belowGate,
    );
  });

  test('an unavailable knowledge base: plain question, and the reason goes '
      'to the UI', () async {
    final retriever = FakeRetriever(
      const Result.ok(
        Retrieval(
          outcome: RetrievalOutcome.unavailable,
          detail: 'indexing 42%',
        ),
      ),
    );

    await turn(responderWith(retriever), 'What is LiteRT?');

    expect(conversation.prompts.single, 'What is LiteRT?');
    final retrieval = (sides.first as ChatRetrieval).retrieval;
    expect(retrieval.outcome, RetrievalOutcome.unavailable);
    expect(retrieval.detail, 'indexing 42%');
  });

  test('a failed search is reported as failed and the turn goes on', () async {
    final retriever = FakeRetriever(
      Result.error(Exception('database is locked')),
    );

    final reply = await turn(responderWith(retriever), 'What is LiteRT?');

    expect(conversation.prompts.single, 'What is LiteRT?');
    final retrieval = (sides.first as ChatRetrieval).retrieval;
    expect(retrieval.outcome, RetrievalOutcome.failed);
    expect(retrieval.detail, contains('database is locked'));
    expect(reply, isNotEmpty);
  });

  test('without a retriever there is no retrieval event', () async {
    await turn(responderWith(null), 'Hello?');

    expect(conversation.prompts.single, 'Hello?');
    expect(sides.single, isA<ChatGenerationDone>());
  });

  test('a stop during the retrieval never asks the model', () async {
    final retriever = FakeRetriever(Result.ok(used([passage(1)])))
      ..gate = Completer<void>();
    final responder = responderWith(retriever);
    final done = responder.respond('What is LiteRT?').drain<void>();
    await pumpEventQueue();

    await responder.stop();
    retriever.gate!.complete();
    await done;

    expect(conversation.prompts, isEmpty);
    expect(conversation.stopCalls, 0, reason: 'no ask of ours to stop');
    expect(sides, isEmpty, reason: 'its excerpts were in no prompt');
  });

  group('context reset', () {
    test('forwarded as ChatContextReset as soon as it happens', () async {
      final responder = responderWith(null);
      final done = responder.respond('Hi').listen((_) {}).asFuture<void>();
      await pumpEventQueue();

      conversation.emitContextReset();
      await pumpEventQueue();
      expect(sides.whereType<ChatContextReset>(), hasLength(1));

      conversation.emit('Hello.');
      await conversation.finish();
      await done;
      expect(sides.whereType<ChatContextReset>(), hasLength(1));
    });

    test('forwarded even when the turn then fails', () async {
      final responder = responderWith(null);
      final done = responder.respond('Hi').listen((_) {}).asFuture<void>();
      await pumpEventQueue();

      conversation.emitContextReset();
      await conversation.fail(Exception('GPU lost'));
      await expectLater(done, throwsA(isA<Exception>()));

      expect(sides.whereType<ChatContextReset>(), hasLength(1));
    });

    test('forwarded even after this turn was stopped (barge-in)', () async {
      final responder = responderWith(null);
      final done = responder.respond('Hi').listen((_) {}).asFuture<void>();
      await pumpEventQueue();
      conversation.stopGate = Completer<void>();
      final stopping = responder.stop();
      await pumpEventQueue();

      conversation.emitContextReset();
      await pumpEventQueue();
      conversation.stopGate!.complete();
      await stopping;
      await done;

      expect(sides.whereType<ChatContextReset>(), hasLength(1));
    });
  });
}
