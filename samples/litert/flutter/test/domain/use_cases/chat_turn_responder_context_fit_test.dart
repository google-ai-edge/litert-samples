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

import 'package:flutter/foundation.dart';
import 'package:flutter_edge_ai/flutter_edge_ai.dart'
    show ModelType, PreferredBackend;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/repositories/conversation_repository.dart';
import 'package:litert_edge_demos/data/repositories/conversation_repository_edge_ai.dart';
import 'package:litert_edge_demos/data/services/llm/llm_service.dart';
import 'package:litert_edge_demos/domain/models/chat_model_config.dart';
import 'package:litert_edge_demos/domain/models/chat_side_event.dart';
import 'package:litert_edge_demos/domain/models/knowledge.dart';
import 'package:litert_edge_demos/domain/use_cases/chat_turn_responder.dart';
import 'package:litert_edge_demos/domain/use_cases/prompt_builder.dart';
import 'package:litert_edge_demos/domain/use_cases/turn_responder_factory.dart';
import 'package:litert_edge_demos/utils/result.dart';
import 'package:litert_edge_demos/utils/wav.dart';

import '../../fakes/fake_knowledge.dart';
import '../../fakes/fake_tool_model.dart';

/// A chat model with its audio encoder; its window is the fake model's.
const _withAudio = ChatModelConfig(
  name: 'Gemma 4 E2B',
  modelType: ModelType.gemma4,
  llm: LlmConfig(
    maxTokens: 1280,
    backend: PreferredBackend.gpu,
    maxNumImages: 1,
    supportImage: true,
    supportAudio: true,
  ),
  tools: false,
);

/// The fake tokenizer's count (length / 4).
int _tokens(String text) => (text.length / 4).ceil();

/// A knowledge-base chunk of about 230 tokens, as the chunker makes them.
Passage _chunk(int n, double similarity) => Passage(
  id: 'litert-overview.md#$n',
  doc: 'litert-overview.md',
  title: 'LiteRT overview',
  section: 'Section $n',
  content: 'LiteRT overview › Section $n\n\n${'Body $n. ' * 112}',
  similarity: similarity,
);

/// Demo 1's responder over the real repository on a small context window
/// (a 1280-token NPU build): the knowledge-base excerpts are fitted to the
/// window before the turn, lowest score first, and the retrieval says how
/// many went in.
void main() {
  late FakeToolModel model;
  late EdgeAiConversationRepository repo;
  late List<ChatSideEvent> sides;
  final chunks = [_chunk(1, 0.84), _chunk(2, 0.79), _chunk(3, 0.72)];
  final retriever = FakeRetriever(
    Result.ok(
      Retrieval(
        outcome: RetrievalOutcome.used,
        passages: chunks,
        candidates: chunks,
        gate: 0.70,
        latency: const Duration(milliseconds: 120),
      ),
    ),
  );
  // 2 s of 16 kHz mono PCM16: 50 tokens at 25 a second.
  final wav = wavFromPcm16(Uint8List(64000), sampleRate: 16000);

  Future<void> openOn(int window) async {
    model = FakeToolModel(maxTokens: window);
    final llm = LlmService(loadModel: (_) async => model);
    expect(await llm.load(_withAudio), isA<Ok<LlmInfo>>());
    repo = EdgeAiConversationRepository(
      llm: llm,
      stopTimeout: const Duration(seconds: 2),
    );
    expect(await repo.open(kVoiceChatProfile), isA<Ok<void>>());
  }

  setUp(() => sides = []);
  tearDown(() => repo.close());

  ChatTurnResponder responders() =>
      ChatTurnResponder(conversation: repo, retriever: retriever);

  Retrieval retrieval() => sides.whereType<ChatRetrieval>().single.retrieval;

  /// What the turn sent to the engine as the user's text.
  String sent() => model.lastSession.queries.where((m) => m.isUser).single.text;

  /// The most leading excerpts whose prompt fits [room] tokens.
  int keepFor(int room, String Function(List<Passage>) prompt) {
    for (var keep = chunks.length; keep >= 0; keep--) {
      if (_tokens(prompt(chunks.sublist(0, keep))) <= room) return keep;
    }
    return -1;
  }

  // What a prompt may take in a 1280-token window: less the headroom, the
  // reply allowance and the turn overhead.
  final room =
      1280 -
      kContextHeadroomTokens -
      kVoiceChatProfile.maxOutputTokens -
      kTurnOverheadTokens;

  group('a 1280-token window', () {
    setUp(() => openOn(1280));

    test('a typed question: the two best excerpts go in, the third (lowest '
        'score) is dropped, the reply streams, and the retrieval says '
        '"2 of 3 excerpts (context)"', () async {
      const question = 'What is LiteRT?';
      String prompt(List<Passage> p) => PromptBuilder.build(question, p);
      expect(
        keepFor(room, prompt),
        2,
        reason: 'the chunks are sized for 2 of 3 to fit',
      );
      expect(
        _tokens(prompt(chunks)) + kVoiceChatProfile.maxOutputTokens,
        greaterThan(1280 - kContextHeadroomTokens - kTurnOverheadTokens),
        reason: 'all three would raise ConversationTooLongException',
      );
      model.script.add(const TextTurn(['It is a runtime [1].']));

      final reply = await responders()
          .prepare(const TurnRequest(typed: true))
          .responder(sides.add)
          .respond(question)
          .join();

      expect(reply, 'It is a runtime [1].');
      expect(sent(), prompt(chunks.sublist(0, 2)));
      expect(sent(), isNot(contains('Body 3.')));
      final shown = retrieval();
      expect(shown.outcome, RetrievalOutcome.used);
      expect(shown.passages, chunks.sublist(0, 2));
      expect(shown.candidates, chunks, reason: 'the search found three');
      expect(shown.detail, '2 of 3 excerpts (context)');
    });

    test('a spoken question (audio mode): its recording counts too, and the '
        'excerpts are fitted beside it', () async {
      String prompt(List<Passage> p) => PromptBuilder.buildSpoken(p);
      final keep = keepFor(room - 50, prompt);
      expect(keep, inInclusiveRange(1, 2), reason: 'some, not all, fit');
      model.script.add(const TextTurn(['Paris [1].']));

      final responder = responders()
          .prepare(const TurnRequest(typed: false, turn: 1))
          .responder(sides.add);
      expect(responder.respondToAudio, isNotNull);
      final reply = await responder.respondToAudio!(wav).join();

      expect(reply, 'Paris [1].');
      expect(sent(), prompt(chunks.sublist(0, keep)));
      expect(retrieval().passages, chunks.sublist(0, keep));
      expect(retrieval().detail, '$keep of 3 excerpts (context)');
    });

    test('not even the question without excerpts fits: the turn fails as too '
        'long, as before, and the retrieval keeps all three', () async {
      final question = 'Why? ' * (room ~/ 5 * 4 + 8);
      expect(_tokens(question), greaterThan(room));

      final asking = responders()
          .prepare(const TurnRequest(typed: true))
          .responder(sides.add)
          .respond(question)
          .join();

      await expectLater(asking, throwsA(isA<ConversationTooLongException>()));
      expect(retrieval().passages, chunks);
      expect(retrieval().detail, isNull);
    });
  });

  test(
    'a roomy window (4096) keeps every excerpt and says nothing more',
    () async {
      await openOn(4096);
      model.script.add(const TextTurn(['It is a runtime [1].']));

      await responders()
          .prepare(const TurnRequest(typed: true))
          .responder(sides.add)
          .respond('What is LiteRT?')
          .join();

      expect(sent(), PromptBuilder.build('What is LiteRT?', chunks));
      expect(retrieval().passages, chunks);
      expect(retrieval().detail, isNull);
    },
  );
}
