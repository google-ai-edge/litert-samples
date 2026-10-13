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

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/domain/models/chat_entry.dart';
import 'package:litert_edge_demos/domain/models/knowledge.dart';
import 'package:litert_edge_demos/ui/features/voice_chat/views/citation_chips.dart';
import 'package:litert_edge_demos/ui/features/voice_chat/views/message_bubble.dart';

import '../../../../fakes/fake_knowledge.dart';

Widget bubble(ReplyKnowledge? knowledge, {String text = 'Answer [1].'}) =>
    MaterialApp(
      home: Scaffold(
        body: MessageBubble(
          entry: ChatEntry(
            role: ChatRole.assistant,
            text: text,
            knowledge: knowledge,
          ),
        ),
      ),
    );

void main() {
  final passages = [
    passage(1, similarity: 0.7, section: 'What LiteRT is'),
    passage(
      2,
      similarity: 0.52,
      source: null,
      doc: 'yolo26n.md',
      title: 'YOLO26n detector',
    ),
  ];
  final used = Retrieval(
    outcome: RetrievalOutcome.used,
    passages: passages,
    candidates: passages,
    gate: 0.4,
  );

  testWidgets('one chip per excerpt, "Title › Section · similarity"; the '
      'uncited one is dimmed', (tester) async {
    await tester.pumpWidget(
      bubble(ReplyKnowledge(retrieval: used, cited: const {1})),
    );

    expect(find.byKey(KnowledgeChipKeys.citation(1)), findsOneWidget);
    expect(find.byKey(KnowledgeChipKeys.citation(2)), findsOneWidget);
    expect(
      find.text('LiteRT overview › What LiteRT is · 0.70'),
      findsOneWidget,
    );
    expect(find.text('YOLO26n detector › Section 2 · 0.52'), findsOneWidget);
    double opacityOf(int n) => tester
        .widget<Opacity>(
          find
              .descendant(
                of: find.byKey(KnowledgeChipKeys.citation(n)),
                matching: find.byType(Opacity),
              )
              .first,
        )
        .opacity;
    expect(opacityOf(1), 1);
    expect(opacityOf(2), lessThan(1));
  });

  test('excerpts of one document are told apart by their section, and by '
      'a part number when they share it too', () {
    const title = 'YOLO26n FP16 object detector for LiteRT';
    final same = [
      passage(1, title: title, section: 'Inputs'),
      passage(2, title: title, section: 'Inputs'),
      passage(3, title: title, section: 'Outputs'),
      passage(4, title: 'LiteRT overview', section: 'What LiteRT is'),
    ];

    expect(citationLabels(same), [
      'Inputs (1)',
      'Inputs (2)',
      'Outputs',
      'LiteRT overview › What LiteRT is',
    ]);
  });

  testWidgets('a tap opens the excerpt and its source', (tester) async {
    await tester.pumpWidget(
      bubble(ReplyKnowledge(retrieval: used, cited: const {1})),
    );

    await tester.tap(find.byKey(KnowledgeChipKeys.citation(1)));
    await tester.pumpAndSettle();

    expect(find.text('[1] LiteRT overview › What LiteRT is'), findsOneWidget);
    expect(find.text('Body of excerpt 1.'), findsOneWidget);
    expect(find.text('https://ai.google.dev/edge/litert'), findsOneWidget);
    await tester.tap(find.text('Close'));
    await tester.pumpAndSettle();
    expect(find.text('Body of excerpt 1.'), findsNothing);
  });

  testWidgets('unavailable and failed show one chip saying so; below the '
      'gate shows nothing', (tester) async {
    await tester.pumpWidget(
      bubble(
        const ReplyKnowledge(
          retrieval: Retrieval(
            outcome: RetrievalOutcome.unavailable,
            detail: 'indexing 42%',
          ),
        ),
      ),
    );
    expect(
      find.text('Knowledge base unavailable: indexing 42%'),
      findsOneWidget,
    );

    await tester.pumpWidget(
      bubble(
        const ReplyKnowledge(
          retrieval: Retrieval(
            outcome: RetrievalOutcome.failed,
            detail: 'database is locked',
          ),
        ),
      ),
    );
    expect(
      find.text('Knowledge base search failed: database is locked'),
      findsOneWidget,
    );

    await tester.pumpWidget(
      bubble(
        const ReplyKnowledge(
          retrieval: Retrieval(outcome: RetrievalOutcome.belowGate),
        ),
      ),
    );
    expect(find.byKey(KnowledgeChipKeys.status), findsNothing);
    expect(find.byType(ActionChip), findsNothing);
  });

  testWidgets('long labels and reasons fit a phone-width screen', (
    tester,
  ) async {
    await tester.binding.setSurfaceSize(const Size(360, 640));
    addTearDown(() => tester.binding.setSurfaceSize(null));
    final long = passage(
      1,
      similarity: 0.62,
      title: 'On-device RAG and embeddings with flutter_gemma',
      section: 'Rules for using RAG with flutter_gemma › Metadata filters',
    );

    await tester.pumpWidget(
      bubble(
        ReplyKnowledge(
          retrieval: Retrieval(
            outcome: RetrievalOutcome.used,
            passages: [long],
            candidates: [long],
          ),
          cited: const {1},
        ),
      ),
    );
    expect(tester.takeException(), isNull);

    await tester.pumpWidget(
      bubble(
        const ReplyKnowledge(
          retrieval: Retrieval(
            outcome: RetrievalOutcome.unavailable,
            detail: 'The built-in EmbeddingGemma is not in this app bundle',
          ),
        ),
      ),
    );
    expect(tester.takeException(), isNull);
    expect(find.byKey(KnowledgeChipKeys.status), findsOneWidget);
  });

  testWidgets('a reply without a retrieval has no chips', (tester) async {
    await tester.pumpWidget(bubble(null, text: 'Hello.'));

    expect(find.byType(KnowledgeChips), findsNothing);
  });
}
