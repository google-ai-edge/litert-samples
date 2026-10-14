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

import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/domain/models/knowledge_query.dart';

void main() {
  test('text and audio questions: their kinds, and the WAV kept as given', () {
    const text = KnowledgeQuery.text('What is LiteRT?');
    final wav = Uint8List(44 + 320);
    final audio = KnowledgeQuery.audio(wav);

    expect(text, isA<KnowledgeTextQuery>());
    expect(text.kind, KnowledgeQueryKind.text);
    expect((text as KnowledgeTextQuery).text, 'What is LiteRT?');
    expect(audio.kind, KnowledgeQueryKind.audio);
    expect((audio as KnowledgeAudioQuery).wav, same(wav));
  });

  test('exhaustive over its two shapes', () {
    String describe(KnowledgeQuery q) => switch (q) {
      KnowledgeTextQuery(:final text) => 'text "$text"',
      KnowledgeAudioQuery(:final wav) => 'audio ${wav.length} B',
    };
    expect(describe(const KnowledgeQuery.text('hi')), 'text "hi"');
    expect(describe(KnowledgeQuery.audio(Uint8List(48))), 'audio 48 B');
  });

  test('toString never prints the question itself', () {
    expect(
      const KnowledgeQuery.text('my secret question').toString(),
      'KnowledgeQuery.text(18 chars)',
    );
    expect(
      KnowledgeQuery.audio(Uint8List(100)).toString(),
      'KnowledgeQuery.audio(100 bytes)',
    );
  });

  test(
    'a question too long to search says what was measured and the limit',
    () {
      const text = KnowledgeQueryTooLongException(
        kind: KnowledgeQueryKind.text,
        length: '1210 UTF-8 bytes',
        limit: '960 UTF-8 bytes',
      );
      const audio = KnowledgeQueryTooLongException(
        kind: KnowledgeQueryKind.audio,
        length: '34.2 s',
        limit: '30 s',
      );

      expect(
        text.toString(),
        allOf(
          contains('1210 UTF-8 bytes'),
          contains('at most 960 UTF-8 bytes'),
        ),
      );
      expect(
        audio.toString(),
        allOf(
          contains('spoken question'),
          contains('34.2 s'),
          contains('30 s'),
        ),
      );
    },
  );
}
