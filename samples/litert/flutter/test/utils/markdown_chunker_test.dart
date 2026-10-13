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

import 'dart:convert';
import 'dart:io';

import 'package:dart_sentencepiece_tokenizer/dart_sentencepiece_tokenizer.dart';
import 'package:flutter_edge_ai/flutter_edge_ai.dart' show TaskType;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/utils/markdown_chunker.dart';

/// The chunker's budget: 1 per prose character, 2 per table character.
int cost(String body) => body
    .split('\n')
    .map((l) => l.trimLeft().startsWith('|') ? l.length * 2 : l.length)
    .fold(0, (sum, n) => sum + n + 1);

/// EmbeddingGemma's tokenizer, built into the app (tool/fetch_models.sh
/// fetches it into assets/models/).
const _tokenizerPath = 'assets/models/sentencepiece.model';

void main() {
  const chunker = MarkdownChunker();

  group('the real knowledge base (assets/kb)', () {
    final files =
        Directory('assets/kb')
            .listSync()
            .whereType<File>()
            .where((f) => f.path.endsWith('.md'))
            .toList()
          ..sort((a, b) => a.path.compareTo(b.path));
    final byDoc = {
      for (final f in files)
        f.uri.pathSegments.last: chunker.chunk(
          f.readAsStringSync(),
          docId: f.uri.pathSegments.last,
        ),
    };
    final all = [for (final chunks in byDoc.values) ...chunks];

    test('16 documents give about 300 chunks within the budget', () {
      expect(files, hasLength(16));
      // 297 today; the range leaves room for edits to the documents.
      expect(all.length, inInclusiveRange(270, 310));
      for (final chunk in all) {
        expect(
          cost(chunk.body),
          lessThanOrEqualTo(1400 + 1),
          reason: '${chunk.id} (${chunk.section}) is over the budget',
        );
        expect(chunk.body.trim(), isNotEmpty, reason: chunk.id);
      }
    });

    test('content is "Title › Section", a blank line, then the body', () {
      for (final chunk in all) {
        expect(
          chunk.content,
          '${chunk.title} › ${chunk.section}\n\n${chunk.body}',
        );
        expect(chunk.title, isNot(contains('\n')));
      }
      final first = byDoc['litert-overview.md']!.first;
      expect(first.title, "LiteRT, Google's on-device AI runtime");
      expect(first.source, startsWith('https://'));
    });

    test('ids are doc#n, sequential per document and stable', () {
      for (final MapEntry(key: doc, value: chunks) in byDoc.entries) {
        expect(chunks.map((c) => c.id), [
          for (var i = 0; i < chunks.length; i++) '$doc#$i',
        ]);
      }
      final again = chunker.chunk(
        File('assets/kb/gemma-4-e2b.md').readAsStringSync(),
        docId: 'gemma-4-e2b.md',
      );
      expect(
        again.map((c) => (c.id, c.content)),
        byDoc['gemma-4-e2b.md']!.map((c) => (c.id, c.content)),
      );
    });

    test('metadata carries doc, title, section, chunk and source', () {
      final chunk = byDoc['flutter-edge-ai-rag.md']![3];
      final meta = jsonDecode(chunk.metadataJson) as Map<String, Object?>;
      expect(meta, {
        'doc': 'flutter-edge-ai-rag.md',
        'title': chunk.title,
        'section': chunk.section,
        'chunk': 3,
        // The front matter lists several URLs separated by " ; ", each
        // followed by the files it was read from in parentheses.
        'source': 'https://pub.dev/packages/flutter_edge_ai_rag/versions/1.0.0',
      });
      expect(meta['source'], isNot(contains(';')));
    });

    test('H3 headings start sections with an "H2 › H3" path', () {
      final api = byDoc['litert-compiled-model-api.md']!;
      expect(
        api.map((c) => c.section),
        contains(
          endsWith(' › Telling a real GPU run from a silent CPU fallback'),
        ),
      );
      final conversion = byDoc['model-conversion-and-quantization.md']!;
      expect(
        conversion.map((c) => c.section),
        contains(endsWith(' › Converting a PyTorch model to a .tflite file')),
      );
    });

    test('fenced code is dropped: no fence or code-only heading leaks', () {
      for (final chunk in all) {
        expect(chunk.body, isNot(contains('```')), reason: chunk.id);
        expect(chunk.body, isNot(contains('~~~')), reason: chunk.id);
      }
      // flutter-edge-ai-rag.md shows FlutterEdgeAi.initialize(...) only in code.
      final rag = byDoc['flutter-edge-ai-rag.md']!.map((c) => c.body).join();
      expect(rag, isNot(contains('await FlutterEdgeAi.initialize(')));
    });

    test('every golden section is present', () {
      final golden = jsonDecode(
        File('test_assets/kb_golden.json').readAsStringSync(),
      ) as Map<String, Object?>;
      for (final entry in golden['on_topic']! as List<Object?>) {
        final e = entry! as Map<String, Object?>;
        final doc = e['expect_doc']! as String;
        final section = e['expect_section']! as String;
        expect(
          byDoc[doc]!.map((c) => c.section),
          contains(anyOf(section, endsWith(' › $section'))),
          reason: '$doc: $section',
        );
      }
    });

    test(
      'no chunk exceeds the 512-token window with the document prefix',
      () {
        final tokenizer = SentencePieceTokenizer.fromModelFileSync(
          _tokenizerPath,
          config: SentencePieceConfig.gemma,
        );
        var maxTokens = 0;
        for (final chunk in all) {
          final tokens = tokenizer
              .encode('${TaskType.retrievalDocument.prefix}${chunk.content}')
              .ids
              .length;
          if (tokens > maxTokens) maxTokens = tokens;
          expect(tokens, lessThanOrEqualTo(512), reason: chunk.id);
        }
        // Reports the knowledge base's chunk count and token peak.
        // ignore: avoid_print
        print('KB chunks=${all.length} max_tokens=$maxTokens');
      },
      skip: File(_tokenizerPath).existsSync()
          ? false
          : 'EmbeddingGemma tokenizer not at $_tokenizerPath (run '
                'tool/fetch_models.sh)',
    );
  });

  group('front matter', () {
    test('title and the first source URL; CRLF and a BOM are handled', () {
      const doc =
          '\uFEFF---\r\ntitle: "Quoted title"\r\nsource: https://a.example/x ; '
          'https://b.example/y\r\nlicense: MIT\r\n---\r\n\r\n# Ignored H1\r\n\r\n'
          '## One\r\n\r\nFirst line.\r\n';
      final chunks = chunker.chunk(doc, docId: 'd.md');
      expect(chunks.single.title, 'Quoted title');
      expect(chunks.single.source, 'https://a.example/x');
      expect(chunks.single.section, 'One');
      expect(chunks.single.body, 'First line.');
    });

    test('without front matter the H1 is the title, else the doc id', () {
      expect(
        chunker.chunk('# Heading one\n\nIntro.', docId: 'd.md').single.title,
        'Heading one',
      );
      expect(chunker.chunk('Intro.', docId: 'd.md').single.title, 'd.md');
    });

    test('an unclosed front matter block is an error', () {
      expect(
        () => chunker.chunk('---\ntitle: x\n\n## A\n\nText.', docId: 'd.md'),
        throwsFormatException,
      );
    });
  });

  group('fences', () {
    test('a heading inside a fence does not start a section', () {
      const doc =
          '## Real\n\nBefore.\n\n```bash\n# not a heading\n## nor this\n```\n\n'
          'After.';
      final chunks = chunker.chunk(doc, docId: 'd.md');
      expect(chunks.single.section, 'Real');
      expect(chunks.single.body, 'Before.\n\nAfter.');
    });

    test('a fence closes only on the same character, at least as long', () {
      const doc =
          '## A\n\n````md\n```\n## inside\n~~~\n````\n\nText.\n\n'
          '~~~\n```\n## inside too\n~~~\n\nMore.';
      final chunks = chunker.chunk(doc, docId: 'd.md');
      expect(chunks.single.body, 'Text.\n\nMore.');
    });

    test('an unclosed fence is an error naming its line', () {
      expect(
        () => chunker.chunk(
          '## A\n\nText.\n\n```dart\nvoid f() {}\n',
          docId: 'd.md',
        ),
        throwsA(
          isA<FormatException>().having(
            (e) => e.message,
            'message',
            allOf(contains('d.md'), contains('line 5')),
          ),
        ),
      );
    });
  });

  group('headings and sections', () {
    test('Overview before the first H2; H3 paths; H4 stays in the body', () {
      const doc =
          '# Title\n\nIntro text.\n\n## Alpha\n\nAlpha text.\n\n'
          '### Beta\n\nBeta text.\n\n#### Gamma\n\nGamma text.\n\n'
          '## Delta\n\n### Epsilon\n\nEpsilon text.';
      final chunks = chunker.chunk(doc, docId: 'd.md');
      expect(chunks.map((c) => c.section), [
        'Overview',
        'Alpha',
        'Alpha › Beta',
        'Delta › Epsilon',
      ]);
      expect(chunks[2].body, 'Beta text.\n\n#### Gamma\n\nGamma text.');
    });

    test('small sections are never merged', () {
      final chunks = chunker.chunk(
        '## A\n\nShort.\n\n## B\n\nAlso short.',
        docId: 'd.md',
      );
      expect(chunks.map((c) => c.body), ['Short.', 'Also short.']);
    });
  });

  group('splitting', () {
    String sentence(int i) =>
        'Sentence number $i talks about LiteRT and the compiled model API '
        'in enough words to take some room.';

    String lastSentence(String body) =>
        RegExp(r'Sentence number \d+[^.]*\.$').firstMatch(body)![0]!;

    test('a long section splits into ceil(cost/1400) balanced parts; each '
        "later part starts with the previous one's last sentence", () {
      // ~1730 characters: two parts of ~870, so the overlap fits.
      final prose = [for (var i = 0; i < 17; i++) sentence(i)].join(' ');
      final chunks = chunker.chunk('## Long\n\n$prose', docId: 'd.md');
      expect(chunks.length, (prose.length / 1400).ceil());
      for (final chunk in chunks) {
        expect(chunk.body.length, lessThanOrEqualTo(1400));
        expect(chunk.body.length, greaterThan(700), reason: 'balanced');
        expect(chunk.section, 'Long');
      }
      for (var i = 1; i < chunks.length; i++) {
        final overlap = lastSentence(chunks[i - 1].body);
        expect(chunks[i].body, startsWith('$overlap Sentence number'));
      }
      final joined = chunks.map((c) => c.body).join(' ');
      for (var i = 0; i < 17; i++) {
        expect(joined, contains(sentence(i)), reason: 'nothing lost');
      }
    });

    test('the overlap is left out when it would not fit', () {
      // Two parts that are each as full as whole sentences allow.
      final perPart = 1401 ~/ (sentence(10).length + 1);
      final count = perPart * 2;
      final prose = [for (var i = 10; i < 10 + count; i++) sentence(i)];
      final chunks = chunker.chunk(
        '## Long\n\n${prose.join(' ')}',
        docId: 'd.md',
      );
      expect(chunks, hasLength(2));
      expect(chunks.every((c) => c.body.length <= 1400), isTrue);
      expect(chunks[1].body, isNot(startsWith(lastSentence(chunks[0].body))));
      expect(
        [for (final c in chunks) ...c.body.split(RegExp(r'(?<=\.) '))],
        prose,
        reason: 'every sentence exactly once, in order',
      );
    });

    test('"e.g." and "i.e." never end a sentence', () {
      final prose = [
        for (var i = 0; i < 40; i++)
          'Item $i uses an accelerator, e.g. GPU or NPU, i.e. not the CPU here.',
      ].join(' ');
      final chunks = chunker.chunk('## Long\n\n$prose', docId: 'd.md');
      expect(chunks.length, greaterThan(1));
      for (final chunk in chunks) {
        expect(chunk.body, isNot(endsWith('e.g.')));
        expect(chunk.body, isNot(endsWith('i.e.')));
        expect(chunk.body, endsWith('here.'));
      }
    });

    test('an oversized table splits into row groups repeating the header; '
        'separator rows are dropped; no overlap crosses a table', () {
      final rows = [
        for (var i = 0; i < 40; i++)
          '| model-$i | ${i * 3} ms | ${i * 7} MB | runs on the GPU |',
      ];
      final doc =
          '## Bench\n\nThe table lists measured numbers.\n\n'
          '| Model | Latency | Size | Note |\n|---|:---:|---:|---|\n'
          '${rows.join('\n')}';
      final chunks = chunker.chunk(doc, docId: 'd.md');
      expect(chunks.length, greaterThan(1));
      for (final chunk in chunks) {
        expect(chunk.body, isNot(contains('|---')));
        expect(cost(chunk.body), lessThanOrEqualTo(1400 + 1));
      }
      final tableParts = chunks.where((c) => c.body.contains('| model-'));
      for (final part in tableParts) {
        final lines = part.body.split('\n\n').last.split('\n');
        expect(lines.first, '| Model | Latency | Size | Note |');
      }
      // Every row is kept exactly once.
      final kept = chunks
          .expand((c) => c.body.split('\n'))
          .where((l) => l.startsWith('| model-'))
          .toList();
      expect(kept, rows);
      // The part after the prose block starts with the table, not with a
      // repeated sentence.
      expect(
        chunks.skip(1).every((c) => c.body.startsWith('| Model |')),
        isTrue,
      );
    });
  });
}
