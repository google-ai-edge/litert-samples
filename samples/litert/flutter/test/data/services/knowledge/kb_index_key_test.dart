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

import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/services/knowledge/kb_documents.dart';
import 'package:litert_edge_demos/data/services/knowledge/kb_index_key.dart';
import 'package:litert_edge_demos/utils/markdown_chunker.dart';

const _key = KbIndexKey(
  documents: '987e8d5ae18cc7542fd9494b929c25e1d53a528768c69a2aae7a7702ef90c478',
  chunker: 'md-chunker-1/1400/300',
  documentPrefix: 'title: none | text: ',
  modelId: 'embeddinggemma-2-740m',
  modelSha256:
      'e7a8a2204b91e0f96e92960e84a09a89212e1633dcb7575a9bf3378b4df77f4c',
  dim: 768,
);

KbIndexKey _with({
  String? documents,
  String? chunker,
  String? documentPrefix,
  String? modelId,
  String? modelSha256,
  int? dim,
}) => KbIndexKey(
  documents: documents ?? _key.documents,
  chunker: chunker ?? _key.chunker,
  documentPrefix: documentPrefix ?? _key.documentPrefix,
  modelId: modelId ?? _key.modelId,
  modelSha256: modelSha256 ?? _key.modelSha256,
  dim: dim ?? _key.dim,
);

KbDocument _doc(String name, String text) =>
    KbDocument(name: name, path: 'assets/kb/$name', bytes: utf8.encode(text));

void main() {
  group('KbIndexKey', () {
    test('the digest is stable and changes with every field', () {
      expect(_with().digest, _key.digest);
      final variants = [
        _with(documents: 'x'),
        _with(chunker: 'md-chunker-2/1400/300'),
        _with(documentPrefix: ''),
        _with(modelId: 'embeddinggemma-v2'),
        _with(modelSha256: 'f' * 64),
        _with(dim: 512),
      ];
      final digests = {_key.digest, for (final v in variants) v.digest};
      expect(digests, hasLength(variants.length + 1));
    });

    test('fields are length-prefixed: moving text between them changes the '
        'digest', () {
      expect(
        _with(modelId: 'ab', documentPrefix: 'c').digest,
        isNot(_with(modelId: 'a', documentPrefix: 'bc').digest),
      );
    });

    test('differences name each field that differs, with short values', () {
      expect(_key.differencesFrom(_key), isEmpty);
      final other = _with(
        documents: '1a2b3c4d${'0' * 56}',
        modelSha256: 'ffffffff${'0' * 56}',
        dim: 512,
      );
      expect(other.differencesFrom(_key), [
        'knowledge-base documents (1a2b3c4d ≠ 987e8d5a)',
        'embedder model file (ffffffff ≠ e7a8a220)',
        'dimension (512 ≠ 768)',
      ]);
      expect(_with(chunker: 'md-chunker-1/1000/300').differencesFrom(_key), [
        'chunker ("md-chunker-1/1000/300" ≠ "md-chunker-1/1400/300")',
      ]);
      expect(
        _with(modelId: 'other').differencesFrom(_key).single,
        'embedder id ("other" ≠ "embeddinggemma-2-740m")',
      );
    });

    test('round-trips through JSON; a missing field is a FormatException', () {
      final json = jsonDecode(jsonEncode(_key.toJson()));
      final back = KbIndexKey.fromJson(json);
      expect(back.digest, _key.digest);
      expect(back.differencesFrom(_key), isEmpty);

      final missing = Map<String, Object?>.of(_key.toJson())..remove('dim');
      expect(() => KbIndexKey.fromJson(missing), throwsFormatException);
      expect(() => KbIndexKey.fromJson('nope'), throwsFormatException);
    });

    test('kb-index/3 keys one self-contained file: no tokenizer digest', () {
      expect(KbIndexKey.format, 'kb-index/3');
      expect(_key.keyFormat, 'kb-index/3');
      expect(_key.toJson().keys, isNot(contains('tokenizerSha256')));
    });

    test('an EmbeddingGemma-300M key (kb-index/2, with its tokenizer digest) '
        'still reads, and differs: never reused', () {
      final old = KbIndexKey.fromJson({
        'format': 'kb-index/2',
        'documents': _key.documents,
        'chunker': _key.chunker,
        'documentPrefix': _key.documentPrefix,
        'modelId': 'embeddinggemma-300M_seq512_mixed-precision',
        'modelSha256':
            'ad09e81557203cb0e177abf9bf8727dfe138a7d394aa0f70f0b2ed16432e121a',
        'tokenizerSha256':
            'd6daa52d93d7aad10e8388bd526c4e501d914b47177398d1d9621f1fe48438c7',
        'dim': 768,
      });

      expect(old.digest, isNot(_key.digest));
      expect(old.differencesFrom(_key), [
        'key format ("kb-index/2" ≠ "kb-index/3")',
        'embedder id ("embeddinggemma-300M_seq512_mixed-precision" ≠ '
            '"embeddinggemma-2-740m")',
        'embedder model file (ad09e815 ≠ e7a8a220)',
      ]);
    });
  });

  group('kbDocumentsHash', () {
    final a = _doc('a.md', '# A');
    final b = _doc('b.md', '# B');

    test('covers order, paths and bytes', () {
      final base = kbDocumentsHash([a, b]);
      expect(kbDocumentsHash([a, b]), base);
      expect(kbDocumentsHash([b, a]), isNot(base));
      expect(kbDocumentsHash([a, _doc('b.md', '# B!')]), isNot(base));
      expect(
        kbDocumentsHash([
          a,
          KbDocument(name: 'b.md', path: 'assets/kb/c/b.md', bytes: b.bytes),
        ]),
        isNot(base),
      );
      expect(kbDocumentsHash([a]), isNot(base));
    });
  });

  test('chunkerId is the version and the budget', () {
    expect(chunkerId(const MarkdownChunker()), 'md-chunker-1/1400/300');
    expect(
      chunkerId(const MarkdownChunker(maxCost: 900, overlapMaxChars: 100)),
      'md-chunker-1/900/100',
    );
  });

  group('PrebuiltKbManifest', () {
    final manifest = PrebuiltKbManifest(
      key: _key,
      profileId: 'embeddinggemma-2-740m-litertlm-test-profile',
      chunks: 290,
      dbBytes: 3616768,
      dbSha256: '0047ec8b${'0' * 56}',
      store: 'flutter_edge_ai_sqlite 2.0.0',
      embedderBackend: 'cpu',
      builtOn: 'macos',
      builtAt: DateTime.utc(2026, 10, 6, 20, 6, 31),
    );

    test('round-trips through JSON', () {
      final back = PrebuiltKbManifest.fromJson(
        jsonDecode(jsonEncode(manifest.toJson())),
      );
      expect(back.key.digest, _key.digest);
      expect(back.profileId, 'embeddinggemma-2-740m-litertlm-test-profile');
      expect(back.chunks, 290);
      expect(back.dbBytes, 3616768);
      expect(back.dbSha256, manifest.dbSha256);
      expect(back.store, 'flutter_edge_ai_sqlite 2.0.0');
      expect(back.embedderBackend, 'cpu');
      expect(back.builtOn, 'macos');
      expect(back.builtAt, DateTime.utc(2026, 10, 6, 20, 6, 31));
    });

    test('another format or a missing field is a FormatException', () {
      final json = manifest.toJson();
      expect(json['format'], 'kb-prebuilt/2');
      expect(
        () => PrebuiltKbManifest.fromJson({...json, 'format': 'kb-prebuilt/3'}),
        throwsA(
          isFormatException.having(
            (e) => e.message,
            'message',
            contains('kb-prebuilt/3'),
          ),
        ),
      );
      expect(
        () => PrebuiltKbManifest.fromJson({...json}..remove('dbSha256')),
        throwsFormatException,
      );
      expect(
        () => PrebuiltKbManifest.fromJson({...json}..remove('profileId')),
        throwsFormatException,
      );
    });

    test('a kb-prebuilt/1 manifest (no profile id) is refused by its format, '
        'not mistaken for a broken one', () {
      final v1 = {...manifest.toJson(), 'format': 'kb-prebuilt/1'}
        ..remove('profileId');
      expect(
        () => PrebuiltKbManifest.fromJson(v1),
        throwsA(
          isFormatException.having(
            (e) => e.message,
            'message',
            allOf(contains('kb-prebuilt/1'), contains('kb-prebuilt/2')),
          ),
        ),
      );
    });
  });
}
