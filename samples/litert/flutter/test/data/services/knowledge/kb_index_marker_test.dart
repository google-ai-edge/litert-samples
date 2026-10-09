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

import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/services/knowledge/kb_index_key.dart';
import 'package:litert_edge_demos/data/services/knowledge/kb_index_marker.dart';
import 'package:litert_edge_demos/domain/models/knowledge.dart';

final _key = KbIndexKey(
  documents: 'a' * 64,
  chunker: 'md-chunker-1/1400/300',
  documentPrefix: 'title: none | text: ',
  modelId: 'embeddinggemma-300M_seq512_mixed-precision',
  modelSha256: 'b' * 64,
  tokenizerSha256: 'c' * 64,
  dim: 768,
);

void main() {
  late Directory dir;
  late KbIndexMarker marker;

  File file() => File('${dir.path}/index.json');
  Map<String, Object?> written() =>
      jsonDecode(file().readAsStringSync()) as Map<String, Object?>;
  void writeJson(Object json) => file().writeAsStringSync(jsonEncode(json));

  setUp(() {
    dir = Directory.systemTemp.createTempSync('kb_index_marker_test');
    marker = KbIndexMarker(dir);
  });

  tearDown(() => dir.deleteSync(recursive: true));

  group('write', () {
    test('an index embedded on the device: the digest, the key and why the '
        'prebuilt index was skipped, through a temp file', () async {
      await marker.write(
        key: _key,
        chunks: 42,
        origin: KnowledgeOrigin.device,
        prebuiltSkipped: 'this build has no assets/kb_index/manifest.json',
      );

      expect(written(), {
        'hash': _key.digest,
        'chunks': 42,
        'dim': 768,
        'origin': 'device',
        'prebuiltSkipped': 'this build has no assets/kb_index/manifest.json',
        'key': _key.toJson(),
      });
      expect(File('${file().path}.tmp').existsSync(), isFalse);
    });

    test('the prebuilt index: no prebuiltSkipped; an earlier marker is '
        'replaced', () async {
      writeJson({'hash': 'old'});

      await marker.write(
        key: _key,
        chunks: 7,
        origin: KnowledgeOrigin.prebuilt,
      );

      expect(written()['origin'], 'prebuilt');
      expect(written()['chunks'], 7);
      expect(written().containsKey('prebuiltSkipped'), isFalse);
    });
  });

  group('read', () {
    test('what write wrote, for the same key, is a match', () async {
      await marker.write(
        key: _key,
        chunks: 42,
        origin: KnowledgeOrigin.device,
        prebuiltSkipped: 'skipped',
      );

      expect(
        await marker.read(_key),
        isA<KbIndexMarkerMatch>()
            .having((m) => m.chunks, 'chunks', 42)
            .having((m) => m.dim, 'dim', 768)
            .having((m) => m.origin, 'origin', KnowledgeOrigin.device)
            .having((m) => m.prebuiltSkipped, 'prebuiltSkipped', 'skipped'),
      );
    });

    test('a match says prebuilt only for origin "prebuilt"', () async {
      await marker.write(
        key: _key,
        chunks: 1,
        origin: KnowledgeOrigin.prebuilt,
      );
      expect(
        await marker.read(_key),
        isA<KbIndexMarkerMatch>()
            .having((m) => m.origin, 'origin', KnowledgeOrigin.prebuilt)
            .having((m) => m.prebuiltSkipped, 'prebuiltSkipped', isNull),
      );

      writeJson({...written(), 'origin': 'cloud'});
      expect(
        await marker.read(_key),
        isA<KbIndexMarkerMatch>().having(
          (m) => m.origin,
          'origin',
          KnowledgeOrigin.device,
        ),
      );
      writeJson({...written()}..remove('origin'));
      expect(
        await marker.read(_key),
        isA<KbIndexMarkerMatch>().having(
          (m) => m.origin,
          'origin',
          KnowledgeOrigin.device,
        ),
      );
    });

    test('no file: missing', () async {
      expect(await marker.read(_key), isA<KbIndexMarkerMissing>());
    });

    test('not JSON: unreadable, with the parse error', () async {
      file().writeAsStringSync('{not json');

      expect(
        await marker.read(_key),
        isA<KbIndexMarkerUnreadable>().having(
          (m) => m.error.message,
          'error',
          isNotEmpty,
        ),
      );
    });

    test('JSON without a string hash, an int chunks or an int dim: '
        'incomplete', () async {
      for (final json in <Object>[
        {'chunks': 3, 'dim': 768},
        {'hash': _key.digest, 'dim': 768},
        {'hash': _key.digest, 'chunks': 3},
        {'hash': _key.digest, 'chunks': '3', 'dim': 768},
        {'hash': 1, 'chunks': 3, 'dim': 768},
        [_key.digest, 3, 768],
      ]) {
        writeJson(json);
        expect(
          await marker.read(_key),
          isA<KbIndexMarkerIncomplete>(),
          reason: '$json',
        );
      }
    });

    group('another key: stale, saying how', () {
      Future<String> builtFrom() async =>
          (await marker.read(_key) as KbIndexMarkerStale).builtFrom;

      test('the fields that differ', () async {
        final other = KbIndexKey(
          documents: 'd' * 64,
          chunker: _key.chunker,
          documentPrefix: _key.documentPrefix,
          modelId: _key.modelId,
          modelSha256: _key.modelSha256,
          tokenizerSha256: _key.tokenizerSha256,
          dim: 512,
        );
        await marker.write(
          key: other,
          chunks: 3,
          origin: KnowledgeOrigin.device,
        );

        expect(
          await builtFrom(),
          'another knowledge-base documents (dddddddd ≠ aaaaaaaa), dimension '
          '(512 ≠ 768)',
        );
      });

      test('a marker without a key: an earlier build', () async {
        writeJson({'hash': 'old', 'chunks': 3, 'dim': 768});

        expect(await builtFrom(), 'a marker of an earlier build');
      });

      test('a key this build cannot read', () async {
        writeJson({
          'hash': 'old',
          'chunks': 3,
          'dim': 768,
          'key': {'format': 'kb-index/0'},
        });

        expect(await builtFrom(), 'a key this build cannot read');
      });

      test('an equal key under another hash', () async {
        await marker.write(
          key: _key,
          chunks: 3,
          origin: KnowledgeOrigin.device,
        );
        writeJson({...written(), 'hash': 'tampered'});

        expect(await builtFrom(), 'an equal key with another digest');
      });
    });
  });

  test('delete removes the marker, and is fine without one', () async {
    await marker.write(key: _key, chunks: 3, origin: KnowledgeOrigin.device);

    await marker.delete();
    expect(file().existsSync(), isFalse);
    await marker.delete();
    expect(await marker.read(_key), isA<KbIndexMarkerMissing>());
  });
}
