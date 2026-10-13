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

// The prebuilt knowledge-base index shipped in assets/kb_index/ still
// matches this checkout: built from these assets/kb documents, this chunker,
// the built-in embedder files and the resolved flutter_edge_ai_sqlite. When
// this fails, the app would ignore the prebuilt index and embed on every
// device (216 s on a Galaxy S24): rebuild it with tool/build_kb_index.sh.
import 'dart:convert';
import 'dart:io';

import 'package:crypto/crypto.dart';
import 'package:flutter_edge_ai/flutter_edge_ai.dart' show TaskType;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/knowledge_config.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/services/knowledge/kb_documents.dart';
import 'package:litert_edge_demos/data/services/knowledge/kb_index_key.dart';
import 'package:litert_edge_demos/data/services/model_store/bundled_model_files.dart';
import 'package:litert_edge_demos/utils/markdown_chunker.dart';

const _rebuild = 'rebuild it with tool/build_kb_index.sh';

/// `assets/kb/*.md` as `AssetKbDocumentSource` lists them: sorted by path.
List<KbDocument> _documents() {
  final files =
      Directory('assets/kb')
          .listSync()
          .whereType<File>()
          .where((f) => f.path.endsWith('.md'))
          .toList()
        ..sort((a, b) => a.path.compareTo(b.path));
  return [
    for (final file in files)
      KbDocument(
        name: file.uri.pathSegments.last,
        path: '$kKbAssetPrefix${file.uri.pathSegments.last}',
        bytes: file.readAsBytesSync(),
      ),
  ];
}

/// The resolved version of [package] in pubspec.lock.
String _locked(String package) {
  final lines = File('pubspec.lock').readAsLinesSync();
  final at = lines.indexOf('  $package:');
  if (at < 0) fail('$package is not in pubspec.lock');
  for (final line in lines.skip(at + 1)) {
    final match = RegExp(r'^    version: "?([^"]+)"?$').firstMatch(line);
    if (match != null) return match.group(1)!;
  }
  fail('no version for $package in pubspec.lock');
}

void main() {
  final manifestFile = File(kKbPrebuiltManifestAsset);
  final database = File(kKbPrebuiltDatabaseAsset);

  test('the manifest and kb.db are shipped and agree', () {
    expect(manifestFile.existsSync(), isTrue, reason: _rebuild);
    expect(database.existsSync(), isTrue, reason: _rebuild);
    final manifest = PrebuiltKbManifest.fromJson(
      jsonDecode(manifestFile.readAsStringSync()),
    );
    final bytes = database.readAsBytesSync();
    expect(bytes.length, manifest.dbBytes);
    expect(sha256.convert(bytes).toString(), manifest.dbSha256);
    expect(
      String.fromCharCodes(bytes.sublist(0, 15)),
      'SQLite format 3',
      reason: 'a sqlite database',
    );
  });

  test('it was built from this checkout ($_rebuild otherwise)', () {
    final manifest = PrebuiltKbManifest.fromJson(
      jsonDecode(manifestFile.readAsStringSync()),
    );
    const chunker = MarkdownChunker();
    final documents = _documents();
    final expected = KbIndexKey(
      documents: kbDocumentsHash(documents),
      chunker: chunkerId(chunker),
      documentPrefix: TaskType.retrievalDocument.prefix,
      modelId: kBundledEmbedderModel.name.replaceAll(RegExp(r'\.tflite$'), ''),
      modelSha256: kBundledEmbedderModel.sha256,
      tokenizerSha256: kBundledEmbedderTokenizer.sha256,
      dim: kEmbedderConfig.dimension,
    );

    expect(
      manifest.key.differencesFrom(expected),
      isEmpty,
      reason: 'the prebuilt index is stale: $_rebuild',
    );
    final chunks = [
      for (final document in documents)
        ...chunker.chunk(utf8.decode(document.bytes), docId: document.name),
    ];
    expect(manifest.chunks, chunks.length);
    expect(
      manifest.store,
      'flutter_edge_ai_sqlite ${_locked('flutter_edge_ai_sqlite')}',
      reason: 'kb.db is the store\'s own vec0 layout: $_rebuild',
    );
    expect(manifest.embedderBackend, kEmbedderConfig.backend.name);
  });
}
