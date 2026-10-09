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
import 'dart:math' as math;

import 'package:crypto/crypto.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_edge_ai_rag/flutter_edge_ai_rag.dart'
    show RetrievalResult;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/repositories/knowledge_repository.dart';
import 'package:litert_edge_demos/data/services/knowledge/embedder_service.dart';
import 'package:litert_edge_demos/data/services/knowledge/kb_documents.dart';
import 'package:litert_edge_demos/data/services/knowledge/kb_index_key.dart';
import 'package:litert_edge_demos/data/services/knowledge/kb_prebuilt_index.dart';
import 'package:litert_edge_demos/domain/models/knowledge.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/utils/markdown_chunker.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../fakes/fake_knowledge.dart';
import 'knowledge_repository_test.dart' show FixedDocuments, corpus;

/// A [FakeVectorStoreService] whose rows live in the file it was opened on,
/// like sqlite: opening a path loads what is there, every write saves it. A
/// prebuilt `kb.db` is such a file, made by indexing into one. [search] ranks
/// the rows by cosine similarity to the query's fake embedding.
class FileBackedStore extends FakeVectorStoreService {
  FileBackedStore(this._model);

  final FakeEmbeddingModel _model;
  String? _path;

  @override
  Future<Result<void>> open(String databasePath) async {
    final opened = await super.open(databasePath);
    if (opened is Error<void>) return opened;
    rows.clear();
    _path = null;
    final file = File(databasePath);
    if (file.existsSync() && file.lengthSync() > 0) {
      try {
        final json = jsonDecode(file.readAsStringSync());
        if (json is! Map<String, Object?>) throw const FormatException('rows');
        for (final MapEntry(:key, :value) in json.entries) {
          if (value case {
            'content': final String content,
            'embedding': final List<Object?> embedding,
            'metadata': final String metadata,
          }) {
            rows[key] = (
              content: content,
              embedding: [for (final v in embedding) (v! as num).toDouble()],
              metadata: metadata,
            );
          } else {
            throw FormatException('row $key', value);
          }
        }
      } on FormatException catch (e) {
        return Result.error(e);
      }
    }
    _path = databasePath;
    return const Result.ok(null);
  }

  @override
  Future<Result<void>> add({
    required String id,
    required String content,
    required List<double> embedding,
    required String metadata,
  }) async {
    final added = await super.add(
      id: id,
      content: content,
      embedding: embedding,
      metadata: metadata,
    );
    _save();
    return added;
  }

  @override
  Future<Result<void>> clear() async {
    final cleared = await super.clear();
    _save();
    return cleared;
  }

  void _save() {
    final path = _path;
    if (path == null) return;
    File(path).writeAsStringSync(
      jsonEncode({
        for (final MapEntry(:key, :value) in rows.entries)
          key: {
            'content': value.content,
            'embedding': value.embedding,
            'metadata': value.metadata,
          },
      }),
    );
  }

  @override
  Future<Result<List<RetrievalResult>>> search(
    String query, {
    required int topK,
  }) async {
    searches.add((query, topK));
    final q = _model.vectorFor(query);
    double cosine(List<double> a, List<double> b) {
      var dot = 0.0;
      var na = 0.0;
      var nb = 0.0;
      for (var i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
      }
      return dot / (math.sqrt(na) * math.sqrt(nb));
    }

    final hits = [
      for (final MapEntry(:key, :value) in rows.entries)
        RetrievalResult(
          id: key,
          content: value.content,
          similarity: cosine(q, value.embedding),
          metadata: value.metadata,
        ),
    ]..sort((a, b) => b.similarity.compareTo(a.similarity));
    return Result.ok(hits.take(topK).toList());
  }
}

const _ready = ModelReady(
  LoadedModelInfo(
    modelId: 'embeddinggemma-300M_seq512_mixed-precision',
    backend: 'cpu',
    loadTime: Duration.zero,
    warmUpTime: Duration.zero,
  ),
);

void main() {
  late Directory dir;
  late FakeEmbeddingModel model;
  late EmbedderService embedder;
  late ValueNotifier<Map<ModelId, ModelState>> models;

  setUp(() async {
    dir = Directory.systemTemp.createTempSync('kb_prebuilt_test');
    model = FakeEmbeddingModel();
    embedder = await loadedEmbedder(model);
    models = ValueNotifier({ModelId.embeddingGemma: _ready});
  });

  tearDown(() {
    models.dispose();
    dir.deleteSync(recursive: true);
  });

  KnowledgeRepository build({
    required String name,
    required FakeVectorStoreService store,
    required PrebuiltKbIndexSource prebuilt,
    List<KbDocument>? documents,
    FixedEmbedderDigests? digests,
    MarkdownChunker chunker = const MarkdownChunker(),
  }) {
    final repo = KnowledgeRepository(
      embedder: embedder,
      store: store,
      models: models,
      documents: FixedDocuments(documents ?? corpus()),
      prebuilt: prebuilt,
      digests: digests ?? FixedEmbedderDigests(),
      indexDir: () async => Directory('${dir.path}/$name'),
      chunker: chunker,
      batchSize: 2,
    );
    addTearDown(repo.close);
    return repo;
  }

  /// The build machine's run: an on-device index of [corpus] into its own
  /// directory and store; its `kb.db` and a manifest with its marker's key.
  Future<({PrebuiltKbManifest manifest, Uint8List db})> buildPrebuilt() async {
    final repo = build(
      name: 'build',
      store: FileBackedStore(model),
      prebuilt: const NoPrebuiltKbIndex('building the prebuilt index'),
    );
    final built = await repo.ensureIndexed();
    expect(built, isA<Ok<KnowledgeReady>>());
    await repo.close();
    final marker = jsonDecode(
      File('${dir.path}/build/index.json').readAsStringSync(),
    ) as Map<String, Object?>;
    final db = File('${dir.path}/build/kb.db').readAsBytesSync();
    return (
      manifest: PrebuiltKbManifest(
        key: KbIndexKey.fromJson(marker['key']),
        chunks: 3,
        dbBytes: db.length,
        dbSha256: sha256.convert(db).toString(),
        store: 'flutter_edge_ai_sqlite 2.0.0',
        embedderBackend: 'cpu',
        builtOn: 'macos',
        builtAt: DateTime.utc(2026, 10, 6),
      ),
      db: db,
    );
  }

  Map<String, Object?> markerOf(String name) =>
      jsonDecode(File('${dir.path}/$name/index.json').readAsStringSync())
          as Map<String, Object?>;

  group('a matching prebuilt index', () {
    test('is installed: nothing embedded, the build\'s rows, the marker says '
        'prebuilt', () async {
      final (:manifest, :db) = await buildPrebuilt();
      final embeddedByBuild = model.documentsEmbedded;
      final store = FileBackedStore(model);
      final prebuilt = FakePrebuiltKbIndex(
        PrebuiltManifestFound(manifest),
        database: db,
      );
      final repo = build(name: 'app', store: store, prebuilt: prebuilt);
      final statuses = <KnowledgeStatus>[];
      repo.status.addListener(() => statuses.add(repo.status.value));

      final result = await repo.ensureIndexed();

      final ready = (result as Ok<KnowledgeReady>).value;
      expect(ready.origin, KnowledgeOrigin.prebuilt);
      expect(ready.reused, isFalse);
      expect(ready.chunks, 3);
      expect(ready.prebuiltSkipped, isNull);
      expect(statuses.whereType<KnowledgeIndexing>(), isEmpty);
      expect(model.documentsEmbedded, embeddedByBuild, reason: 'no embedding');
      expect(store.clearCalls, 0);
      expect(store.openedPaths, [
        '${dir.path}/app/kb.db',
        '${dir.path}/app/kb.db',
      ], reason: 'opened, then reopened on the copy');
      expect(store.rows.keys, ['alpha.md#0', 'alpha.md#1', 'beta.md#0']);
      expect(
        File('${dir.path}/app/kb.db').readAsBytesSync(),
        db,
        reason: 'the shipped file, byte for byte',
      );
      final marker = markerOf('app');
      expect(marker['origin'], 'prebuilt');
      expect(marker['hash'], manifest.key.digest);
      expect(marker['chunks'], 3);
      expect(marker.containsKey('prebuiltSkipped'), isFalse);
      expect(File('${dir.path}/app/kb.db.prebuilt').existsSync(), isFalse);
    });

    test('the next launch reuses it, still saying prebuilt', () async {
      final (:manifest, :db) = await buildPrebuilt();
      final store = FileBackedStore(model);
      final prebuilt = FakePrebuiltKbIndex(
        PrebuiltManifestFound(manifest),
        database: db,
      );
      final first = build(name: 'app', store: store, prebuilt: prebuilt);
      await first.ensureIndexed();
      await first.close();

      // The next launch: a new store over the same file.
      final second = build(
        name: 'app',
        store: FileBackedStore(model),
        prebuilt: prebuilt,
      );
      final result = await second.ensureIndexed();

      final ready = (result as Ok<KnowledgeReady>).value;
      expect(ready.reused, isTrue);
      expect(ready.origin, KnowledgeOrigin.prebuilt);
      expect(prebuilt.databaseReads, 1, reason: 'copied once');
    });

    test('retrieval from it equals retrieval from an index embedded on the '
        'device', () async {
      final (:manifest, :db) = await buildPrebuilt();
      final onDevice = FileBackedStore(model);
      final device = build(
        name: 'device',
        store: onDevice,
        prebuilt: const NoPrebuiltKbIndex('the reference'),
      );
      expect(
        (await device.ensureIndexed() as Ok<KnowledgeReady>).value.origin,
        KnowledgeOrigin.device,
      );
      final fromPrebuilt = FileBackedStore(model);
      final installed = build(
        name: 'installed',
        store: fromPrebuilt,
        prebuilt: FakePrebuiltKbIndex(
          PrebuiltManifestFound(manifest),
          database: db,
        ),
      );
      expect(
        (await installed.ensureIndexed() as Ok<KnowledgeReady>).value.origin,
        KnowledgeOrigin.prebuilt,
      );

      for (final question in [
        'First section',
        'What is in the third section?',
        'Alpha two',
        'Something else entirely',
      ]) {
        final a = (await device.retrieve(question) as Ok<Retrieval>).value;
        final b = (await installed.retrieve(question) as Ok<Retrieval>).value;
        expect(
          [for (final p in b.candidates) (p.id, p.similarity, p.content)],
          [for (final p in a.candidates) (p.id, p.similarity, p.content)],
          reason: question,
        );
        expect(b.outcome, a.outcome, reason: question);
        expect(a.candidates, isNotEmpty);
      }
    });
  });

  group('a prebuilt index that is not used: indexed on the device, and why '
      'is in the status and the marker', () {
    Future<KnowledgeReady> indexWith(
      KnowledgeRepository repo,
      List<KnowledgeStatus> statuses,
    ) async {
      repo.status.addListener(() => statuses.add(repo.status.value));
      final result = await repo.ensureIndexed();
      return (result as Ok<KnowledgeReady>).value;
    }

    test('other documents', () async {
      final (:manifest, :db) = await buildPrebuilt();
      final embeddedByBuild = model.documentsEmbedded;
      final prebuilt = FakePrebuiltKbIndex(
        PrebuiltManifestFound(manifest),
        database: db,
      );
      final statuses = <KnowledgeStatus>[];
      final ready = await indexWith(
        build(
          name: 'app',
          store: FileBackedStore(model),
          prebuilt: prebuilt,
          documents: corpus(alphaTwo: 'Edited text.'),
        ),
        statuses,
      );

      expect(ready.origin, KnowledgeOrigin.device);
      expect(
        ready.prebuiltSkipped,
        allOf(
          contains('knowledge-base documents'),
          contains('tool/build_kb_index.sh'),
        ),
      );
      expect(
        statuses.whereType<KnowledgeIndexing>().map((s) => s.prebuiltSkipped),
        everyElement(ready.prebuiltSkipped),
      );
      expect(statuses.whereType<KnowledgeIndexing>(), isNotEmpty);
      expect(model.documentsEmbedded, embeddedByBuild + 3);
      expect(prebuilt.databaseReads, 0, reason: 'the key decided');
      expect(markerOf('app')['origin'], 'device');
      expect(markerOf('app')['prebuiltSkipped'], ready.prebuiltSkipped);
    });

    test('other embedder files (same id)', () async {
      final (:manifest, :db) = await buildPrebuilt();
      final ready = await indexWith(
        build(
          name: 'app',
          store: FileBackedStore(model),
          prebuilt: FakePrebuiltKbIndex(
            PrebuiltManifestFound(manifest),
            database: db,
          ),
          digests: FixedEmbedderDigests(model: 'another-model-sha'),
        ),
        [],
      );

      expect(ready.origin, KnowledgeOrigin.device);
      expect(ready.prebuiltSkipped, contains('embedder model file'));
      expect(ready.prebuiltSkipped, isNot(contains('documents')));
    });

    test('another chunker budget', () async {
      final (:manifest, :db) = await buildPrebuilt();
      final ready = await indexWith(
        build(
          name: 'app',
          store: FileBackedStore(model),
          prebuilt: FakePrebuiltKbIndex(
            PrebuiltManifestFound(manifest),
            database: db,
          ),
          chunker: const MarkdownChunker(maxCost: 1000),
        ),
        [],
      );

      expect(ready.prebuiltSkipped, contains('chunker'));
    });

    test('none in this build', () async {
      final ready = await indexWith(
        build(
          name: 'app',
          store: FileBackedStore(model),
          prebuilt: const NoPrebuiltKbIndex(
            'this build has no assets/kb_index/manifest.json',
          ),
        ),
        [],
      );

      expect(
        ready.prebuiltSkipped,
        'this build has no assets/kb_index/manifest.json',
      );
      expect(ready.chunks, 3);
    });

    test('an unreadable manifest', () async {
      final prebuilt = FakePrebuiltKbIndex(
        const PrebuiltManifestMissing('unused'),
      )..manifestError = const FormatException('Not a prebuilt index manifest');
      final ready = await indexWith(
        build(name: 'app', store: FileBackedStore(model), prebuilt: prebuilt),
        [],
      );

      expect(ready.prebuiltSkipped, startsWith('its manifest is unreadable'));
      expect(ready.chunks, 3);
    });

    test('a kb.db that does not match its manifest is never copied', () async {
      final (:manifest, :db) = await buildPrebuilt();
      final tampered = Uint8List.fromList(db)..[0] ^= 0x20;
      final store = FileBackedStore(model);
      final ready = await indexWith(
        build(
          name: 'app',
          store: store,
          prebuilt: FakePrebuiltKbIndex(
            PrebuiltManifestFound(manifest),
            database: tampered,
          ),
        ),
        [],
      );

      expect(ready.prebuiltSkipped, contains('SHA-256'));
      expect(store.openedPaths, hasLength(1), reason: 'never reopened');
      expect(store.rows, hasLength(3));

      final short = await indexWith(
        build(
          name: 'app2',
          store: FileBackedStore(model),
          prebuilt: FakePrebuiltKbIndex(
            PrebuiltManifestFound(manifest),
            database: Uint8List.sublistView(db, 1),
          ),
        ),
        [],
      );
      expect(short.prebuiltSkipped, contains('bytes, the manifest says'));
    });

    test('a copy with other rows than its manifest is cleared and indexed '
        'on the device', () async {
      final (:manifest, :db) = await buildPrebuilt();
      final wrongCount = PrebuiltKbManifest(
        key: manifest.key,
        chunks: 4,
        dbBytes: manifest.dbBytes,
        dbSha256: manifest.dbSha256,
        store: manifest.store,
        embedderBackend: manifest.embedderBackend,
        builtOn: manifest.builtOn,
        builtAt: manifest.builtAt,
      );
      final store = FileBackedStore(model);
      final ready = await indexWith(
        build(
          name: 'app',
          store: store,
          prebuilt: FakePrebuiltKbIndex(
            PrebuiltManifestFound(wrongCount),
            database: db,
          ),
        ),
        [],
      );

      expect(ready.origin, KnowledgeOrigin.device);
      expect(ready.prebuiltSkipped, contains('holds 3 rows'));
      expect(store.clearCalls, 1);
      expect(store.rows, hasLength(3));
      expect(markerOf('app')['chunks'], 3);
    });

    test('a copy that does not open is deleted, the store reopened empty, '
        'and indexed on the device', () async {
      final (:manifest, db: _) = await buildPrebuilt();
      final garbage = Uint8List.fromList(utf8.encode('not a database'));
      final unreadable = PrebuiltKbManifest(
        key: manifest.key,
        chunks: 3,
        dbBytes: garbage.length,
        dbSha256: sha256.convert(garbage).toString(),
        store: manifest.store,
        embedderBackend: manifest.embedderBackend,
        builtOn: manifest.builtOn,
        builtAt: manifest.builtAt,
      );
      final store = FileBackedStore(model);
      final ready = await indexWith(
        build(
          name: 'app',
          store: store,
          prebuilt: FakePrebuiltKbIndex(
            PrebuiltManifestFound(unreadable),
            database: garbage,
          ),
        ),
        [],
      );

      expect(ready.origin, KnowledgeOrigin.device);
      expect(ready.prebuiltSkipped, startsWith('it did not open'));
      expect(store.openedPaths, hasLength(4), reason: 'open, copy, reset x2');
      expect(store.rows, hasLength(3));
    });
  });

  test('the installed files are what the key hashes', () {
    expect(embedder.installedFiles?.model, endsWith('/model.tflite'));
    expect(
      embedder.installedFiles?.tokenizer,
      endsWith('/sentencepiece.model'),
    );
  });
}
