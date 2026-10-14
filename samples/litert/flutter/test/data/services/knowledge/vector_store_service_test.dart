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
import 'dart:io';

import 'package:flutter_edge_ai_rag/flutter_edge_ai_rag.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/knowledge_config.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/services/knowledge/vector_store_service.dart';
import 'package:litert_edge_demos/data/services/model_store/bundled_model_files.dart';
import 'package:litert_edge_demos/domain/models/knowledge_query.dart';
import 'package:litert_edge_demos/utils/result.dart';

/// An in-memory sqlite stand-in: rows and the profile binding per location,
/// kept across opens like the database file.
final class _Disk {
  final Map<String, Map<String, (String, List<double>, String?)>> rows = {};
  final Map<String, EmbeddingProfile> profiles = {};
}

final class _FakeStore extends VectorStoreRepository {
  _FakeStore(this.disk);

  final _Disk disk;
  String? location;
  int closeCalls = 0;

  Map<String, (String, List<double>, String?)> get _rows =>
      disk.rows.putIfAbsent(location!, () => {});

  @override
  void configure(FilterSchema schema) {}

  @override
  Future<void> initialize(String location) async => this.location = location;

  @override
  Future<VectorStoreBinding?> readBinding() async {
    final profile = disk.profiles[location];
    return profile == null ? null : VectorStoreBinding(profile: profile);
  }

  @override
  Future<void> bind(VectorStoreBinding binding) async {
    final bound = disk.profiles[location];
    if (bound != null && bound != binding.profile) {
      throw StateError('already bound to $bound');
    }
    disk.profiles[location!] = binding.profile;
  }

  @override
  Future<void> addDocument({
    required String id,
    required String content,
    required List<double> embedding,
    String? metadata,
  }) async => _rows[id] = (content, embedding, metadata);

  @override
  Future<void> removeDocument({required String id}) async => _rows.remove(id);

  @override
  Future<List<RetrievalResult>> searchSimilar({
    required List<double> queryEmbedding,
    required int topK,
    double threshold = 0.0,
    Filter? filter,
  }) async => [
    for (final MapEntry(:key, :value) in _rows.entries)
      RetrievalResult(
        id: key,
        content: value.$1,
        similarity: value.$2.first == queryEmbedding.first ? 1 : 0.1,
        metadata: value.$3,
      ),
  ].take(topK).toList();

  @override
  Future<VectorStoreStats> getStats() async => VectorStoreStats(
    documentCount: _rows.length,
    vectorDimension: _rows.isEmpty ? 0 : _rows.values.first.$2.length,
  );

  @override
  Future<void> clear() async => _rows.clear();

  @override
  Future<void> flush() async {}

  @override
  Future<void> close() async => closeCalls++;

  @override
  bool get isInitialized => location != null;

  @override
  FilterSchema get filterSchema => FilterSchema.empty;
}

final class _FakeProvider implements VectorStoreProvider {
  _FakeProvider(this.disk);

  final _Disk disk;
  final List<_FakeStore> stores = [];

  /// When set, [createStore] waits for it (an open in flight).
  Completer<void>? createGate;

  @override
  String get id => 'sqlite';

  @override
  String get name => 'fake sqlite';

  @override
  int get priority => 0;

  @override
  bool canHandle(VectorStoreSpec spec) => true;

  @override
  Future<VectorStoreRepository> createStore(VectorStoreSpec spec) async {
    await createGate?.future;
    final store = _FakeStore(disk);
    stores.add(store);
    return store;
  }
}

List<double> vector(double first) => [
  first,
  for (var i = 1; i < kKbEmbeddingDimension; i++) 0.5,
];

void main() {
  late _Disk disk;
  late _FakeProvider provider;
  final queries = <KnowledgeQuery>[];

  VectorStoreService service({String profileId = kKbEmbeddingProfileId}) =>
      VectorStoreService(
        embedQuery: (q) async {
          queries.add(q);
          return Result.ok(vector(0.9));
        },
        rag: FlutterEdgeAiRag(providers: [provider]),
        profileId: profileId,
      );

  setUp(() {
    disk = _Disk();
    provider = _FakeProvider(disk);
    queries.clear();
  });

  test('a new database is bound to the knowledge base\'s profile; search '
      'embeds the query and searches by vector', () async {
    final store = service();

    expect(await store.open('/kb/kb.db'), isA<Ok<void>>());
    expect(
      disk.profiles['/kb/kb.db'],
      EmbeddingProfile(id: kKbEmbeddingProfileId, dimension: 768),
    );
    await store.add(
      id: 'a#1',
      content: 'A',
      embedding: vector(0.9),
      metadata: '{}',
    );

    final hits =
        (await store.search(
              const KnowledgeQuery.text('what?'),
              topK: 3,
            ) as Ok).value
            as List<RetrievalResult>;
    expect(queries.single, isA<KnowledgeTextQuery>());
    expect((queries.single as KnowledgeTextQuery).text, 'what?');
    expect(store.profileId, kKbEmbeddingProfileId);
    expect(hits.single.id, 'a#1');
  });

  test('a database written before profiles existed (rows, no profile) is '
      'refused: its EmbeddingGemma-300M vectors have the same 768 dimensions '
      'but another space', () async {
    disk.rows['/kb/kb.db'] = {'old#1': ('old', vector(0.9), '{}')};

    final opened = await service().open('/kb/kb.db');

    expect('${(opened as Error).error}', contains('legacy vectors'));
    expect(disk.profiles['/kb/kb.db'], isNull, reason: 'nothing bound');
  });

  test('an index of EmbeddingGemma-300M (its profile) is refused', () async {
    disk.profiles['/kb/kb.db'] = EmbeddingProfile(
      id: 'embeddinggemma-300m-seq512-mp-ad09e815-sp-d6daa52d-retrieval-prefix-v1',
      dimension: 768,
    );

    final opened = await service().open('/kb/kb.db');

    expect(opened, isA<Error<void>>());
  });

  test('the profile names the built-in EmbeddingGemma 2 bundle at its '
      'pinned revision (tool/models.lock) and SHA-256, and holds no control '
      'character', () {
    final lock = File('tool/models.lock')
        .readAsLinesSync()
        .map((line) => line.split(RegExp(r'\s+')))
        .singleWhere(
          (f) => f.length == 8 && f[1] == kBundledEmbedderModel.name,
        );
    final revision = lock[6];

    expect(kKbEmbeddingProfileId, startsWith('embeddinggemma-2-740m-'));
    expect(kKbEmbeddingProfileId, contains(revision.substring(0, 8)));
    expect(
      kKbEmbeddingProfileId,
      contains(kBundledEmbedderModel.sha256.substring(0, 8)),
    );
    expect(kKbEmbeddingProfileId.codeUnits.every((u) => u > 0x1F), isTrue);
    expect(kKbEmbeddingDimension, kEmbedderConfig.dimension);
  });

  test('a database bound to another profile is refused', () async {
    disk.profiles['/kb/kb.db'] = EmbeddingProfile(
      id: 'another-embedder-v1',
      dimension: 768,
    );

    final opened = await service().open('/kb/kb.db');

    expect('${(opened as Error).error}', contains('another-embedder-v1'));
  });

  test('open closes the database opened before; close closes it', () async {
    final store = service();
    await store.open('/kb/a.db');
    await store.open('/kb/b.db');

    expect(provider.stores.first.closeCalls, 1);
    expect(provider.stores.last.closeCalls, 0);

    await store.close();
    expect(provider.stores.last.closeCalls, 1);
  });

  test('after close, open is refused: nothing reopens the database '
      'without an owner', () async {
    final store = service();
    await store.open('/kb/kb.db');
    await store.close();

    expect(await store.open('/kb/kb.db'), isA<Error<void>>());
    expect(provider.stores, hasLength(1));
    expect(await store.stats(), isA<Error<VectorStoreStats>>());
  });

  test('a close during an open closes the index the open produces', () async {
    provider.createGate = Completer<void>();
    final store = service();
    final opening = store.open('/kb/kb.db');
    await pumpEventQueue();

    await store.close();
    provider.createGate!.complete();

    expect(await opening, isA<Error<void>>());
    expect(provider.stores.single.closeCalls, 1);
    expect(await store.stats(), isA<Error<VectorStoreStats>>());
  });

  test('a call before open is an error, not an exception', () async {
    expect(await service().stats(), isA<Error<VectorStoreStats>>());
  });
}
