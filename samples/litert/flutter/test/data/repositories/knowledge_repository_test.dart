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
import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter_edge_ai/flutter_edge_ai.dart' show TaskType;
import 'package:flutter_edge_ai_rag/flutter_edge_ai_rag.dart'
    show RetrievalResult;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/repositories/knowledge_repository.dart';
import 'package:litert_edge_demos/data/services/knowledge/embedder_digests.dart';
import 'package:litert_edge_demos/data/services/knowledge/embedder_service.dart';
import 'package:litert_edge_demos/data/services/knowledge/kb_documents.dart';
import 'package:litert_edge_demos/data/services/knowledge/kb_prebuilt_index.dart';
import 'package:litert_edge_demos/domain/models/knowledge.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../fakes/fake_knowledge.dart';

final class FixedDocuments implements KbDocumentSource {
  FixedDocuments(this.documents);

  List<KbDocument> documents;

  @override
  Future<List<KbDocument>> load() async => documents;
}

KbDocument doc(String name, String markdown) => KbDocument(
  name: name,
  path: 'assets/kb/$name',
  bytes: utf8.encode(markdown),
);

/// Three chunks: alpha.md#0 (One), alpha.md#1 (Two), beta.md#0 (Three).
List<KbDocument> corpus({String alphaTwo = 'Second section text.'}) => [
  doc(
    'alpha.md',
    '# Alpha\n\n## One\n\nFirst section text.\n\n## Two\n\n$alphaTwo',
  ),
  doc(
    'beta.md',
    '---\ntitle: Beta doc\nsource: https://beta.example/doc\n---\n\n'
        '## Three\n\nThird section text.',
  ),
];

/// [FixedEmbedderDigests] that answer once [gate] completes ([started]
/// completes when asked).
final class _GatedDigests implements EmbedderDigests {
  final Completer<void> gate = Completer();
  final Completer<void> started = Completer();

  @override
  Future<EmbedderDigest> of({
    required String modelPath,
    required String tokenizerPath,
  }) async {
    if (!started.isCompleted) started.complete();
    await gate.future;
    return FixedEmbedderDigests().of(
      modelPath: modelPath,
      tokenizerPath: tokenizerPath,
    );
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

/// [FakeEmbeddingModel] that says when a batch arrives, so a test waits for
/// it instead of polling.
final class _BatchSignallingModel extends FakeEmbeddingModel {
  final List<(int, Completer<void>)> _waiters = [];

  /// Completes once [count] batches have reached [generateEmbeddings]
  /// (recorded, before [gate]): at once when they already have.
  Future<void> untilBatches(int count) {
    if (batches.length >= count) return Future.value();
    final waiter = Completer<void>();
    _waiters.add((count, waiter));
    return waiter.future.timeout(const Duration(seconds: 10));
  }

  @override
  Future<List<List<double>>> generateEmbeddings(
    List<String> texts, {
    TaskType taskType = TaskType.retrievalQuery,
  }) {
    // Runs up to its first await: the batch is recorded on return.
    final result = super.generateEmbeddings(texts, taskType: taskType);
    _waiters.removeWhere((waiter) {
      if (waiter.$1 > batches.length) return false;
      waiter.$2.complete();
      return true;
    });
    return result;
  }
}

void main() {
  late Directory dir;
  late _BatchSignallingModel model;
  late EmbedderService embedder;
  late FakeVectorStoreService store;
  late ValueNotifier<Map<ModelId, ModelState>> models;

  File marker() => File('${dir.path}/kb/index.json');

  KnowledgeRepository build({
    List<KbDocument>? documents,
    EmbedderService? withEmbedder,
    Duration closeWait = const Duration(seconds: 5),
    EmbedderDigests? digests,
    Future<void>? indexDirGate,
  }) {
    final repo = KnowledgeRepository(
      embedder: withEmbedder ?? embedder,
      store: store,
      models: models,
      documents: FixedDocuments(documents ?? corpus()),
      prebuilt: const NoPrebuiltKbIndex('test: no prebuilt index'),
      digests: digests ?? FixedEmbedderDigests(),
      indexDir: () async {
        await indexDirGate;
        return Directory('${dir.path}/kb');
      },
      batchSize: 2,
      closeWait: closeWait,
    );
    addTearDown(repo.close);
    return repo;
  }

  /// A hit for a stored chunk, with the metadata the index wrote for it.
  RetrievalResult hit(String id, double similarity) => RetrievalResult(
    id: id,
    content: store.rows[id]!.content,
    similarity: similarity,
    metadata: store.rows[id]!.metadata,
  );

  setUp(() async {
    dir = Directory.systemTemp.createTempSync('knowledge_repository_test');
    model = _BatchSignallingModel();
    embedder = await loadedEmbedder(model);
    store = FakeVectorStoreService();
    models = ValueNotifier(const {});
  });

  tearDown(() {
    models.dispose();
    dir.deleteSync(recursive: true);
  });

  group('status follows the embedder', () {
    test('waiting, then unavailable with the reason; retrieval says so and '
        'never searches', () async {
      final repo = build();
      expect(repo.status.value, isA<KnowledgeWaiting>());

      models.value = {
        ModelId.embeddingGemma: const ModelUnavailable(
          'The built-in EmbeddingGemma is not in this app bundle',
        ),
      };
      expect(
        repo.status.value,
        isA<KnowledgeUnavailable>().having(
          (s) => s.reason,
          'reason',
          'The built-in EmbeddingGemma is not in this app bundle',
        ),
      );

      final retrieval =
          (await repo.retrieve('What is LiteRT?') as Ok<Retrieval>).value;
      expect(retrieval.outcome, RetrievalOutcome.unavailable);
      expect(retrieval.detail, contains('not in this app bundle'));
      expect(store.searches, isEmpty);
      expect(store.openedPaths, isEmpty, reason: 'nothing indexed');
    });

    test('a failed embedder makes the knowledge base unavailable', () {
      final repo = build();
      models.value = {ModelId.embeddingGemma: const ModelFailed('boom')};

      expect(
        repo.status.value,
        isA<KnowledgeUnavailable>().having(
          (s) => s.reason,
          'reason',
          'EmbeddingGemma failed: boom',
        ),
      );
    });
  });

  group('indexing (hash-gated)', () {
    test('a ready embedder starts indexing by itself: progress per batch, '
        'chunks with metadata, the marker last', () async {
      final repo = build();
      final statuses = <KnowledgeStatus>[];
      repo.status.addListener(() => statuses.add(repo.status.value));

      models.value = {ModelId.embeddingGemma: _ready};
      final result = await repo.ensureIndexed(); // joins the automatic run

      final ready = (result as Ok<KnowledgeReady>).value;
      expect(ready.chunks, 3);
      expect(ready.reused, isFalse);
      expect(
        statuses.map(
          (s) => switch (s) {
            KnowledgeIndexing(:final done, :final total) =>
              'indexing $done/$total',
            KnowledgeReady(:final reused) => 'ready reused=$reused',
            _ => s.runtimeType.toString(),
          },
        ),
        ['indexing 0/3', 'indexing 2/3', 'indexing 3/3', 'ready reused=false'],
      );
      expect(store.openedPaths, ['${dir.path}/kb/kb.db']);
      expect(store.clearCalls, 1);
      expect(store.rows.keys, ['alpha.md#0', 'alpha.md#1', 'beta.md#0']);
      expect(
        store.rows['alpha.md#1']!.content,
        'Alpha › Two\n\nSecond section text.',
      );
      expect(jsonDecode(store.rows['beta.md#0']!.metadata), {
        'doc': 'beta.md',
        'title': 'Beta doc',
        'section': 'Three',
        'chunk': 0,
        'source': 'https://beta.example/doc',
      });
      expect(model.batches.map((b) => b.$2).toSet(), {
        TaskType.retrievalDocument,
      });
      expect(model.documentsEmbedded, 3);

      final written =
          jsonDecode(marker().readAsStringSync()) as Map<String, Object?>;
      expect(written['chunks'], 3);
      expect(written['dim'], 768);
      expect(written['hash'], isA<String>());
      expect(File('${marker().path}.tmp').existsSync(), isFalse);
    });

    test('the same documents, chunker and model on the next launch: reused, '
        'nothing cleared or embedded', () async {
      models.value = {ModelId.embeddingGemma: _ready};
      final first = build();
      await first.ensureIndexed();
      await first.close();
      store.relaunch();
      final embedded = model.documentsEmbedded;

      final second = build(); // same store: the sqlite file survives
      final result = await second.ensureIndexed();

      final ready = (result as Ok<KnowledgeReady>).value;
      expect(ready.reused, isTrue);
      expect(ready.chunks, 3);
      expect(second.status.value, isA<KnowledgeReady>());
      expect(model.documentsEmbedded, embedded);
      expect(store.clearCalls, 1, reason: 'only the first launch cleared');
    });

    test('a changed document clears the store and re-indexes', () async {
      models.value = {ModelId.embeddingGemma: _ready};
      final first = build();
      await first.ensureIndexed();
      await first.close();
      store.relaunch();
      final hashBefore = (jsonDecode(
        marker().readAsStringSync(),
      ) as Map<String, Object?>)['hash'];

      final second = build(documents: corpus(alphaTwo: 'Edited text.'));
      final result = await second.ensureIndexed();

      expect((result as Ok<KnowledgeReady>).value.reused, isFalse);
      expect(store.clearCalls, 2);
      expect(model.documentsEmbedded, 6);
      expect(store.rows['alpha.md#1']!.content, endsWith('Edited text.'));
      expect(
        (jsonDecode(marker().readAsStringSync())
            as Map<String, Object?>)['hash'],
        isNot(hashBefore),
      );
    });

    test('another model id re-indexes', () async {
      models.value = {ModelId.embeddingGemma: _ready};
      final first = build();
      await first.ensureIndexed();
      await first.close();
      store.relaunch();

      final other = await loadedEmbedder(model, modelId: 'embeddinggemma-v2');
      final second = build(withEmbedder: other);
      final result = await second.ensureIndexed();

      expect((result as Ok<KnowledgeReady>).value.reused, isFalse);
      expect(store.clearCalls, 2);
    });

    test(
      'a store that lost rows behind a matching marker re-indexes',
      () async {
        models.value = {ModelId.embeddingGemma: _ready};
        final first = build();
        await first.ensureIndexed();
        await first.close();
        store.relaunch();
        store.rows.remove('beta.md#0');

        final second = build();
        final result = await second.ensureIndexed();

        expect((result as Ok<KnowledgeReady>).value.reused, isFalse);
        expect(store.rows, hasLength(3));
      },
    );

    test('while re-indexing there is no marker; it is written last', () async {
      models.value = {ModelId.embeddingGemma: _ready};
      final first = build();
      await first.ensureIndexed();
      await first.close();
      store.relaunch();
      expect(marker().existsSync(), isTrue);

      model.gate = Completer<void>();
      final batchesBefore = model.batches.length;
      final second = build(documents: corpus(alphaTwo: 'Changed.'));
      await model.untilBatches(batchesBefore + 1);
      expect(second.status.value, isA<KnowledgeIndexing>());
      expect(marker().existsSync(), isFalse);

      model.gate!.complete();
      await second.ensureIndexed();
      expect(marker().existsSync(), isTrue);
    });

    test('a failure mid-way leaves no marker and is visible; the next call '
        'tries again', () async {
      model.failFromBatch = 2;
      models.value = {ModelId.embeddingGemma: _ready};
      final repo = build();

      final failed = await repo.ensureIndexed();

      expect(failed, isA<Error<KnowledgeReady>>());
      expect(
        repo.status.value,
        isA<KnowledgeFailed>().having(
          (s) => s.message,
          'message',
          contains('embedding worker died'),
        ),
      );
      expect(marker().existsSync(), isFalse);
      final retrieval = (await repo.retrieve('q') as Ok<Retrieval>).value;
      expect(retrieval.outcome, RetrievalOutcome.unavailable);
      expect(retrieval.detail, startsWith('indexing failed: '));

      model.failFromBatch = null;
      final retried = await repo.ensureIndexed();
      expect((retried as Ok<KnowledgeReady>).value.chunks, 3);
      expect(marker().existsSync(), isTrue);
    });

    test('a database that will not open is deleted with its marker and '
        'opened again once; then it re-indexes', () async {
      final kbDir = Directory('${dir.path}/kb')..createSync(recursive: true);
      final db = File('${kbDir.path}/kb.db')..writeAsStringSync('garbage');
      final wal = File('${kbDir.path}/kb.db-wal')..writeAsStringSync('x');
      marker().writeAsStringSync('{"hash":"old","chunks":3,"dim":768}');
      final onDisk = <bool>[];
      store
        ..failOpens = 1
        ..onOpen = (_) => onDisk.add(db.existsSync());
      models.value = {ModelId.embeddingGemma: _ready};
      final repo = build();

      final result = await repo.ensureIndexed();

      expect(store.openedPaths, ['${kbDir.path}/kb.db', '${kbDir.path}/kb.db']);
      expect(onDisk, [isTrue, isFalse], reason: 'deleted before the retry');
      expect(wal.existsSync(), isFalse);
      final ready = (result as Ok<KnowledgeReady>).value;
      expect(ready.reused, isFalse);
      expect(ready.chunks, 3);
      expect(marker().existsSync(), isTrue);
      expect(repo.status.value, isA<KnowledgeReady>());
    });

    test(
      'when the reopened database fails too, the failure is visible',
      () async {
        store
          ..failOpens = 2
          ..openError = Exception('no such module: vec0');
        models.value = {ModelId.embeddingGemma: _ready};
        final repo = build();

        final result = await repo.ensureIndexed();

        expect(store.openedPaths, hasLength(2), reason: 'one retry only');
        expect(result, isA<Error<KnowledgeReady>>());
        expect(
          repo.status.value,
          isA<KnowledgeFailed>().having(
            (s) => s.message,
            'message',
            allOf(contains('no such module: vec0'), contains('reset')),
          ),
        );
      },
    );

    test('a malformed document fails indexing visibly (the chunker error '
        'crosses its isolate)', () async {
      models.value = {ModelId.embeddingGemma: _ready};
      final repo = build(
        documents: [
          doc('broken.md', '## A\n\nText.\n\n```dart\nvoid f() {}\n'),
        ],
      );

      final result = await repo.ensureIndexed();

      expect((result as Error<KnowledgeReady>).error, isA<FormatException>());
      expect(
        repo.status.value,
        isA<KnowledgeFailed>().having(
          (s) => s.message,
          'message',
          contains('broken.md: the code fence opened on line 5'),
        ),
      );
      expect(marker().existsSync(), isFalse);
      expect(model.batches, isEmpty);
    });

    test('close() during indexing stops it without a marker', () async {
      model.gate = Completer<void>();
      models.value = {ModelId.embeddingGemma: _ready};
      final repo = build();
      await model.untilBatches(1);
      final run = repo.ensureIndexed();

      final closing = repo.close();
      model.gate!.complete();

      expect(await run, isA<Error<KnowledgeReady>>());
      await closing;
      expect(marker().existsSync(), isFalse);
    });

    group('close waits for the native calls in flight', () {
      /// Whether [future] has completed, polled.
      bool Function() completion(Future<void> future) {
        var done = false;
        unawaited(future.then((_) => done = true));
        return () => done;
      }

      test('an add in flight: close does not complete and the store is not '
          'closed until the add returns; the run then stops', () async {
        store.addGate = Completer<void>();
        models.value = {ModelId.embeddingGemma: _ready};
        final repo = build();
        await store.addStarted.future;

        final closing = repo.close();
        final closed = completion(closing);
        await Future<void>.delayed(const Duration(milliseconds: 20));
        expect(closed(), isFalse);
        expect(store.closeCalls, 0, reason: 'the add is still running');

        store.addGate!.complete();
        await closing;

        expect(store.closeCalls, 1);
        expect(store.closedWithCallsInFlight, 0);
        expect(store.usedWhileClosed, isEmpty);
        expect(store.addCalls, 1, reason: 'stopped at the next check');
        expect(marker().existsSync(), isFalse);
      });

      test('a retrieval in flight: its search finishes before the store '
          'closes; afterwards retrieval is an error and searches '
          'nothing', () async {
        models.value = {ModelId.embeddingGemma: _ready};
        final repo = build();
        expect(await repo.ensureIndexed(), isA<Ok<KnowledgeReady>>());
        store.searchGate = Completer<void>();
        final retrieving = repo.retrieve('what is litert?');
        await store.searchStarted.future;

        final closing = repo.close();
        final closed = completion(closing);
        await Future<void>.delayed(const Duration(milliseconds: 20));
        expect(closed(), isFalse);
        expect(store.closeCalls, 0);

        store.searchGate!.complete();
        expect(await retrieving, isA<Ok<Retrieval>>());
        await closing;
        expect(store.closeCalls, 1);
        expect(store.closedWithCallsInFlight, 0);

        expect(await repo.retrieve('again'), isA<Error<Retrieval>>());
        expect(await repo.ensureIndexed(), isA<Ok<KnowledgeReady>>());
        expect(store.searches, hasLength(1));
        expect(store.usedWhileClosed, isEmpty);
      });

      test('twice, also concurrently: one shutdown', () async {
        models.value = {ModelId.embeddingGemma: _ready};
        final repo = build();
        await repo.ensureIndexed();

        await Future.wait([repo.close(), repo.close()]);
        await repo.close();

        expect(store.closeCalls, 1);
      });

      test('an add that never returns holds close only for closeWait '
          '(logged), not forever', () async {
        store.addGate = Completer<void>();
        models.value = {ModelId.embeddingGemma: _ready};
        final repo = build(closeWait: const Duration(milliseconds: 50));
        await store.addStarted.future;

        await repo.close();

        expect(store.closeCalls, 1);
      });
    });

    group('a run still going when close gave up on it', () {
      test('reaching the marker check: no store call, the marker '
          'kept', () async {
        models.value = {ModelId.embeddingGemma: _ready};
        final first = build();
        expect(await first.ensureIndexed(), isA<Ok<KnowledgeReady>>());
        await first.close();
        expect(marker().existsSync(), isTrue);
        // The next launch: the same database, a slow key.
        store = FakeVectorStoreService()..rows.addAll(store.rows);
        final digests = _GatedDigests();
        final second = build(
          digests: digests,
          closeWait: const Duration(milliseconds: 50),
        );
        await digests.started.future;
        final run = second.ensureIndexed();

        await second.close();
        digests.gate.complete();

        expect(await run, isA<Error<KnowledgeReady>>());
        expect(marker().existsSync(), isTrue, reason: 'no re-index forced');
        expect(store.usedWhileClosed, isEmpty);
      });

      test('before the store opened: it is not opened again, nothing is '
          'deleted', () async {
        models.value = {ModelId.embeddingGemma: _ready};
        final first = build();
        expect(await first.ensureIndexed(), isA<Ok<KnowledgeReady>>());
        await first.close();
        store = FakeVectorStoreService()..rows.addAll(store.rows);
        final gate = Completer<void>();
        final second = build(
          indexDirGate: gate.future,
          closeWait: const Duration(milliseconds: 50),
        );
        final run = second.ensureIndexed();

        await second.close();
        gate.complete();

        expect(await run, isA<Error<KnowledgeReady>>());
        expect(store.openedPaths, isEmpty);
        expect(store.usedWhileClosed, isEmpty);
        expect(marker().existsSync(), isTrue);
      });
    });

    test(
      'while indexing, retrieval is unavailable with the progress',
      () async {
        model.gate = Completer<void>();
        models.value = {ModelId.embeddingGemma: _ready};
        final repo = build();
        await model.untilBatches(1);

        final retrieval = (await repo.retrieve('q') as Ok<Retrieval>).value;

        expect(retrieval.outcome, RetrievalOutcome.unavailable);
        expect(retrieval.detail, 'indexing 0%');
        expect(store.searches, isEmpty);
        model.gate!.complete();
        await repo.ensureIndexed();
      },
    );
  });

  group('retrieval', () {
    late KnowledgeRepository repo;

    setUp(() async {
      models.value = {ModelId.embeddingGemma: _ready};
      repo = build();
      await repo.ensureIndexed();
    });

    test('top-3, gated per hit at 0.40; passages carry the citation '
        'metadata', () async {
      store.searchResults['q'] = [
        hit('beta.md#0', 0.45),
        hit('alpha.md#0', 0.62),
        hit('alpha.md#1', 0.39),
      ];

      final retrieval = (await repo.retrieve('q') as Ok<Retrieval>).value;

      expect(store.searches, [('q', 3)]);
      expect(retrieval.outcome, RetrievalOutcome.used);
      expect(retrieval.gate, 0.40);
      expect(retrieval.latency, isNotNull);
      expect(retrieval.topSimilarity, 0.62);
      expect(retrieval.candidates.map((p) => (p.id, p.similarity)), [
        ('alpha.md#0', 0.62),
        ('beta.md#0', 0.45),
        ('alpha.md#1', 0.39),
      ], reason: 'best first');
      expect(retrieval.passages.map((p) => p.id), ['alpha.md#0', 'beta.md#0']);
      final beta = retrieval.passages[1];
      expect(beta.doc, 'beta.md');
      expect(beta.title, 'Beta doc');
      expect(beta.section, 'Three');
      expect(beta.source, 'https://beta.example/doc');
      expect(beta.label, 'Beta doc › Three');
      expect(beta.content, 'Beta doc › Three\n\nThird section text.');
      expect(retrieval.passages[0].source, isNull, reason: 'alpha has none');
    });

    test(
      'nothing at or above the gate: below gate, with the top similarity',
      () async {
        store.searchResults['weather?'] = [hit('alpha.md#0', 0.21)];

        final retrieval =
            (await repo.retrieve('weather?') as Ok<Retrieval>).value;

        expect(retrieval.outcome, RetrievalOutcome.belowGate);
        expect(retrieval.passages, isEmpty);
        expect(retrieval.topSimilarity, 0.21);
      },
    );

    test('exactly at the gate counts', () async {
      store.searchResults['q'] = [hit('alpha.md#0', 0.40)];

      final retrieval = (await repo.retrieve('q') as Ok<Retrieval>).value;

      expect(retrieval.outcome, RetrievalOutcome.used);
    });

    test('a failed search is an Error', () async {
      store.searchError = Exception('database is locked');

      final result = await repo.retrieve('q');

      expect(result, isA<Error<Retrieval>>());
      expect(result.toString(), contains('database is locked'));
    });

    test('a hit without our metadata is a corrupt index: an Error', () async {
      store.searchResults['q'] = [
        const RetrievalResult(
          id: 'x#0',
          content: 'x',
          similarity: 0.9,
          metadata: '{"doc": 1}',
        ),
      ];

      final result = await repo.retrieve('q');

      expect((result as Error<Retrieval>).error, isA<FormatException>());
    });
  });
}
