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

import 'package:crypto/crypto.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_edge_ai_rag/flutter_edge_ai_rag.dart'
    show VectorStoreStats;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/repositories/knowledge_repository.dart';
import 'package:litert_edge_demos/data/services/knowledge/embedder_service.dart';
import 'package:litert_edge_demos/data/services/knowledge/kb_documents.dart';
import 'package:litert_edge_demos/data/services/knowledge/kb_index_key.dart';
import 'package:litert_edge_demos/data/services/knowledge/kb_prebuilt_index.dart';
import 'package:litert_edge_demos/domain/models/knowledge.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../fakes/fake_knowledge.dart';
import 'knowledge_repository_test.dart' show FixedDocuments, corpus;

// The index lifecycle's branches the other knowledge tests leave out, pinned
// down as they are: every `[Knowledge]` log line and status transition of
// each path (reuse, prebuilt, on the device, each failure), and the runs
// that close overtakes at each of its checks. knowledge_repository_test.dart
// and knowledge_repository_prebuilt_test.dart cover the main paths.

const _ready = ModelReady(
  LoadedModelInfo(
    modelId: 'embeddinggemma-300M_seq512_mixed-precision',
    backend: 'cpu',
    loadTime: Duration.zero,
    warmUpTime: Duration.zero,
  ),
);

/// A [FakeVectorStoreService] with scripted answers: [statsAnswers] are
/// returned by the next [stats] calls in order, [clearError] fails [clear],
/// and [openGates] hold the next [open] calls (each fails with [gatedError]
/// once released, like an open that close overtook); [opened] completes
/// when a gated open waits. Like native calls in flight, the next [stats]
/// waits for [statsGate] and the [gatedAdd]-th [add] for [gatedAddGate], each
/// after its answer is made; [statsStarted] and [gatedAddStarted] complete
/// then.
class _ScriptedStore extends FakeVectorStoreService {
  final List<Result<VectorStoreStats>> statsAnswers = [];
  Exception? clearError;
  final List<Completer<void>?> openGates = [];
  Exception gatedError = Exception('the knowledge-base index is closed');
  Completer<void> opened = Completer<void>();
  Completer<void>? statsGate;
  final Completer<void> statsStarted = Completer<void>();
  int? gatedAdd;
  final Completer<void> gatedAddGate = Completer<void>();
  final Completer<void> gatedAddStarted = Completer<void>();

  @override
  Future<Result<VectorStoreStats>> stats() async {
    if (statsAnswers.isNotEmpty) return statsAnswers.removeAt(0);
    final gate = statsGate;
    if (gate == null) return super.stats();
    statsGate = null;
    final result = await super.stats();
    statsStarted.complete();
    await gate.future;
    return result;
  }

  @override
  Future<Result<void>> add({
    required String id,
    required String content,
    required List<double> embedding,
    required String metadata,
  }) async {
    final result = await super.add(
      id: id,
      content: content,
      embedding: embedding,
      metadata: metadata,
    );
    if (addCalls == gatedAdd) {
      gatedAddStarted.complete();
      await gatedAddGate.future;
    }
    return result;
  }

  @override
  Future<Result<void>> clear() async {
    if (clearError case final error?) return Result.error(error);
    return super.clear();
  }

  @override
  Future<Result<void>> open(String databasePath) async {
    final gate = openGates.isEmpty ? null : openGates.removeAt(0);
    if (gate == null) return super.open(databasePath);
    final result = await super.open(databasePath);
    if (!opened.isCompleted) opened.complete();
    await gate.future;
    return result is Error<void> ? result : Result.error(gatedError);
  }
}

/// A [FakePrebuiltKbIndex] whose [manifest] waits for [manifestGate] and
/// whose [database] waits for [databaseGate], when set; [manifestStarted]
/// and [databaseStarted] complete when they do.
class _GatedPrebuilt extends FakePrebuiltKbIndex {
  _GatedPrebuilt(super.manifestRead, {super.database});

  Completer<void>? manifestGate;
  Completer<void>? databaseGate;
  final Completer<void> manifestStarted = Completer<void>();
  final Completer<void> databaseStarted = Completer<void>();

  @override
  Future<PrebuiltManifestRead> manifest() async {
    if (manifestGate case final gate?) {
      manifestStarted.complete();
      await gate.future;
    }
    return super.manifest();
  }

  @override
  Future<Uint8List> database() async {
    if (databaseGate case final gate?) {
      databaseStarted.complete();
      await gate.future;
    }
    return super.database();
  }
}

/// A status as one word plus its facts, for comparing transitions.
String describe(KnowledgeStatus status) => switch (status) {
  KnowledgeWaiting() => 'waiting',
  KnowledgeUnavailable(:final reason) => 'unavailable: $reason',
  KnowledgeIndexing(:final done, :final total, :final prebuiltSkipped) =>
    'indexing $done/$total skipped=$prebuiltSkipped',
  KnowledgeReady(
    :final chunks,
    :final reused,
    :final origin,
    :final prebuiltSkipped,
  ) =>
    'ready $chunks reused=$reused ${origin.name} skipped=$prebuiltSkipped',
  KnowledgeFailed(:final message) => 'failed: $message',
};

/// A manifest for [db] built from [key].
PrebuiltKbManifest manifestFor(
  KbIndexKey key,
  Uint8List db, {
  int chunks = 3,
}) => PrebuiltKbManifest(
  key: key,
  chunks: chunks,
  dbBytes: db.length,
  dbSha256: sha256.convert(db).toString(),
  store: 'flutter_edge_ai_sqlite 2.0.0',
  embedderBackend: 'cpu',
  builtOn: 'macos',
  builtAt: DateTime.utc(2026, 10, 6),
);

void main() {
  late Directory dir;
  late FakeEmbeddingModel model;
  late EmbedderService embedder;
  late _ScriptedStore store;
  late ValueNotifier<Map<ModelId, ModelState>> models;

  /// Every `[Knowledge]` message, with measured timings as `<n>ms` and the
  /// reuse hash as `<hash>`.
  late List<String> logs;

  /// [logs] without the stack traces some of them carry.
  List<String> heads() => [for (final line in logs) line.split('\n').first];

  File marker([String name = 'kb']) => File('${dir.path}/$name/index.json');

  setUp(() async {
    dir = Directory.systemTemp.createTempSync('kb_branches_test');
    model = FakeEmbeddingModel();
    embedder = await loadedEmbedder(model);
    store = _ScriptedStore();
    models = ValueNotifier(const {});
    logs = [];
    final original = debugPrint;
    debugPrint = (message, {wrapWidth}) {
      final text = message ?? '';
      if (!text.startsWith('[Knowledge]')) return;
      logs.add(
        text
            .replaceAll(RegExp(r'\bin \d+ms'), 'in <n>ms')
            .replaceAll(RegExp(r'\(\d+ms\)'), '(<n>ms)')
            .replaceAll(RegExp(r'hash [0-9a-f]{12}'), 'hash <hash>'),
      );
    };
    addTearDown(() => debugPrint = original);
  });

  tearDown(() {
    models.dispose();
    dir.deleteSync(recursive: true);
  });

  KnowledgeRepository build({
    String name = 'kb',
    FakeVectorStoreService? withStore,
    EmbedderService? withEmbedder,
    List<KbDocument>? documents,
    PrebuiltKbIndexSource prebuilt = const NoPrebuiltKbIndex(
      'test: no prebuilt index',
    ),
    Duration closeWait = const Duration(seconds: 5),
  }) {
    final repo = KnowledgeRepository(
      embedder: withEmbedder ?? embedder,
      store: withStore ?? store,
      models: models,
      documents: FixedDocuments(documents ?? corpus()),
      prebuilt: prebuilt,
      digests: FixedEmbedderDigests(),
      indexDir: () async => Directory('${dir.path}/$name'),
      batchSize: 2,
      closeWait: closeWait,
    );
    addTearDown(repo.close);
    return repo;
  }

  /// Runs [repo] to its end, recording its status transitions.
  Future<(Result<KnowledgeReady>, List<String>)> run(
    KnowledgeRepository repo,
  ) async {
    final statuses = <String>[];
    repo.status.addListener(() => statuses.add(describe(repo.status.value)));
    final result = await repo.ensureIndexed();
    return (result, statuses);
  }

  /// A finished on-device index in `<dir>/kb` over [store]; its log lines
  /// are dropped. The next [build] is the next launch.
  Future<void> indexedBefore() async {
    final first = build();
    expect(await first.ensureIndexed(), isA<Ok<KnowledgeReady>>());
    await first.close();
    store.relaunch();
    logs.clear();
  }

  /// This app's index key, as an on-device run in its own directory writes
  /// it into the marker.
  Future<KbIndexKey> appKey() async {
    final repo = build(name: 'key', withStore: FakeVectorStoreService());
    expect(await repo.ensureIndexed(), isA<Ok<KnowledgeReady>>());
    await repo.close();
    logs.clear();
    final json = jsonDecode(marker('key').readAsStringSync());
    return KbIndexKey.fromJson((json as Map<String, Object?>)['key']);
  }

  void writeMarker(Object json) {
    marker().parent.createSync(recursive: true);
    marker().writeAsStringSync(jsonEncode(json));
  }

  const onDevice = [
    '[Knowledge] prebuilt index not used: test: no prebuilt index. Indexing '
        'on the device',
    '[Knowledge] indexed 3 chunks from 2 documents in <n>ms',
  ];
  const onDeviceStatuses = [
    'indexing 0/3 skipped=test: no prebuilt index',
    'indexing 2/3 skipped=test: no prebuilt index',
    'indexing 3/3 skipped=test: no prebuilt index',
    'ready 3 reused=false device skipped=test: no prebuilt index',
  ];

  group('reuse: the log says why an index is not reused', () {
    test('no marker', () async {
      final (result, statuses) = await run(build());

      expect(result, isA<Ok<KnowledgeReady>>());
      expect(logs, ['[Knowledge] no index marker: indexing', ...onDevice]);
      expect(statuses, onDeviceStatuses);
      final written = jsonDecode(marker().readAsStringSync());
      expect(written, {
        'hash': isA<String>(),
        'chunks': 3,
        'dim': 768,
        'origin': 'device',
        'prebuiltSkipped': 'test: no prebuilt index',
        'key': isA<Map<String, Object?>>(),
      });
    });

    test('a matching marker: reused with its origin and the reason the '
        'prebuilt index was skipped then', () async {
      await indexedBefore();

      final (result, statuses) = await run(build());

      final ready = (result as Ok<KnowledgeReady>).value;
      expect(ready.reused, isTrue);
      expect(ready.origin, KnowledgeOrigin.device);
      expect(ready.prebuiltSkipped, 'test: no prebuilt index');
      expect(logs, [
        '[Knowledge] index reused: 3 chunks (device), hash <hash> (<n>ms)',
      ]);
      expect(statuses, [
        'ready 3 reused=true device skipped=test: no prebuilt index',
      ]);
    });

    test('a marker that is not JSON', () async {
      await indexedBefore();
      marker().writeAsStringSync('{not json');

      final (result, statuses) = await run(build());

      expect(result, isA<Ok<KnowledgeReady>>());
      final String error;
      try {
        jsonDecode('{not json');
        fail('decoded');
      } on FormatException catch (e) {
        error = '$e';
      }
      expect(logs, [
        '[Knowledge] unreadable index marker ($error): re-indexing',
        ...onDevice,
      ]);
      expect(statuses, onDeviceStatuses);
    });

    test('a marker without hash, chunks or dim', () async {
      await indexedBefore();
      writeMarker({'hash': 'x', 'chunks': '3', 'dim': 768});

      await run(build());

      expect(logs, [
        '[Knowledge] index marker without hash/chunks/dim: re-indexing',
        ...onDevice,
      ]);
    });

    test('a marker of an earlier build (no key)', () async {
      await indexedBefore();
      writeMarker({'hash': 'old', 'chunks': 3, 'dim': 768});

      await run(build());

      expect(logs, [
        '[Knowledge] the index was built from a marker of an earlier build: '
            're-indexing',
        ...onDevice,
      ]);
    });

    test('a key this build cannot read', () async {
      await indexedBefore();
      writeMarker({
        'hash': 'old',
        'chunks': 3,
        'dim': 768,
        'key': {'format': 'kb-index/0'},
      });

      await run(build());

      expect(logs, [
        '[Knowledge] the index was built from a key this build cannot read: '
            're-indexing',
        ...onDevice,
      ]);
    });

    test('an equal key with another digest', () async {
      await indexedBefore();
      final json =
          jsonDecode(marker().readAsStringSync()) as Map<String, Object?>;
      writeMarker({...json, 'hash': 'tampered'});

      await run(build());

      expect(logs, [
        '[Knowledge] the index was built from an equal key with another '
            'digest: re-indexing',
        ...onDevice,
      ]);
    });

    test('another key: what differs', () async {
      await indexedBefore();

      await run(build(documents: corpus(alphaTwo: 'Edited text.')));

      expect(logs, hasLength(3));
      expect(
        logs.first,
        matches(
          RegExp(
            r'^\[Knowledge\] the index was built from another knowledge-base '
            r'documents \([0-9a-f]{8} ≠ [0-9a-f]{8}\): re-indexing$',
          ),
        ),
      );
      expect(logs.skip(1), onDevice);
    });

    test('a store with other rows than the marker says', () async {
      await indexedBefore();
      store.rows.remove('beta.md#0');

      await run(build());

      expect(logs, [
        '[Knowledge] the store holds 2 rows of 768 dims, the marker says 3 of '
            '768: re-indexing',
        ...onDevice,
      ]);
    });

    test('store stats that fail', () async {
      await indexedBefore();
      store.statsAnswers.add(Result.error(Exception('database is locked')));

      await run(build());

      expect(logs, [
        '[Knowledge] store stats failed (Exception: database is locked): '
            're-indexing',
        ...onDevice,
      ]);
    });
  });

  group('on the device: each failure is logged and shown', () {
    test('an embedder that is not loaded', () async {
      final (result, statuses) = await run(
        build(withEmbedder: EmbedderService()),
      );

      expect(
        (result as Error<KnowledgeReady>).error.toString(),
        'Bad state: EmbeddingGemma is not loaded',
      );
      expect(logs, [
        '[Knowledge] index failed: Bad state: EmbeddingGemma is not loaded',
      ]);
      expect(statuses, ['failed: Bad state: EmbeddingGemma is not loaded']);
      expect(store.openedPaths, isEmpty);
    });

    test('a clear that fails', () async {
      store.clearError = Exception('readonly database');

      final (result, statuses) = await run(build());

      expect(result, isA<Error<KnowledgeReady>>());
      expect(logs, [
        '[Knowledge] no index marker: indexing',
        '[Knowledge] prebuilt index not used: test: no prebuilt index. '
            'Indexing on the device',
        '[Knowledge] index failed: Exception: readonly database',
      ]);
      expect(statuses, ['failed: Exception: readonly database']);
      expect(model.batches, isEmpty);
      expect(marker().existsSync(), isFalse);
    });

    test('documents that give no chunks', () async {
      final (result, statuses) = await run(build(documents: const []));

      expect(result, isA<Error<KnowledgeReady>>());
      expect(
        logs.last,
        '[Knowledge] index failed: Bad state: The documents '
        'gave no chunks',
      );
      expect(statuses, ['failed: Bad state: The documents gave no chunks']);
      expect(store.clearCalls, 1);
    });

    test('an add that fails', () async {
      store.failAddFrom = 2;

      final (result, statuses) = await run(build());

      expect(result, isA<Error<KnowledgeReady>>());
      expect(logs.last, '[Knowledge] index failed: Exception: disk full');
      expect(statuses, [
        'indexing 0/3 skipped=test: no prebuilt index',
        'failed: Exception: disk full',
      ]);
      expect(store.addCalls, 2);
      expect(marker().existsSync(), isFalse);
    });

    test('a store that holds another row count after indexing', () async {
      store.statsAnswers.add(
        const Result.ok(
          VectorStoreStats(documentCount: 2, vectorDimension: 768),
        ),
      );

      final (result, statuses) = await run(build());

      expect(result, isA<Error<KnowledgeReady>>());
      expect(
        logs.last,
        '[Knowledge] index failed: Bad state: The store holds 2 rows after '
        'indexing 3 chunks',
      );
      expect(
        statuses.last,
        'failed: Bad state: The store holds 2 rows after '
        'indexing 3 chunks',
      );
      expect(statuses, hasLength(4), reason: 'three progress steps first');
      expect(marker().existsSync(), isFalse);
    });

    test('stats that fail after indexing', () async {
      store.statsAnswers.add(Result.error(Exception('database is locked')));

      final (result, statuses) = await run(build());

      expect(result, isA<Error<KnowledgeReady>>());
      expect(
        logs.last,
        '[Knowledge] index failed: Exception: database is '
        'locked',
      );
      expect(statuses.last, 'failed: Exception: database is locked');
      expect(marker().existsSync(), isFalse);
    });

    test('a database that will not open: the reset is logged', () async {
      store.failOpens = 1;

      final (result, _) = await run(build());

      expect(result, isA<Ok<KnowledgeReady>>());
      expect(logs, [
        '[Knowledge] opening the index failed (Exception: file is not a '
            'database): deleting it and re-indexing',
        '[Knowledge] no index marker: indexing',
        ...onDevice,
      ]);
    });

    test('a database that will not open after the reset either', () async {
      store.failOpens = 2;

      final (result, statuses) = await run(build());

      expect(result, isA<Error<KnowledgeReady>>());
      expect(logs, [
        '[Knowledge] opening the index failed (Exception: file is not a '
            'database): deleting it and re-indexing',
        '[Knowledge] index failed: The knowledge-base index did not open, '
            'even after a reset: Exception: file is not a database',
      ]);
      expect(statuses, [
        'failed: The knowledge-base index did not open, even after a reset: '
            'Exception: file is not a database',
      ]);
    });
  });

  group('prebuilt', () {
    test('installed: logged, shown, the marker says prebuilt', () async {
      final key = await appKey();
      final db = Uint8List.fromList(utf8.encode('prebuilt rows'));
      // The fake store keeps its rows across open: these stand for the
      // copied file's.
      for (final id in ['a#0', 'a#1', 'b#0']) {
        store.rows[id] = (
          content: id,
          embedding: List.filled(768, 0.5),
          metadata: '{}',
        );
      }

      final (result, statuses) = await run(
        build(
          prebuilt: FakePrebuiltKbIndex(
            PrebuiltManifestFound(manifestFor(key, db)),
            database: db,
          ),
        ),
      );

      final ready = (result as Ok<KnowledgeReady>).value;
      expect(ready.origin, KnowledgeOrigin.prebuilt);
      expect(logs, [
        '[Knowledge] no index marker: indexing',
        '[Knowledge] prebuilt index installed: 3 chunks in <n>ms',
      ]);
      expect(statuses, ['ready 3 reused=false prebuilt skipped=null']);
      expect(File('${dir.path}/kb/kb.db').readAsBytesSync(), db);
      final written =
          jsonDecode(marker().readAsStringSync()) as Map<String, Object?>;
      expect(written['origin'], 'prebuilt');
      expect(written.containsKey('prebuiltSkipped'), isFalse);
    });

    test('installed, but its marker cannot be written: the run fails '
        'visibly, it does not throw', () async {
      final key = await appKey();
      final db = Uint8List.fromList(utf8.encode('prebuilt rows'));
      for (final id in ['a#0', 'a#1', 'b#0']) {
        store.rows[id] = (
          content: id,
          embedding: List.filled(768, 0.5),
          metadata: '{}',
        );
      }
      // A directory where the marker's temp file goes.
      Directory('${dir.path}/kb/index.json.tmp').createSync(recursive: true);

      final (result, statuses) = await run(
        build(
          prebuilt: FakePrebuiltKbIndex(
            PrebuiltManifestFound(manifestFor(key, db)),
            database: db,
          ),
        ),
      );

      expect(
        (result as Error<KnowledgeReady>).error,
        isA<FileSystemException>(),
      );
      expect(heads(), [
        '[Knowledge] no index marker: indexing',
        startsWith('[Knowledge] indexing failed: FileSystemException: '),
        startsWith('[Knowledge] index failed: FileSystemException: '),
      ]);
      expect(statuses, [startsWith('failed: FileSystemException: ')]);
      expect(marker().existsSync(), isFalse);
    });

    test('a copy that cannot be written: logged, then indexed on the '
        'device', () async {
      final key = await appKey();
      final db = Uint8List.fromList(utf8.encode('prebuilt rows'));
      // A directory where the copy goes: writing the file fails.
      Directory('${dir.path}/kb/kb.db.prebuilt').createSync(recursive: true);

      final (result, statuses) = await run(
        build(
          prebuilt: FakePrebuiltKbIndex(
            PrebuiltManifestFound(manifestFor(key, db)),
            database: db,
          ),
        ),
      );

      final ready = (result as Ok<KnowledgeReady>).value;
      expect(ready.origin, KnowledgeOrigin.device);
      expect(ready.prebuiltSkipped, startsWith('copying it failed: '));
      expect(logs, hasLength(4));
      expect(logs[0], '[Knowledge] no index marker: indexing');
      expect(
        logs[1],
        startsWith('[Knowledge] copying the prebuilt index failed: '),
      );
      expect(logs[1], contains('\n'), reason: 'with its stack trace');
      expect(
        logs[2],
        allOf(
          startsWith(
            '[Knowledge] prebuilt index not used: copying it failed: ',
          ),
          endsWith('. Indexing on the device'),
        ),
      );
      expect(logs[3], '[Knowledge] indexed 3 chunks from 2 documents in <n>ms');
      expect(statuses.first, startsWith('indexing 0/3 skipped=copying it'));
    });

    test('a copy that does not open, and a reset that does not open either: '
        'the run fails', () async {
      final key = await appKey();
      final db = Uint8List.fromList(utf8.encode('not a database'));
      store.onOpen = (_) {
        // The app's own database opens; the copy and both opens of the reset
        // fail.
        if (store.openedPaths.length == 2) store.failOpens = 3;
      };

      final (result, statuses) = await run(
        build(
          prebuilt: FakePrebuiltKbIndex(
            PrebuiltManifestFound(manifestFor(key, db)),
            database: db,
          ),
        ),
      );

      expect(
        (result as Error<KnowledgeReady>).error,
        isA<KnowledgeIndexOpenException>(),
      );
      expect(store.openedPaths, hasLength(4));
      expect(heads(), [
        '[Knowledge] no index marker: indexing',
        '[Knowledge] opening the index failed (Exception: file is not a '
            'database): deleting it and re-indexing',
        '[Knowledge] indexing failed: The knowledge-base index did not open, '
            'even after a reset: Exception: file is not a database',
        '[Knowledge] index failed: The knowledge-base index did not open, '
            'even after a reset: Exception: file is not a database',
      ]);
      expect(statuses, [
        'failed: The knowledge-base index did not open, even after a reset: '
            'Exception: file is not a database',
      ]);
    });
  });

  group('a run that close gave up on stops at its next check', () {
    test('an open that fails because close ran meanwhile: no reset, the files '
        'kept', () async {
      await indexedBefore();
      final db = File('${dir.path}/kb/kb.db')..writeAsStringSync('rows');
      final gate = Completer<void>();
      store.openGates.add(gate);
      final repo = build(closeWait: const Duration(milliseconds: 50));
      final run = repo.ensureIndexed();
      await store.opened.future;

      await repo.close();
      gate.complete();

      expect(
        (await run as Error<KnowledgeReady>).error.toString(),
        'Bad state: KnowledgeRepository closed',
      );
      expect(
        store.openedPaths,
        hasLength(2),
        reason: 'the first launch, then the gated open; never reopened',
      );
      expect(marker().existsSync(), isTrue);
      expect(db.existsSync(), isTrue);
      expect(store.usedWhileClosed, isEmpty);
      expect(logs, [
        '[Knowledge] close: indexing did not stop within 50ms; closing the '
            'store anyway',
      ]);
    });

    test('while the prebuilt database is read: the copy is not '
        'opened', () async {
      final key = await appKey();
      final db = Uint8List.fromList(utf8.encode('prebuilt rows'));
      final prebuilt = _GatedPrebuilt(
        PrebuiltManifestFound(manifestFor(key, db)),
        database: db,
      )..databaseGate = Completer<void>();
      final repo = build(
        prebuilt: prebuilt,
        closeWait: const Duration(milliseconds: 50),
      );
      final run = repo.ensureIndexed();
      await prebuilt.databaseStarted.future;

      await repo.close();
      prebuilt.databaseGate!.complete();

      expect(
        (await run as Error<KnowledgeReady>).error.toString(),
        'Bad state: KnowledgeRepository closed',
      );
      expect(store.openedPaths, hasLength(1), reason: 'not reopened');
      expect(store.usedWhileClosed, isEmpty);
      expect(logs, [
        '[Knowledge] no index marker: indexing',
        '[Knowledge] close: indexing did not stop within 50ms; closing the '
            'store anyway',
      ]);
    });

    test('while the copy is reopened, which then fails: no reset', () async {
      final key = await appKey();
      final db = Uint8List.fromList(utf8.encode('prebuilt rows'));
      final gate = Completer<void>();
      store.openGates.addAll([null, gate]); // the app's open, then the copy's
      final repo = build(
        prebuilt: FakePrebuiltKbIndex(
          PrebuiltManifestFound(manifestFor(key, db)),
          database: db,
        ),
        closeWait: const Duration(milliseconds: 50),
      );
      final run = repo.ensureIndexed();
      await store.opened.future;

      await repo.close();
      gate.complete();

      expect(
        (await run as Error<KnowledgeReady>).error.toString(),
        'Bad state: KnowledgeRepository closed',
      );
      expect(store.openedPaths, hasLength(2), reason: 'no reset');
      expect(store.usedWhileClosed, isEmpty);
      expect(logs, [
        '[Knowledge] no index marker: indexing',
        '[Knowledge] close: indexing did not stop within 50ms; closing the '
            'store anyway',
      ]);
    });

    const closeGaveUp =
        '[Knowledge] close: indexing did not stop within 50ms; closing the '
        'store anyway';

    test('while the prebuilt manifest is read: nothing more is logged and the '
        'store is not cleared', () async {
      final prebuilt = _GatedPrebuilt(
        const PrebuiltManifestMissing('test: no prebuilt index'),
      )..manifestGate = Completer<void>();
      final repo = build(
        prebuilt: prebuilt,
        closeWait: const Duration(milliseconds: 50),
      );
      final run = repo.ensureIndexed();
      await prebuilt.manifestStarted.future;

      await repo.close();
      prebuilt.manifestGate!.complete();

      expect(
        (await run as Error<KnowledgeReady>).error.toString(),
        'Bad state: KnowledgeRepository closed',
      );
      expect(store.clearCalls, 0);
      expect(store.usedWhileClosed, isEmpty);
      expect(logs, ['[Knowledge] no index marker: indexing', closeGaveUp]);
    });

    test('while the installed copy is checked, which then passes: no marker, '
        'not ready', () async {
      final key = await appKey();
      final db = Uint8List.fromList(utf8.encode('prebuilt rows'));
      for (final id in ['a#0', 'a#1', 'b#0']) {
        store.rows[id] = (
          content: id,
          embedding: List.filled(768, 0.5),
          metadata: '{}',
        );
      }
      // The first stats call of a run without a marker: the copy's check.
      final gate = store.statsGate = Completer<void>();
      final repo = build(
        prebuilt: FakePrebuiltKbIndex(
          PrebuiltManifestFound(manifestFor(key, db)),
          database: db,
        ),
        closeWait: const Duration(milliseconds: 50),
      );
      final run = repo.ensureIndexed();
      await store.statsStarted.future;

      await repo.close();
      gate.complete();

      expect(
        (await run as Error<KnowledgeReady>).error.toString(),
        'Bad state: KnowledgeRepository closed',
      );
      expect(marker().existsSync(), isFalse);
      expect(logs, ['[Knowledge] no index marker: indexing', closeGaveUp]);
    });

    test('while the reuse check reads the store: the marker is '
        'kept', () async {
      await indexedBefore();
      store.rows.remove('beta.md#0'); // a mismatch the check reports late
      final gate = store.statsGate = Completer<void>();
      final repo = build(closeWait: const Duration(milliseconds: 50));
      final run = repo.ensureIndexed();
      await store.statsStarted.future;

      await repo.close();
      gate.complete();

      expect(
        (await run as Error<KnowledgeReady>).error.toString(),
        'Bad state: KnowledgeRepository closed',
      );
      expect(marker().existsSync(), isTrue, reason: 'no re-index forced');
      expect(store.clearCalls, 1, reason: 'the first launch only');
      expect(store.usedWhileClosed, isEmpty);
      expect(logs, [
        closeGaveUp,
        '[Knowledge] the store holds 2 rows of 768 dims, the marker says 3 of '
            '768: re-indexing',
      ]);
    });

    test("while a batch's last add runs: the next batch is not "
        'embedded', () async {
      store.gatedAdd = 2; // the end of the first batch of two
      final repo = build(closeWait: const Duration(milliseconds: 50));
      final run = repo.ensureIndexed();
      await store.gatedAddStarted.future;

      await repo.close();
      store.gatedAddGate.complete();

      expect(
        (await run as Error<KnowledgeReady>).error.toString(),
        'Bad state: KnowledgeRepository closed',
      );
      expect(model.batches, hasLength(1));
      expect(store.addCalls, 2);
      expect(store.usedWhileClosed, isEmpty);
    });

    test('while the last add runs: no row count is read, no marker '
        'written', () async {
      store.gatedAdd = 3;
      final repo = build(closeWait: const Duration(milliseconds: 50));
      final run = repo.ensureIndexed();
      await store.gatedAddStarted.future;

      await repo.close();
      store.gatedAddGate.complete();

      expect(
        (await run as Error<KnowledgeReady>).error.toString(),
        'Bad state: KnowledgeRepository closed',
      );
      expect(store.usedWhileClosed, isEmpty, reason: 'no stats');
      expect(marker().existsSync(), isFalse);
      expect(logs, [
        '[Knowledge] no index marker: indexing',
        ...onDevice.take(1),
        closeGaveUp,
      ]);
    });

    test('while the row count is read, which then matches: no marker, not '
        'ready', () async {
      // The only stats call of a run that indexes on the device.
      final gate = store.statsGate = Completer<void>();
      final repo = build(closeWait: const Duration(milliseconds: 50));
      final run = repo.ensureIndexed();
      await store.statsStarted.future;

      await repo.close();
      gate.complete();

      expect(
        (await run as Error<KnowledgeReady>).error.toString(),
        'Bad state: KnowledgeRepository closed',
      );
      expect(marker().existsSync(), isFalse);
      expect(logs, [
        '[Knowledge] no index marker: indexing',
        ...onDevice.take(1),
        closeGaveUp,
      ]);
    });

    test('a retrieval close gave up on is logged', () async {
      models.value = {ModelId.embeddingGemma: _ready};
      final repo = build(closeWait: const Duration(milliseconds: 50));
      expect(await repo.ensureIndexed(), isA<Ok<KnowledgeReady>>());
      store.searchGate = Completer<void>();
      final retrieving = repo.retrieve('q');
      await store.searchStarted.future;
      logs.clear();

      await repo.close();
      store.searchGate!.complete();
      await retrieving;

      expect(
        logs.first,
        '[Knowledge] close: 1 retrieval(s) still searching after 50ms; '
        'closing the store anyway',
      );
    });
  });
}
