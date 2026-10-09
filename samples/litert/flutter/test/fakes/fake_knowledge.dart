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
import 'dart:typed_data';

import 'package:flutter_edge_ai/flutter_edge_ai.dart'
    show EmbeddingModel, PreferredBackend, TaskType;
import 'package:flutter_edge_ai_rag/flutter_edge_ai_rag.dart'
    show RetrievalResult, VectorStoreStats;
import 'package:litert_edge_demos/data/services/knowledge/embedder_digests.dart';
import 'package:litert_edge_demos/data/services/knowledge/embedder_service.dart';
import 'package:litert_edge_demos/data/services/knowledge/kb_prebuilt_index.dart';
import 'package:litert_edge_demos/data/services/knowledge/vector_store_service.dart';
import 'package:litert_edge_demos/domain/models/knowledge.dart';
import 'package:litert_edge_demos/domain/ports/knowledge_retriever.dart';
import 'package:litert_edge_demos/utils/result.dart';

/// An [EmbeddingModel] at the package edge: deterministic non-zero vectors
/// of [dimension], with the calls recorded. [gate] pauses
/// [generateEmbeddings]; from batch [failFromBatch] (1-based) it throws.
class FakeEmbeddingModel extends EmbeddingModel {
  FakeEmbeddingModel({
    this.dimension = 768,
    this.backend = PreferredBackend.cpu,
    this.zeroVectors = false,
  });

  final int dimension;
  final PreferredBackend? backend;
  final bool zeroVectors;

  final List<(String, TaskType)> queries = [];
  final List<(List<String>, TaskType)> batches = [];
  Completer<void>? gate;
  int? failFromBatch;
  int closeCalls = 0;

  /// When set, [generateEmbedding] (the warm-up and queries) throws it.
  StateError? queryError;

  int get documentsEmbedded =>
      batches.fold(0, (sum, batch) => sum + batch.$1.length);

  List<double> vectorFor(String text) => [
    for (var i = 0; i < dimension; i++)
      zeroVectors ? 0.0 : ((text.hashCode + i * 31) % 97 + 1) / 97,
  ];

  @override
  PreferredBackend? get activeBackend => backend;

  @override
  bool get isClosed => closeCalls > 0;

  @override
  Future<List<double>> generateEmbedding(
    String text, {
    TaskType taskType = TaskType.retrievalQuery,
  }) async {
    queries.add((text, taskType));
    if (queryError case final error?) throw error;
    return vectorFor(text);
  }

  @override
  Future<List<List<double>>> generateEmbeddings(
    List<String> texts, {
    TaskType taskType = TaskType.retrievalQuery,
  }) async {
    batches.add((texts, taskType));
    await gate?.future;
    if (failFromBatch case final n? when batches.length >= n) {
      throw StateError('embedding worker died');
    }
    return [for (final text in texts) vectorFor(text)];
  }

  @override
  Future<int> getDimension() async => dimension;

  @override
  void addCloseListener(void Function() listener) {}

  @override
  Future<void> close() async => closeCalls++;
}

/// A real [EmbedderService] whose install and load are fakes, already
/// installed and loaded with [model].
Future<EmbedderService> loadedEmbedder(
  FakeEmbeddingModel model, {
  String modelId = 'embeddinggemma-300M_seq512_mixed-precision',
}) async {
  final service = EmbedderService(
    install: (config, source, onProgress) async => modelId,
    load: (config) async => model,
  );
  // Two empty files: the install checks they exist; the fake install
  // registers nothing.
  final dir = Directory.systemTemp.createTempSync('fake_embedder');
  final installed = await service.install(
    EmbedderFromFiles(
      modelPath: (File('${dir.path}/model.tflite')..createSync()).path,
      tokenizerPath: (File(
        '${dir.path}/sentencepiece.model',
      )..createSync()).path,
    ),
    onProgress: (_) {},
  );
  if (installed is! Ok<String>) throw StateError('fake install: $installed');
  final loaded = await service.load();
  if (loaded is! Ok<EmbedderInfo>) throw StateError('fake load: $loaded');
  return service;
}

/// An in-memory [VectorStoreService]: rows survive as long as the instance,
/// like the sqlite file across launches. [search] answers from
/// [searchResults] by query (an empty list when absent).
class FakeVectorStoreService implements VectorStoreService {
  final Map<String, ({String content, List<double> embedding, String metadata})>
  rows = {};
  final List<String> openedPaths = [];
  final List<(String, int)> searches = [];
  int clearCalls = 0;
  int addCalls = 0;
  Map<String, List<RetrievalResult>> searchResults = {};
  Exception? searchError;

  /// When set, [add] fails from this call on (1-based).
  int? failAddFrom;

  /// When set, [add] and [search] wait for them, like a native call in
  /// flight; [addStarted] and [searchStarted] complete when one waits.
  Completer<void>? addGate;
  Completer<void>? searchGate;
  final Completer<void> addStarted = Completer<void>();
  final Completer<void> searchStarted = Completer<void>();

  /// Calls in flight, and the calls made on the store after its close
  /// (each one a use-after-free of the native index, or a reopen nothing
  /// owns, in the app).
  int _inFlight = 0;
  bool _closed = false;
  final List<String> usedWhileClosed = [];

  /// Like the real one: a call on a closed store is an error.
  Result<T>? _refusedWhenClosed<T>(String what) {
    if (!_closed) return null;
    usedWhileClosed.add(what);
    return Result.error(
      asException(StateError('the knowledge-base index is closed')),
    );
  }

  Future<T> _call<T>(String what, Future<T> Function() body) async {
    if (_closed) usedWhileClosed.add(what);
    _inFlight++;
    try {
      return await body();
    } finally {
      _inFlight--;
    }
  }

  /// The next this-many [open] calls fail with [openError], like a corrupt
  /// `kb.db` or a schema the store's release does not expect. [onOpen] sees
  /// each path first (to check what is on disk at that moment).
  int failOpens = 0;
  Exception openError = Exception('file is not a database');
  void Function(String path)? onOpen;

  @override
  Future<Result<void>> open(String databasePath) async {
    if (_refusedWhenClosed<void>('open') case final refused?) return refused;
    openedPaths.add(databasePath);
    onOpen?.call(databasePath);
    if (failOpens > 0) {
      failOpens--;
      return Result.error(openError);
    }
    return const Result.ok(null);
  }

  @override
  Future<Result<void>> add({
    required String id,
    required String content,
    required List<double> embedding,
    required String metadata,
  }) => _call('add', () async {
    addCalls++;
    if (addGate case final gate?) {
      if (!addStarted.isCompleted) addStarted.complete();
      await gate.future;
    }
    if (failAddFrom case final n? when addCalls >= n) {
      return Result.error(Exception('disk full'));
    }
    rows[id] = (content: content, embedding: embedding, metadata: metadata);
    return const Result.ok(null);
  });

  @override
  Future<Result<List<RetrievalResult>>> search(
    String query, {
    required int topK,
  }) => _call('search', () async {
    searches.add((query, topK));
    if (searchGate case final gate?) {
      if (!searchStarted.isCompleted) searchStarted.complete();
      await gate.future;
    }
    if (searchError case final error?) return Result.error(error);
    return Result.ok(searchResults[query] ?? const []);
  });

  @override
  Future<Result<VectorStoreStats>> stats() async =>
      _refusedWhenClosed('stats') ??
      Result.ok(
        VectorStoreStats(
          documentCount: rows.length,
          vectorDimension: rows.isEmpty
              ? 0
              : rows.values.first.embedding.length,
        ),
      );

  @override
  Future<Result<void>> clear() async {
    if (_refusedWhenClosed<void>('clear') case final refused?) return refused;
    clearCalls++;
    rows.clear();
    return const Result.ok(null);
  }

  int closeCalls = 0;

  /// Calls that were still in flight when [close] ran.
  int closedWithCallsInFlight = 0;

  /// The next launch: a new service over the same database (its rows and
  /// the counters are kept), open to a new repository.
  void relaunch() => _closed = false;

  /// Like the real one, the store stays closed: [open] refuses afterwards.
  @override
  Future<void> close() async {
    closeCalls++;
    _closed = true;
    closedWithCallsInFlight += _inFlight;
  }
}

/// A [KnowledgeRetriever] that answers [result] and records the questions;
/// [gate] holds the answer back.
class FakeRetriever implements KnowledgeRetriever {
  FakeRetriever(this.result);

  Result<Retrieval> result;
  final List<String> questions = [];
  Completer<void>? gate;

  @override
  Future<Result<Retrieval>> retrieve(String question) async {
    questions.add(question);
    await gate?.future;
    return result;
  }
}

/// A passage as the index would return it.
Passage passage(
  int n, {
  double similarity = 0.6,
  String doc = 'litert-overview.md',
  String title = 'LiteRT overview',
  String? section,
  String? source = 'https://ai.google.dev/edge/litert',
}) {
  final s = section ?? 'Section $n';
  return Passage(
    id: '$doc#$n',
    doc: doc,
    title: title,
    section: s,
    content: '$title › $s\n\nBody of excerpt $n.',
    similarity: similarity,
    source: source,
  );
}

/// Fixed [EmbedderDigests]: no hashing.
class FixedEmbedderDigests implements EmbedderDigests {
  FixedEmbedderDigests({this.model = 'model-sha', this.tokenizer = 'tok-sha'});

  String model;
  String tokenizer;

  @override
  Future<EmbedderDigest> of({
    required String modelPath,
    required String tokenizerPath,
  }) async => EmbedderDigest(model: model, tokenizer: tokenizer);
}

/// A [PrebuiltKbIndexSource] serving [manifestRead] and [databaseBytes];
/// [manifestError] makes [manifest] throw it.
class FakePrebuiltKbIndex implements PrebuiltKbIndexSource {
  FakePrebuiltKbIndex(this.manifestRead, {Uint8List? database})
    : databaseBytes = database ?? Uint8List(0);

  PrebuiltManifestRead manifestRead;
  Uint8List databaseBytes;
  Exception? manifestError;
  int databaseReads = 0;

  @override
  Future<PrebuiltManifestRead> manifest() async {
    if (manifestError case final error?) throw error;
    return manifestRead;
  }

  @override
  Future<Uint8List> database() async {
    databaseReads++;
    return databaseBytes;
  }
}
