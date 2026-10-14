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
    show
        CancelToken,
        EmbeddingInput,
        EmbeddingInputTooLongException,
        EmbeddingModality,
        EmbeddingModel,
        PreferredBackend,
        TaskType;
import 'package:flutter_edge_ai_rag/flutter_edge_ai_rag.dart'
    show RetrievalResult, VectorStoreStats;
import 'package:litert_edge_demos/config/knowledge_config.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/services/knowledge/embedder_digests.dart';
import 'package:litert_edge_demos/data/services/knowledge/embedder_service.dart';
import 'package:litert_edge_demos/data/services/knowledge/kb_prebuilt_index.dart';
import 'package:litert_edge_demos/data/services/knowledge/vector_store_service.dart';
import 'package:litert_edge_demos/domain/models/knowledge.dart';
import 'package:litert_edge_demos/domain/models/knowledge_query.dart';
import 'package:litert_edge_demos/domain/ports/knowledge_retriever.dart';
import 'package:litert_edge_demos/utils/result.dart';

/// An [EmbeddingModel] at the package edge: deterministic non-zero vectors
/// of [dimension], with the calls recorded. [gate] pauses [embedAll]; from
/// batch [failFromBatch] (1-based) it throws. Like the engine it reports the
/// media it was loaded with: text plus the config's ([loadedWith]; the
/// app's setup load, text and audio, until then), unless a test fixes
/// [modalities]; and it refuses an image or a recording it was not loaded
/// for. A 1024-token limit unless told otherwise.
base class FakeEmbeddingModel extends EmbeddingModel {
  FakeEmbeddingModel({
    this.dimension = 768,
    this.backend = PreferredBackend.cpu,
    this.zeroVectors = false,
    Set<EmbeddingModality>? modalities,
    this.maxInputTokens = kEmbedderMaxInputTokens,
  }) : _fixed = modalities;

  @override
  final int dimension;
  final PreferredBackend? backend;
  final bool zeroVectors;

  final Set<EmbeddingModality>? _fixed;
  Set<EmbeddingModality> _loaded = {
    EmbeddingModality.text,
    ...kEmbedderConfig.media,
  };

  @override
  Set<EmbeddingModality> get modalities => _fixed ?? _loaded;

  /// This model as a load with [config] makes it: it reports text plus
  /// [config]'s media (unless the test fixed them). What the fake loaders
  /// return.
  FakeEmbeddingModel loadedWith(EmbedderConfig config) {
    _loaded = {EmbeddingModality.text, ...config.media};
    return this;
  }

  @override
  final int? maxInputTokens;

  /// Every [embed] call's input (text, audio and images), in order.
  final List<(EmbeddingInput, TaskType)> inputs = [];

  /// The text [embed] calls (the warm-up and text questions).
  final List<(String, TaskType)> queries = [];

  /// When set, [embed] fails with it (an input the engine refused as too
  /// long after counting it).
  EmbeddingInputTooLongException? tooLong;
  final List<(List<String>, TaskType)> batches = [];
  Completer<void>? gate;
  int? failFromBatch;
  int closeCalls = 0;

  /// When set, [embed] (the warm-up and queries) throws it.
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
  Future<List<double>> embed(
    EmbeddingInput input, {
    required TaskType taskType,
  }) async {
    inputs.add((input, taskType));
    if (input.images.isNotEmpty &&
        !modalities.contains(EmbeddingModality.image)) {
      throw StateError('the embedder was loaded without images');
    }
    if (input.audio != null && !modalities.contains(EmbeddingModality.audio)) {
      throw StateError('the embedder was loaded without audio');
    }
    final text =
        input.text ??
        switch (input.audio) {
          final audio? => 'audio:${audio.length}',
          null => 'image:${input.images.first.length}',
        };
    if (input.text != null) queries.add((text, taskType));
    if (queryError case final error?) throw error;
    if (tooLong case final error?) throw error;
    return vectorFor(text);
  }

  @override
  Future<List<List<double>>> embedAll(
    List<EmbeddingInput> inputs, {
    required TaskType taskType,
    CancelToken? cancelToken,
  }) async {
    final texts = [for (final input in inputs) input.text!];
    batches.add((texts, taskType));
    await gate?.future;
    if (failFromBatch case final n? when batches.length >= n) {
      throw StateError('embedding worker died');
    }
    return [for (final text in texts) vectorFor(text)];
  }

  @override
  void addCloseListener(void Function() listener) {}

  @override
  Future<void> close() async => closeCalls++;
}

/// A real [EmbedderService] whose install and load are fakes, already
/// installed and loaded with [model].
Future<EmbedderService> loadedEmbedder(
  FakeEmbeddingModel model, {
  String modelId = 'embeddinggemma-2-740m',
  Future<Uint8List> Function()? warmUpAudio,
  EmbedderConfig config = kEmbedderConfig,
}) async {
  final service = EmbedderService(
    config: config,
    install: (config, source, onProgress) async => modelId,
    load: (config) async => model.loadedWith(config),
    warmUpAudio: warmUpAudio,
  );
  // An empty file: the install checks it exists; the fake install
  // registers nothing.
  final dir = Directory.systemTemp.createTempSync('fake_embedder');
  final installed = await service.install(
    EmbedderFromFiles(
      modelPath: (File('${dir.path}/model.litertlm')..createSync()).path,
    ),
    onProgress: (_) {},
  );
  if (installed is! Ok<String>) throw StateError('fake install: $installed');
  final loaded = await service.load();
  if (loaded is! Ok<EmbedderInfo>) throw StateError('fake load: $loaded');
  return service;
}

/// How the fakes key a question: a text question by its text, a spoken one
/// as `audio:<bytes>`.
String queryKey(KnowledgeQuery query) => switch (query) {
  KnowledgeTextQuery(:final text) => text,
  KnowledgeAudioQuery(:final wav) => 'audio:${wav.length}',
};

/// An in-memory [VectorStoreService]: rows survive as long as the instance,
/// like the sqlite file across launches. [search] answers from
/// [searchResults] by [queryKey] (an empty list when absent).
class FakeVectorStoreService implements VectorStoreService {
  FakeVectorStoreService({this.profileId = kKbEmbeddingProfileId});

  @override
  final String profileId;

  final Map<String, ({String content, List<double> embedding, String metadata})>
  rows = {};
  final List<String> openedPaths = [];
  final List<(String, int)> searches = [];

  /// The questions [search] got, as they were passed.
  final List<KnowledgeQuery> searchQueries = [];
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
    KnowledgeQuery query, {
    required int topK,
  }) => _call('search', () async {
    searchQueries.add(query);
    searches.add((queryKey(query), topK));
    if (searchGate case final gate?) {
      if (!searchStarted.isCompleted) searchStarted.complete();
      await gate.future;
    }
    if (searchError case final error?) return Result.error(error);
    return Result.ok(searchResults[queryKey(query)] ?? const []);
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

  /// Every question asked, as it was passed.
  final List<KnowledgeQuery> queries = [];
  Completer<void>? gate;

  /// The [queries] by [queryKey] (a text question's text).
  List<String> get questions => [for (final q in queries) queryKey(q)];

  @override
  Future<Result<Retrieval>> retrieve(KnowledgeQuery query) async {
    queries.add(query);
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
  FixedEmbedderDigests({this.model = 'model-sha'});

  String model;

  @override
  Future<String> digestOf(String path) async => model;
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
