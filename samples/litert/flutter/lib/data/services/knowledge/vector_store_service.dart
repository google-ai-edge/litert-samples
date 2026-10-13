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

import 'package:flutter/foundation.dart';
import 'package:flutter_edge_ai_rag/flutter_edge_ai_rag.dart'
    show
        EmbeddingProfile,
        FlutterEdgeAiRag,
        RagIndex,
        RetrievalResult,
        VectorStoreSpec,
        VectorStoreStats;
import 'package:flutter_edge_ai_sqlite/flutter_edge_ai_sqlite.dart'
    show SqliteVectorStoreProvider;

import '../../../config/knowledge_config.dart';
import '../../../utils/result.dart';

/// The query embedding [VectorStoreService.search] searches with
/// (`EmbedderService.embedQuery`: the `retrievalQuery` prefix).
typedef QueryEmbedder = Future<Result<List<double>>> Function(String query);

/// The sqlite-vec knowledge-base index: one `flutter_edge_ai_rag` index over
/// `flutter_edge_ai_sqlite`. Only `KnowledgeRepository` uses it.
///
/// - [open] opens (or creates) the database at a path, bound to the
///   knowledge base's embedding profile ([kKbEmbeddingProfileId]). A
///   database an earlier build wrote before the RAG package had profiles
///   is adopted: its vectors came from the same built-in EmbeddingGemma,
///   and the package checks their dimension.
/// - [search] embeds the query through [QueryEmbedder] and returns
///   similarity = 1 − cosine distance, best first. No threshold: the
///   repository gates.
/// - [clear] drops the rows (the profile binding stays); the next [add]
///   recreates the table.
/// - Every [add] is its own autocommit transaction (no batch insert).
class VectorStoreService {
  VectorStoreService({
    required this._embedQuery,
    FlutterEdgeAiRag? rag,
    String profileId = kKbEmbeddingProfileId,
    int dimension = kKbEmbeddingDimension,
  }) : _rag =
           rag ??
           FlutterEdgeAiRag(providers: const [SqliteVectorStoreProvider()]),
       _profile = EmbeddingProfile(id: profileId, dimension: dimension);

  final QueryEmbedder _embedQuery;
  final FlutterEdgeAiRag _rag;
  final EmbeddingProfile _profile;
  RagIndex? _index;
  bool _closed = false;

  /// Opens [databasePath], closing the database opened before. Refused
  /// after [close]: an index opened then would have no owner.
  Future<Result<void>> open(String databasePath) => _guard('open', () async {
    if (_closed) throw StateError('the knowledge-base index is closed');
    final previous = _index;
    _index = null;
    await previous?.dispose();
    final index = await _rag.open(
      spec: VectorStoreSpec(
        providerId: 'sqlite',
        location: databasePath,
        allowLegacyProfileAdoption: true,
      ),
      embeddingProfile: _profile,
    );
    if (_closed) {
      // close() ran during the open.
      await index.dispose();
      throw StateError('the knowledge-base index is closed');
    }
    _index = index;
  });

  Future<Result<void>> add({
    required String id,
    required String content,
    required List<double> embedding,
    required String metadata,
  }) => _guard(
    'add',
    () => _opened.addVector(
      id: id,
      content: content,
      embedding: embedding,
      metadata: metadata,
    ),
  );

  Future<Result<List<RetrievalResult>>> search(
    String query, {
    required int topK,
  }) async {
    final List<double> embedding;
    switch (await _embedQuery(query)) {
      case Ok(:final value):
        embedding = value;
      case Error(:final error):
        return Result.error(error);
    }
    return _guard(
      'search',
      () => _opened.searchVector(embedding: embedding, topK: topK),
    );
  }

  Future<Result<VectorStoreStats>> stats() =>
      _guard('stats', () => _opened.stats());

  Future<Result<void>> clear() => _guard('clear', () => _opened.clear());

  /// Closes the open database; [open] refuses from now on.
  Future<void> close() async {
    _closed = true;
    final index = _index;
    _index = null;
    await index?.dispose();
  }

  RagIndex get _opened =>
      _index ?? (throw StateError('the knowledge-base index is not open'));

  static Future<Result<T>> _guard<T>(
    String what,
    Future<T> Function() action,
  ) async {
    try {
      return Result.ok(await action());
    } catch (e, st) {
      debugPrint('[VectorStoreService] $what failed: $e\n$st');
      return Result.error(asException(e));
    }
  }
}
