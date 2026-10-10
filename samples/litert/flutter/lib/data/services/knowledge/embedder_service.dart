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

import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter_edge_ai/flutter_edge_ai.dart'
    show EmbeddingModel, FlutterEdgeAi, PreferredBackend, TaskType;

import '../../../config/model_catalog.dart';
import '../../../domain/models/model_id.dart';
import '../../../utils/result.dart';

/// Where EmbeddingGemma comes from.
sealed class const EmbedderSource();

/// The files built into the app (absolute paths): in place in the app
/// bundle, or Android's verified extracted copy. Registered in place, never
/// copied.
final class const EmbedderFromFiles({
  required final String modelPath,
  required final String tokenizerPath,
}) extends EmbedderSource;

/// Registers (local) or downloads (network) the model and tokenizer and
/// makes them the active embedder; returns the model id. [source]'s paths
/// are absolute and both files exist.
typedef EmbedderInstaller = Future<String> Function(
  EmbedderConfig config,
  EmbedderSource source,
  void Function(int percent) onProgress,
);

/// Builds the active embedder.
typedef EmbedderLoader = Future<EmbeddingModel> Function(EmbedderConfig config);

/// `installEmbedder` from files (`modelFromFile` registers the path; the
/// load reads the active spec's own path even when the file name was
/// registered from elsewhere before).
Future<String> installEmbedderFrom(
  EmbedderConfig config,
  EmbedderSource source,
  void Function(int percent) onProgress,
) async {
  final builder = FlutterEdgeAi.installEmbedder();
  switch (source) {
    case EmbedderFromFiles(:final modelPath, :final tokenizerPath):
      builder.modelFromFile(modelPath).tokenizerFromFile(tokenizerPath);
  }
  final installation = await builder.install();
  return installation.modelId;
}

/// `FlutterEdgeAi.getActiveEmbedder` with the [EmbedderConfig] backend (a
/// worker isolate; LiteRT forces the CPU and reports it).
Future<EmbeddingModel> getActiveEmbedderFor(EmbedderConfig config) =>
    FlutterEdgeAi.getActiveEmbedder(preferredBackend: config.backend);

/// A file the embedder needs is not there (a built-in file missing from
/// this build, or an extracted copy that vanished): shown as "unavailable"
/// rather than a failure to retry.
final class EmbedderFilesMissingException implements Exception {
  const EmbedderFilesMissingException(this.paths);

  final List<String> paths;

  @override
  String toString() =>
      'EmbeddingGemma files not found or not readable: ${paths.join(', ')} '
      '(a build fetches them with tool/fetch_models.sh first)';
}

/// The loaded embedder is not the one the catalog describes (backend,
/// dimension, or zero vectors); it is closed, never used.
final class EmbedderMismatchException implements Exception {
  const EmbedderMismatchException(this.message);

  final String message;

  @override
  String toString() => message;
}

/// The service was closed while an operation was in flight.
final class EmbedderServiceClosedException implements Exception {
  const EmbedderServiceClosedException();

  @override
  String toString() => 'The embedder service was closed';
}

/// What [EmbedderService.load] reports.
final class const EmbedderInfo({
  required final String modelId,
  required final PreferredBackend backend,
  required final int dimension,
  required final Duration loadTime,
});

/// Owns EmbeddingGemma: install, load (checked against the catalog), warm
/// up, embed documents, close. `ModelRepository` drives its lifecycle (with
/// it, the only caller of `getActiveEmbedder`); `KnowledgeRepository`
/// borrows [embedDocuments] for indexing. Queries are embedded by the vector
/// store's `searchSimilar`, which uses this same active embedder.
class EmbedderService {
  EmbedderService({
    this._config = kEmbedderConfig,
    this._install = installEmbedderFrom,
    this._load = getActiveEmbedderFor,
  });

  final EmbedderConfig _config;
  final EmbedderInstaller _install;
  final EmbedderLoader _load;

  EmbeddingModel? _model;
  String? _modelId;
  ({String model, String tokenizer})? _installedFiles;
  bool _closed = false;

  bool get isLoaded => _model != null;

  /// The installed model's id (its file name without the extension); null
  /// before [install] succeeds. Part of the knowledge-base index hash.
  String? get modelId => _modelId;

  /// The absolute paths of the model and tokenizer files [install]
  /// registered; null before it succeeds. The knowledge base keys its index
  /// on their SHA-256.
  ({String model, String tokenizer})? get installedFiles => _installedFiles;

  /// The vector size every embedding has; checked at [load].
  int get dimension => _config.dimension;

  /// Registers the files and makes them the active embedder. A missing
  /// file fails with [EmbedderFilesMissingException] before anything is
  /// registered.
  Future<Result<String>> install(
    EmbedderSource source, {
    required void Function(int percent) onProgress,
  }) async {
    try {
      final files = switch (source) {
        EmbedderFromFiles(:final modelPath, :final tokenizerPath) => [
          modelPath,
          tokenizerPath,
        ],
      };
      final missing = [
        for (final path in files)
          if (!await File(path).exists()) path,
      ];
      if (missing.isNotEmpty) {
        return Result.error(EmbedderFilesMissingException(missing));
      }
      final id = await _install(_config, source, onProgress);
      _modelId = id;
      _installedFiles = (model: files[0], tokenizer: files[1]);
      return Result.ok(id);
    } catch (e, st) {
      debugPrint('[EmbedderService] install failed: $e\n$st');
      return Result.error(asException(e));
    }
  }

  /// Loads the active embedder. Fails, and closes what was loaded, unless it
  /// reports the catalog's backend and dimension.
  Future<Result<EmbedderInfo>> load() async {
    final watch = Stopwatch()..start();
    try {
      final model = await _load(_config);
      watch.stop();
      if (_closed) {
        await _closeQuietly(model);
        return const Result.error(EmbedderServiceClosedException());
      }
      final backend = model.activeBackend;
      if (backend != _config.backend) {
        await _closeQuietly(model);
        return Result.error(
          EmbedderMismatchException(
            'EmbeddingGemma requested ${_config.backend.name} but reports '
            '${backend?.name ?? 'an unknown backend'}',
          ),
        );
      }
      final dimension = await model.getDimension();
      if (dimension != _config.dimension) {
        await _closeQuietly(model);
        return Result.error(
          EmbedderMismatchException(
            'EmbeddingGemma reports $dimension dimensions, the catalog '
            'expects ${_config.dimension}',
          ),
        );
      }
      if (_closed) {
        await _closeQuietly(model);
        return const Result.error(EmbedderServiceClosedException());
      }
      _model = model;
      final info = EmbedderInfo(
        modelId: _modelId ?? ModelId.embeddingGemma.name,
        backend: backend!,
        dimension: dimension,
        loadTime: watch.elapsed,
      );
      debugPrint(
        '[EmbedderService] loaded ${info.modelId} backend=${backend.name} '
        'dim=$dimension load=${watch.elapsedMilliseconds}ms',
      );
      return Result.ok(info);
    } catch (e, st) {
      debugPrint('[EmbedderService] load failed: $e\n$st');
      return Result.error(asException(e));
    }
  }

  /// One query embedding, so the first question does not pay for lazy
  /// setup; fails on a wrong-sized or all-zero vector. A failed warm-up
  /// closes and unloads the model, like a mismatch at [load]: the row shows
  /// the failure and the ~0.3 GB are released.
  Future<Result<Duration>> warmUp() async {
    final model = _model;
    if (_closed) return const Result.error(EmbedderServiceClosedException());
    if (model == null) {
      return Result.error(asException(StateError('warmUp() before load()')));
    }
    final watch = Stopwatch()..start();
    try {
      final vector = await model.generateEmbedding(
        'What is LiteRT?',
        taskType: TaskType.retrievalQuery,
      );
      if (_checkVector(vector) case final String problem) {
        await _unload(model);
        return Result.error(EmbedderMismatchException(problem));
      }
      debugPrint('[EmbedderService] warm-up ${watch.elapsedMilliseconds}ms');
      return Result.ok(watch.elapsed);
    } catch (e, st) {
      debugPrint('[EmbedderService] warm-up failed: $e\n$st');
      await _unload(model);
      return Result.error(asException(e));
    }
  }

  /// A query embedding (the `retrievalQuery` prefix), checked like the
  /// warm-up's: what the knowledge base searches with.
  Future<Result<List<double>>> embedQuery(String text) async {
    final model = _model;
    if (_closed) return const Result.error(EmbedderServiceClosedException());
    if (model == null) {
      return Result.error(
        asException(StateError('embedQuery() before load()')),
      );
    }
    try {
      final vector = await model.generateEmbedding(
        text,
        taskType: TaskType.retrievalQuery,
      );
      if (_checkVector(vector) case final String problem) {
        return Result.error(EmbedderMismatchException(problem));
      }
      return Result.ok(vector);
    } catch (e, st) {
      debugPrint('[EmbedderService] embedQuery failed: $e\n$st');
      return Result.error(asException(e));
    }
  }

  /// Drops [model] if it is still the loaded one, then closes it.
  Future<void> _unload(EmbeddingModel model) async {
    if (identical(_model, model)) _model = null;
    await _closeQuietly(model);
  }

  /// Document embeddings (the `retrievalDocument` prefix), one per text, in
  /// order. Each is checked like the warm-up's.
  Future<Result<List<List<double>>>> embedDocuments(List<String> texts) async {
    final model = _model;
    if (_closed) return const Result.error(EmbedderServiceClosedException());
    if (model == null) {
      return Result.error(
        asException(StateError('embedDocuments() before load()')),
      );
    }
    try {
      final vectors = await model.generateEmbeddings(
        texts,
        taskType: TaskType.retrievalDocument,
      );
      if (vectors.length != texts.length) {
        return Result.error(
          EmbedderMismatchException(
            '${vectors.length} embeddings for ${texts.length} texts',
          ),
        );
      }
      for (final vector in vectors) {
        if (_checkVector(vector) case final String problem) {
          return Result.error(EmbedderMismatchException(problem));
        }
      }
      return Result.ok(vectors);
    } catch (e, st) {
      debugPrint('[EmbedderService] embedDocuments failed: $e\n$st');
      return Result.error(asException(e));
    }
  }

  String? _checkVector(List<double> vector) {
    if (vector.length != _config.dimension) {
      return 'EmbeddingGemma returned ${vector.length} values, expected '
          '${_config.dimension}';
    }
    if (!vector.any((v) => v != 0)) {
      return 'EmbeddingGemma returned an all-zero vector';
    }
    return null;
  }

  /// Closes the embedder, and any a load in flight delivers later. Safe to
  /// call more than once.
  Future<void> close() async {
    _closed = true;
    final model = _model;
    _model = null;
    if (model != null) await _closeQuietly(model);
  }

  static Future<void> _closeQuietly(EmbeddingModel model) async {
    try {
      await model.close();
    } catch (e, st) {
      debugPrint('[EmbedderService] close failed: $e\n$st');
    }
  }
}
