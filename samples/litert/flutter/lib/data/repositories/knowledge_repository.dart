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
import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:flutter_edge_ai/flutter_edge_ai.dart' show TaskType;
import 'package:flutter_edge_ai_rag/flutter_edge_ai_rag.dart'
    show RetrievalResult;

import '../../config/knowledge_config.dart';
import '../../domain/models/knowledge.dart';
import '../../domain/models/model_id.dart';
import '../../domain/models/model_state.dart';
import '../../domain/ports/knowledge_retriever.dart';
import '../../utils/markdown_chunker.dart';
import '../../utils/result.dart';
import '../services/knowledge/embedder_digests.dart';
import '../services/knowledge/embedder_service.dart';
import '../services/knowledge/kb_background.dart';
import '../services/knowledge/kb_documents.dart';
import '../services/knowledge/kb_index_key.dart';
import '../services/knowledge/kb_index_marker.dart';
import '../services/knowledge/kb_prebuilt_index.dart';
import '../services/knowledge/vector_store_service.dart';

/// The index database did not open even after it was deleted and created
/// again.
final class KnowledgeIndexOpenException implements Exception {
  const KnowledgeIndexOpenException(this.cause);

  final Exception cause;

  @override
  String toString() =>
      'The knowledge-base index did not open, even after a reset: $cause';
}

/// The knowledge base: indexes `assets/kb/*.md` once per content version and
/// answers one gated search per Demo 1 turn.
///
/// Its [status] follows the embedder's model state: waiting while setup
/// loads it, unavailable (with the reason) when it is missing or failed —
/// chat keeps working and says so — and, once it is ready, indexing with
/// progress, then ready. Indexing starts by itself when the embedder becomes
/// ready and never blocks setup.
///
/// Index lifecycle: the database is `<dir>/kb.db`, the marker
/// `<dir>/index.json` = `{hash, chunks, dim, origin, key}` ([KbIndexMarker]).
/// The [KbIndexKey] covers every document's path and bytes, the chunker
/// (version and budget), the document prefix, the embedder's id and the SHA-256
/// of its model and tokenizer files; `hash` is its digest. A matching marker
/// over a store with that many rows is reused. Anything else deletes the marker
/// and then:
///
/// 1. **Prebuilt** (`assets/kb_index/`, `tool/build_kb_index.sh`): when the
///    manifest's key equals this app's, its `kb.db` (checked against the
///    manifest's size and SHA-256) is copied over `<dir>/kb.db` and reopened;
///    nothing is embedded.
/// 2. **On the device**, when there is no prebuilt index, its key differs, or
///    it does not install: the store is cleared, every chunk embedded and
///    inserted with progress. Why the prebuilt index was not used is logged
///    and carried in [status] (`prebuiltSkipped`), never silent.
///
/// Either way the marker is written last (temp file + rename), so an
/// interrupted run is never mistaken for a finished one.
class KnowledgeRepository implements KnowledgeRetriever {
  KnowledgeRepository({
    required this._embedder,
    required this._store,
    required this._models,
    required this._documents,
    required this._prebuilt,
    required this._digests,
    this._indexDir = defaultKbIndexDir,
    this._chunker = const MarkdownChunker(),
    this._minSimilarity = kKbMinSimilarity,
    this._topK = kKbTopK,
    this._batchSize = kKbEmbedBatch,
    this._closeWait = const Duration(seconds: 5),
  }) {
    _models.addListener(_onModels);
    _onModels();
  }

  final EmbedderService _embedder;
  final VectorStoreService _store;
  final ValueListenable<Map<ModelId, ModelState>> _models;
  final KbDocumentSource _documents;
  final PrebuiltKbIndexSource _prebuilt;
  final EmbedderDigests _digests;
  final Future<Directory> Function() _indexDir;
  final MarkdownChunker _chunker;
  final double _minSimilarity;
  final int _topK;
  final int _batchSize;

  /// How long [close] waits for an indexing run or a retrieval in flight.
  final Duration _closeWait;

  final ValueNotifier<KnowledgeStatus> _status = ValueNotifier(
    const KnowledgeWaiting(),
  );
  Future<Result<KnowledgeReady>>? _indexing;
  bool _indexStarted = false;
  bool _closed = false;
  Future<void>? _closing;

  /// Retrievals whose store search is in flight; [close] waits for them
  /// ([_retrievalsDone] completes when the last one ends).
  int _retrievals = 0;
  Completer<void>? _retrievalsDone;

  /// The knowledge base's state, for the home tile and the overlay.
  ValueListenable<KnowledgeStatus> get status => _status;

  void _onModels() {
    if (_closed) return;
    switch (_models.value[ModelId.embeddingGemma] ?? const ModelPending()) {
      case ModelReady():
        if (_indexStarted) return;
        _indexStarted = true;
        // ensureIndexed never throws: failures land in [status] and the log.
        unawaited(ensureIndexed());
      case ModelUnavailable(:final reason):
        _set(KnowledgeUnavailable(reason));
      case ModelFailed(:final message):
        _set(KnowledgeUnavailable('EmbeddingGemma failed: $message'));
      case ModelPending() ||
          ModelInstalling() ||
          ModelLoading() ||
          ModelWarmingUp():
        if (!_indexStarted) _set(const KnowledgeWaiting());
    }
  }

  /// Makes the index match the bundled documents: reuses it when the marker
  /// matches, otherwise rebuilds it with progress in [status]. Concurrent
  /// calls share one run; after a failure the next call tries again.
  Future<Result<KnowledgeReady>> ensureIndexed() {
    final running = _indexing;
    if (running != null) return running;
    if (_closed) return Future.value(_closedError());
    final run = _ensureIndexed();
    _indexing = run;
    unawaited(
      run.then((result) {
        if (result is Error<KnowledgeReady> && identical(_indexing, run)) {
          _indexing = null;
        }
      }),
    );
    return run;
  }

  /// One run: the store opened, then the first strategy that applies
  /// (reuse, the prebuilt index, indexing on the device). Never throws:
  /// failures land in [status] and the log.
  Future<Result<KnowledgeReady>> _ensureIndexed() async {
    final watch = Stopwatch()..start();
    try {
      final files = _embedder.installedFiles;
      if (!_embedder.isLoaded || files == null) {
        return _fail(StateError('EmbeddingGemma is not loaded'));
      }
      final dir = await _indexDir();
      await dir.create(recursive: true);
      final marker = KbIndexMarker(dir);
      _stopIfClosed();
      if (await _openStore(dir, marker) case Error(:final error)) {
        _stopIfClosed();
        return _fail(error);
      }
      final documents = await _documents.load();
      final key = await _keyFor(documents, files);
      _stopIfClosed();

      if (await _reuse(marker, key, watch) case final reused?) return reused;
      // Not reusable because close ran meanwhile: the marker stays.
      _stopIfClosed();
      // The marker goes first: from here until it is rewritten, a crash
      // leaves no marker, so the next launch re-indexes.
      await marker.delete();

      switch (await _installPrebuilt(dir, marker, key)) {
        case _PrebuiltInstalled(:final chunks):
          return await _prebuiltReady(marker, key, chunks, watch);
        case _PrebuiltSkipped(:final reason):
          return await _indexOnDevice(
            marker,
            key,
            documents,
            watch,
            prebuiltSkipped: reason,
          );
      }
    } on _Closed {
      return _closedError();
    } catch (e, st) {
      debugPrint('[Knowledge] indexing failed: $e\n$st');
      return _fail(e);
    }
  }

  /// The one way a run stops for [close], called before each store call and
  /// marker write or delete of a run. A run close gave up on (closeWait)
  /// goes on until its next call here, so it neither uses the closed store
  /// nor forces a re-index; [_ensureIndexed] turns the throw into
  /// [_closedError]. Two places do not stop: [_openStore], whose reset
  /// (marker and database deleted, store reopened) only follows an open
  /// that failed while the repository was still open, and the prebuilt
  /// copy in [_installPrebuilt], which a run close overtook still finishes
  /// (the store is not reopened on it).
  void _stopIfClosed() {
    if (_closed) throw const _Closed();
  }

  /// Strategy 1: the index as it is, when the marker was written for [key]
  /// and the store holds that many rows of the embedder's dimension. Null
  /// otherwise, with the reason logged.
  Future<Result<KnowledgeReady>?> _reuse(
    KbIndexMarker marker,
    KbIndexKey key,
    Stopwatch watch,
  ) async {
    final KbIndexMarkerMatch marked;
    switch (await marker.read(key)) {
      case final KbIndexMarkerMatch match:
        marked = match;
      case KbIndexMarkerMissing():
        debugPrint('[Knowledge] no index marker: indexing');
        return null;
      case KbIndexMarkerUnreadable(:final error):
        debugPrint('[Knowledge] unreadable index marker ($error): re-indexing');
        return null;
      case KbIndexMarkerIncomplete():
        debugPrint(
          '[Knowledge] index marker without hash/chunks/dim: re-indexing',
        );
        return null;
      case KbIndexMarkerStale(:final builtFrom):
        debugPrint(
          '[Knowledge] the index was built from $builtFrom: re-indexing',
        );
        return null;
    }
    // Closed meanwhile: the run stops (the marker is kept).
    _stopIfClosed();
    final mismatch = switch (await _store.stats()) {
      Ok(:final value)
          when value.documentCount == marked.chunks &&
              value.vectorDimension == marked.dim &&
              marked.dim == _embedder.dimension =>
        null,
      Ok(:final value) =>
        'the store holds ${value.documentCount} rows of '
            '${value.vectorDimension} dims, the marker says ${marked.chunks} '
            'of ${marked.dim}: re-indexing',
      Error(:final error) => 'store stats failed ($error): re-indexing',
    };
    if (mismatch != null) {
      debugPrint('[Knowledge] $mismatch');
      return null;
    }
    final ready = KnowledgeReady(
      chunks: marked.chunks,
      reused: true,
      elapsed: watch.elapsed,
      origin: marked.origin,
      prebuiltSkipped: marked.prebuiltSkipped,
    );
    debugPrint(
      '[Knowledge] index reused: ${marked.chunks} chunks '
      '(${marked.origin.name}), hash ${key.digest.substring(0, 12)} '
      '(${watch.elapsedMilliseconds}ms)',
    );
    _set(ready);
    return Result.ok(ready);
  }

  /// Strategy 2, once [_installPrebuilt] put the prebuilt index in place:
  /// its marker, then ready.
  Future<Result<KnowledgeReady>> _prebuiltReady(
    KbIndexMarker marker,
    KbIndexKey key,
    int chunks,
    Stopwatch watch,
  ) async {
    _stopIfClosed();
    await marker.write(
      key: key,
      chunks: chunks,
      origin: KnowledgeOrigin.prebuilt,
    );
    final ready = KnowledgeReady(
      chunks: chunks,
      reused: false,
      elapsed: watch.elapsed,
      origin: KnowledgeOrigin.prebuilt,
    );
    debugPrint(
      '[Knowledge] prebuilt index installed: $chunks chunks in '
      '${watch.elapsedMilliseconds}ms',
    );
    _set(ready);
    return Result.ok(ready);
  }

  /// Strategy 3: the store cleared, every chunk embedded and inserted with
  /// progress, the row count checked, the marker written. [prebuiltSkipped]
  /// (why the prebuilt index was not used) is logged, shown while indexing
  /// and kept in the marker.
  Future<Result<KnowledgeReady>> _indexOnDevice(
    KbIndexMarker marker,
    KbIndexKey key,
    List<KbDocument> documents,
    Stopwatch watch, {
    required String prebuiltSkipped,
  }) async {
    _stopIfClosed();
    debugPrint(
      '[Knowledge] prebuilt index not used: $prebuiltSkipped. Indexing on the '
      'device',
    );
    if (await _store.clear() case Error(:final error)) return _fail(error);

    final chunks = await chunkInBackground(_chunker, documents);
    if (chunks.isEmpty) {
      return _fail(StateError('The documents gave no chunks'));
    }
    _set(
      KnowledgeIndexing(
        done: 0,
        total: chunks.length,
        prebuiltSkipped: prebuiltSkipped,
      ),
    );

    for (var start = 0; start < chunks.length; start += _batchSize) {
      _stopIfClosed();
      final batch = chunks.sublist(
        start,
        math.min(start + _batchSize, chunks.length),
      );
      final List<List<double>> vectors;
      switch (await _embedder.embedDocuments([
        for (final chunk in batch) chunk.content,
      ])) {
        case Ok(:final value):
          vectors = value;
        case Error(:final error):
          return _fail(error);
      }
      for (var i = 0; i < batch.length; i++) {
        _stopIfClosed();
        final added = await _store.add(
          id: batch[i].id,
          content: batch[i].content,
          embedding: vectors[i],
          metadata: batch[i].metadataJson,
        );
        if (added case Error(:final error)) return _fail(error);
      }
      _set(
        KnowledgeIndexing(
          done: start + batch.length,
          total: chunks.length,
          prebuiltSkipped: prebuiltSkipped,
        ),
      );
    }

    _stopIfClosed();
    switch (await _store.stats()) {
      case Ok(:final value) when value.documentCount != chunks.length:
        return _fail(
          StateError(
            'The store holds ${value.documentCount} rows after indexing '
            '${chunks.length} chunks',
          ),
        );
      case Ok():
        break;
      case Error(:final error):
        return _fail(error);
    }
    _stopIfClosed();
    await marker.write(
      key: key,
      chunks: chunks.length,
      origin: KnowledgeOrigin.device,
      prebuiltSkipped: prebuiltSkipped,
    );

    final ready = KnowledgeReady(
      chunks: chunks.length,
      reused: false,
      elapsed: watch.elapsed,
      prebuiltSkipped: prebuiltSkipped,
    );
    debugPrint(
      '[Knowledge] indexed ${chunks.length} chunks from '
      '${documents.length} documents in ${watch.elapsedMilliseconds}ms',
    );
    _set(ready);
    return Result.ok(ready);
  }

  /// This app's [KbIndexKey]: the documents, the chunker, the document
  /// prefix and the embedder's files (their SHA-256, see
  /// [FileEmbedderDigests]).
  Future<KbIndexKey> _keyFor(
    List<KbDocument> documents,
    ({String model, String tokenizer}) files,
  ) async {
    final digest = await _digests.of(
      modelPath: files.model,
      tokenizerPath: files.tokenizer,
    );
    return KbIndexKey(
      documents: kbDocumentsHash(documents),
      chunker: chunkerId(_chunker),
      documentPrefix: TaskType.retrievalDocument.prefix,
      modelId: _embedder.modelId ?? '',
      modelSha256: digest.model,
      tokenizerSha256: digest.tokenizer,
      dim: _embedder.dimension,
    );
  }

  /// Strategy 2: puts the prebuilt index in place of `<dir>/kb.db` when its
  /// manifest's key is [key]: the asset is checked against the manifest's
  /// size and SHA-256, written beside the database, renamed over it, and the
  /// store reopened on it (it closes its handle on the old file first), then
  /// its rows and dimension are checked. Anything short of that is skipped
  /// with the reason, and the store is left open (reset when the copy would
  /// not open) for indexing on the device.
  Future<_Prebuilt> _installPrebuilt(
    Directory dir,
    KbIndexMarker marker,
    KbIndexKey key,
  ) async {
    final PrebuiltKbManifest manifest;
    try {
      switch (await _prebuilt.manifest()) {
        case PrebuiltManifestMissing(:final reason):
          return _PrebuiltSkipped(reason);
        case PrebuiltManifestFound(manifest: final found):
          manifest = found;
      }
    } catch (e) {
      return _PrebuiltSkipped('its manifest is unreadable: $e');
    }
    final differences = manifest.key.differencesFrom(key);
    if (differences.isNotEmpty) {
      return _PrebuiltSkipped(
        'it was built from another ${differences.join(', ')}; rebuild it '
        'with tool/build_kb_index.sh',
      );
    }
    final path = '${dir.path}/kb.db';
    final temp = File('$path.prebuilt');
    try {
      final bytes = await _prebuilt.database();
      if (bytes.length != manifest.dbBytes) {
        return _PrebuiltSkipped(
          'its kb.db is ${bytes.length} bytes, the manifest says '
          '${manifest.dbBytes}',
        );
      }
      final hex = await sha256InBackground(bytes);
      if (hex != manifest.dbSha256) {
        return _PrebuiltSkipped(
          'its kb.db has SHA-256 ${hex.substring(0, 8)}, the manifest says '
          '${manifest.dbSha256.substring(0, 8)}',
        );
      }
      await temp.writeAsBytes(bytes, flush: true);
      for (final suffix in ['-wal', '-shm', '-journal']) {
        final side = File('$path$suffix');
        if (await side.exists()) await side.delete();
      }
      await temp.rename(path);
    } catch (e, st) {
      debugPrint('[Knowledge] copying the prebuilt index failed: $e\n$st');
      return _PrebuiltSkipped('copying it failed: $e');
    } finally {
      if (await temp.exists()) await temp.delete();
    }
    _stopIfClosed();
    final problem = switch (await _store.open(path)) {
      Error(:final error) => 'it did not open: $error',
      Ok() => switch (await _store.stats()) {
        Ok(:final value)
            when value.documentCount == manifest.chunks &&
                value.vectorDimension == key.dim =>
          null,
        Ok(:final value) =>
          'it holds ${value.documentCount} rows of '
              '${value.vectorDimension} dims, the manifest says '
              '${manifest.chunks} of ${key.dim}',
        Error(:final error) => 'reading its stats failed: $error',
      },
    };
    if (problem == null) return _PrebuiltInstalled(manifest.chunks);
    _stopIfClosed();
    // Back to a store that opens (deleted and created again if need be);
    // indexing on the device clears it.
    // (A second failure ends indexing: _ensureIndexed catches it.)
    if (await _openStore(dir, marker) case Error(:final error)) throw error;
    return _PrebuiltSkipped(problem);
  }

  /// Opens `<dir>/kb.db`. The database is only a cache of the bundled
  /// documents, so when it will not open (a corrupt file, or a schema an
  /// upgraded store rejects) it is deleted with its side files and the
  /// marker, and opened once more, empty; the caller then re-indexes. A
  /// second failure is returned, saying a reset was tried.
  Future<Result<void>> _openStore(Directory dir, KbIndexMarker marker) async {
    final path = '${dir.path}/kb.db';
    final first = await _store.open(path);
    if (first is! Error<void>) return first;
    // A store closed meanwhile refuses to open: that says nothing about the
    // file, which is kept (the caller decides whether the run stops).
    if (_closed) return first;
    debugPrint(
      '[Knowledge] opening the index failed (${first.error}): deleting it '
      'and re-indexing',
    );
    await marker.delete();
    for (final file in [
      File(path),
      for (final suffix in ['-wal', '-shm', '-journal']) File('$path$suffix'),
    ]) {
      if (await file.exists()) await file.delete();
    }
    return switch (await _store.open(path)) {
      Error(:final error) => Result.error(KnowledgeIndexOpenException(error)),
      final ok => ok,
    };
  }

  /// One gated search: the top-k hits, of which those at or above the gate
  /// become the prompt's excerpts.
  @override
  Future<Result<Retrieval>> retrieve(String question) async {
    if (_closed) {
      return Result.error(
        asException(StateError('KnowledgeRepository closed')),
      );
    }
    final unavailable = switch (_status.value) {
      KnowledgeReady() => null,
      KnowledgeWaiting() => 'the embedder is still loading',
      KnowledgeUnavailable(:final reason) => reason,
      KnowledgeIndexing(:final percent) => 'indexing $percent%',
      KnowledgeFailed(:final message) => 'indexing failed: $message',
    };
    if (unavailable != null) {
      return Result.ok(
        Retrieval(
          outcome: RetrievalOutcome.unavailable,
          gate: _minSimilarity,
          detail: unavailable,
        ),
      );
    }
    final watch = Stopwatch()..start();
    final Result<List<RetrievalResult>> searched;
    _retrievals++;
    try {
      searched = await _store.search(question, topK: _topK);
    } finally {
      if (--_retrievals == 0) {
        _retrievalsDone?.complete();
        _retrievalsDone = null;
      }
    }
    watch.stop();
    switch (searched) {
      case Error(:final error):
        return Result.error(error);
      case Ok(:final value):
        try {
          final candidates = [for (final hit in value) _passage(hit)]
            ..sort((a, b) => b.similarity.compareTo(a.similarity));
          final passages = [
            for (final passage in candidates)
              if (passage.similarity >= _minSimilarity) passage,
          ];
          final retrieval = Retrieval(
            outcome: passages.isEmpty
                ? RetrievalOutcome.belowGate
                : RetrievalOutcome.used,
            passages: passages,
            candidates: candidates,
            gate: _minSimilarity,
            latency: watch.elapsed,
          );
          debugPrint(
            '[Knowledge] retrieve ${watch.elapsedMilliseconds}ms '
            'top=${retrieval.topSimilarity?.toStringAsFixed(3) ?? '–'} '
            '${retrieval.outcome.name} excerpts=${passages.length}',
          );
          return Result.ok(retrieval);
        } on FormatException catch (e) {
          return Result.error(e);
        }
    }
  }

  /// A hit with the metadata the index stored for it. Anything else is a
  /// corrupt index, reported as a failed retrieval.
  static Passage _passage(RetrievalResult hit) {
    final raw = hit.metadata;
    if (raw == null) throw FormatException('${hit.id} has no metadata');
    final Object? json = jsonDecode(raw);
    if (json case <String, Object?>{
      'doc': final String doc,
      'title': final String title,
      'section': final String section,
      'source': final String? source,
    }) {
      return Passage(
        id: hit.id,
        doc: doc,
        title: title,
        section: section,
        content: hit.content,
        similarity: hit.similarity,
        source: source,
      );
    }
    throw FormatException('${hit.id} has unexpected metadata', raw);
  }

  Result<KnowledgeReady> _fail(Object error) {
    final exception = asException(error);
    debugPrint('[Knowledge] index failed: $exception');
    _set(KnowledgeFailed(exception.toString()));
    return Result.error(exception);
  }

  static Result<KnowledgeReady> _closedError() =>
      Result.error(asException(StateError('KnowledgeRepository closed')));

  void _set(KnowledgeStatus status) {
    if (_closed) return;
    _status.value = status;
  }

  /// Stops following the model states and waits for the work in flight
  /// before the store closes, so no add, stats or search races the native
  /// index's dispose: an indexing run stops at its next check without
  /// writing the marker, a retrieval finishes its search. Each wait is
  /// bounded by `closeWait` (logged when it runs out). The status is
  /// disposed last. Close before the model repository it listens to (the
  /// embedder an indexing run uses). Safe to call more than once.
  Future<void> close() => _closing ??= _close();

  Future<void> _close() async {
    _closed = true;
    _models.removeListener(_onModels);
    if (_indexing case final indexing?) {
      await indexing.timeout(
        _closeWait,
        onTimeout: () {
          debugPrint(
            '[Knowledge] close: indexing did not stop within '
            '${_closeWait.inMilliseconds}ms; closing the store anyway',
          );
          return _closedError();
        },
      );
    }
    if (_retrievals > 0) {
      await (_retrievalsDone ??= Completer<void>()).future.timeout(
        _closeWait,
        onTimeout: () => debugPrint(
          '[Knowledge] close: $_retrievals retrieval(s) still searching '
          'after ${_closeWait.inMilliseconds}ms; closing the store anyway',
        ),
      );
    }
    await _store.close();
    _status.dispose();
  }
}

/// Thrown by `KnowledgeRepository._stopIfClosed`: the run stops where it is.
final class const _Closed() implements Exception;

/// What [KnowledgeRepository] did with the prebuilt index.
sealed class const _Prebuilt();

final class const _PrebuiltInstalled(final int chunks) extends _Prebuilt;

final class const _PrebuiltSkipped(final String reason) extends _Prebuilt;
