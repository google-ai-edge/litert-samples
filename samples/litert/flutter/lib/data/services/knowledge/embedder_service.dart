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
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_edge_ai/flutter_edge_ai.dart'
    show
        EmbeddingInput,
        EmbeddingInputTooLongException,
        EmbeddingModality,
        EmbeddingModel,
        FlutterEdgeAi,
        PreferredBackend,
        TaskType;

import '../../../config/knowledge_config.dart';
import '../../../config/model_catalog.dart';
import '../../../domain/hardware/diagnostics_report.dart' show formatBytes;
import '../../../domain/models/knowledge_query.dart';
import '../../../domain/models/model_id.dart';
import '../../../domain/ports/visual_memory_embedder.dart';
import '../../../utils/result.dart';
import '../../../utils/wav.dart';

/// Where EmbeddingGemma comes from.
sealed class const EmbedderSource();

/// The file built into the app (an absolute path): one self-contained
/// `.litertlm` that carries its own tokenizer, in place in the app bundle, or
/// Android's verified extracted copy. Registered in place, never copied.
final class const EmbedderFromFiles({required final String modelPath})
    extends EmbedderSource;

/// Registers (local) or downloads (network) the model and makes it the
/// active embedder; returns the model id. [source]'s path is absolute and
/// the file exists.
typedef EmbedderInstaller = Future<String> Function(
  EmbedderConfig config,
  EmbedderSource source,
  void Function(int percent) onProgress,
);

/// Builds the active embedder.
typedef EmbedderLoader = Future<EmbeddingModel> Function(EmbedderConfig config);

/// `installEmbedder` from a file (`modelFromFile` registers the path; the
/// load reads the active spec's own path even when the file name was
/// registered from elsewhere before). A `.litertlm` takes no tokenizer: the
/// builder refuses one with an `ArgumentError`.
Future<String> installEmbedderFrom(
  EmbedderConfig config,
  EmbedderSource source,
  void Function(int percent) onProgress,
) async {
  final builder = FlutterEdgeAi.installEmbedder();
  switch (source) {
    case EmbedderFromFiles(:final modelPath):
      builder.modelFromFile(modelPath);
  }
  final installation = await builder.install();
  return installation.modelId;
}

/// `FlutterEdgeAi.getActiveEmbedder` with the [EmbedderConfig] backend and
/// media (a worker isolate; the model reports the backend it runs on and
/// the modalities it loaded). Each declared medium loads its encoder.
Future<EmbeddingModel> getActiveEmbedderFor(EmbedderConfig config) =>
    FlutterEdgeAi.getActiveEmbedder(
      preferredBackend: config.backend,
      supportAudio: config.media.contains(EmbeddingModality.audio),
      supportImage: config.media.contains(EmbeddingModality.image),
    );

/// The file the embedder needs is not there (a built-in file missing from
/// this build, or an extracted copy that vanished): shown as "unavailable"
/// rather than a failure to retry.
final class EmbedderFilesMissingException implements Exception {
  const EmbedderFilesMissingException(this.paths);

  final List<String> paths;

  @override
  String toString() =>
      'EmbeddingGemma file not found or not readable: ${paths.join(', ')} '
      '(a build fetches it with tool/fetch_models.sh first)';
}

/// The loaded embedder is not the one the catalog describes (backend,
/// dimension, media, input limit, or zero vectors); it is closed, never
/// used.
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

/// Loading the image encoder failed ([EmbedderService.enableImages]): the
/// visual memory is off with this reason. [restored]: the text and audio
/// embedder was loaded again (or never left), so the knowledge base goes on;
/// otherwise no embedder is loaded.
final class EmbedderImagesException implements Exception {
  const EmbedderImagesException(this.cause, {required this.restored});

  final Object cause;
  final bool restored;

  @override
  String toString() =>
      "EmbeddingGemma's image encoder did not load: $cause"
      '${restored ? '' : ' (and the text and audio embedder could not be loaded again)'}';
}

/// What [EmbedderService.enableImages] reports: the reload's time and the
/// process's resident memory around it (`ProcessInfo.currentRss`), or that
/// the image encoder was loaded already.
final class const EmbedderImagesLoad({
  required final Duration loadTime,
  final int? rssBeforeBytes,
  final int? rssAfterBytes,
  final bool alreadyLoaded = false,
});

/// What [EmbedderService.load] reports.
final class const EmbedderInfo({
  required final String modelId,
  required final PreferredBackend backend,
  required final int dimension,
  required final Duration loadTime,

  /// What the model accepts (text, plus the media declared at load).
  final Set<EmbeddingModality> modalities = const {EmbeddingModality.text},

  /// The input limit the model reports, in tokens; null when it declares
  /// none.
  final int? maxInputTokens,

  /// The process's resident memory before and after the load
  /// (`ProcessInfo.currentRss`).
  final int? rssBeforeBytes,
  final int? rssAfterBytes,
});

/// The app's audio warm-up clip ([kKbAudioWarmUpAsset], a spoken "What time
/// is it?") from the asset bundle: what [EmbedderService.warmUp] embeds after
/// the text question, in the app and in the headless self-test.
Future<Uint8List> loadAudioWarmUpClip() async {
  final data = await rootBundle.load(kKbAudioWarmUpAsset);
  return data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
}

/// The [EmbedderService] the app builds (setup's, and the headless
/// self-test's): warmed up with the text question and [loadAudioWarmUpClip].
/// [install] and [load] are for tests.
EmbedderService appEmbedderService({
  EmbedderInstaller install = installEmbedderFrom,
  EmbedderLoader load = getActiveEmbedderFor,
}) => EmbedderService(
  install: install,
  load: load,
  warmUpAudio: loadAudioWarmUpClip,
);

/// Owns EmbeddingGemma: install, load (checked against the catalog), warm
/// up, embed documents and questions, close. `ModelRepository` drives its
/// lifecycle (with it, the only caller of `getActiveEmbedder`);
/// `KnowledgeRepository` borrows [embedDocuments] for indexing, and the
/// vector store [embedQuery] for each search. The visual memory borrows
/// [embedImage] for its frames and [embedText] for its questions. The
/// package's input type stays in here: callers pass a [KnowledgeQuery], PNG
/// bytes or a phrase.
///
/// Setup loads text and audio ([kEmbedderConfig]); [enableImages] adds the
/// image encoder when the visual memory first needs it: flutter_edge_ai
/// closes the loaded embedder and builds one with the wider media (about
/// 2 s). Meanwhile every embedding call waits for the swap and then runs on
/// the new model (none fails for it); the calls already running finish on
/// the old one first. The images stay until [close] (media only widen).
class EmbedderService implements VisualMemoryEmbedder {
  /// [warmUpAudio]: the spoken clip [warmUp] embeds when the catalog
  /// declares audio (the app: [kKbAudioWarmUpAsset]); null skips that part.
  EmbedderService({
    this._config = kEmbedderConfig,
    this._install = installEmbedderFrom,
    this._load = getActiveEmbedderFor,
    this._warmUpAudio,
  });

  final EmbedderConfig _config;
  final EmbedderInstaller _install;
  final EmbedderLoader _load;
  final Future<Uint8List> Function()? _warmUpAudio;

  EmbeddingModel? _model;
  String? _modelId;
  String? _installedModelPath;
  bool _closed = false;

  /// An image has been embedded since the load: the first one's memory cost
  /// is logged with the RSS around it.
  bool _imageEmbedded = false;

  /// [enableImages]'s reload, while it runs: every embedding call waits for
  /// it first.
  Future<Result<EmbedderImagesLoad>>? _widening;

  /// Embedding calls running on the current model; [enableImages] waits
  /// until none does before the model is replaced.
  int _running = 0;
  Completer<void>? _drained;

  bool get isLoaded => _model != null;

  /// The installed model's id (its file name without the extension); null
  /// before [install] succeeds. Part of the knowledge-base index hash.
  String? get modelId => _modelId;

  /// The absolute path of the model file [install] registered; null before
  /// it succeeds. The knowledge base keys its index on its SHA-256.
  String? get installedModelPath => _installedModelPath;

  /// The vector size every embedding has; checked at [load].
  int get dimension => _config.dimension;

  /// What the loaded model accepts (text, plus the declared media); empty
  /// when none is loaded.
  Set<EmbeddingModality> get modalities => _model?.modalities ?? const {};

  /// Registers the file and makes it the active embedder. A missing file
  /// fails with [EmbedderFilesMissingException] before anything is
  /// registered.
  Future<Result<String>> install(
    EmbedderSource source, {
    required void Function(int percent) onProgress,
  }) async {
    try {
      final modelPath = switch (source) {
        EmbedderFromFiles(:final modelPath) => modelPath,
      };
      if (!await File(modelPath).exists()) {
        return Result.error(EmbedderFilesMissingException([modelPath]));
      }
      final id = await _install(_config, source, onProgress);
      _modelId = id;
      _installedModelPath = modelPath;
      return Result.ok(id);
    } catch (e, st) {
      debugPrint('[EmbedderService] install failed: $e\n$st');
      return Result.error(asException(e));
    }
  }

  /// Loads the active embedder with the catalog's media. Fails, and closes
  /// what was loaded, unless it reports the catalog's backend, dimension,
  /// input limit and every declared medium. A load the plugin refuses fails
  /// with its `EmbeddingLoadException` as it is: among them a load that wrote
  /// no weight cache at all, which the app's registration makes a failure
  /// (`kStrictEmbeddingBackend`, `requireWeightCache`).
  Future<Result<EmbedderInfo>> load() async {
    final rssBefore = _rss();
    final watch = Stopwatch()..start();
    try {
      final model = await _load(_config);
      watch.stop();
      if (_closed) {
        await _closeQuietly(model);
        return const Result.error(EmbedderServiceClosedException());
      }
      final backend = model.activeBackend;
      final dimension = model.dimension;
      final modalities = model.modalities;
      final maxInputTokens = model.maxInputTokens;
      final mismatch = _mismatch(model, _config);
      if (mismatch != null) {
        await _closeQuietly(model);
        return Result.error(EmbedderMismatchException(mismatch));
      }
      if (_closed) {
        await _closeQuietly(model);
        return const Result.error(EmbedderServiceClosedException());
      }
      _model = model;
      _imageEmbedded = false;
      final rssAfter = _rss();
      final info = EmbedderInfo(
        modelId: _modelId ?? ModelId.embeddingGemma.name,
        backend: backend!,
        dimension: dimension,
        loadTime: watch.elapsed,
        modalities: modalities,
        maxInputTokens: maxInputTokens,
        rssBeforeBytes: rssBefore,
        rssAfterBytes: rssAfter,
      );
      debugPrint(
        '[EmbedderService] loaded ${info.modelId} backend=${backend.name} '
        'dim=$dimension media=${_names(modalities)} '
        'maxInputTokens=$maxInputTokens load=${watch.elapsedMilliseconds}ms '
        'rss ${_mb(rssBefore)} -> ${_mb(rssAfter)}',
      );
      return Result.ok(info);
    } catch (e, st) {
      debugPrint('[EmbedderService] load failed: $e\n$st');
      return Result.error(asException(e));
    }
  }

  static String _names(Set<EmbeddingModality> media) => media.isEmpty
      ? 'no media'
      : (media.map((m) => m.name).toList()..sort()).join('+');

  static int? _rss() {
    try {
      return ProcessInfo.currentRss;
    } catch (_) {
      return null;
    }
  }

  static String _mb(int? bytes) => bytes == null ? 'n/a' : formatBytes(bytes);

  /// One query embedding, so the first question does not pay for lazy
  /// setup; fails on a wrong-sized or all-zero vector. With audio declared
  /// and a clip given, one spoken query too (the audio encoder's first use,
  /// otherwise paid by the first spoken question). A failed warm-up — either
  /// part, or a clip that cannot be read — closes and unloads the model, like
  /// a mismatch at [load]: the row shows the failure and the model's memory
  /// is released.
  Future<Result<Duration>> warmUp() async {
    final model = _model;
    if (_closed) return const Result.error(EmbedderServiceClosedException());
    if (model == null) {
      return Result.error(asException(StateError('warmUp() before load()')));
    }
    final watch = Stopwatch()..start();
    try {
      final vector = await model.embed(
        EmbeddingInput.text('What is LiteRT?'),
        taskType: TaskType.retrievalQuery,
      );
      if (_checkVector(vector) case final String problem) {
        await _unload(model);
        return Result.error(EmbedderMismatchException(problem));
      }
      final textMs = watch.elapsedMilliseconds;
      final clip = _warmUpAudio;
      var audio = 'no audio';
      if (clip != null && _config.media.contains(EmbeddingModality.audio)) {
        final wav = await clip();
        final started = watch.elapsed;
        final spoken = await model.embed(
          EmbeddingInput.audio(wav),
          taskType: TaskType.none,
        );
        if (_checkVector(spoken) case final String problem) {
          await _unload(model);
          return Result.error(
            EmbedderMismatchException('audio warm-up: $problem'),
          );
        }
        audio =
            'audio ${(watch.elapsed - started).inMilliseconds}ms '
            '(${wavDuration(wav).inMilliseconds}ms clip)';
      }
      debugPrint(
        '[EmbedderService] warm-up ${watch.elapsedMilliseconds}ms '
        '(text ${textMs}ms, $audio)',
      );
      return Result.ok(watch.elapsed);
    } catch (e, st) {
      debugPrint('[EmbedderService] warm-up failed: $e\n$st');
      await _unload(model);
      return Result.error(asException(e));
    }
  }

  /// A query embedding, checked like the warm-up's: what the knowledge base
  /// searches with. A text question gets the `retrievalQuery` prefix; a
  /// spoken one is embedded as audio, with no prefix (a task type reaches
  /// text only). A question over its cap ([kKbMaxTextQueryUtf8Bytes],
  /// [kKbMaxAudioQuery]) fails with [KnowledgeQueryTooLongException] before
  /// anything is embedded, as does one the embedder refuses as too long
  /// (`EmbeddingInputTooLongException`); nothing is cut to fit. A spoken
  /// question that is not a 16 kHz mono PCM16 WAV fails with a
  /// [FormatException].
  Future<Result<List<double>>> embedQuery(KnowledgeQuery query) =>
      _use('embedQuery', (model) => _embedQuery(model, query));

  Future<Result<List<double>>> _embedQuery(
    EmbeddingModel model,
    KnowledgeQuery query,
  ) async {
    final EmbeddingInput input;
    final TaskType taskType;
    try {
      (input, taskType) = queryInput(query);
    } on KnowledgeQueryTooLongException catch (e) {
      debugPrint('[EmbedderService] $e');
      return Result.error(e);
    } on FormatException catch (e) {
      debugPrint('[EmbedderService] $query: $e');
      return Result.error(e);
    }
    try {
      final vector = await model.embed(input, taskType: taskType);
      if (_checkVector(vector) case final String problem) {
        return Result.error(EmbedderMismatchException(problem));
      }
      return Result.ok(vector);
    } on EmbeddingInputTooLongException catch (e) {
      debugPrint('[EmbedderService] embedQuery refused as too long: $e');
      return Result.error(
        KnowledgeQueryTooLongException(
          kind: query.kind,
          length: e.actualTokens == null
              ? 'more than ${e.maxInputTokens} tokens'
              : '${e.actualTokens} tokens',
          limit: '${e.maxInputTokens} tokens',
        ),
      );
    } catch (e, st) {
      debugPrint('[EmbedderService] embedQuery failed: $e\n$st');
      return Result.error(asException(e));
    }
  }

  /// The embedder's input and task type for [query], or a throw: a
  /// [KnowledgeQueryTooLongException] over the caps, a [FormatException]
  /// for audio that is not a 16 kHz mono PCM16 WAV.
  @visibleForTesting
  static (EmbeddingInput, TaskType) queryInput(KnowledgeQuery query) {
    switch (query) {
      case KnowledgeTextQuery(:final text):
        final bytes = utf8.encode(text).length;
        if (bytes > kKbMaxTextQueryUtf8Bytes) {
          throw KnowledgeQueryTooLongException(
            kind: KnowledgeQueryKind.text,
            length: '$bytes UTF-8 bytes',
            limit: '$kKbMaxTextQueryUtf8Bytes UTF-8 bytes',
          );
        }
        return (EmbeddingInput.text(text), TaskType.retrievalQuery);
      case KnowledgeAudioQuery(:final wav):
        final duration = wavDuration(wav);
        if (duration > kKbMaxAudioQuery) {
          throw KnowledgeQueryTooLongException(
            kind: KnowledgeQueryKind.audio,
            length: '${(duration.inMilliseconds / 1000).toStringAsFixed(1)} s',
            limit: '${kKbMaxAudioQuery.inSeconds} s',
          );
        }
        if (duration == Duration.zero) {
          throw const FormatException('The spoken question has no samples');
        }
        // TaskType.none: the prefix would reach a text part only, and there
        // is none.
        return (EmbeddingInput.audio(wav), TaskType.none);
    }
  }

  /// Null once [enableImages] loaded the image encoder.
  @override
  String? get imageUnavailableReason {
    if (_closed) return 'the embedder was closed';
    final model = _model;
    if (model == null) return 'EmbeddingGemma is not loaded';
    if (!model.modalities.contains(EmbeddingModality.image)) {
      return 'EmbeddingGemma was loaded without images';
    }
    return null;
  }

  /// A memory frame's vector: the encoded image (PNG or JPEG) as it is,
  /// with the `retrievalDocument` task type (a prefix reaches text only, so
  /// it changes nothing for an image). Checked like the warm-up's. The
  /// first image since the load logs the process's RSS before and after.
  /// Measured (`docs/design/eg2-multimodal.md`): with images declared at the
  /// load (M4 Pro Mac, a test app with the embedder alone, 2026-10-13) the
  /// load itself added 407 MB (648 → 1055 MB) and the first image 75–175 MB
  /// more; as the app does it (the [enableImages] widening reload, the same
  /// Mac, 2026-10-14) the reload added 327 MB and the first image 227 MB;
  /// on phones the first image added 280 MB (Galaxy S24) and 311 MB (S25
  /// Ultra).
  @override
  Future<Result<List<double>>> embedImage(Uint8List encoded) =>
      _use('embedImage', (model) => _embedImage(model, encoded));

  Future<Result<List<double>>> _embedImage(
    EmbeddingModel model,
    Uint8List encoded,
  ) async {
    if (!model.modalities.contains(EmbeddingModality.image)) {
      return const Result.error(
        EmbedderMismatchException('EmbeddingGemma was loaded without images'),
      );
    }
    final first = !_imageEmbedded;
    final rssBefore = first ? _rss() : null;
    final watch = Stopwatch()..start();
    try {
      final vector = await model.embed(
        EmbeddingInput.image(encoded),
        taskType: TaskType.retrievalDocument,
      );
      if (first) {
        _imageEmbedded = true;
        debugPrint(
          '[EmbedderService] first image embedded in '
          '${watch.elapsedMilliseconds}ms (${encoded.length} B): rss '
          '${_mb(rssBefore)} -> ${_mb(_rss())}',
        );
      }
      if (_checkVector(vector) case final String problem) {
        return Result.error(EmbedderMismatchException(problem));
      }
      return Result.ok(vector);
    } catch (e, st) {
      debugPrint('[EmbedderService] embedImage failed: $e\n$st');
      return Result.error(asException(e));
    }
  }

  /// A memory question's vector: [text] raw ([TaskType.none]; a text query
  /// over images works best without a prefix). A phrase over the engine's
  /// input limit fails as the engine refuses it; moonshine's 5 s
  /// transcripts are far below it.
  @override
  Future<Result<List<double>>> embedText(String text) =>
      _use('embedText', (model) => _embedText(model, text));

  Future<Result<List<double>>> _embedText(
    EmbeddingModel model,
    String text,
  ) async {
    try {
      final vector = await model.embed(
        EmbeddingInput.text(text),
        taskType: TaskType.none,
      );
      if (_checkVector(vector) case final String problem) {
        return Result.error(EmbedderMismatchException(problem));
      }
      return Result.ok(vector);
    } catch (e, st) {
      debugPrint('[EmbedderService] embedText failed: $e\n$st');
      return Result.error(asException(e));
    }
  }

  /// Runs [body] on the loaded model, after an [enableImages] reload in
  /// flight (so a knowledge-base question waits ~2 s instead of failing on a
  /// closing model); counted, so the next reload waits for it to end.
  Future<Result<T>> _use<T>(
    String what,
    Future<Result<T>> Function(EmbeddingModel model) body,
  ) async {
    if (_widening case final widening?) await widening;
    if (_closed) return const Result.error(EmbedderServiceClosedException());
    final model = _model;
    if (model == null) {
      return Result.error(asException(StateError('$what() before load()')));
    }
    _running++;
    try {
      return await body(model);
    } finally {
      if (--_running == 0) {
        final drained = _drained;
        _drained = null;
        drained?.complete();
      }
    }
  }

  /// Adds the image encoder to the loaded embedder (the visual memory's
  /// first start): `getActiveEmbedder` with [kEmbedderMemoryMedia], which
  /// closes the loaded model and builds one with the wider media. Calls
  /// already running finish first; calls made meanwhile wait and run on the
  /// new model. Checked like [load] plus one text embedding. Ok at once when
  /// images are loaded already; concurrent calls share one reload.
  ///
  /// A failure is an [EmbedderImagesException]: the text and audio embedder
  /// is loaded again if the failed reload took it down
  /// ([EmbedderImagesException.restored]), so the knowledge base goes on.
  /// Never narrowed back: the images stay until [close].
  Future<Result<EmbedderImagesLoad>> enableImages() {
    if (_closed) {
      return Future.value(const Result.error(EmbedderServiceClosedException()));
    }
    final model = _model;
    if (model == null) {
      return Future.value(
        Result.error(asException(StateError('enableImages() before load()'))),
      );
    }
    if (model.modalities.contains(EmbeddingModality.image)) {
      return Future.value(
        const Result.ok(
          EmbedderImagesLoad(loadTime: Duration.zero, alreadyLoaded: true),
        ),
      );
    }
    return _widening ??= _widen(model).whenComplete(() => _widening = null);
  }

  Future<Result<EmbedderImagesLoad>> _widen(EmbeddingModel old) async {
    // The calls running on the old model end before it is closed.
    if (_running > 0) await (_drained ??= Completer<void>()).future;
    if (_closed) return const Result.error(EmbedderServiceClosedException());
    final widened = _config.withMedia({
      ..._config.media,
      ...kEmbedderMemoryMedia,
    });
    final rssBefore = _rss();
    final watch = Stopwatch()..start();
    Object failure;
    try {
      final model = await _load(widened);
      watch.stop();
      if (_closed) {
        await _closeQuietly(model);
        return const Result.error(EmbedderServiceClosedException());
      }
      final problem = _mismatch(model, widened) ?? await _probe(model);
      if (problem == null) {
        _model = model;
        _imageEmbedded = false;
        final rssAfter = _rss();
        debugPrint(
          '[EmbedderService] image encoder loaded: media '
          '${_names(model.modalities)} reload ${watch.elapsedMilliseconds}ms '
          'rss ${_mb(rssBefore)} -> ${_mb(rssAfter)}',
        );
        return Result.ok(
          EmbedderImagesLoad(
            loadTime: watch.elapsed,
            rssBeforeBytes: rssBefore,
            rssAfterBytes: rssAfter,
          ),
        );
      }
      failure = EmbedderMismatchException(problem);
      await _closeQuietly(model);
    } catch (e, st) {
      debugPrint('[EmbedderService] image encoder load failed: $e\n$st');
      failure = e;
    }
    debugPrint('[EmbedderService] image encoder: $failure');
    return Result.error(
      EmbedderImagesException(failure, restored: await _restore(old)),
    );
  }

  /// After a failed [enableImages]: keeps [old] when the failed reload left
  /// it open, otherwise loads the text and audio embedder again (the
  /// knowledge base's). Whether one is loaded afterwards.
  Future<bool> _restore(EmbeddingModel old) async {
    if (_closed) return false;
    if (!old.isClosed) {
      _model = old;
      return true;
    }
    if (identical(_model, old)) _model = null;
    try {
      final model = await _load(_config);
      if (_closed) {
        await _closeQuietly(model);
        return false;
      }
      final problem = _mismatch(model, _config) ?? await _probe(model);
      if (problem != null) {
        debugPrint('[EmbedderService] reload without images: $problem');
        await _closeQuietly(model);
        return false;
      }
      _model = model;
      debugPrint(
        '[EmbedderService] reloaded without images: media '
        '${_names(model.modalities)}',
      );
      return true;
    } catch (e, st) {
      debugPrint('[EmbedderService] reload without images failed: $e\n$st');
      return false;
    }
  }

  /// Why [model] is not the one [config] describes; null when it is.
  String? _mismatch(EmbeddingModel model, EmbedderConfig config) {
    final backend = model.activeBackend;
    final dimension = model.dimension;
    final modalities = model.modalities;
    final maxInputTokens = model.maxInputTokens;
    return switch (()) {
      _ when backend != config.backend =>
        'EmbeddingGemma requested ${config.backend.name} but reports '
            '${backend?.name ?? 'an unknown backend'}',
      _ when dimension != config.dimension =>
        'EmbeddingGemma reports $dimension dimensions, the catalog '
            'expects ${config.dimension}',
      _ when !modalities.containsAll(config.media) =>
        'EmbeddingGemma was asked for '
            '${_names(config.media)} but accepts ${_names(modalities)}',
      _ when maxInputTokens != config.maxInputTokens =>
        'EmbeddingGemma takes ${maxInputTokens ?? 'unlimited'} input '
            'tokens, the catalog expects ${config.maxInputTokens} (the '
            'question caps are derived from it)',
      _ => null,
    };
  }

  /// One text query on [model], checked like the warm-up's; the problem, or
  /// null.
  Future<String?> _probe(EmbeddingModel model) async {
    try {
      final vector = await model.embed(
        EmbeddingInput.text('What is LiteRT?'),
        taskType: TaskType.retrievalQuery,
      );
      return _checkVector(vector);
    } catch (e) {
      return 'a text embedding failed: $e';
    }
  }

  /// Drops [model] if it is still the loaded one, then closes it.
  Future<void> _unload(EmbeddingModel model) async {
    if (identical(_model, model)) _model = null;
    await _closeQuietly(model);
  }

  /// Document embeddings (the `retrievalDocument` prefix), one per text, in
  /// order. Each is checked like the warm-up's.
  Future<Result<List<List<double>>>> embedDocuments(List<String> texts) =>
      _use('embedDocuments', (model) => _embedDocuments(model, texts));

  Future<Result<List<List<double>>>> _embedDocuments(
    EmbeddingModel model,
    List<String> texts,
  ) async {
    try {
      final vectors = await model.embedAll([
        for (final text in texts) EmbeddingInput.text(text),
      ], taskType: TaskType.retrievalDocument);
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
