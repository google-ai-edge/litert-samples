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

import 'package:flutter/foundation.dart';
import 'package:flutter_edge_ai/flutter_edge_ai.dart' hide ModelSpec;

import '../../../config/model_catalog.dart';
import '../../../domain/models/model_id.dart';
import '../../../utils/result.dart';
import '../../../utils/serial_queue.dart';

/// Where a recognizer's two files come from.
sealed class const SttSource();

/// Verified model-store files (the app).
final class const SttFromFiles({
  required final String modelPath,
  required final String tokenizerPath,
}) extends SttSource;

/// Installs the STT model and its tokenizer from [SttSource] and makes it
/// the active one; returns the model id. Progress is 0–100 over both files.
typedef SttInstaller = Future<String> Function(
  SttConfig config,
  SttSource source,
  void Function(int percent) onProgress,
);

/// Loads the active STT model with [SttConfig]'s backend and language.
typedef RecognizerLoader = Future<SpeechRecognizer> Function(SttConfig config);

/// `installStt` with the files of [source]. Must run on every launch: it
/// sets the active spec, and flutter_edge_ai 2.1.0 restores a saved STT
/// spec only from its own model directory (`_restoreActiveSttModel` in
/// `mobile_model_manager.dart`), never files registered in place like these.
/// Files register in place (measured: 0 ms).
Future<String> installStt(
  SttConfig config,
  SttSource source,
  void Function(int percent) onProgress,
) async {
  switch (source) {
    case SttFromFiles(:final modelPath, :final tokenizerPath):
      final installation = await FlutterEdgeAi.installStt()
          .modelFromFile(modelPath)
          .tokenizerFromFile(tokenizerPath)
          .ofType(config.type)
          .withModelProgress(onProgress)
          .install();
      return installation.modelId;
  }
}

/// `FlutterEdgeAi.getActiveStt` with exactly the [SttConfig] arguments
/// (moonshine gets no language: any value throws).
Future<SpeechRecognizer> getActiveSttFor(SttConfig config) =>
    FlutterEdgeAi.getActiveStt(
      preferredBackend: config.backend,
      language: config.language,
    );

/// The service was closed while an operation was in flight.
final class SpeechServiceClosedException implements Exception {
  const SpeechServiceClosedException(this.what);

  final String what;

  @override
  String toString() => 'The $what service was closed';
}

/// A turn needs a model that is not loaded (setup did not finish, or the
/// last recognizer switch failed).
final class SpeechNotReadyException implements Exception {
  const SpeechNotReadyException(this.what);

  final String what;

  @override
  String toString() => 'The $what is not loaded';
}

/// The recognizer that is loaded now.
final class const ActiveStt({
  required final ModelId id,

  /// The installed model's id (`moonshine_tiny_5s_f32`).
  required final String modelId,
  required final Duration loadTime,

  /// Null when this load skipped the warm-up (a switch, [kSttWarmUpOnSwitch]).
  final Duration? warmUpTime,

  /// For a switch on demo entry: request → ready, including the wait for a
  /// transcription still running on the previous model.
  final Duration? switchTime,
});

/// Owns the speech recognizer: install, load, warm up, switch, close. The
/// STT model is a process-wide singleton in flutter_edge_ai 2.1.0
/// (`createSttModel` in `lib/desktop/flutter_edge_ai_desktop.dart` and the
/// mobile shell): loading one closes the other.
/// So this service holds one *active* recognizer for several configs
/// ([kSttConfigs]): Whisper for Demo 1, moonshine for Demo 3.
///
/// Every operation that touches the singleton (install, which sets the
/// active spec; load; warm-up; close) runs one at a time. A switch waits
/// (bounded) for a transcription still running on the old model before it
/// closes it, and [transcribe] waits for a pending switch, so a question
/// asked right after entering a demo is transcribed by that demo's model.
///
/// Called only by `ModelRepository` (with it, the only caller of
/// `getActiveStt`); the speech repository transcribes through it per turn.
class SttService {
  SttService({
    this._configs = kSttConfigs,
    this._install = installStt,
    this._load = getActiveSttFor,
    this._drainTimeout = const Duration(seconds: 3),
  });

  final Map<ModelId, SttConfig> _configs;
  final SttInstaller _install;
  final RecognizerLoader _load;
  final Duration _drainTimeout;

  final ValueNotifier<ActiveStt?> _active = ValueNotifier(null);
  final ValueNotifier<String?> _switchError = ValueNotifier(null);
  SpeechRecognizer? _recognizer;
  final Map<ModelId, String> _modelIds = {};

  /// Each recognizer's files, from its [install]: a switch re-installs from
  /// them to restore flutter_edge_ai's active spec.
  final Map<ModelId, SttSource> _sources = {};

  /// The model whose spec flutter_edge_ai holds as active (the last
  /// install).
  ModelId? _specFor;

  /// The model the newest [activate] asked for.
  ModelId? _requested;

  /// Installs, loads, switches and unloads, one at a time. After the first
  /// none starts inside the call that queues it, so an [activate] made right
  /// after another supersedes it. Made without a future: a completed future
  /// schedules its listeners in the zone it was made in (a widget test's fake
  /// zone would then hold up every later operation).
  final SerialQueue _ops = SerialQueue.firstAtOnce(
    onError: (e, st) => debugPrint('[SttService] operation failed: $e\n$st'),
  );

  /// Loads queued or running: [transcribe] waits for them.
  int _loading = 0;
  int _transcribing = 0;
  Completer<void>? _idle;
  bool _closed = false;

  /// The loaded recognizer; null before a load and after a failed one.
  ValueListenable<ActiveStt?> get active => _active;

  /// Why the last load or switch failed; null after a successful one.
  ValueListenable<String?> get switchError => _switchError;

  bool get isLoaded => _recognizer != null;

  /// A load or switch is queued or running.
  bool get isSwitching => _loading > 0;

  /// The installed model's id; null before [install] succeeded for [id].
  String? modelIdOf(ModelId id) => _modelIds[id];

  SttConfig configOf(ModelId id) =>
      _configs[id] ?? (throw ArgumentError.value(id, 'id', 'not an STT model'));

  /// The loaded recognizer. Only valid after a successful load.
  SpeechRecognizer get recognizer =>
      _recognizer ?? (throw StateError('SttService.load() has not succeeded'));

  /// Registers [id] from [source]; makes its spec the active one, so a load
  /// must follow before another install.
  Future<Result<String>> install(
    ModelId id, {
    required SttSource source,
    required void Function(int percent) onProgress,
  }) => _ops.run(() async {
    if (_closed) return const Result.error(SpeechServiceClosedException('STT'));
    try {
      final modelId = await _install(configOf(id), source, onProgress);
      _sources[id] = source;
      _modelIds[id] = modelId;
      _specFor = id;
      return Result.ok(modelId);
    } catch (e, st) {
      debugPrint('[SttService] install $id failed: $e\n$st');
      return Result.error(asException(e));
    }
  });

  /// Loads [id] (closing the active one first); returns the load time. The
  /// backend is the one requested ([SttConfig.backend]); nothing reports
  /// another.
  Future<Result<Duration>> load(ModelId id) {
    _loading++;
    return _ops.run(() => _loadNow(id)).whenComplete(() => _loading--);
  }

  /// Transcribes 0.5 s of silence on the active recognizer so the first
  /// question does not pay for lazy setup. Whisper always pads to its 30 s
  /// window, so this takes as long as a real turn's STT. The text is
  /// ignored.
  Future<Result<Duration>> warmUp() => _ops.run(_warmUpNow);

  /// Makes [id] the active recognizer (on a demo's entry): a
  /// no-op when it already is; otherwise waits for a transcription still
  /// running on the old one, then loads [id] (warming it up only when
  /// [warmUp]). Queued behind whatever runs; a request that a newer one for
  /// another model replaces before it runs is skipped (Ok).
  Future<Result<ActiveStt>> activate(ModelId id, {bool warmUp = false}) {
    _loading++;
    _requested = id;
    final requestedAt = Stopwatch()..start();
    return _ops
        .run(() async {
          if (_closed) {
            return const Result<ActiveStt>.error(
              SpeechServiceClosedException('STT'),
            );
          }
          final current = _active.value;
          if (current != null && current.id == id && _recognizer != null) {
            return Result.ok(current);
          }
          if (_requested != id) {
            debugPrint('[SttService] switch to $id skipped (superseded)');
            return current != null
                ? Result.ok(current)
                : const Result<ActiveStt>.error(SpeechNotReadyException('STT'));
          }
          await _drain();
          final loaded = await _loadNow(id, warmUp: warmUp);
          switch (loaded) {
            case Error(:final error):
              return Result<ActiveStt>.error(error);
            case Ok():
              final active = _active.value!;
              final switched = ActiveStt(
                id: active.id,
                modelId: active.modelId,
                loadTime: active.loadTime,
                warmUpTime: active.warmUpTime,
                switchTime: requestedAt.elapsed,
              );
              _setActive(switched);
              debugPrint(
                '[SttService] switched ${current?.modelId ?? 'none'} → '
                '${switched.modelId} in ${requestedAt.elapsedMilliseconds}ms '
                '(load ${switched.loadTime.inMilliseconds}ms, warm-up '
                '${switched.warmUpTime?.inMilliseconds ?? 'skipped'})',
              );
              return Result.ok(switched);
          }
        })
        .whenComplete(() => _loading--);
  }

  /// Transcribes [pcm] with the active recognizer once any queued switch has
  /// finished. Throws [SpeechNotReadyException] when none is loaded, or when
  /// [expected] (the demo's model) is not the active one: a switch that
  /// failed before releasing the old model leaves it loaded, and serving a
  /// Demo 1 question with moonshine (English only, 5 s cut) would be a
  /// silent fallback. The identity is checked at use time for that reason.
  Future<String> transcribe(Uint8List pcm, {ModelId? expected}) async {
    while (_loading > 0) {
      await _ops.idle;
    }
    final recognizer = _recognizer;
    if (_closed || recognizer == null) {
      throw const SpeechNotReadyException('speech recognizer');
    }
    final active = _active.value;
    if (expected != null && active?.id != expected) {
      throw SpeechNotReadyException(
        '${expected.spec.displayName} (the last switch failed; '
        '${active?.modelId ?? 'no model'} is loaded)',
      );
    }
    _transcribing++;
    try {
      return await recognizer.transcribe(pcm);
    } finally {
      if (--_transcribing == 0) {
        final idle = _idle;
        _idle = null;
        idle?.complete();
      }
    }
  }

  /// Closes the active recognizer and keeps the service usable: the next
  /// load or switch builds one again. After a failed warm-up, so a broken
  /// recognizer does not hold its memory until a Retry. Queued behind
  /// whatever runs; a no-op when none is loaded.
  Future<void> unload() => _ops.run(() async {
    final recognizer = _recognizer;
    _recognizer = null;
    _setActive(null);
    if (recognizer != null) {
      debugPrint('[SttService] unloading the active recognizer');
      await _closeQuietly(recognizer);
    }
  });

  /// Closes the recognizer, and any a load in flight delivers later. Safe to
  /// call more than once.
  Future<void> close() async {
    if (_closed) return;
    _closed = true;
    await _ops.run(() async {
      final recognizer = _recognizer;
      _recognizer = null;
      _setActive(null);
      if (recognizer != null) await _closeQuietly(recognizer);
    });
    _active.dispose();
    _switchError.dispose();
  }

  Future<Result<Duration>> _loadNow(ModelId id, {bool warmUp = false}) async {
    if (_closed) return const Result.error(SpeechServiceClosedException('STT'));
    final config = configOf(id);
    try {
      if (_specFor != id) {
        // Restores this model's spec as flutter_edge_ai's active one (files
        // are on disk: no download).
        final source =
            _sources[id] ??
            (throw StateError('${id.spec.displayName} was never installed'));
        _modelIds[id] = await _install(config, source, (_) {});
        _specFor = id;
      }
      // The singleton holds one model: close ours explicitly rather than
      // leave a stale handle that flutter_edge_ai closes behind our back.
      final previous = _recognizer;
      _recognizer = null;
      _setActive(null);
      if (previous != null) await _closeQuietly(previous);
      final watch = Stopwatch()..start();
      final recognizer = await _load(config);
      final loadTime = watch.elapsed;
      if (_closed) {
        await _closeQuietly(recognizer);
        return const Result.error(SpeechServiceClosedException('STT'));
      }
      _recognizer = recognizer;
      _setActive(
        ActiveStt(
          id: id,
          modelId: _modelIds[id] ?? id.name,
          loadTime: loadTime,
        ),
      );
      debugPrint(
        '[SttService] loaded ${_modelIds[id] ?? id.name} '
        'backend=${config.backend.name} (requested) '
        'language=${config.language} load=${loadTime.inMilliseconds}ms',
      );
      if (warmUp) {
        if (await _warmUpNow() case Error(:final error)) {
          _setSwitchError('$error');
          return Result.error(error);
        }
      }
      _setSwitchError(null);
      return Result.ok(loadTime);
    } catch (e, st) {
      debugPrint('[SttService] load $id failed: $e\n$st');
      _setSwitchError('$e');
      return Result.error(asException(e));
    }
  }

  void _setSwitchError(String? value) {
    if (!_closed) _switchError.value = value;
  }

  Future<Result<Duration>> _warmUpNow() async {
    final recognizer = _recognizer;
    final active = _active.value;
    if (_closed) return const Result.error(SpeechServiceClosedException('STT'));
    if (recognizer == null || active == null) {
      return Result.error(asException(StateError('warmUp() before load()')));
    }
    final watch = Stopwatch()..start();
    try {
      await recognizer.transcribe(Uint8List(configOf(active.id).sampleRate));
      debugPrint(
        '[SttService] warm-up ${active.modelId} '
        '${watch.elapsedMilliseconds}ms',
      );
      _setActive(
        ActiveStt(
          id: active.id,
          modelId: active.modelId,
          loadTime: active.loadTime,
          warmUpTime: watch.elapsed,
          switchTime: active.switchTime,
        ),
      );
      return Result.ok(watch.elapsed);
    } catch (e, st) {
      debugPrint('[SttService] warm-up failed: $e\n$st');
      return Result.error(asException(e));
    }
  }

  /// Waits (bounded) until no transcription runs on the active recognizer.
  Future<void> _drain() async {
    if (_transcribing == 0) return;
    final idle = _idle ??= Completer<void>();
    await idle.future.timeout(
      _drainTimeout,
      onTimeout: () => debugPrint(
        '[SttService] a transcription still runs after '
        '${_drainTimeout.inSeconds}s; switching anyway',
      ),
    );
  }

  void _setActive(ActiveStt? value) {
    if (!_closed) _active.value = value;
  }

  static Future<void> _closeQuietly(SpeechRecognizer recognizer) async {
    try {
      await recognizer.close();
    } catch (e, st) {
      debugPrint('[SttService] close failed: $e\n$st');
    }
  }
}
