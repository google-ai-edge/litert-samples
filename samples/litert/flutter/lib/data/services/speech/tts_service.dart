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
import 'package:flutter_edge_ai/flutter_edge_ai.dart' hide ModelSpec;

import '../../../config/model_catalog.dart';
import '../../../utils/result.dart';
import 'stt_service.dart' show SpeechServiceClosedException;

/// Installs the TTS bundle from the directory holding its files and makes
/// it the active one; returns the model id. Progress is 0–100 over the
/// bundle's files.
typedef TtsInstaller = Future<String> Function(
  TtsConfig config,
  String directory,
  void Function(int percent) onProgress,
);

/// Loads the active TTS model with [TtsConfig]'s backend.
typedef SynthesizerLoader = Future<SpeechSynthesizer> Function(
  TtsConfig config,
);

/// `installTts().fromFile`: the bundle built into the app, used where it is
/// (flutter_edge_ai 2.1.0 `tts_installation_builder.dart:57`; no network,
/// no copy). Never uninstalled: that would delete the files.
Future<String> installTtsFromFiles(
  TtsConfig config,
  String directory,
  void Function(int percent) onProgress,
) async {
  final installation = await FlutterEdgeAi.installTts()
      .fromFile(directory)
      .ofType(config.type)
      .withProgress(onProgress)
      .install();
  return installation.modelId;
}

/// `FlutterEdgeAi.getActiveTts` with exactly the [TtsConfig] arguments.
Future<SpeechSynthesizer> getActiveTtsFor(TtsConfig config) =>
    FlutterEdgeAi.getActiveTts(preferredBackend: config.backend);

/// The synthesizer reports another rate than the catalog's; playing at the
/// wrong rate would shift the pitch, so setup stops instead.
final class TtsSampleRateException implements Exception {
  const TtsSampleRateException({required this.expected, required this.actual});

  final int expected;
  final int actual;

  @override
  String toString() =>
      'TTS reports $actual Hz, the catalog expects $expected Hz';
}

/// The warm-up synthesized nothing for a plain sentence.
final class TtsSilentException implements Exception {
  const TtsSilentException(this.text);

  final String text;

  @override
  String toString() => 'TTS returned no audio for "$text"';
}

/// Owns the speech synthesizer: install, load, warm up, close. Called only
/// by `ModelRepository` (with it, the only caller of `getActiveTts`); the
/// speech repository borrows [synthesizer] per turn.
class TtsService {
  TtsService({
    this._config = kTtsConfig,
    this._install = installTtsFromFiles,
    this._load = getActiveTtsFor,
  });

  final TtsConfig _config;
  final TtsInstaller _install;
  final SynthesizerLoader _load;

  SpeechSynthesizer? _synthesizer;
  String? _modelId;
  bool _closed = false;

  bool get isLoaded => _synthesizer != null;

  String? get modelId => _modelId;

  /// The loaded synthesizer. Only valid after a successful [load].
  SpeechSynthesizer get synthesizer =>
      _synthesizer ?? (throw StateError('TtsService.load() has not succeeded'));

  /// The rate reply audio plays at: the loaded synthesizer's, which [load]
  /// checked against the catalog; the catalog's before that.
  int get sampleRate => _synthesizer?.sampleRate ?? _config.sampleRate;

  /// Installs the bundle in [directory] (the built-in files).
  Future<Result<String>> install({
    required String directory,
    required void Function(int percent) onProgress,
  }) async {
    try {
      final id = await _install(_config, directory, onProgress);
      _modelId = id;
      return Result.ok(id);
    } catch (e, st) {
      debugPrint('[TtsService] install failed: $e\n$st');
      return Result.error(asException(e));
    }
  }

  /// Loads the synthesizer and checks its sample rate; returns the load time.
  Future<Result<Duration>> load() async {
    final watch = Stopwatch()..start();
    try {
      final synthesizer = await _load(_config);
      watch.stop();
      if (_closed) {
        await _closeQuietly(synthesizer);
        return const Result.error(SpeechServiceClosedException('TTS'));
      }
      if (synthesizer.sampleRate != _config.sampleRate) {
        final error = TtsSampleRateException(
          expected: _config.sampleRate,
          actual: synthesizer.sampleRate,
        );
        debugPrint('[TtsService] $error');
        await _closeQuietly(synthesizer);
        return Result.error(error);
      }
      _synthesizer = synthesizer;
      debugPrint(
        '[TtsService] loaded ${_modelId ?? '?'} '
        'backend=${_config.backend.name} (requested) '
        'rate=${synthesizer.sampleRate}Hz load=${watch.elapsedMilliseconds}ms',
      );
      return Result.ok(watch.elapsed);
    } catch (e, st) {
      debugPrint('[TtsService] load failed: $e\n$st');
      return Result.error(asException(e));
    }
  }

  /// Synthesizes a short sentence (not played): the first clause of the
  /// first reply then does not pay for lazy setup. Empty audio is a failure.
  Future<Result<Duration>> warmUp({String text = 'Ready.'}) async {
    final synthesizer = _synthesizer;
    if (_closed) return const Result.error(SpeechServiceClosedException('TTS'));
    if (synthesizer == null) {
      return Result.error(asException(StateError('warmUp() before load()')));
    }
    final watch = Stopwatch()..start();
    try {
      final pcm = await synthesizer.synthesize(text);
      if (pcm.isEmpty) return Result.error(TtsSilentException(text));
      debugPrint(
        '[TtsService] warm-up ${watch.elapsedMilliseconds}ms '
        '(${pcm.length} bytes)',
      );
      return Result.ok(watch.elapsed);
    } catch (e, st) {
      debugPrint('[TtsService] warm-up failed: $e\n$st');
      return Result.error(asException(e));
    }
  }

  /// Closes the loaded synthesizer and keeps the service usable: the next
  /// [load] builds it again. After a failed warm-up, so a broken synthesizer
  /// does not hold its memory until a Retry. A no-op when none is loaded.
  Future<void> unload() async {
    final synthesizer = _synthesizer;
    _synthesizer = null;
    if (synthesizer != null) {
      debugPrint('[TtsService] unloading ${_modelId ?? '?'}');
      await _closeQuietly(synthesizer);
    }
  }

  /// Closes the synthesizer, and any a load in flight delivers later. Safe to
  /// call more than once.
  Future<void> close() async {
    _closed = true;
    final synthesizer = _synthesizer;
    _synthesizer = null;
    if (synthesizer != null) await _closeQuietly(synthesizer);
  }

  static Future<void> _closeQuietly(SpeechSynthesizer synthesizer) async {
    try {
      await synthesizer.close();
    } catch (e, st) {
      debugPrint('[TtsService] close failed: $e\n$st');
    }
  }
}
