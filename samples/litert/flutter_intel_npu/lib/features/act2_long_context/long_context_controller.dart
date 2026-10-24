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
import 'package:flutter_edge_ai/flutter_edge_ai.dart';

import '../../app/demo_config.dart';
import '../../core/engines/engine_pool.dart';
import '../../core/engines/engine_slot.dart';
import '../../core/transcript/transcript.dart';

enum Act2Phase { idle, loadingEngine, prefilling, ready, answering, failed }

/// Act 2 — the whole source goes into the context ONCE (prefill happens
/// before the audience asks anything), then every question reuses the warm
/// KV cache. A small model with a short window (the contrast engine) shows
/// the same source not fitting at all.
class LongContextController extends ChangeNotifier {
  LongContextController({
    required this.pool,
    required this.spec,
    required this.fallbackModelPath,
    required this.transcript,
  });

  final EnginePool pool;
  final Act2Spec spec;
  final String fallbackModelPath;
  final Transcript transcript;

  Act2Phase phase = Act2Phase.idle;
  String? error;

  EngineSlot? _slot;
  EngineTurn? _turn;
  bool _disposed = false;

  PreferredBackend? get activeBackend => _slot?.activeBackend;
  Duration? get loadTime => _slot?.loadTime;
  int? transcriptTokens;
  Duration? prefillTime;
  SessionMetrics? prefillMetrics;

  String question = '';
  String answer = '';
  List<Citation> citations = const [];
  SessionMetrics? answerMetrics;

  /// Tokens of the last answer alone. LiteRT-LM's counts add up every turn of
  /// the conversation (the prefill's "Ready" and every earlier answer).
  int? answerOutputTokens;
  Duration? answerTtft;

  // Contrast engine (short window) state.
  String? contrastResult;
  bool contrastRunning = false;

  String get modelPath => spec.modelPathOr(fallbackModelPath);
  String get modelFileName => modelPath.split(RegExp(r'[\\/]')).last;
  String get contrastModelPath => spec.contrastModelPathOr(fallbackModelPath);
  String get contrastModelFileName => contrastModelPath.split(RegExp(r'[\\/]')).last;

  static const _prefillInstruction =
      '\n\nAbove is the full transcript of a meeting. Read it carefully. '
      'Later I will ask questions; answer strictly from the transcript, and '
      'for every fact give the [mm:ss] timestamp of the line it comes from '
      'and quote that line. For now reply with exactly one word: Ready.';

  static const _questionSuffix =
      '\n(Answer strictly from the transcript. Cite the [mm:ss] timestamp and '
      'quote the line.)';

  /// Loads the engine and prefills the transcript. Idempotent while running.
  Future<void> start() async {
    if (phase != Act2Phase.idle && phase != Act2Phase.failed) return;
    error = null;
    _set(Act2Phase.loadingEngine);
    try {
      _slot = await pool.acquire(spec.backend, maxTokens: spec.maxTokens, modelPath: modelPath);
      if (_disposed) return;
      _turn = await _slot!.openTurn(maxOutputTokens: spec.maxOutputTokens);
      if (_disposed) return await _closeTurn();
      transcriptTokens = await _turn!.countTokens(transcript.asPromptText);
      if (_disposed) return await _closeTurn();
      notifyListeners();

      _set(Act2Phase.prefilling);
      final sw = Stopwatch()..start();
      final buf = StringBuffer();
      await for (final chunk in _turn!.generate(transcript.asPromptText + _prefillInstruction)) {
        buf.write(chunk);
      }
      // Left the act during the prefill: nobody will ask on this context.
      if (_disposed) return await _closeTurn();
      sw.stop();
      prefillTime = sw.elapsed;
      prefillMetrics = _turn!.metrics;
      _set(Act2Phase.ready);
    } catch (e, st) {
      error = '$e\n$st';
      _set(Act2Phase.failed);
    }
  }

  /// Asks on the warm context. Streams into [answer].
  Future<void> ask(String q) async {
    final turn = _turn;
    if (turn == null || phase != Act2Phase.ready) return;
    question = q;
    answer = '';
    citations = const [];
    answerMetrics = null;
    answerOutputTokens = null;
    answerTtft = null;
    _set(Act2Phase.answering);
    try {
      final outBefore = turn.metrics.outputTokens;
      final sw = Stopwatch()..start();
      await for (final chunk in turn.generate(q + _questionSuffix)) {
        answerTtft ??= sw.elapsed;
        answer += chunk;
        citations = transcript.citationsIn(answer);
        notifyListeners();
      }
      answerMetrics = turn.metrics;
      answerOutputTokens = answerMetrics!.outputTokens - outBefore;
      _set(Act2Phase.ready);
    } catch (e, st) {
      error = '$e\n$st';
      _set(Act2Phase.failed);
    }
  }

  /// Same source into an engine whose window is too small: expected to fail
  /// with the runtime's own "Input token ids are too long" error.
  Future<void> runContrast() async {
    if (contrastRunning) return;
    contrastRunning = true;
    contrastResult = null;
    notifyListeners();
    EngineTurn? t;
    try {
      final slot = await pool.acquire(spec.backend, maxTokens: spec.contrastMaxTokens, modelPath: contrastModelPath);
      t = await slot.openTurn(maxOutputTokens: 32);
      final buf = StringBuffer();
      await for (final chunk in t.generate(transcript.asPromptText + _prefillInstruction)) {
        buf.write(chunk);
      }
      contrastResult = 'Unexpectedly fitted: "${buf.toString().trim()}"';
    } catch (e) {
      contrastResult = e.toString().split('\n').first;
    } finally {
      await t?.close();
      // The contrast engine is done for good; keep its memory for the act.
      await pool.release(spec.backend, maxTokens: spec.contrastMaxTokens, modelPath: contrastModelPath);
      contrastRunning = false;
      notifyListeners();
    }
    // Under `sequential` the pool unloaded the main engine to load the
    // contrast one, and the warm context went with it: fill it again rather
    // than send the next question to a closed model.
    if (pool.policy == EnginePolicy.sequential && !_disposed && !(_slot?.isLoaded ?? false)) {
      _turn = null;
      phase = Act2Phase.idle;
      await start();
    }
  }

  Future<void> _closeTurn() async {
    final t = _turn;
    _turn = null;
    await t?.close();
  }

  /// Drops the conversation and prefills again (new KV cache).
  Future<void> reset() async {
    await _turn?.close();
    _turn = null;
    answer = '';
    question = '';
    citations = const [];
    prefillTime = null;
    phase = Act2Phase.idle;
    notifyListeners();
    await start();
  }

  Map<String, Object?> toJson() => {
        'model': modelFileName,
        'backend': spec.backend.name,
        'active_backend': activeBackend?.name,
        'max_tokens': spec.maxTokens,
        'transcript_tokens': transcriptTokens,
        'load_ms': loadTime?.inMilliseconds,
        'prefill_ms': prefillTime?.inMilliseconds,
        'prefill_input_tokens': prefillMetrics?.inputTokens,
        'prefill_ttft_ms': prefillMetrics?.timeToFirstTokenMs,
        'question': question,
        'answer': answer,
        'citations': citations.map((c) => c.stamp).toList(),
        'answer_ttft_ms': answerTtft?.inMilliseconds,
        'answer_tps': answerMetrics?.tokensPerSecond,
        'answer_output_tokens': answerOutputTokens,
        'contrast_model': contrastModelFileName,
        'contrast_max_tokens': spec.contrastMaxTokens,
        'contrast_result': contrastResult,
        'error': error,
      };

  void _set(Act2Phase p) {
    phase = p;
    if (!_disposed) notifyListeners();
  }

  @override
  void notifyListeners() {
    if (!_disposed) super.notifyListeners();
  }

  @override
  void dispose() {
    _disposed = true;
    // The warm context dies with the turn, so a later visit prefills again
    // anyway; keeping the 12B resident would only hold ~6 GB through Act 3.
    unawaited(_closeTurn().then((_) => pool.release(spec.backend, maxTokens: spec.maxTokens, modelPath: modelPath)));
    super.dispose();
  }
}
