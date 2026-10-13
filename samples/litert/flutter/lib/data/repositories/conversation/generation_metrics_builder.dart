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

import 'dart:math' as math;

import 'package:flutter_edge_ai/flutter_edge_ai.dart' show SessionMetrics;

import '../../../domain/models/assistant_event.dart';
import '../../../domain/models/skill_step.dart';

/// One turn's clock and facts, and the one place both turn paths (the plain
/// chat and the agent chat) build its [GenerationMetrics].
///
/// The clock starts when the builder is created. The turn reports its text
/// chunks ([chunk]) and a context reset ([contextWasReset]) as they happen,
/// stops the clock when the reply has ended ([stop]), then builds the
/// metrics once: [stoppedBeforeSending] or [finished].
final class GenerationMetricsBuilder {
  /// [stopwatch] is for tests; it is started here.
  GenerationMetricsBuilder({required this.imageAttached, Stopwatch? stopwatch})
    : _watch = (stopwatch ?? Stopwatch())..start();

  /// The turn has an image attached: sent now, or already in the context.
  final bool imageAttached;

  final Stopwatch _watch;
  Duration? _firstChunkAt;
  Duration? _lastChunkAt;
  int _chunks = 0;
  bool _contextReset = false;
  ContextResetReason _resetReason = ContextResetReason.budget;

  /// Time since the turn started; fixed once [stop] is called.
  Duration get elapsed => _watch.elapsed;

  /// A piece of visible reply text arrived now.
  void chunk() {
    _firstChunkAt ??= _watch.elapsed;
    _lastChunkAt = _watch.elapsed;
    _chunks++;
  }

  /// The chat was recreated before this turn. The first reset's [reason] is
  /// the one reported: it is the one the turn announced first.
  void contextWasReset(ContextResetReason reason) {
    if (!_contextReset) _resetReason = reason;
    _contextReset = true;
  }

  /// The reply has ended: the turn's total time is fixed from here.
  void stop() => _watch.stop();

  /// A turn stopped before anything reached the model: no text, no rates,
  /// nothing sent.
  GenerationMetrics stoppedBeforeSending({
    required Duration? stopLatency,
    required int promptTokens,
  }) => GenerationMetrics(
    timeToFirstToken: null,
    chunks: 0,
    tokensPerSecond: null,
    tokensPerSecondSource: TokenRateSource.chunks,
    total: _watch.elapsed,
    stopped: true,
    stopLatency: stopLatency,
    imageAttached: imageAttached,
    contextReset: _contextReset,
    contextResetReason: _resetReason,
    promptTokens: promptTokens,
  );

  /// A turn that reached the model and ended, normally or [stopped].
  ///
  /// [before] and [after]: LiteRT-LM's metrics for the live conversation
  /// around the turn (null when they could not be read). LiteRT-LM reports
  /// the conversation's LAST decode turn; that is this turn only when it ran
  /// to the end and produced text, so only then is its rate used (otherwise
  /// the chunk rate). The prefill is the input-token delta, unless
  /// [rebuildPending] (LiteRT-LM rebuilt the native conversation when this
  /// turn started) or the turn had [toolRounds] (the delta then also holds
  /// tool calls and responses, so it says nothing about the image).
  /// [chatTokens]: flutter_edge_ai's own count after the turn.
  GenerationMetrics finished({
    required bool stopped,
    required Duration? stopLatency,
    required bool imageSent,
    required ImageLoss? imageResent,
    required SessionMetrics? before,
    required SessionMetrics? after,
    required bool rebuildPending,
    required int chatTokens,
    required int promptTokens,
    int toolRounds = 0,
    List<SkillStep> skillSteps = const [],
  }) {
    final nativeRate = !stopped && _chunks > 0 ? after?.tokensPerSecond : null;
    final prefill =
        toolRounds == 0 && !rebuildPending && before != null && after != null
        ? after.inputTokens - before.inputTokens
        : null;
    return GenerationMetrics(
      timeToFirstToken: _firstChunkAt,
      chunks: _chunks,
      tokensPerSecond: nativeRate ?? _chunkRate(),
      tokensPerSecondSource: nativeRate != null
          ? TokenRateSource.native
          : TokenRateSource.chunks,
      total: _watch.elapsed,
      stopped: stopped,
      stopLatency: stopLatency,
      imageAttached: imageAttached,
      imageSent: imageSent,
      imageResent: imageResent,
      contextReset: _contextReset,
      contextResetReason: _resetReason,
      contextTokens: math.max(after?.totalTokens ?? 0, chatTokens),
      prefillTokens: prefill != null && prefill > 0 ? prefill : null,
      promptTokens: promptTokens,
      toolRounds: toolRounds,
      skillSteps: skillSteps,
    );
  }

  /// Chunks after the first over the time between the first and the last.
  double? _chunkRate() {
    final first = _firstChunkAt;
    final last = _lastChunkAt;
    if (_chunks < 2 || first == null || last == null || last <= first) {
      return null;
    }
    return (_chunks - 1) / ((last - first).inMicroseconds / 1e6);
  }
}
