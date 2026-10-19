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

import 'package:flutter_edge_ai/flutter_edge_ai.dart';

import '../power/power_source.dart';

enum BenchmarkRunStatus { pending, loading, warming, generating, ok, failed, unavailable }

/// One streamed chunk and when it arrived, counted from the moment the
/// prompt was sent (so the first one includes the prefill).
class TimedChunk {
  const TimedChunk(this.at, this.text);
  final Duration at;
  final String text;
}

/// What a recorded generation showed at [elapsed] into it: the replay of a
/// run, so runs measured one after another can be raced side by side.
String textAt(List<TimedChunk> timeline, Duration elapsed) {
  final b = StringBuffer();
  for (final c in timeline) {
    if (c.at > elapsed) break;
    b.write(c.text);
  }
  return b.toString();
}

/// One backend's result for one prompt. Immutable; the runner emits a fresh
/// copy on every state change so the UI can rebuild a single row.
class BenchmarkRun {
  const BenchmarkRun({
    required this.modelFile,
    required this.requestedBackend,
    required this.status,
    this.actualBackend,
    this.loadTime,
    this.wallTimeToFirstToken,
    this.wallTotal,
    this.chunks = 0,
    this.metrics,
    this.responsePreview = '',
    this.error,
    this.reason,
    this.power,
    this.timeline = const [],
  });

  final String modelFile;
  final PreferredBackend requestedBackend;
  final BenchmarkRunStatus status;

  /// What the runtime actually initialised (may differ from requested).
  final PreferredBackend? actualBackend;

  /// Wall clock around `engine_create` (model load + accelerator init).
  final Duration? loadTime;

  /// Wall clock from `getResponseAsync()` subscription to the first chunk.
  /// Cross-check for `metrics.timeToFirstTokenMs`.
  final Duration? wallTimeToFirstToken;

  /// Wall clock for the whole measured generation.
  final Duration? wallTotal;

  /// Number of streamed chunks (≈ tokens on the FFI path).
  final int chunks;

  /// Runtime-reported numbers (LiteRT-LM benchmark info). Null until done.
  final SessionMetrics? metrics;

  final String responsePreview;
  final String? error;

  /// Why a backend was skipped (e.g. "Windows + Intel Lunar/Panther Lake only").
  final String? reason;

  /// Power / utilisation averaged over the measured generation (null = no sidecar).
  final PowerSummary? power;

  /// Every chunk of the measured generation with its arrival time.
  final List<TimedChunk> timeline;

  /// When the last chunk arrived: the run's finishing time in a race.
  Duration? get finishedAt => timeline.isEmpty ? null : timeline.last.at;

  /// Decode speed from wall clock: chunks after the first, divided by the
  /// time after the first chunk. Falls back to null when not measurable.
  double? get wallDecodeTokensPerSecond {
    final ttft = wallTimeToFirstToken;
    final total = wallTotal;
    if (ttft == null || total == null || chunks < 2) return null;
    final decodeSeconds = (total - ttft).inMicroseconds / 1e6;
    if (decodeSeconds <= 0) return null;
    return (chunks - 1) / decodeSeconds;
  }

  BenchmarkRun copyWith({
    BenchmarkRunStatus? status,
    PreferredBackend? actualBackend,
    Duration? loadTime,
    Duration? wallTimeToFirstToken,
    Duration? wallTotal,
    int? chunks,
    SessionMetrics? metrics,
    String? responsePreview,
    String? error,
    String? reason,
    PowerSummary? power,
    List<TimedChunk>? timeline,
  }) =>
      BenchmarkRun(
        modelFile: modelFile,
        requestedBackend: requestedBackend,
        status: status ?? this.status,
        actualBackend: actualBackend ?? this.actualBackend,
        loadTime: loadTime ?? this.loadTime,
        wallTimeToFirstToken: wallTimeToFirstToken ?? this.wallTimeToFirstToken,
        wallTotal: wallTotal ?? this.wallTotal,
        chunks: chunks ?? this.chunks,
        metrics: metrics ?? this.metrics,
        responsePreview: responsePreview ?? this.responsePreview,
        error: error ?? this.error,
        reason: reason ?? this.reason,
        power: power ?? this.power,
        timeline: timeline ?? this.timeline,
      );

  Map<String, Object?> toJson() => {
        'model': modelFile,
        'requested': requestedBackend.name,
        'actual': actualBackend?.name,
        'status': status.name,
        'load_ms': loadTime?.inMilliseconds,
        'wall_ttft_ms': wallTimeToFirstToken?.inMilliseconds,
        'wall_total_ms': wallTotal?.inMilliseconds,
        'wall_decode_tps': wallDecodeTokensPerSecond,
        'chunks': chunks,
        'input_tokens': metrics?.inputTokens,
        'output_tokens': metrics?.outputTokens,
        'metrics_ttft_ms': metrics?.timeToFirstTokenMs,
        'metrics_tps': metrics?.tokensPerSecond,
        'metrics_init_ms': metrics?.initTimeMs,
        'error': error,
        'reason': reason,
        'power': power?.toJson(),
        'finished_ms': finishedAt?.inMilliseconds,
        // Arrival time of every chunk: enough to replay or plot the run.
        'chunk_ms': [for (final c in timeline) c.at.inMilliseconds],
        'response': responsePreview.length > 2000
            ? '${responsePreview.substring(0, 2000)}…'
            : responsePreview,
      };
}
