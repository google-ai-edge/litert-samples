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

import 'package:flutter_edge_ai/flutter_edge_ai.dart';

import 'backend_catalog.dart';
import 'benchmark_run.dart';
import 'engine_pool.dart';
import '../power/power_source.dart';

/// Runs the same prompt on each backend, ONE AT A TIME, and streams the row
/// state as it changes. Sequential by measurement design: the iGPU and the
/// NPU share memory bandwidth, so concurrent generation would corrupt both
/// tokens/sec and watts.
class BenchmarkRunner {
  BenchmarkRunner({
    required this.pool,
    required this.catalog,
    this.power,
  });

  final EnginePool pool;
  final BackendCatalog catalog;

  /// Sidecar readings averaged over each measured generation.
  final PowerSource? power;

  /// Emits a [BenchmarkRun] for every state change of every backend. The
  /// last emission per backend is its final result.
  /// [modelPathFor] picks each backend's file: the NPU wants an NPU build.
  Stream<BenchmarkRun> runComparison({
    required String Function(PreferredBackend backend) modelPathFor,
    required int maxTokens,
    required List<PreferredBackend> backends,
    required String prompt,
    bool warmup = true,
    int? maxOutputTokens,
    int topK = 1,
    void Function(PreferredBackend backend, String chunk)? onChunk,
  }) async* {
    for (final backend in backends) {
      final modelPath = modelPathFor(backend);
      var run = BenchmarkRun(
        modelFile: modelPath.split(RegExp(r'[\\/]')).last,
        requestedBackend: backend,
        status: BenchmarkRunStatus.pending,
      );

      final status = catalog.status(backend);
      if (!status.isAvailable) {
        yield run.copyWith(
          status: BenchmarkRunStatus.unavailable,
          reason: status.reason,
        );
        continue;
      }

      yield run = run.copyWith(status: BenchmarkRunStatus.loading, reason: status.reason);
      try {
        final slot = await pool.acquire(backend, maxTokens: maxTokens, modelPath: modelPath);
        yield run = run.copyWith(
          actualBackend: slot.activeBackend,
          loadTime: slot.loadTime,
        );

        if (warmup) {
          yield run = run.copyWith(status: BenchmarkRunStatus.warming);
          final t = await slot.openTurn(maxOutputTokens: 8);
          try {
            await for (final _ in t.generate('Say hi.')) {}
          } finally {
            await t.close();
          }
        }

        yield run = run.copyWith(status: BenchmarkRunStatus.generating);
        final turn = await slot.openTurn(maxOutputTokens: maxOutputTokens, topK: topK);
        try {
          final sw = Stopwatch()..start();
          Duration? firstChunkAt;
          var chunks = 0;
          final buffer = StringBuffer();
          final timeline = <TimedChunk>[];
          Future<void> generate() async {
            await for (final chunk in turn.generate(prompt)) {
              final at = sw.elapsed;
              firstChunkAt ??= at;
              chunks++;
              buffer.write(chunk);
              timeline.add(TimedChunk(at, chunk));
              onChunk?.call(backend, chunk);
            }
          }

          PowerSummary? powerSummary;
          if (power != null) {
            final (_, p) = await power!.measure(generate);
            powerSummary = p;
          } else {
            await generate();
          }
          sw.stop();
          final metrics = turn.metrics;
          yield run = run.copyWith(
            status: BenchmarkRunStatus.ok,
            actualBackend: slot.activeBackend,
            wallTimeToFirstToken: firstChunkAt,
            wallTotal: sw.elapsed,
            chunks: chunks,
            metrics: metrics,
            responsePreview: buffer.toString(),
            power: powerSummary,
            timeline: List.unmodifiable(timeline),
          );
        } finally {
          await turn.close();
        }
      } catch (e, st) {
        yield run = run.copyWith(
          status: BenchmarkRunStatus.failed,
          error: '$e\n$st',
        );
      }
    }
  }
}
