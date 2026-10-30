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
import 'package:flutter_test/flutter_test.dart';
import 'package:intel_npu_demo/core/engines/backend_catalog.dart';
import 'package:intel_npu_demo/core/engines/benchmark_run.dart';
import 'package:intel_npu_demo/core/engines/benchmark_runner.dart';
import 'package:intel_npu_demo/core/engines/engine_pool.dart';
import 'package:intel_npu_demo/core/engines/engine_slot.dart';
import 'package:intel_npu_demo/features/act1_inference/race_view.dart';

/// Every backend is available, whatever OS the test runs on.
class _AllBackends extends BackendCatalog {
  const _AllBackends();
  @override
  BackendStatus status(PreferredBackend backend) =>
      BackendStatus(backend: backend, availability: BackendAvailability.available);
}

class _Turn implements EngineTurn {
  _Turn(this.chunks);
  final List<String> chunks;
  @override
  Stream<String> generate(String prompt) async* {
    for (final c in chunks) {
      await Future<void>.delayed(const Duration(milliseconds: 2));
      yield c;
    }
  }

  @override
  Future<int?> countTokens(String text) async => null;
  @override
  SessionMetrics get metrics => SessionMetrics(outputTokens: chunks.length, tokensPerSecond: 50);
  @override
  Future<void> stop() async {}
  @override
  Future<void> close() async {}
}

class _Slot implements EngineSlot {
  _Slot(this.backend);
  @override
  final PreferredBackend backend;
  @override
  PreferredBackend? get activeBackend => backend;
  @override
  bool get isLoaded => true;
  @override
  Duration? get loadTime => Duration.zero;
  @override
  Future<void> load({required String modelPath, required int maxTokens}) async {}
  @override
  Future<EngineTurn> openTurn({double temperature = 0.8, int topK = 1, int? maxOutputTokens}) async =>
      _Turn(['a', 'b', 'c']);
  @override
  Future<InferenceChat> openChat({
    required List<Tool> tools,
    required String systemInstruction,
    required bool enableThinking,
    required double temperature,
    required int topK,
    required int randomSeed,
    int? maxOutputTokens,
  }) =>
      throw UnimplementedError();
  @override
  Future<void> unload() async {}
}

class _Pool extends EnginePool {
  _Pool() : super(policy: EnginePolicy.resident, modelPath: '/m/standard.litertlm');
  final Map<PreferredBackend, String?> askedFor = {};
  @override
  Future<EngineSlot> acquire(PreferredBackend backend, {required int maxTokens, String? modelPath}) async {
    askedFor[backend] = modelPath;
    return _Slot(backend);
  }
}

void main() {
  test('each backend runs its own model file and keeps a replayable timeline', () async {
    final pool = _Pool();
    final runner = BenchmarkRunner(pool: pool, catalog: const _AllBackends());
    final last = <PreferredBackend, BenchmarkRun>{};
    await for (final r in runner.runComparison(
      modelPathFor: (b) => b == PreferredBackend.npu ? '/m/npu_build.litertlm' : '/m/standard.litertlm',
      maxTokens: 4096,
      backends: const [PreferredBackend.cpu, PreferredBackend.npu],
      prompt: 'hi',
      warmup: false,
    )) {
      last[r.requestedBackend] = r;
    }
    expect(pool.askedFor[PreferredBackend.cpu], '/m/standard.litertlm');
    expect(pool.askedFor[PreferredBackend.npu], '/m/npu_build.litertlm');
    expect(last[PreferredBackend.npu]!.modelFile, 'npu_build.litertlm');

    final npu = last[PreferredBackend.npu]!;
    expect(npu.status, BenchmarkRunStatus.ok);
    expect(npu.timeline.map((c) => c.text).join(), 'abc');
    expect(npu.finishedAt, npu.timeline.last.at);
    // Replay: nothing before the first chunk, everything at the finish.
    expect(textAt(npu.timeline, Duration.zero), '');
    expect(textAt(npu.timeline, npu.finishedAt!), 'abc');
    expect(textAt(npu.timeline, npu.timeline[1].at), 'ab');
  });

  test('the race is ordered by when each answer finished, failures left out', () {
    BenchmarkRun run(PreferredBackend b, int finishMs, {bool ok = true}) => BenchmarkRun(
          modelFile: 'm',
          requestedBackend: b,
          status: ok ? BenchmarkRunStatus.ok : BenchmarkRunStatus.failed,
          timeline: [TimedChunk(Duration(milliseconds: finishMs), 'x')],
        );
    final places = placesOf([
      run(PreferredBackend.cpu, 21000),
      run(PreferredBackend.gpu, 6800),
      run(PreferredBackend.npu, 4100),
    ]);
    expect(places, {PreferredBackend.npu: 1, PreferredBackend.gpu: 2, PreferredBackend.cpu: 3});
    expect(placesOf([run(PreferredBackend.npu, 10, ok: false), run(PreferredBackend.gpu, 20)]), {PreferredBackend.gpu: 1});
  });
}
