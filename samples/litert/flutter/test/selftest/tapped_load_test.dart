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

import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/domain/models/accelerator_evidence.dart';
import 'package:litert_edge_demos/selftest/tapped_load.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../fakes/fake_hardware.dart';

void main() {
  test('only the lines printed during the load count', () async {
    final tap = FakeNativeLogTap()
      ..lines.add('I0000 delegate_metal.mm:88] Created a Metal device.');
    final load = await loadWithNativeLog(tap, () async {
      tap.lines.add('I0000 webgpu.cc:1] Created a WebGPU environment');
      return const Result.ok('loaded');
    });

    expect(load.result, isA<Ok<String>>());
    expect(load.logLines, [
      'log: I0000 webgpu.cc:1] Created a WebGPU environment',
    ]);
    final evidence = load.evidence(
      requested: 'gpu',
      reported: 'gpu',
      hardware: kFakeMacProfile,
    );
    expect(evidence.api, 'WebGPU');
    expect(evidence.apiSource, EvidenceSource.log);
    expect(evidence.adapter, 'Apple M4 Pro');
  });

  test('a failure: the error, the evidence lines, then the native problems '
      'not already shown', () async {
    final tap = FakeNativeLogTap();
    final load = await loadWithNativeLog<String>(tap, () async {
      tap.lines.addAll(const [
        'INFO: attempting the GPU',
        'W0000 x.cc:1] GPU accelerator could not be loaded',
        'E0000 webgpu.cc:12] Failed to create a Vulkan instance',
      ]);
      return Result.error(Exception('no delegate'));
    });

    final error = (load.result as Error<String>).error;
    expect(load.failure(error), [
      'load failed (no fallback): Exception: no delegate',
      'log: W0000 x.cc:1] GPU accelerator could not be loaded',
      'log: E0000 webgpu.cc:12] Failed to create a Vulkan instance',
    ]);
  });
}
