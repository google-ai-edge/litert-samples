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
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/repositories/live_detection_repository.dart';
import 'package:litert_edge_demos/domain/models/frame_source_spec.dart';
import 'package:litert_edge_demos/domain/models/live_state.dart';
import 'package:litert_edge_demos/domain/use_cases/gpu_arbiter.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../fakes/fake_detector.dart';
import '../../fakes/fake_frame_source.dart';

void main() {
  late ValueNotifier<bool> generating;
  late FakeFrameSource source;
  late LiveDetectionRepository live;

  setUp(() {
    generating = ValueNotifier(false);
    source = FakeFrameSource();
    live = LiveDetectionRepository(
      detector: FakeDetector(autoComplete: true),
      createSource: (_) => Result.ok(source),
    );
  });

  tearDown(() async {
    await live.close();
    generating.dispose();
  });

  test('pauses detection while the chat model generates, naming it, and '
      'resumes after', () async {
    final arbiter = GpuArbiter(
      llmBusy: generating,
      setDetectorDuty: live.setDuty,
      duringGeneration: DetectorDuty.paused,
      chatModelName: () => 'Gemma 4 E2B',
    );
    addTearDown(arbiter.dispose);
    await live.start(const FixtureSourceSpec(['/x']), owner: Object());
    expect(live.state.value, isA<LiveRunning>());

    generating.value = true;
    final paused = live.state.value;
    expect(paused, isA<LivePaused>());
    expect((paused as LivePaused).reason, 'Gemma 4 E2B');
    expect(live.duty, DetectorDuty.paused);

    generating.value = false;
    expect(live.state.value, isA<LiveRunning>());
    expect(live.duty, DetectorDuty.live);
  });

  test('without a name the pause says "the chat model"', () async {
    final arbiter = GpuArbiter(
      llmBusy: generating,
      setDetectorDuty: live.setDuty,
      duringGeneration: DetectorDuty.paused,
      chatModelName: () => null,
    );
    addTearDown(arbiter.dispose);
    await live.start(const FixtureSourceSpec(['/x']), owner: Object());

    generating.value = true;

    expect((live.state.value as LivePaused).reason, 'the chat model');
  });

  test('a generation already running when the arbiter is created pauses at '
      'once; after dispose it no longer reacts', () {
    generating.value = true;
    final arbiter = GpuArbiter(
      llmBusy: generating,
      setDetectorDuty: live.setDuty,
      duringGeneration: DetectorDuty.paused,
    );
    expect(live.duty, DetectorDuty.paused);

    arbiter.dispose();
    generating.value = false;
    expect(live.duty, DetectorDuty.paused, reason: 'detached');
  });

  test('duringGeneration: live keeps detecting (the measurement mode)', () {
    final arbiter = GpuArbiter(
      llmBusy: generating,
      setDetectorDuty: live.setDuty,
      duringGeneration: DetectorDuty.live,
    );
    addTearDown(arbiter.dispose);

    generating.value = true;

    expect(live.duty, DetectorDuty.live);
  });
}
