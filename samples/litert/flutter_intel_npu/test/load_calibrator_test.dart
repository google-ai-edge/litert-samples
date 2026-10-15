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

import 'dart:ui' show FrameTiming;

import 'package:flutter_test/flutter_test.dart';
import 'package:intel_npu_demo/core/telemetry/frame_time_probe.dart';
import 'package:intel_npu_demo/features/act3_scene/runner/load_calibrator.dart';

/// A probe whose frame rate is whatever the current ladder level makes it.
class _LadderProbe extends FrameTimeProbe {
  _LadderProbe(this.fpsAt);
  final double Function(int level) fpsAt;
  int level = 0;
  @override
  double get sustainedFps => fpsAt(level);
}

Future<int> calibrate(FrameBudget budget, double Function(int level) fpsAt, {int start = 1}) {
  final probe = _LadderProbe(fpsAt);
  final c = LoadCalibrator(
    probe: probe,
    budget: budget,
    startLevel: start,
    settle: Duration.zero,
    sample: Duration.zero,
    apply: (step) => probe.level = LoadCalibrator.ladder.indexOf(step),
  );
  return c.run();
}

void main() {
  group('FrameBudget', () {
    test('a display that cannot show clearly more than the target sets the budget', () {
      final at60 = FrameBudget.resolve(requested: 60, refreshRate: 60);
      expect((at60.fps, at60.vsyncCapped), (60, true));
      final at75 = FrameBudget.resolve(requested: 60, refreshRate: 75);
      expect((at75.fps, at75.vsyncCapped), (75, true));
    });

    test('a fast display keeps the requested target', () {
      final mac = FrameBudget.resolve(requested: 60, refreshRate: 120);
      expect((mac.fps, mac.vsyncCapped), (60, false));
    });

    test('0 means the refresh rate; an unknown refresh rate is 60', () {
      expect(FrameBudget.resolve(requested: 0, refreshRate: 120).fps, 120);
      expect(FrameBudget.resolve(requested: 60, refreshRate: 0).fps, 60);
    });
  });

  group('LoadCalibrator', () {
    test('on a 60 Hz display it keeps the heaviest rung that still holds 60', () async {
      // Vsync caps every light rung at exactly 60: the old "climb above
      // 1.3 × target" rule never climbed and locked the lightest rung.
      final level = await calibrate(
        FrameBudget.resolve(requested: 60, refreshRate: 60),
        (l) => l <= 6 ? 60 : 52,
      );
      expect(level, 6);
    });

    test('on a 60 Hz display it comes down from a rung that is too heavy', () async {
      final level = await calibrate(FrameBudget.resolve(requested: 60, refreshRate: 60), (l) => l == 0 ? 60 : 48);
      expect(level, 0);
    });

    test('on a 120 Hz display aiming at 60 it settles a little above 60', () async {
      final level = await calibrate(
        FrameBudget.resolve(requested: 60, refreshRate: 120),
        (l) => switch (l) { <= 3 => 120, 4 => 70, _ => 55 },
      );
      expect(level, 4);
    });
  });

  group('FrameTimeProbe', () {
    FrameTiming at(int vsyncUs) => FrameTiming(
          vsyncStart: vsyncUs,
          buildStart: vsyncUs,
          buildFinish: vsyncUs + 1000,
          rasterStart: vsyncUs + 1000,
          rasterFinish: vsyncUs + 2000,
          rasterFinishWallTime: vsyncUs + 2000,
        );

    test('a stall counts every frame slot it swallowed', () {
      final probe = FrameTimeProbe(targetFps: 60);
      const slot = 16667;
      probe.debugAddTimings([at(0), at(slot), at(slot + 1000000)]); // a one-second stall
      expect(probe.missedFrames, inInclusiveRange(58, 60));
    });

    test('frames on time miss nothing', () {
      final probe = FrameTimeProbe(targetFps: 60);
      probe.debugAddTimings([for (var i = 0; i < 10; i++) at(i * 16667)]);
      expect(probe.missedFrames, 0);
    });
  });
}
