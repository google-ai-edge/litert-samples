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

import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/domain/models/accelerator_evidence.dart';
import 'package:litert_edge_demos/domain/models/detection.dart';
import 'package:litert_edge_demos/domain/models/hardware_profile.dart';
import 'package:litert_edge_demos/selftest/evidence_judge.dart';
import 'package:litert_edge_demos/selftest/step_recorder.dart';
import 'package:litert_edge_demos/utils/pcm.dart';

import '../fakes/fake_self_test_adapters.dart';

const _strict = EvidenceJudge(allowSoftwareGpu: false);
const _allowing = EvidenceJudge(allowSoftwareGpu: true);

AcceleratorEvidence _gpu({
  String actual = 'gpu',
  String? adapter = 'Tesla T4 (Discrete GPU)',
  bool software = false,
}) => AcceleratorEvidence(
  requested: 'gpu',
  actual: actual,
  backendSource: EvidenceSource.api,
  adapter: adapter,
  adapterSource: EvidenceSource.log,
  softwareGpu: software,
);

PcmStats _stats(double peak, {int samples = 16000, bool allZero = false}) =>
    PcmStats(samples: samples, allZero: allZero, peakFrameDbfs: peak);

DetectionFrame _withRun(int frameId, int runMicros) =>
    fakeGoldenCatsFrame(frameId, runMicros: runMicros);

void main() {
  group('accelerator', () {
    test('a named hardware GPU passes clean', () {
      final v = _strict.accelerator(_gpu(), platform: HostPlatform.linux);
      expect(v.status, StepStatus.pass);
      expect(v.errors, isEmpty);
      expect(v.softwareGpuAllowed, isFalse);
    });

    test('a confirmed mismatch fails, whatever --allow-software-gpu '
        'says', () {
      for (final judge in [_strict, _allowing]) {
        final v = judge.accelerator(
          _gpu(actual: 'cpu', adapter: null),
          platform: HostPlatform.macos,
        );
        expect(v.status, StepStatus.fail);
        expect(v.errors, ['requested gpu but runs on cpu']);
        expect(v.softwareGpuAllowed, isFalse);
      }
    });

    test('a mismatch that is only requested (not confirmed) is no '
        'mismatch', () {
      const e = AcceleratorEvidence(
        requested: 'gpu',
        actual: 'cpu',
        backendSource: EvidenceSource.requested,
      );
      expect(e.mismatch, isFalse);
      expect(
        _strict.accelerator(e, platform: HostPlatform.macos).status,
        StepStatus.pass,
      );
    });

    test('a software rasterizer fails; allowed, it passes labelled and '
        'says so', () {
      final e = _gpu(
        adapter: 'llvmpipe (LLVM 19.1.1, 256 bits)',
        software: true,
      );
      final strict = _strict.accelerator(e, platform: HostPlatform.linux);
      expect(strict.status, StepStatus.fail);
      expect(strict.errors, [
        'the GPU is a software rasterizer (llvmpipe (LLVM 19.1.1, 256 bits), '
            'confirmed: native log): this is the CPU, not a GPU',
      ]);
      expect(strict.softwareGpuAllowed, isFalse);

      final allowed = _allowing.accelerator(e, platform: HostPlatform.linux);
      expect(allowed.status, StepStatus.pass);
      expect(allowed.softwareGpuAllowed, isTrue);
      expect(
        allowed.errors.single,
        endsWith(
          ' (allowed by --allow-software-gpu: the GPU code path is tested, '
          'not GPU speed)',
        ),
      );
    });

    test('an unknown software adapter is named "unknown"', () {
      final v = _strict.accelerator(
        _gpu(adapter: null, software: true),
        platform: HostPlatform.macos,
      );
      expect(
        v.errors.single,
        startsWith(
          'the GPU is a software rasterizer '
          '(unknown, ',
        ),
      );
    });

    test('an adapter nothing names: a failure on Linux only; allowed, a '
        'labelled pass', () {
      final unnamed = _gpu(adapter: null);
      final linux = _strict.accelerator(unnamed, platform: HostPlatform.linux);
      expect(linux.status, StepStatus.fail);
      expect(linux.errors, [
        'cannot rule out a software GPU: no "Selected adapter" line in the '
            'native log and the probe found no hardware GPU (install '
            'vulkan-tools for vulkaninfo)',
      ]);

      final allowed = _allowing.accelerator(
        unnamed,
        platform: HostPlatform.linux,
      );
      expect(allowed.status, StepStatus.pass);
      expect(allowed.softwareGpuAllowed, isTrue);
      expect(
        allowed.errors.single,
        endsWith('(allowed by --allow-software-gpu)'),
      );

      for (final platform in [HostPlatform.macos, null]) {
        final v = _strict.accelerator(unnamed, platform: platform);
        expect(v.status, StepStatus.pass, reason: '$platform');
        expect(v.errors, isEmpty);
      }
    });

    test('the CPU needs no adapter', () {
      const cpu = AcceleratorEvidence(
        requested: 'cpu',
        actual: 'cpu',
        backendSource: EvidenceSource.api,
      );
      final v = _strict.accelerator(cpu, platform: HostPlatform.linux);
      expect(v.status, StepStatus.pass);
      expect(v.errors, isEmpty);
    });
  });

  group('cats', () {
    test('ten golden runs: identical, the golden matches, the median leaves '
        'out the first run', () {
      final frames = [
        _withRun(1, 44800),
        for (var i = 2; i <= 10; i++) _withRun(i, 1000 * i),
      ];
      final v = _strict.cats(frames, golden: true);
      expect(v.identical, isTrue);
      expect(v.golden!.passed, isTrue);
      expect(v.passed, isTrue);
      // Runs 2..10 → 2..10 ms; the middle of nine is 6 ms.
      expect(v.medianRunMs, 6.0);
    });

    test('an even count takes the upper middle', () {
      final frames = [for (var i = 1; i <= 5; i++) _withRun(i, 1000 * i)];
      expect(_strict.cats(frames, golden: true).medianRunMs, 4.0);
    });

    test('runs that differ fail, even when the first matches the golden', () {
      final frames = [
        fakeGoldenCatsFrame(1),
        fakeGoldenCatsFrame(2),
        fakeWrongFrame(3),
      ];
      final v = _strict.cats(frames, golden: true);
      expect(v.identical, isFalse);
      expect(v.golden!.passed, isTrue);
      expect(v.passed, isFalse);
    });

    test('wrong classes fail the golden', () {
      final frames = [fakeWrongFrame(1), fakeWrongFrame(1)];
      final v = _strict.cats(frames, golden: true);
      expect(v.identical, isTrue);
      expect(v.golden!.classesMatch, isFalse);
      expect(v.passed, isFalse);
    });

    test('an image without a golden: consistency only', () {
      final frames = [fakeWrongFrame(1), fakeWrongFrame(1)];
      final v = _strict.cats(frames, golden: false);
      expect(v.golden, isNull);
      expect(v.passed, isTrue);
    });
  });

  test('step 5 compares bit for bit', () {
    expect(
      _strict.sameAsReference(
        fakeGoldenCatsFrame(1000),
        fakeGoldenCatsFrame(1),
      ),
      isTrue,
    );
    final nudged = fakeGoldenCatsFrame(1000);
    final boxes = Float32List.fromList(nudged.boxes)..[0] += 1e-4;
    expect(
      _strict.sameAsReference(
        DetectionFrame(
          frameId: 1000,
          width: 640,
          height: 480,
          boxes: boxes,
          preMicros: 0,
          runMicros: 0,
          postMicros: 0,
          backend: DetectorBackend.gpu,
        ),
        fakeGoldenCatsFrame(1),
      ),
      isFalse,
    );
  });

  test('generation: at least $kSelfTestMinChunks chunks', () {
    expect(_strict.enoughChunks(kSelfTestMinChunks - 1), isFalse);
    expect(_strict.enoughChunks(kSelfTestMinChunks), isTrue);
    expect(_strict.enoughChunks(64), isTrue);
  });

  test('the sink monitor: heard from $kMonitorPassDbfs dBFS, too quiet down '
      'to $kMonitorNothingDbfs, nothing below or all zeros', () {
    expect(_strict.monitor(_stats(-9)), MonitorVerdict.heard);
    expect(_strict.monitor(_stats(kMonitorPassDbfs)), MonitorVerdict.heard);
    expect(_strict.monitor(_stats(-40.1)), MonitorVerdict.tooQuiet);
    expect(
      _strict.monitor(_stats(kMonitorNothingDbfs)),
      MonitorVerdict.tooQuiet,
    );
    expect(_strict.monitor(_stats(-80.1)), MonitorVerdict.nothing);
    expect(
      _strict.monitor(_stats(kPcmFloorDbfs, allZero: true)),
      MonitorVerdict.nothing,
    );
  });

  test('the microphone: empty, digital zeros, silent below '
      '$kMicSilenceDbfs dBFS, else heard', () {
    expect(_strict.mic(_stats(kPcmFloorDbfs, samples: 0)), MicVerdict.empty);
    expect(
      _strict.mic(_stats(kPcmFloorDbfs, allZero: true)),
      MicVerdict.digitalZeros,
    );
    expect(_strict.mic(_stats(-60.1)), MicVerdict.silent);
    expect(_strict.mic(_stats(kMicSilenceDbfs)), MicVerdict.heard);
    expect(_strict.mic(_stats(-35)), MicVerdict.heard);
  });
}
