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

import 'package:flutter_litert/native.dart' show Accelerator;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/services/detector/detector_engine.dart';
import 'package:litert_edge_demos/domain/models/detection.dart';
import 'package:litert_edge_demos/domain/models/detector_spec.dart';

import '../../../fakes/fake_detector_runtime.dart';
import '../../../support/frames.dart';

const _fileBytes = 64;

DetectorEngine load(
  DetectorRuntime runtime, {
  DetectorBackend backend = DetectorBackend.gpu,
  int bytes = _fileBytes,
  List<String>? logs,
}) => DetectorEngine.load(
  modelBytes: Uint8List(bytes),
  backend: backend,
  runtime: runtime,
  expectedBytes: _fileBytes,
  log: (m) => logs?.add(m),
);

Matcher loadError(String fragment) => throwsA(
  isA<DetectorLoadException>().having(
    (e) => e.message,
    'message',
    contains(fragment),
  ),
);

void main() {
  test('strict GPU, fully accelerated: loads once on {gpu}, warms up, and '
      'detects with boxes in frame pixels', () {
    final runtime = RecordingRuntime();

    final engine = load(runtime);

    expect(runtime.requested, [
      {Accelerator.gpu},
    ]);
    expect(engine.info.backend, DetectorBackend.gpu);
    expect(engine.info.label, 'GPU fp32 full');
    expect(engine.info.verifyAbsolute, 1.8e-3);
    expect(runtime.models.single.runs, 2, reason: 'verify + warm-up');

    final frame = engine.detect(rgbaFrame(frameId: 7));
    expect(frame.frameId, 7);
    expect((frame.width, frame.height), (640, 480));
    expect(frame.count, 1);
    expect(
      [frame.x1(0), frame.y1(0), frame.x2(0), frame.y2(0)],
      [100, 100, 300, 300],
      reason: 'letterbox padY 80 undone',
    );
    expect(frame.classId(0), kFakeClass);
    expect(frame.score(0), closeTo(kFakeScore, 1e-6));
    expect(frame.backend, DetectorBackend.gpu);
    engine.close();
    expect(runtime.models.single.closed, isTrue);
  });

  test('fail fast: a GPU model reporting partial acceleration is an error, '
      'closed, and never retried on the CPU', () {
    final runtime = RecordingRuntime(
      const FakeDetectorRuntime(fullyAccelerated: false),
    );

    expect(() => load(runtime), loadError('only partly on the GPU'));

    expect(runtime.requested, [
      {Accelerator.gpu},
    ], reason: 'no silent CPU retry');
    expect(runtime.models.single.closed, isTrue);
  });

  test(
    'fail fast: a GPU model narrowed to another accelerator is an error',
    () {
      final runtime = RecordingRuntime(
        const FakeDetectorRuntime(effectiveAccelerators: {Accelerator.cpu}),
      );

      expect(() => load(runtime), loadError('instead of strict GPU'));
      expect(runtime.models.single.closed, isTrue);
    },
  );

  test('a GPU compile failure names the explicit CPU mode and builds nothing '
      'else', () {
    final runtime = RecordingRuntime(
      const FakeDetectorRuntime(
        createError: 'LiteRtCreateCompiledModel failed with LiteRtStatus=504',
      ),
    );

    expect(
      () => load(runtime),
      loadError('GPU rejected YOLO26n: Bad state: LiteRtCreateCompiledModel'),
    );
    expect(() => load(runtime), loadError('DETECTOR_BACKEND=cpu'));
    expect(runtime.models, isEmpty);
    expect(runtime.requested.toSet(), {
      {Accelerator.gpu},
    });
  });

  test("the wrong I/O contract (Arm's [1,300,6] head) is an error", () {
    final runtime = RecordingRuntime(
      const FakeDetectorRuntime(outputBytes: 300 * 6 * 4),
    );

    expect(() => load(runtime), loadError('Wrong model I/O'));
    expect(runtime.models.single.closed, isTrue);
  });

  test('the reference the runtime compared against reaches DetectorInfo', () {
    final interpreter = load(RecordingRuntime());
    expect(interpreter.info.verifyReference, VerifyReference.interpreter);

    final liteRt = load(
      RecordingRuntime(
        const FakeDetectorRuntime(reference: VerifyReference.liteRtCpu),
      ),
    );
    expect(liteRt.info.verifyReference, VerifyReference.liteRtCpu);
    expect(liteRt.info.toString(), contains('vs LiteRT CPU'));
  });

  test('a disagreement names the reference it was measured against', () {
    final runtime = RecordingRuntime(
      const FakeDetectorRuntime(
        agrees: false,
        deviation: 12,
        reference: VerifyReference.liteRtCpu,
      ),
    );
    expect(
      () => load(runtime),
      loadError('disagrees with the CPU reference (LiteRT CPU)'),
    );
  });

  test('disagreeing with the CPU reference is an error', () {
    final runtime = RecordingRuntime(
      const FakeDetectorRuntime(agrees: false, deviation: 12),
    );

    expect(() => load(runtime), loadError('disagrees with the CPU reference'));
    expect(runtime.models.single.closed, isTrue);
  });

  test("a file of the wrong size is rejected before compiling; Arm's "
      'original is named', () {
    final runtime = RecordingRuntime();

    expect(
      () => load(runtime, bytes: kDetArmOriginalBytes),
      loadError("Arm's original"),
    );
    expect(() => load(runtime, bytes: 10), loadError('has 10 bytes'));
    expect(runtime.requested, isEmpty);
  });

  test('DETECTOR_BACKEND=cpu builds on {cpu} without the full-acceleration '
      'gate and is labelled CPU (chosen)', () {
    final runtime = RecordingRuntime(
      const FakeDetectorRuntime(fullyAccelerated: false),
    );

    final engine = load(runtime, backend: DetectorBackend.cpu);

    expect(runtime.requested, [
      {Accelerator.cpu},
    ]);
    expect(engine.info.label, 'CPU (chosen)');
    expect(engine.detect(rgbaFrame()).backend, DetectorBackend.cpu);
    engine.close();
  });

  test('a GPU output bit-identical to the CPU reference is logged as a '
      'warning, not a gate', () {
    final logs = <String>[];

    load(const FakeDetectorRuntime(deviation: 0), logs: logs).close();

    expect(logs, contains(contains('bit-identical')));
  });

  test('detect after close is an error', () {
    final engine = load(const FakeDetectorRuntime())..close();

    expect(() => engine.detect(rgbaFrame()), throwsStateError);
  });
}
