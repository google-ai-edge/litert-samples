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
import 'package:litert_edge_demos/data/repositories/live_detection_repository.dart';
import 'package:litert_edge_demos/data/services/frames/frame_source.dart';
import 'package:litert_edge_demos/domain/models/detection.dart';
import 'package:litert_edge_demos/domain/models/frame_source_info.dart';
import 'package:litert_edge_demos/domain/models/frame_source_spec.dart';
import 'package:litert_edge_demos/domain/models/live_state.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../fakes/fake_detector.dart';
import '../../fakes/fake_frame_source.dart';
import '../../support/frames.dart';

/// The live figures the debug overlay and diagnostics show, and the
/// black-frames warning, as the repository publishes them: when they are
/// published, what each one counts, and what a resume, a stop and a restart
/// do to them. live_detection_repository_test.dart covers the pipeline
/// itself.

/// Lets queued microtasks and zero-length timers run.
Future<void> settle() => Future<void>.delayed(Duration.zero);

const _spec = FixtureSourceSpec(['/fixtures']);
const _period = 66667; // µs at 15 fps

/// A 640×480 RGBA frame of vertical 16-pixel stripes, [dark] and [light]
/// grey in turn: the luma sample (every 16th pixel) sees half of each.
FrameView _stripes({required int dark, required int light}) {
  const width = 640;
  const height = 480;
  final bytes = Uint8List(width * height * 4);
  for (var y = 0; y < height; y++) {
    for (var x = 0; x < width; x++) {
      final value = (x ~/ 16).isOdd ? light : dark;
      final i = (y * width + x) * 4;
      bytes
        ..[i] = value
        ..[i + 1] = value
        ..[i + 2] = value
        ..[i + 3] = 255;
    }
  }
  return TestFrame(width, height, FramePixelFormat.rgba8888, 0, [
    FramePlane(bytes: bytes, bytesPerRow: width * 4, bytesPerPixel: 4),
  ]);
}

void main() {
  late FakeDetector detector;
  late FakeFrameSource source;
  late int clock; // µs
  final owner = Object();

  setUp(() {
    detector = FakeDetector();
    clock = 0;
  });

  LiveDetectionRepository build({
    Duration statsInterval = Duration.zero,
    int statsWindow = 30,
  }) {
    final repo = LiveDetectionRepository(
      detector: detector,
      createSource: (_) => Result.ok(source = FakeFrameSource()),
      statsInterval: statsInterval,
      statsWindow: statsWindow,
      clockMicros: () => clock,
    );
    addTearDown(repo.close);
    return repo;
  }

  Future<void> start(LiveDetectionRepository repo) async {
    expect(await repo.start(_spec, owner: owner), isA<Ok<FrameSourceInfo>>());
  }

  /// [n] frames, one per 15 fps period, each detected at once.
  Future<void> emitDetected(int n) async {
    for (var i = 0; i < n; i++) {
      clock += _period;
      source.emit();
      await settle();
    }
  }

  test('stats are published on the first frame, then on the first frame at '
      'least the interval after the last publication', () async {
    detector.autoComplete = true;
    final repo = build(statsInterval: const Duration(milliseconds: 250));
    await start(repo);
    final published = <int>[];
    repo.stats.addListener(() => published.add(repo.stats.value.processed));

    await emitDetected(15);

    // Frames every 66.7 ms: published at 66.7, 333.3, 600 and 866.7 ms.
    expect(published, [1, 5, 9, 13]);
  });

  test('a dropped frame publishes too, at the same cadence; every source '
      'frame is counted', () async {
    final repo = build(statsInterval: const Duration(milliseconds: 250));
    repo.setDuty(DetectorDuty.paused, reason: 'Gemma');
    await start(repo);
    final published = <(int, int)>[];
    repo.stats.addListener(
      () => published.add((
        repo.stats.value.sourceFrames,
        repo.stats.value.droppedPaused,
      )),
    );

    for (var i = 0; i < 10; i++) {
      clock += 100000;
      source.emit();
    }

    expect(published, [(1, 1), (4, 4), (7, 7), (10, 10)]);
    expect(detector.calls, isEmpty);
  });

  test(
    "each figure has its source: pre, run and post from the detector's "
    'frame, latency from the send to the result, fps from the results and '
    'the source rate and size from the source frames, over statsWindow '
    '(medians and windows themselves: live_stats_tracker_test.dart)',
    () async {
      detector.results = (frame) => DetectionFrame(
        frameId: frame.frameId,
        width: frame.width,
        height: frame.height,
        boxes: Float32List(0),
        preMicros: 2000,
        runMicros: 7000,
        postMicros: 500,
        backend: DetectorBackend.gpu,
      );
      final repo = build(statsWindow: 2);
      await start(repo);
      final sent = <int>[];
      final returned = <int>[];

      /// One frame whose result comes back [latency] µs after the send.
      Future<void> frame(int latency) async {
        clock += _period;
        sent.add(clock);
        source.emit();
        clock += latency;
        returned.add(clock);
        detector.complete();
        await settle();
      }

      await frame(3000);
      var stats = repo.stats.value;
      expect((stats.processed, stats.sourceFrames), (1, 1));
      expect((stats.preMs, stats.runMs, stats.postMs), (2.0, 7.0, 0.5));
      expect(stats.latencyMs, 3.0);
      expect(stats.copyMs, 0.0, reason: 'the fake clock stands still');
      expect((stats.sourceWidth, stats.sourceHeight), (640, 480));

      // Uneven latencies: results and sends are spaced differently, and a
      // window of three would differ from one of two.
      await frame(9000);
      await frame(1000);
      stats = repo.stats.value;
      expect(stats.fps, 1e6 / (returned[2] - returned[1]));
      expect(stats.sourceFps, 1e6 / (sent[2] - sent[1]));
    },
  );

  test('a resume restarts the fps window; the source rate and the counters '
      'carry on', () async {
    detector.autoComplete = true;
    final repo = build();
    await start(repo);
    await emitDetected(5);
    expect(repo.stats.value.fps, closeTo(15, 0.01));

    repo.setDuty(DetectorDuty.paused, reason: 'Gemma');
    for (var i = 0; i < 3; i++) {
      clock += _period;
      source.emit();
    }
    repo.setDuty(DetectorDuty.live);
    await emitDetected(1);

    final stats = repo.stats.value;
    expect(stats.fps, 0, reason: 'one frame since the resume');
    expect(stats.processed, 6);
    expect(stats.sourceFrames, 9);
    expect(stats.droppedPaused, 3);
    expect(stats.sourceFps, closeTo(1e6 / _period, 1e-6));
    expect(stats.preMs, 1.0);
  });

  test(
    'a stop keeps the last figures; a restart starts them from zero',
    () async {
      detector.autoComplete = true;
      final repo = build();
      await start(repo);
      await emitDetected(5);

      await repo.stop(owner: owner);
      expect(repo.stats.value.processed, 5);

      await start(repo);
      final reset = repo.stats.value;
      expect(
        (
          reset.processed,
          reset.sourceFrames,
          reset.droppedBusy,
          reset.droppedPaused,
          reset.droppedRate,
          reset.fps,
          reset.sourceFps,
          reset.sourceWidth,
          reset.preMs,
          reset.latencyMs,
        ),
        (0, 0, 0, 0, 0, 0.0, 0.0, 0, null, null),
      );

      await emitDetected(1);
      final stats = repo.stats.value;
      expect(
        (stats.processed, stats.sourceFrames, stats.fps, stats.sourceFps),
        (1, 1, 0.0, 0.0),
      );
      expect((stats.sourceWidth, stats.sourceHeight), (640, 480));
    },
  );

  test('a dark but textured scene (a dim room) is not black frames: only a '
      'flat one is', () async {
    detector.autoComplete = true;
    final repo = build();
    await start(repo);
    // Sampled luma 0 and 40 in turn: mean 20 (below 24), spread 20.
    final dim = _stripes(dark: 0, light: 40);

    for (var i = 0; i < 60; i++) {
      clock += _period;
      source.emit(dim);
      await settle();
    }

    expect(repo.luma, closeTo(20, 1e-9));
    expect(repo.blackFrames.value, isFalse, reason: '4 s of dim frames');

    final flat = _stripes(dark: 20, light: 20);
    for (var i = 0; i < 60; i++) {
      clock += _period;
      source.emit(flat);
      await settle();
    }
    expect(repo.luma, closeTo(20, 1e-9));
    expect(repo.blackFrames.value, isTrue, reason: '4 s of flat dark frames');
  });
}
