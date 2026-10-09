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
import 'package:litert_edge_demos/data/repositories/live_detection/live_stats_tracker.dart';
import 'package:litert_edge_demos/domain/models/live_state.dart';

const _second = 1000000; // µs

void main() {
  late LiveStatsTracker tracker;

  setUp(() {
    tracker = LiveStatsTracker(
      window: 4,
      interval: const Duration(milliseconds: 250),
    );
  });

  /// A result at [at] whose timings are all [micros], latency 2·[micros].
  void result(int at, {int micros = 1000}) => tracker.processed(
    atMicros: at,
    copyMicros: micros,
    preMicros: micros,
    runMicros: micros,
    postMicros: micros,
    latencyMicros: 2 * micros,
  );

  test('before any event: zero rates and counts, no timings', () {
    final stats = tracker.current();
    expect(
      (
        stats.fps,
        stats.sourceFps,
        stats.sourceWidth,
        stats.sourceHeight,
        stats.processed,
        stats.sourceFrames,
        stats.droppedBusy,
        stats.droppedPaused,
        stats.droppedRate,
      ),
      (0.0, 0.0, 0, 0, 0, 0, 0, 0, 0),
    );
    expect([
      stats.copyMs,
      stats.preMs,
      stats.runMs,
      stats.postMs,
      stats.latencyMs,
    ], everyElement(isNull));
  });

  group('publishDue', () {
    test('is due the first time, then once the interval has passed, counted '
        'from the last time it was due', () {
      expect(tracker.publishDue(1000), isNotNull);
      expect(tracker.publishDue(1000 + 249999), isNull);
      expect(tracker.publishDue(1000 + 250000), isNotNull, reason: 'at 250 ms');
      expect(tracker.publishDue(1000 + 499999), isNull);
      expect(tracker.publishDue(1000 + 600000), isNotNull);
      expect(tracker.publishDue(1000 + 849999), isNull);
    });

    test('returns the current figures', () {
      tracker
        ..sourceFrame(0, width: 640, height: 480)
        ..dropped(FrameDrop.rate);
      final stats = tracker.publishDue(0)!;
      expect((stats.sourceFrames, stats.droppedRate), (1, 1));
    });

    test('a zero interval is always due', () {
      final always = LiveStatsTracker(window: 4, interval: Duration.zero);
      expect(always.publishDue(5), isNotNull);
      expect(always.publishDue(5), isNotNull);
      expect(always.publishDue(6), isNotNull);
    });

    test('a reset makes it due again at once', () {
      expect(tracker.publishDue(0), isNotNull);
      tracker.reset();
      expect(tracker.publishDue(1), isNotNull);
    });

    test('clearing the samples does not', () {
      expect(tracker.publishDue(0), isNotNull);
      tracker.clearSamples();
      expect(tracker.publishDue(1), isNull);
    });
  });

  group('fps', () {
    test('needs two results that are apart in time', () {
      result(0);
      expect(tracker.current().fps, 0, reason: 'one result');
      result(0);
      expect(tracker.current().fps, 0, reason: 'two at the same time');
      result(_second ~/ 10);
      expect(tracker.current().fps, closeTo(20, 1e-9), reason: '2 in 100 ms');
    });

    test('counts the last window results only', () {
      // Slow at first, then four results 50 ms apart.
      for (final at in [0, 500000, 1000000, 1050000, 1100000, 1150000]) {
        result(at);
      }
      expect(tracker.current().fps, closeTo(20, 1e-9));
    });
  });

  group('p50 timings', () {
    test('are in ms; of an odd count the middle one, of an even count the '
        'upper middle one', () {
      result(0, micros: 3000);
      result(1, micros: 1000);
      result(2, micros: 2000);
      expect(tracker.current().preMs, 2.0, reason: 'of 1, 2, 3');
      result(3, micros: 4000);
      expect(tracker.current().preMs, 3.0, reason: 'of 1, 2, 3, 4');
    });

    test('each figure from its own field', () {
      tracker.processed(
        atMicros: 0,
        copyMicros: 100,
        preMicros: 1000,
        runMicros: 4000,
        postMicros: 1500,
        latencyMicros: 9000,
      );
      final stats = tracker.current();
      expect(
        (stats.copyMs, stats.preMs, stats.runMs, stats.postMs, stats.latencyMs),
        (0.1, 1.0, 4.0, 1.5, 9.0),
      );
    });

    test('over the last window results only', () {
      for (var i = 1; i <= 6; i++) {
        result(i, micros: i * 1000);
      }
      // 3, 4, 5, 6 ms: the upper middle one is 5.
      expect(tracker.current().preMs, 5.0);
      expect(tracker.current().latencyMs, 10.0);
    });
  });

  test('source frames: the rate over the last window frames, the newest '
      'size, and every frame counted', () {
    tracker
      ..sourceFrame(0, width: 1280, height: 720)
      ..sourceFrame(_second, width: 1280, height: 720);
    for (var i = 0; i < 4; i++) {
      tracker.sourceFrame(2 * _second + i * 20000, width: 640, height: 480);
    }
    final stats = tracker.current();
    expect(stats.sourceFps, closeTo(50, 1e-9), reason: 'one per 20 ms');
    expect((stats.sourceWidth, stats.sourceHeight), (640, 480));
    expect(stats.sourceFrames, 6);
    expect(stats.fps, 0, reason: 'source frames are not results');
  });

  test('drops are counted by reason', () {
    tracker
      ..dropped(FrameDrop.busy)
      ..dropped(FrameDrop.paused)
      ..dropped(FrameDrop.paused)
      ..dropped(FrameDrop.rate)
      ..dropped(FrameDrop.rate)
      ..dropped(FrameDrop.rate);
    final stats = tracker.current();
    expect(
      (stats.droppedBusy, stats.droppedPaused, stats.droppedRate),
      (1, 2, 3),
    );
  });

  test('clearSamples forgets the results window only', () {
    result(0);
    result(50000);
    tracker
      ..sourceFrame(0, width: 640, height: 480)
      ..sourceFrame(50000, width: 640, height: 480)
      ..dropped(FrameDrop.busy)
      ..clearSamples();
    final stats = tracker.current();
    expect(stats.fps, 0);
    expect(stats.preMs, isNull);
    expect(stats.processed, 2);
    expect(stats.sourceFps, closeTo(20, 1e-9));
    expect((stats.sourceFrames, stats.droppedBusy), (2, 1));
  });

  test('reset returns to the figures before any event', () {
    result(0);
    result(50000);
    tracker
      ..sourceFrame(0, width: 640, height: 480)
      ..sourceFrame(50000, width: 640, height: 480)
      ..dropped(FrameDrop.busy)
      ..dropped(FrameDrop.paused)
      ..dropped(FrameDrop.rate)
      ..reset();
    final stats = tracker.current();
    const empty = LiveStats();
    expect(
      (
        stats.fps,
        stats.sourceFps,
        stats.sourceWidth,
        stats.sourceHeight,
        stats.processed,
        stats.sourceFrames,
        stats.droppedBusy,
        stats.droppedPaused,
        stats.droppedRate,
        stats.preMs,
      ),
      (
        empty.fps,
        empty.sourceFps,
        empty.sourceWidth,
        empty.sourceHeight,
        empty.processed,
        empty.sourceFrames,
        empty.droppedBusy,
        empty.droppedPaused,
        empty.droppedRate,
        empty.preMs,
      ),
    );
  });
}
