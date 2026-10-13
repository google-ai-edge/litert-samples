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
import 'package:litert_edge_demos/config/live_camera_config.dart';
import 'package:litert_edge_demos/data/repositories/live_detection/black_frame_detector.dart';

import '../../../support/frames.dart';

const _ms = 1000; // µs

void main() {
  late BlackFrameDetector detector;

  setUp(() {
    detector = BlackFrameDetector(
      sampleEvery: 3,
      darkLuma: 24,
      flatSpread: 6,
      after: const Duration(milliseconds: 2000),
    );
  });

  BlackFrameChange? dark(int atMs) =>
      detector.sample(mean: 1, spread: 0, nowMicros: atMs * _ms);

  test('the defaults are the app config', () {
    final defaults = BlackFrameDetector();
    for (var i = 0; i < kLumaSampleEvery; i++) {
      defaults.frameSent(TestFrame.rgba(fill: i == 0 ? 1 : 200), 0);
    }
    expect(defaults.luma, closeTo(1, 1e-9), reason: 'one sample per 15');
    expect(
      defaults.sample(mean: kBlackFrameLuma - 0.1, spread: 0, nowMicros: 0),
      isNull,
    );
    final started = defaults.sample(
      mean: kBlackFrameLuma - 0.1,
      spread: kBlackFrameSpread - 0.1,
      nowMicros: kBlackFramesAfter.inMicroseconds,
    );
    expect(started, isA<BlackFramesStarted>());
  });

  group('sample', () {
    test('dark and flat for the whole delay turns the warning on, once, with '
        'how long it has been dark; not a moment before', () {
      expect(dark(0), isNull, reason: 'auto-exposure warm-up');
      expect(dark(1999), isNull);
      expect(detector.black, isFalse);

      final started = dark(2000);
      expect(
        started,
        isA<BlackFramesStarted>()
            .having((c) => c.luma, 'luma', 1.0)
            .having((c) => c.spread, 'spread', 0.0)
            .having((c) => c.darkMicros, 'darkMicros', 2000 * _ms),
      );
      expect(detector.black, isTrue);
      expect(dark(2500), isNull, reason: 'already on');
      expect(detector.black, isTrue);
    });

    test('the dark run is counted from its first sample', () {
      dark(500);
      expect(dark(2499), isNull);
      expect(
        dark(2600),
        isA<BlackFramesStarted>().having(
          (c) => c.darkMicros,
          'darkMicros',
          2100 * _ms,
        ),
      );
    });

    test('a bright sample turns it off, once, and restarts the dark run', () {
      dark(0);
      dark(2000);
      final ended = detector.sample(
        mean: 128,
        spread: 0,
        nowMicros: 2100 * _ms,
      );
      expect(
        ended,
        isA<BlackFramesEnded>().having((c) => c.luma, 'luma', 128.0),
      );
      expect(detector.black, isFalse);
      expect(
        detector.sample(mean: 128, spread: 0, nowMicros: 2200 * _ms),
        isNull,
        reason: 'already off',
      );

      expect(dark(2300), isNull);
      expect(dark(4299), isNull, reason: 'a new dark run');
      expect(dark(4300), isA<BlackFramesStarted>());
    });

    test('a textured sample (a dim room) is not black, and interrupts a dark '
        'run', () {
      expect(detector.sample(mean: 20, spread: 6, nowMicros: 0), isNull);
      dark(100);
      detector.sample(mean: 20, spread: 6, nowMicros: 1000 * _ms);
      expect(dark(2100), isNull, reason: 'the run restarted at 2.1 s');
      expect(detector.black, isFalse);
    });

    test('the thresholds are exclusive: luma at the limit is bright, spread '
        'at the limit is textured', () {
      detector.sample(mean: 24, spread: 0, nowMicros: 0);
      detector.sample(mean: 24, spread: 0, nowMicros: 3000 * _ms);
      expect(detector.black, isFalse);
      detector.sample(mean: 1, spread: 6, nowMicros: 3100 * _ms);
      detector.sample(mean: 1, spread: 6, nowMicros: 6000 * _ms);
      expect(detector.black, isFalse);
      detector.sample(mean: 23.9, spread: 5.9, nowMicros: 6100 * _ms);
      expect(
        detector.sample(mean: 23.9, spread: 5.9, nowMicros: 8100 * _ms),
        isA<BlackFramesStarted>(),
      );
    });

    test('luma is the newest sample, bright or dark', () {
      expect(detector.luma, isNull);
      detector.sample(mean: 80.5, spread: 30, nowMicros: 0);
      expect(detector.luma, 80.5);
      dark(10);
      expect(detector.luma, 1.0);
    });
  });

  group('frameSent', () {
    test('samples the first frame and every sampleEvery-th after it', () {
      final fills = [10, 20, 30, 40, 50, 60, 70];
      final seen = <double?>[];
      for (final (i, fill) in fills.indexed) {
        detector.frameSent(TestFrame.rgba(fill: fill), i * _ms);
        seen.add(detector.luma);
      }
      expect(seen, [10, 10, 10, 40, 40, 40, 70]);
    });

    test('the sampled frames drive the warning', () {
      final black = TestFrame.rgba();
      final changes = <BlackFrameChange>[];
      // One frame every 100 ms: samples at 0, 300, 600 ... ms.
      for (var i = 0; i < 30; i++) {
        if (detector.frameSent(black, i * 100 * _ms) case final change?) {
          changes.add(change);
        }
      }
      expect(changes, [
        isA<BlackFramesStarted>()
            .having((c) => c.luma, 'luma', 0.0)
            .having((c) => c.darkMicros, 'darkMicros', 2100 * _ms),
      ]);
    });
  });

  test('reset forgets the samples, the dark run and the warning; the next '
      'frame is sampled', () {
    dark(0);
    dark(2000);
    detector.frameSent(TestFrame.rgba(), 2100 * _ms); // the 1st: sampled
    detector.frameSent(TestFrame.rgba(), 2200 * _ms); // the 2nd: not
    expect(detector.black, isTrue);

    detector.reset();
    expect(detector.black, isFalse);
    expect(detector.luma, isNull);

    detector.frameSent(TestFrame.rgba(fill: 99), 2300 * _ms);
    expect(detector.luma, closeTo(99, 1e-9), reason: 'sampled after a reset');
    expect(dark(2400), isNull);
    expect(dark(4399), isNull, reason: 'a new dark run from 2.4 s');
    expect(dark(4400), isA<BlackFramesStarted>());
  });
}
