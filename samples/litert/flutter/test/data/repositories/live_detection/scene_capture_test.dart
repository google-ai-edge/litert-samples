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
import 'dart:ui' as ui;

import 'package:fake_async/fake_async.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/live_camera_config.dart';
import 'package:litert_edge_demos/data/repositories/live_detection/scene_capture.dart';
import 'package:litert_edge_demos/domain/models/detection.dart';
import 'package:litert_edge_demos/domain/models/frame_source_info.dart';
import 'package:litert_edge_demos/domain/models/scene_snapshot.dart';
import 'package:litert_edge_demos/utils/result.dart';

const _cat = 15;
const _dog = 16;
const _ms = 1000; // µs

/// Frame [id] with one box per entry of [boxes]: (class id, score).
DetectionFrame _frame(int id, [List<(int, double)> boxes = const []]) =>
    DetectionFrame(
      frameId: id,
      width: 4,
      height: 2,
      boxes: Float32List.fromList([
        for (final (cls, score) in boxes) ...[
          0,
          0,
          1,
          1,
          score,
          cls.toDouble(),
        ],
      ]),
      preMicros: 0,
      runMicros: 0,
      postMicros: 0,
      backend: DetectorBackend.gpu,
    );

/// [width]×[height] RGBA, left half red, right half blue.
RgbaPixels _pixels({int width = 4, int height = 2}) {
  final bytes = Uint8List(width * height * 4);
  for (var i = 0; i < width * height; i++) {
    final left = i % width < width ~/ 2;
    bytes
      ..[i * 4] = left ? 255 : 0
      ..[i * 4 + 2] = left ? 0 : 255
      ..[i * 4 + 3] = 255;
  }
  return RgbaPixels(width: width, height: height, bytes: bytes);
}

const _mirroring = FrameSourceInfo(
  label: 'desktop',
  width: 4,
  height: 2,
  format: FramePixelFormat.rgba8888,
  mirrored: true,
  previewMirrored: true,
);

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  late bool starting;
  late int? session;
  late int clock;

  SceneCapture build({
    Duration timeout = const Duration(milliseconds: 50),
    int llmImageMaxSide = 1024,
  }) => SceneCapture(
    isStarting: () => starting,
    runningSession: () => session,
    clockMicros: () => clock,
    timeout: timeout,
    countScore: 0.4,
    summaryMaxAge: const Duration(milliseconds: 500),
    llmImageMaxSide: llmImageMaxSide,
  );

  setUp(() {
    starting = false;
    session = 1;
    clock = 0;
  });

  Matcher unavailable(CaptureFailure kind, [Object? message]) =>
      isA<Error<SceneSnapshot>>().having(
        (r) => r.error,
        'error',
        isA<CaptureUnavailableException>()
            .having((e) => e.kind, 'kind', kind)
            .having((e) => e.message, 'message', message ?? anything),
      );

  /// Captures one frame of [scene] (the next frame id, 7) at [clock].
  Future<SceneSnapshot> captureOne(
    SceneCapture scene, {
    FrameSourceInfo? source,
  }) async {
    final capturing = scene.capture();
    scene.claimFrame(7);
    final frame = _frame(7, [(_cat, 0.9)]);
    scene
      ..record(frame, atMicros: clock)
      ..complete(frame, _pixels(), source: source);
    return ((await capturing) as Ok<SceneSnapshot>).value;
  }

  group('the summary window', () {
    test('keeps the last kSummaryWindow frames, oldest first', () {
      final scene = build();
      for (var id = 1; id <= kSummaryWindow + 2; id++) {
        scene.record(_frame(id), atMicros: id);
      }

      expect(scene.recent.map((f) => f.frameId), [
        for (var id = 3; id <= kSummaryWindow + 2; id++) id,
      ]);
      expect(() => scene.recent.add(_frame(0)), throwsUnsupportedError);
    });

    test('clearWindow empties it', () {
      final scene = build()..record(_frame(1), atMicros: 0);

      scene.clearWindow();

      expect(scene.recent, isEmpty);
    });
  });

  group('capture', () {
    test('without a running source it fails at once', () async {
      session = null;
      final scene = build();

      expect(
        await scene.capture(),
        unavailable(CaptureFailure.notRunning, "The camera isn't running"),
      );
      expect(scene.wantsFrame(1), isFalse);
    });

    test('asks for the next frame of the running session until one is '
        'claimed', () {
      final scene = build();
      expect(scene.wantsFrame(1), isFalse, reason: 'nothing pending');

      unawaited(scene.capture());

      expect(scene.wantsFrame(1), isTrue);
      expect(scene.wantsFrame(2), isFalse, reason: 'another session');
      scene.claimFrame(7);
      expect(scene.wantsFrame(1), isFalse, reason: 'claimed');
      scene.fail('done');
    });

    test('the claimed frame completes it: its detections and pixels, the '
        'summary of the fresh window frames scored at countScore, the '
        "source's mirroring and the latency from the request", () async {
      final scene = build()
        // Too old (600 ms before the result), then two fresh frames: one
        // with a dog under countScore.
        ..record(_frame(1, [(_dog, 0.9), (_dog, 0.9)]), atMicros: 0)
        ..record(_frame(2, [(_cat, 0.9), (_dog, 0.3)]), atMicros: 400 * _ms)
        ..record(_frame(3, [(_cat, 0.9)]), atMicros: 500 * _ms);
      clock = 520 * _ms;
      final capturing = scene.capture();
      scene.claimFrame(4);
      clock = 600 * _ms;
      final frame = _frame(4, [(_cat, 0.9), (_cat, 0.8)]);
      final pixels = _pixels();
      scene
        ..record(frame, atMicros: clock)
        ..complete(frame, pixels, source: _mirroring);

      final snapshot = ((await capturing) as Ok<SceneSnapshot>).value;
      expect(snapshot.frameId, 4);
      expect(snapshot.detections, same(frame));
      expect(snapshot.pixels, same(pixels));
      expect(snapshot.summary.frames, 3, reason: 'frames 2 to 4');
      expect(snapshot.summary.counts, {_cat: 1}, reason: 'medians: 1, 0, 2');
      expect(snapshot.mirrored, isTrue);
      expect(snapshot.previewMirrored, isTrue);
      expect(snapshot.latency, const Duration(milliseconds: 80));
      expect(scene.wantsFrame(1), isFalse, reason: 'nothing pending');
    });

    test('without source info the snapshot is not mirrored', () async {
      final snapshot = await captureOne(build());

      expect(snapshot.mirrored, isFalse);
      expect(snapshot.previewMirrored, isFalse);
    });

    test('another frame does not complete it', () async {
      final scene = build();
      var done = false;
      final capturing = scene.capture();
      unawaited(capturing.then((_) => done = true));
      scene
        ..claimFrame(7)
        ..complete(_frame(6), _pixels(), source: null);
      await pumpEventQueue();
      expect(done, isFalse);

      scene.complete(_frame(7), _pixels(), source: null);
      expect(await capturing, isA<Ok<SceneSnapshot>>());
    });

    test(
      'captures while one is pending share it; the next one is new',
      () async {
        final scene = build();
        final a = scene.capture();
        final b = scene.capture();
        scene.claimFrame(7);
        scene.complete(_frame(7), _pixels(), source: null);

        final first = ((await a) as Ok<SceneSnapshot>).value;
        expect(((await b) as Ok<SceneSnapshot>).value, same(first));
        unawaited(scene.capture());
        expect(scene.wantsFrame(1), isTrue, reason: 'a new request');
        scene.fail('done');
      },
    );

    test('a capture in a new session is a new request; the old one runs out '
        'its timeout', () {
      fakeAsync((async) {
        final scene = build();
        Result<SceneSnapshot>? old;
        unawaited(scene.capture().then((r) => old = r));
        session = 2;
        Result<SceneSnapshot>? fresh;
        unawaited(scene.capture().then((r) => fresh = r));

        expect(scene.wantsFrame(2), isTrue);
        expect(scene.wantsFrame(1), isFalse);
        async.elapse(const Duration(milliseconds: 50));

        expect(old, unavailable(CaptureFailure.timedOut));
        expect(fresh, unavailable(CaptureFailure.timedOut));
      });
    });

    test('no frame within the timeout: timed out, saying how long', () {
      fakeAsync((async) {
        final scene = build();
        Result<SceneSnapshot>? result;
        unawaited(scene.capture().then((r) => result = r));

        async.elapse(const Duration(milliseconds: 49));
        expect(result, isNull);
        async.elapse(const Duration(milliseconds: 1));

        expect(
          result,
          unavailable(
            CaptureFailure.timedOut,
            'No camera frame was detected within 50 ms',
          ),
        );
        expect(scene.wantsFrame(1), isFalse);
      });
    });

    test('fail ends the pending capture with its message and kind, and stops '
        'its timer; with nothing pending it does nothing', () {
      fakeAsync((async) {
        final scene = build();
        Result<SceneSnapshot>? result;
        unawaited(scene.capture().then((r) => result = r));

        scene.fail('Detector failed: GPU lost', kind: CaptureFailure.failed);
        async.flushMicrotasks();

        expect(
          result,
          unavailable(CaptureFailure.failed, 'Detector failed: GPU lost'),
        );
        expect(scene.wantsFrame(1), isFalse);
        scene.fail('again');
        async.elapse(const Duration(seconds: 1));
        expect(async.pendingTimers, isEmpty);
      });
    });
  });

  group('while the source starts', () {
    test('a capture waits for the start, then asks for a frame of the '
        'running session', () {
      fakeAsync((async) {
        starting = true;
        session = null;
        final scene = build();
        unawaited(scene.capture());
        async.flushMicrotasks();
        expect(scene.wantsFrame(3), isFalse);

        starting = false;
        session = 3;
        scene.releaseStartWaiters();
        async.flushMicrotasks();

        expect(scene.wantsFrame(3), isTrue);
        scene.fail('done');
      });
    });

    test('a start that ends without a source fails it as not running', () {
      fakeAsync((async) {
        starting = true;
        session = null;
        final scene = build();
        Result<SceneSnapshot>? result;
        unawaited(scene.capture().then((r) => result = r));

        starting = false;
        scene.releaseStartWaiters();
        async.flushMicrotasks();

        expect(result, unavailable(CaptureFailure.notRunning));
      });
    });

    test('a start that outlasts the timeout fails it as not running', () {
      fakeAsync((async) {
        starting = true;
        session = null;
        final scene = build();
        Result<SceneSnapshot>? result;
        unawaited(scene.capture().then((r) => result = r));

        async.elapse(const Duration(milliseconds: 50));

        expect(result, unavailable(CaptureFailure.notRunning));
        scene.releaseStartWaiters(); // nothing waits any more
      });
    });
  });

  group('encodings', () {
    test('the PNG for Gemma: at most llmImageMaxSide, mirrored back for a '
        'mirroring source, with its frame id', () async {
      final scene = build(llmImageMaxSide: 2);
      final snapshot = await captureOne(scene, source: _mirroring);

      final png =
          ((await scene.encodeForLlm(snapshot)) as Ok<EncodedSnapshot>).value;

      expect(png.frameId, 7);
      expect((png.width, png.height), (2, 1));
      expect(png.unmirrored, isTrue);
      final codec = await ui.instantiateImageCodec(png.png);
      final image = (await codec.getNextFrame()).image;
      addTearDown(image.dispose);
      addTearDown(codec.dispose);
      final rgba = (await image.toByteData())!;
      expect(
        [
          rgba.getUint8(0),
          rgba.getUint8(2),
          rgba.getUint8(4),
          rgba.getUint8(6),
        ],
        [0, 255, 255, 0],
        reason: 'blue left, red right: flipped',
      );
    });

    test('a frame that does not mirror is sent as it is', () async {
      final scene = build();
      final snapshot = await captureOne(scene);

      final png =
          ((await scene.encodeForLlm(snapshot)) as Ok<EncodedSnapshot>).value;

      expect((png.width, png.height), (4, 2));
      expect(png.unmirrored, isFalse);
    });

    test('the snapshot image is the frame at full size', () async {
      final scene = build(llmImageMaxSide: 2);
      final snapshot = await captureOne(scene);

      final image =
          ((await scene.snapshotImage(snapshot)) as Ok<ui.Image>).value;
      addTearDown(image.dispose);

      expect((image.width, image.height), (4, 2));
    });
  });
}
