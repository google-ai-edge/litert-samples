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

import 'package:flutter/foundation.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/repositories/live_detection_repository.dart';
import 'package:litert_edge_demos/data/services/frames/frame_source.dart';
import 'package:litert_edge_demos/domain/models/frame_source_info.dart';
import 'package:litert_edge_demos/domain/models/frame_source_spec.dart';
import 'package:litert_edge_demos/domain/models/live_state.dart';
import 'package:litert_edge_demos/domain/models/scene_snapshot.dart';
import 'package:litert_edge_demos/domain/vision/coco_vocabulary.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../fakes/fake_detector.dart';
import '../../fakes/fake_frame_source.dart';
import '../../support/frames.dart';

Future<void> settle() => pumpEventQueue();

const _spec = FixtureSourceSpec(['/fixtures']);
const _period = 66667; // µs at 15 fps

/// [w]×[h] RGBA, left half red, right half blue.
TestFrame halvesFrame({int w = 640, int h = 480}) {
  final bytes = Uint8List(w * h * 4);
  for (var y = 0; y < h; y++) {
    for (var x = 0; x < w; x++) {
      final o = (y * w + x) * 4;
      final left = x < w ~/ 2;
      bytes[o] = left ? 255 : 0;
      bytes[o + 2] = left ? 0 : 255;
      bytes[o + 3] = 255;
    }
  }
  return TestFrame(w, h, FramePixelFormat.rgba8888, 0, [
    FramePlane(bytes: bytes, bytesPerRow: w * 4, bytesPerPixel: 4),
  ]);
}

Future<(int, int, int, int)> decodeCorners(Uint8List png) async {
  final codec = await ui.instantiateImageCodec(png);
  final image = (await codec.getNextFrame()).image;
  final rgba = (await image.toByteData())!;
  final w = image.width;
  final h = image.height;
  int r(int x) => rgba.getUint8(((h ~/ 2) * w + x) * 4);
  int b(int x) => rgba.getUint8(((h ~/ 2) * w + x) * 4 + 2);
  final out = (r(2), b(2), r(w - 3), b(w - 3));
  image.dispose();
  codec.dispose();
  return out;
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  late FakeDetector detector;
  late FakeFrameSource source;
  late int clock;
  late LiveDetectionRepository repo;
  final owner = Object();

  LiveDetectionRepository build({
    Duration captureTimeout = const Duration(seconds: 5),
  }) => LiveDetectionRepository(
    detector: detector,
    createSource: (spec) => Result.ok(source = FakeFrameSource()),
    statsInterval: Duration.zero,
    captureTimeout: captureTimeout,
    clockMicros: () => clock,
  );

  setUp(() {
    detector = FakeDetector();
    clock = 0;
    repo = build();
  });

  tearDown(() => repo.close());

  Future<void> start() async {
    expect(await repo.start(_spec, owner: owner), isA<Ok<FrameSourceInfo>>());
  }

  void emitAfter(int dtMicros) {
    clock += dtMicros;
    source.emit();
  }

  SceneSnapshot snapshotOf(Result<SceneSnapshot> result) => switch (result) {
    Ok(:final value) => value,
    Error(:final error) => throw StateError('capture failed: $error'),
  };

  test('without a running source it fails at once', () async {
    final result = await repo.capture();
    expect(
      (result as Error<SceneSnapshot>).error,
      isA<CaptureUnavailableException>()
          .having((e) => e.message, 'message', "The camera isn't running")
          .having((e) => e.kind, 'kind', CaptureFailure.notRunning),
    );
  });

  test('a capture while the source is still starting waits for it, then '
      'takes the first frame', () async {
    await repo.close();
    final gate = Completer<void>();
    repo = LiveDetectionRepository(
      detector: detector,
      createSource: (spec) =>
          Result.ok(source = FakeFrameSource()..startGate = gate),
      statsInterval: Duration.zero,
      captureTimeout: const Duration(seconds: 5),
      clockMicros: () => clock,
    );
    final starting = repo.start(_spec, owner: owner);
    await settle();
    expect(repo.state.value, isA<LiveStarting>());

    final capturing = repo.capture();
    await settle();
    gate.complete();
    await starting;
    await settle(); // the waiting capture registers for the next frame
    emitAfter(_period);
    detector.complete();

    expect(snapshotOf(await capturing).frameId, detector.calls.single.frameId);
  });

  test('a start that fails while a capture waits fails the capture as '
      '"not running"', () async {
    await repo.close();
    final gate = Completer<void>();
    repo = LiveDetectionRepository(
      detector: detector,
      createSource: (spec) => Result.ok(
        source = FakeFrameSource()
          ..startGate = gate
          ..startResult = const Result.error(
            FrameSourceUnavailableException('no cam'),
          ),
      ),
      statsInterval: Duration.zero,
      clockMicros: () => clock,
    );
    final starting = repo.start(_spec, owner: owner);
    await settle();
    final capturing = repo.capture();
    gate.complete();
    await starting;
    final result = await capturing;
    expect(
      (result as Error<SceneSnapshot>).error,
      isA<CaptureUnavailableException>().having(
        (e) => e.kind,
        'kind',
        CaptureFailure.notRunning,
      ),
    );
  });

  test('takes the next frame even when the rate gate would drop it, and '
      'returns its own detection plus the window summary', () async {
    await start();
    emitAfter(_period);
    detector.complete();
    await settle();
    expect(detector.calls, hasLength(1));

    final capturing = repo.capture();
    emitAfter(1000); // 1 ms later: the 15 fps gate alone would drop it
    expect(detector.calls, hasLength(2));
    clock += 9000;
    detector.complete();
    final snapshot = snapshotOf(await capturing);

    expect(snapshot.frameId, detector.calls.last.frameId);
    expect(snapshot.detections.frameId, snapshot.frameId);
    expect(snapshot.latency, const Duration(milliseconds: 10));
    expect(snapshot.summary.counts, {kCocoNames.indexOf('cat'): 1});
    expect(snapshot.summary.frames, 2);
    expect(snapshot.mirrored, isFalse);
    expect(repo.frames.value?.frameId, snapshot.frameId, reason: 'published');
  });

  test('after a long pause the summary holds only fresh frames, not the '
      'pre-pause window (a barge-in question while Gemma generated)', () async {
    var cats = 2;
    detector.results = (frame) => FakeDetector.catsFor(frame, cats: cats);
    await start();
    for (var i = 0; i < 5; i++) {
      emitAfter(_period);
      detector.complete();
      await settle();
    }
    expect(repo.recent, hasLength(5));

    repo.setDuty(DetectorDuty.paused, reason: 'Gemma');
    clock += 10 * 1000000; // ten seconds of generation; the cats left
    cats = 0;
    final capturing = repo.capture();
    emitAfter(_period);
    detector.complete();
    final snapshot = snapshotOf(await capturing);

    expect(snapshot.summary.frames, 1, reason: 'only the captured frame');
    expect(snapshot.summary.counts, isEmpty, reason: 'no stale cats');
  });

  test('frames older than the summary age are dropped even without a '
      'pause (a stalled source)', () async {
    var cats = 2;
    detector.results = (frame) => FakeDetector.catsFor(frame, cats: cats);
    await start();
    for (var i = 0; i < 4; i++) {
      emitAfter(_period);
      detector.complete();
      await settle();
    }
    clock += 2 * 1000000; // the source stalled for two seconds
    cats = 1;
    final capturing = repo.capture();
    emitAfter(_period);
    detector.complete();
    final snapshot = snapshotOf(await capturing);

    expect(snapshot.summary.frames, 1);
    expect(snapshot.summary.counts, {kCocoNames.indexOf('cat'): 1});
  });

  test(
    'the snapshot carries its own frame as upright RGBA, and its '
    'PNG keeps the frame id; a non-mirroring source is not flipped',
    () async {
      await start();
      final capturing = repo.capture();
      clock += _period;
      source.emit(halvesFrame());
      detector.complete();
      final snapshot = snapshotOf(await capturing);

      expect(detector.snapshotCalls.single, isTrue);
      expect((snapshot.pixels.width, snapshot.pixels.height), (640, 480));
      expect(snapshot.pixels.bytes.sublist(0, 4), [255, 0, 0, 255]);
      expect(snapshot.mirrored, isFalse);

      final encoded = await repo.encodeForLlm(snapshot);
      final png = (encoded as Ok<EncodedSnapshot>).value;
      expect(png.frameId, snapshot.frameId);
      expect((png.width, png.height), (640, 480));
      expect(png.unmirrored, isFalse);
      expect(await decodeCorners(png.png), (255, 0, 0, 255), reason: 'as is');
    },
  );

  test('a mirroring source (camera_desktop) is mirrored back for Gemma; '
      'a 16:9 frame becomes 1024×576', () async {
    await repo.close();
    repo = LiveDetectionRepository(
      detector: detector,
      createSource: (spec) => Result.ok(
        source = FakeFrameSource()
          ..startResult = const Result.ok(
            FrameSourceInfo(
              label: 'mirrored',
              width: 1280,
              height: 720,
              format: FramePixelFormat.rgba8888,
              mirrored: true,
              previewMirrored: true,
            ),
          ),
      ),
      statsInterval: Duration.zero,
      clockMicros: () => clock,
    );
    await start();
    final capturing = repo.capture();
    clock += _period;
    source.emit(halvesFrame(w: 1280, h: 720));
    detector.complete();
    final snapshot = snapshotOf(await capturing);
    expect(snapshot.mirrored, isTrue);
    expect(snapshot.previewMirrored, isTrue);

    final png =
        (await repo.encodeForLlm(snapshot) as Ok<EncodedSnapshot>).value;
    expect((png.width, png.height), (1024, 576));
    expect(png.unmirrored, isTrue);
    expect(png.frameId, snapshot.frameId);
    expect(await decodeCorners(png.png), (0, 255, 255, 0), reason: 'flipped');
  });

  test('live frames never pay for a snapshot', () async {
    await start();
    for (var i = 0; i < 3; i++) {
      emitAfter(_period);
      detector.complete();
      await settle();
    }
    expect(detector.snapshotCalls, [false, false, false]);
  });

  test('works while the detector is paused (Gemma generating)', () async {
    await start();
    repo.setDuty(DetectorDuty.paused, reason: 'Gemma');
    emitAfter(_period);
    expect(detector.calls, isEmpty, reason: 'paused: no live frames');

    final capturing = repo.capture();
    emitAfter(_period);
    expect(detector.calls, hasLength(1));
    detector.complete();
    expect(snapshotOf(await capturing).frameId, detector.calls.single.frameId);
    emitAfter(_period);
    expect(detector.calls, hasLength(1), reason: 'still paused afterwards');
  });

  test('waits for the frame in flight, then takes the next one', () async {
    await start();
    emitAfter(_period); // live frame in flight
    final capturing = repo.capture();
    emitAfter(_period); // dropped: the detector is busy
    expect(detector.calls, hasLength(1));
    detector.complete();
    await settle();
    emitAfter(1000);
    expect(detector.calls, hasLength(2));
    detector.complete();
    expect(snapshotOf(await capturing).frameId, detector.calls.last.frameId);
  });

  test('concurrent captures share one frame', () async {
    await start();
    final a = repo.capture();
    final b = repo.capture();
    emitAfter(_period);
    detector.complete();
    expect(snapshotOf(await a).frameId, snapshotOf(await b).frameId);
    expect(detector.calls, hasLength(1));
  });

  test('no frame within the timeout is a failure the turn can speak', () async {
    await repo.close();
    repo = build(captureTimeout: const Duration(milliseconds: 30));
    await start();
    final result = await repo.capture();
    expect(
      (result as Error<SceneSnapshot>).error,
      isA<CaptureUnavailableException>()
          .having(
            (e) => e.message,
            'message',
            contains('No camera frame was detected within 30 ms'),
          )
          .having((e) => e.kind, 'kind', CaptureFailure.timedOut),
    );
  });

  test('stopping the source fails a pending capture', () async {
    await start();
    final capturing = repo.capture();
    await repo.stop(owner: owner);
    expect(await capturing, isA<Error<SceneSnapshot>>());
  });

  test('a detector failure fails the capture with its message', () async {
    await start();
    final capturing = repo.capture();
    emitAfter(_period);
    detector.fail(Exception('GPU lost'));
    final result = await capturing;
    expect(
      (result as Error<SceneSnapshot>).error,
      isA<CaptureUnavailableException>()
          .having((e) => e.message, 'message', contains('GPU lost'))
          .having((e) => e.kind, 'kind', CaptureFailure.failed),
    );
  });

  test('a start that outlasts the capture timeout fails a waiting '
      'capture as "not running"', () async {
    await repo.close();
    final gate = Completer<void>();
    repo = LiveDetectionRepository(
      detector: detector,
      createSource: (spec) =>
          Result.ok(source = FakeFrameSource()..startGate = gate),
      statsInterval: Duration.zero,
      captureTimeout: const Duration(milliseconds: 30),
      clockMicros: () => clock,
    );
    final starting = repo.start(_spec, owner: owner);
    await settle();

    final result = await repo.capture();

    expect(repo.state.value, isA<LiveStarting>());
    expect(
      (result as Error<SceneSnapshot>).error,
      isA<CaptureUnavailableException>()
          .having((e) => e.message, 'message', "The camera isn't running")
          .having((e) => e.kind, 'kind', CaptureFailure.notRunning),
    );
    gate.complete();
    await starting;
  });

  test('close while a capture waits for the start fails it as "not '
      'running"', () async {
    await repo.close();
    final gate = Completer<void>();
    repo = LiveDetectionRepository(
      detector: detector,
      createSource: (spec) =>
          Result.ok(source = FakeFrameSource()..startGate = gate),
      statsInterval: Duration.zero,
      clockMicros: () => clock,
    );
    unawaited(repo.start(_spec, owner: owner));
    await settle();
    final capturing = repo.capture();
    await settle();

    final closing = repo.close();
    gate.complete();
    await closing;

    expect(
      ((await capturing) as Error<SceneSnapshot>).error,
      isA<CaptureUnavailableException>().having(
        (e) => e.kind,
        'kind',
        CaptureFailure.notRunning,
      ),
    );
  });

  test('after close a capture fails at once', () async {
    await start();
    await repo.close();

    final result = await repo.capture();

    expect(
      (result as Error<SceneSnapshot>).error,
      isA<CaptureUnavailableException>()
          .having((e) => e.message, 'message', "The camera isn't running")
          .having((e) => e.kind, 'kind', CaptureFailure.notRunning),
    );
  });

  test('the snapshot image is its frame at full size', () async {
    await start();
    final capturing = repo.capture();
    emitAfter(_period);
    detector.complete();
    final snapshot = snapshotOf(await capturing);

    final image = (await repo.snapshotImage(snapshot) as Ok<ui.Image>).value;
    addTearDown(image.dispose);

    expect((image.width, image.height), (640, 480));
  });

  test('the summary window is cleared by a pause and by a stop', () async {
    detector.autoComplete = true;
    await start();
    for (var i = 0; i < 3; i++) {
      emitAfter(_period);
      await settle();
    }
    expect(repo.recent, hasLength(3));

    repo.setDuty(DetectorDuty.paused, reason: 'Gemma');
    expect(repo.recent, isEmpty);

    repo.setDuty(DetectorDuty.live);
    emitAfter(_period);
    await settle();
    expect(repo.recent, hasLength(1));

    await repo.stop(owner: owner);
    expect(repo.recent, isEmpty);
  });

  test('captures, their failures and the encoding are logged', () async {
    final logs = <String>[];
    final original = debugPrint;
    debugPrint = (message, {wrapWidth}) {
      if (message case final m? when m.contains(RegExp('capture|encoded'))) {
        logs.add(m);
      }
    };
    addTearDown(() => debugPrint = original);
    await repo.close();
    repo = build(captureTimeout: const Duration(milliseconds: 30));
    await start();

    // A capture, and its PNG.
    final capturing = repo.capture();
    emitAfter(_period);
    detector.complete();
    final snapshot = snapshotOf(await capturing);
    final png =
        (await repo.encodeForLlm(snapshot) as Ok<EncodedSnapshot>).value;
    // No frame in time; the detector fails; the source stops.
    await repo.capture();
    final failing = repo.capture();
    emitAfter(_period);
    detector.fail(Exception('GPU lost'));
    await failing;
    await settle();
    expect(await repo.start(_spec, owner: owner), isA<Ok<FrameSourceInfo>>());
    final stopped = repo.capture();
    await repo.stop(owner: owner);
    await stopped;

    expect(logs, [
      '[LiveDetection] capture frame=${snapshot.frameId} latency=66ms '
          'boxes=${snapshot.detections.count} '
          'summary=${snapshot.summary.counts}',
      '[LiveDetection] encoded frame=${png.frameId} ${png.width}x'
          '${png.height} PNG ${png.png.length} B unmirrored=false in '
          '${png.encodeTime.inMilliseconds}ms',
      '[LiveDetection] capture failed (timedOut): No camera frame was '
          'detected within 30 ms',
      '[LiveDetection] capture failed (failed): Detector failed: Exception: '
          'GPU lost',
      "[LiveDetection] capture failed (notRunning): The camera isn't running",
    ]);
  });
}
