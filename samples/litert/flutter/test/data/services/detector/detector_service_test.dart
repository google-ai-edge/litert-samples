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

import 'dart:io';

import 'package:crypto/crypto.dart' show sha256;
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/services/detector/detector_service.dart';
import 'package:litert_edge_demos/data/services/detector/frame_message.dart';
import 'package:litert_edge_demos/data/services/frames/frame_source.dart';
import 'package:litert_edge_demos/domain/models/detection.dart';
import 'package:litert_edge_demos/domain/models/detector_spec.dart';
import 'package:litert_edge_demos/domain/models/frame_source_info.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../../fakes/fake_detector_runtime.dart';
import '../../../support/frames.dart';

const _fileBytes = 64;

void main() {
  late Directory dir;
  late String modelPath;

  setUp(() {
    dir = Directory.systemTemp.createTempSync('detector_service_test');
    modelPath = '${dir.path}/yolo26n_fp16_rawhead.tflite';
    File(modelPath).writeAsBytesSync(Uint8List(_fileBytes));
  });

  tearDown(() => dir.deleteSync(recursive: true));

  DetectorService service([
    FakeDetectorRuntime runtime = const FakeDetectorRuntime(),
  ]) {
    final s = DetectorService(runtime: runtime, expectedModelBytes: _fileBytes);
    addTearDown(s.close);
    return s;
  }

  test('loads in the worker isolate and detects there', () async {
    final detector = service();

    final loaded = await detector.load(
      source: DetectorFile(modelPath),
      backend: DetectorBackend.gpu,
    );

    expect(loaded, isA<Ok<DetectorInfo>>());
    expect(detector.info?.label, 'GPU fp32 full');
    final result = await detector.detect(rgbaFrame(frameId: 3));
    final frame = (result as Ok<DetectionFrame>).value;
    expect(frame.frameId, 3);
    expect(frame.count, 1);
    expect(frame.classId(0), kFakeClass);
    expect(frame.preMicros, greaterThan(0));
  });

  group('the built-in detector (an asset, no file)', () {
    test('its bytes reach the worker (moved, not copied) and it loads on '
        'the strict GPU like a file', () async {
      final detector = service();

      final loaded = await detector.load(
        source: DetectorBytes(Uint8List(_fileBytes)),
        backend: DetectorBackend.gpu,
      );

      expect(loaded, isA<Ok<DetectorInfo>>());
      expect(detector.info?.label, 'GPU fp32 full');
      final result = await detector.detect(rgbaFrame(frameId: 7));
      expect((result as Ok<DetectionFrame>).value.frameId, 7);
    });

    test('the size check applies to the asset too', () async {
      final detector = service();

      final loaded = await detector.load(
        source: DetectorBytes(Uint8List(_fileBytes + 1)),
        backend: DetectorBackend.gpu,
      );

      expect(loaded, isA<Error<DetectorInfo>>());
      expect(loaded.toString(), contains('${_fileBytes + 1}'));
      expect(detector.isLoaded, isFalse);
    });

    test('loadBundledDetector reads the asset key from the bundle', () async {
      final bundle = _FakeBundle({
        'assets/models/yolo26n_fp16_rawhead.tflite': Uint8List.fromList([
          1,
          2,
          3,
        ]),
      });

      final bytes = await loadBundledDetector(bundle);

      expect(bytes, [1, 2, 3]);
      expect(bundle.loads, ['assets/models/yolo26n_fp16_rawhead.tflite']);
    });

    test(
      'the real bundle: the asset in the app is the raw-head file',
      () async {
        TestWidgetsFlutterBinding.ensureInitialized();

        final bytes = await loadBundledDetector();

        expect(bytes.length, kDetModelBytes);
        expect(sha256.convert(bytes).toString(), kDetModelSha256);
      },
    );
  });

  test('detectWithSnapshot: the detection plus the same frame as upright '
      'RGBA, back from the worker', () async {
    final detector = service();
    await detector.load(
      source: DetectorFile(modelPath),
      backend: DetectorBackend.gpu,
    );
    final bytes = Uint8List(640 * 480 * 4);
    for (var i = 0; i < bytes.length; i++) {
      bytes[i] = i % 251;
    }
    final frame = FrameMessage.copyOf(
      TestFrame(640, 480, FramePixelFormat.rgba8888, 0, [
        FramePlane(bytes: bytes, bytesPerRow: 640 * 4, bytesPerPixel: 4),
      ]),
      frameId: 9,
    );

    final result = await detector.detectWithSnapshot(frame);

    final value = (result as Ok<SnapshotDetection>).value;
    expect(value.frame.frameId, 9);
    expect(value.frame.classId(0), kFakeClass);
    expect((value.pixels.width, value.pixels.height), (640, 480));
    expect(value.pixels.bytes.sublist(0, 3), bytes.sublist(0, 3));
    expect(value.pixels.bytes[3], 255);
    expect(value.pixels.bytes.sublist(4000, 4003), bytes.sublist(4000, 4003));
    // The slot is free again: a plain frame goes through.
    expect(await detector.detect(rgbaFrame(frameId: 10)), isA<Ok<Object>>());
  });

  test('fail fast through the isolate: partial GPU acceleration is a load '
      'error and nothing is loaded', () async {
    final detector = service(
      const FakeDetectorRuntime(fullyAccelerated: false),
    );

    final loaded = await detector.load(
      source: DetectorFile(modelPath),
      backend: DetectorBackend.gpu,
    );

    expect(loaded, isA<Error<DetectorInfo>>());
    expect(
      (loaded as Error<DetectorInfo>).error.toString(),
      contains('only partly on the GPU'),
    );
    expect(detector.info, isNull);
    expect(detector.isLoaded, isFalse);
    expect(await detector.detect(rgbaFrame()), isA<Error<DetectionFrame>>());
  });

  test('a missing file is a load error', () async {
    final detector = service();

    final loaded = await detector.load(
      source: DetectorFile('${dir.path}/missing.tflite'),
      backend: DetectorBackend.gpu,
    );

    expect((loaded as Error).error.toString(), contains('not found'));
  });

  test(
    'close() waits for the frame in flight, which still completes',
    () async {
      final detector = service(
        const FakeDetectorRuntime(runDelay: Duration(milliseconds: 300)),
      );
      await detector.load(
        source: DetectorFile(modelPath),
        backend: DetectorBackend.gpu,
      );
      final order = <String>[];

      final detecting = detector.detect(rgbaFrame()).then((r) {
        order.add('detect ${r.runtimeType}');
        return r;
      });
      await Future<void>.delayed(const Duration(milliseconds: 50));
      await detector.close().then((_) => order.add('closed'));
      await detecting;

      expect(order, ['detect Ok<DetectionFrame>', 'closed']);
      expect(await detector.detect(rgbaFrame()), isA<Error<DetectionFrame>>());
    },
  );

  test(
    'one frame at a time: a second detect while one runs is refused',
    () async {
      final detector = service(
        const FakeDetectorRuntime(runDelay: Duration(milliseconds: 200)),
      );
      await detector.load(
        source: DetectorFile(modelPath),
        backend: DetectorBackend.gpu,
      );

      final first = detector.detect(rgbaFrame());
      final second = await detector.detect(rgbaFrame(frameId: 2));

      expect(
        (second as Error).error.toString(),
        contains('already being detected'),
      );
      expect(await first, isA<Ok<DetectionFrame>>());
    },
  );

  test('a worker that dies mid-frame fails the frame instead of hanging, and '
      'later frames fail at once', () async {
    // Runs 1 and 2 are load's verify and warm-up; run 3 is the first frame.
    final detector = service(const FakeDetectorRuntime(exitOnRun: 3));
    expect(
      await detector.load(
        source: DetectorFile(modelPath),
        backend: DetectorBackend.gpu,
      ),
      isA<Ok<DetectorInfo>>(),
    );

    final result = await detector
        .detect(rgbaFrame())
        .timeout(const Duration(seconds: 10));

    expect(result, isA<Error<DetectionFrame>>());
    expect((result as Error).error.toString(), contains('worker'));
    expect(await detector.detect(rgbaFrame()), isA<Error<DetectionFrame>>());
  });

  test('close() gives up on a frame that never comes back: bounded, the '
      'worker is ended and the frame fails', () async {
    final detector = DetectorService(
      runtime: const FakeDetectorRuntime(
        runDelay: Duration(seconds: 4),
        delayFromRun: 3, // only the first frame hangs, not the load
      ),
      expectedModelBytes: _fileBytes,
      closeTimeout: const Duration(milliseconds: 200),
    );
    await detector.load(
      source: DetectorFile(modelPath),
      backend: DetectorBackend.gpu,
    );
    final detecting = detector.detect(rgbaFrame());
    await Future<void>.delayed(const Duration(milliseconds: 50));

    final watch = Stopwatch()..start();
    await detector.close();
    final closeMs = watch.elapsedMilliseconds;

    expect(closeMs, lessThan(1500), reason: 'not the 4 s native call');
    expect(await detecting, isA<Error<DetectionFrame>>());
  });

  test('close() with a stuck frame takes ~1x closeTimeout: once the frame '
      'wait has timed out, the worker is killed without a close request it '
      'cannot answer', () async {
    // 1 s, so one timeout (~1 s) and two (~2 s, a close request after the
    // frame wait) stay far apart on a loaded machine.
    const closeTimeout = Duration(seconds: 1);
    final detector = DetectorService(
      runtime: const FakeDetectorRuntime(
        runDelay: Duration(seconds: 4),
        delayFromRun: 3,
      ),
      expectedModelBytes: _fileBytes,
      closeTimeout: closeTimeout,
    );
    await detector.load(
      source: DetectorFile(modelPath),
      backend: DetectorBackend.gpu,
    );
    final detecting = detector.detect(rgbaFrame());
    await Future<void>.delayed(const Duration(milliseconds: 50));
    final logs = <String>[];
    final previous = debugPrint;
    debugPrint = (message, {wrapWidth}) => logs.add('$message');
    addTearDown(() => debugPrint = previous);

    final watch = Stopwatch()..start();
    await detector.close();
    final closeMs = watch.elapsedMilliseconds;

    expect(logs, [
      contains(
        'the frame in flight did not finish within 1000 ms of close; '
        'ending the worker',
      ),
    ]);
    expect(closeMs, greaterThanOrEqualTo(950), reason: 'waits for the frame');
    expect(closeMs, lessThan(1900), reason: 'one closeTimeout (1 s), not two');
    expect(await detecting, isA<Error<DetectionFrame>>());
  });
}

/// An [AssetBundle] over a map of keys to bytes.
final class _FakeBundle extends CachingAssetBundle {
  _FakeBundle(this.assets);

  final Map<String, Uint8List> assets;
  final List<String> loads = [];

  @override
  Future<ByteData> load(String key) async {
    loads.add(key);
    final bytes = assets[key];
    if (bytes == null) throw StateError('no asset $key');
    return ByteData.sublistView(bytes);
  }
}
