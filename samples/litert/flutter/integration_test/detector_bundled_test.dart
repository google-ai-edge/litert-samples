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

// The built-in YOLO26n raw-head detector (assets/models/) on any target,
// without the app's UI or a chat model: the bytes the app ships, loaded
// exactly as the app loads them (the asset bundle → DetectorService worker →
// CompiledModel.fromBuffer). No dart-define needed: the default backend is
// the platform's standard one (the GPU).
//
//   flutter test integration_test/detector_bundled_test.dart -d <device> \
//     [--dart-define=DETECTOR_BACKEND=cpu|gpu]
//
// Guards the iOS model-loading ABI: with the wrong model-loading signature
// step 1 fails on iOS, simulator and device, with LiteRtStatus=501
// (kLiteRtStatusErrorInvalidFlatbuffer).
//
// 1. flutter_litert itself: CompiledModel.fromBuffer on the CPU builds.
// 2. The app's DetectorService loads it on exactly the requested backend and
//    passes the CPU-reference check.
// 3. Cats (test_assets/cats.jpg): the golden classes, box ≤ 3 px,
//    |Δscore| ≤ 0.03, 10 identical runs.
//
// Prints `DETBUNDLED os=… det=… vs=… box=… run=…`.

import 'dart:io';
import 'dart:ui' as ui;

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_litert/native.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:litert_edge_demos/config/env.dart';
import 'package:litert_edge_demos/config/live_camera_config.dart'
    show standardDetectorBackend;
import 'package:litert_edge_demos/data/services/detector/detector_service.dart';
import 'package:litert_edge_demos/data/services/detector/frame_message.dart';
import 'package:litert_edge_demos/data/services/frames/frame_source.dart';
import 'package:litert_edge_demos/domain/models/detection.dart';
import 'package:litert_edge_demos/domain/models/detector_choice.dart';
import 'package:litert_edge_demos/domain/models/frame_source_info.dart';
import 'package:litert_edge_demos/selftest/cats_golden.dart';
import 'package:litert_edge_demos/utils/result.dart';

final class _RgbaFrame implements FrameView {
  _RgbaFrame(this.width, this.height, Uint8List rgba)
    : planes = [
        FramePlane(bytes: rgba, bytesPerRow: width * 4, bytesPerPixel: 4),
      ];

  @override
  final int width;
  @override
  final int height;
  @override
  FramePixelFormat get format => FramePixelFormat.rgba8888;
  @override
  int get rotationDeg => 0;
  @override
  final List<FramePlane> planes;
}

Future<_RgbaFrame> _decodeAsset(String asset) async {
  final data = await rootBundle.load(asset);
  final codec = await ui.instantiateImageCodec(data.buffer.asUint8List());
  final image = (await codec.getNextFrame()).image;
  final bytes = await image.toByteData();
  final frame = _RgbaFrame(
    image.width,
    image.height,
    bytes!.buffer.asUint8List(),
  );
  image.dispose();
  codec.dispose();
  return frame;
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('the built-in detector loads and detects', (tester) async {
    final backend = switch (resolveDetectorBackend(
      define: kDetectorBackend,
      standard: standardDetectorBackend(),
    )) {
      Ok(:final value) => value.backend,
      Error(:final error) => fail('$error'),
    };
    final bytes = (await tester.runAsync(loadBundledDetector))!;

    // 1. flutter_litert alone, on this isolate.
    final raw = CompiledModel.fromBuffer(
      bytes,
      accelerators: {Accelerator.cpu},
      precision: Precision.fp32,
    );
    debugPrint(
      'DETBUNDLED raw fromBuffer ok: in ${raw.inputByteSizes} B, out ${raw.outputByteSizes} B',
    );
    raw.close();

    // 2. The app's path.
    final detector = DetectorService();
    final loaded = await tester.runAsync(
      () => detector.load(source: DetectorBytes(bytes), backend: backend),
    );
    final info = switch (loaded) {
      Ok(:final value) => value,
      Error(:final error) => fail('detector load failed: $error'),
      null => fail('detector load did not complete'),
    };
    debugPrint('DETBUNDLED detector $info');
    expect(info.backend, backend, reason: 'exactly the requested backend');

    // 3. Cats, ten times.
    final cats = (await tester.runAsync(() => _decodeAsset(kCatsAsset)))!;
    final frames = <DetectionFrame>[];
    for (var i = 0; i < 10; i++) {
      final result = await tester.runAsync(
        () => detector.detect(FrameMessage.copyOf(cats, frameId: i + 1)),
      );
      switch (result) {
        case Ok(:final value):
          frames.add(value);
        case Error(:final error):
          fail('detect failed: $error');
        case null:
          fail('detect did not complete');
      }
    }
    await tester.runAsync(detector.close);

    final golden = checkCatsGolden(frames.first);
    debugPrint(
      'DETBUNDLED os="${Platform.operatingSystemVersion}" '
      'det=${info.label} vs=${info.verifyReference.label} '
      'verify=${info.verifyRelative.toStringAsExponential(1)} '
      'box=${golden.maxBoxPx.toStringAsFixed(2)}px '
      'dscore=${golden.maxScoreDelta.toStringAsFixed(4)} '
      'run=${(frames.last.runMicros / 1000).toStringAsFixed(1)}ms',
    );
    expect(golden.classes, [
      for (final g in kCatsGolden) g.cls,
    ], reason: 'golden classes in score order');
    expect(golden.maxBoxPx, lessThanOrEqualTo(kCatsMaxBoxPx));
    expect(golden.maxScoreDelta, lessThanOrEqualTo(kCatsMaxScoreDelta));
    expect(frames.every((f) => f.sameBoxes(frames.first)), isTrue);
  }, timeout: const Timeout(Duration(minutes: 5)));
}
