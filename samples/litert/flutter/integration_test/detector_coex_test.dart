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

// YOLO26n + Gemma 4 E2B in one process, without the app's UI: the platform
// check for a new target (Linux x86_64 / arm64). On Linux both run on the
// LiteRT-LM native bundle's single LiteRT copy, each in its own LiteRT
// environment.
//
//   flutter test integration_test/detector_coex_test.dart -d linux \
//     --dart-define=GEMMA_MODEL_PATH=<path/to/gemma-4-E2B-it.litertlm> \
//     [--dart-define=DETECTOR_BACKEND=cpu] [--dart-define=GEMMA_BACKEND=cpu]
//
// The detector is the one built into the app (the YOLO26n raw-head asset,
// loaded from memory as the app loads it); the cats image is the app's
// test_assets/cats.jpg (COCO val2017 39769). Headless Linux: wrap in
// `xvfb-run -a`.
//
// 1. The built-in detector loads with exactly the requested backend and
//    passes the CPU-reference check (which reference is printed).
// 2. Cats: the golden classes, box ≤ 3 px, |Δscore| ≤ 0.03, 10 identical runs.
// 3. Gemma loads on GEMMA_BACKEND (default gpu, no fallback) next to the
//    loaded detector and generates up to 64 tokens.
// 4. The detector's output afterwards is bit-identical to before.
//
// Prints `DETCOEX os=… det=… vs=… box=… gemma=… tokps=… identical=…`.

import 'dart:io';
import 'dart:ui' as ui;

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_edge_ai/flutter_edge_ai.dart';
import 'package:flutter_edge_ai_litertlm/flutter_edge_ai_litertlm.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:litert_edge_demos/config/env.dart';
import 'package:litert_edge_demos/data/services/detector/detector_service.dart';
import 'package:litert_edge_demos/data/services/detector/frame_message.dart';
import 'package:litert_edge_demos/data/services/frames/frame_source.dart';
import 'package:litert_edge_demos/domain/models/detection.dart';
import 'package:litert_edge_demos/domain/models/detector_choice.dart';
import 'package:litert_edge_demos/domain/models/frame_source_info.dart';
import 'package:litert_edge_demos/domain/vision/coco_vocabulary.dart';
import 'package:litert_edge_demos/selftest/cats_golden.dart';
import 'package:litert_edge_demos/utils/result.dart';

const _gemmaBackendName = String.fromEnvironment(
  'GEMMA_BACKEND',
  defaultValue: 'gpu',
);

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

Future<_RgbaFrame> _decode(Uint8List encoded) async {
  final codec = await ui.instantiateImageCodec(encoded);
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

String _describe(DetectionFrame f) => [
  for (var i = 0; i < f.count; i++)
    '${cocoName(f.classId(i))} ${f.score(i).toStringAsFixed(3)}',
].join(', ');

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('YOLO26n and Gemma 4 E2B coexist in one process', (tester) async {
    if (kGemmaModelPath.isEmpty) fail('Pass --dart-define=GEMMA_MODEL_PATH=…');
    // Empty (the default) is the GPU.
    final detBackend = switch (resolveDetectorBackend(
      define: kDetectorBackend,
    )) {
      Ok(:final value) => value.backend,
      Error(:final error) => fail('$error'),
    };
    final gemmaBackend = PreferredBackend.values.byName(_gemmaBackendName);

    // 1. The built-in detector, exactly as requested.
    final detector = DetectorService();
    final loaded = await tester.runAsync(
      () async => detector.load(
        source: DetectorBytes(await loadBundledDetector()),
        backend: detBackend,
      ),
    );
    final info = switch (loaded) {
      Ok(:final value) => value,
      Error(:final error) => fail('detector load failed: $error'),
      null => fail('detector load did not complete'),
    };
    debugPrint('DETCOEX detector $info');

    // 2. Cats, ten times.
    final cats = (await tester.runAsync(() async {
      final data = await rootBundle.load('test_assets/cats.jpg');
      return _decode(data.buffer.asUint8List());
    }))!;
    Future<List<DetectionFrame>> detectCats(int n, int firstId) async {
      final frames = <DetectionFrame>[];
      for (var i = 0; i < n; i++) {
        final result = await tester.runAsync(
          () =>
              detector.detect(FrameMessage.copyOf(cats, frameId: firstId + i)),
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
      return frames;
    }

    final before = await detectCats(10, 1);
    final golden = checkCatsGolden(before.first);
    debugPrint(
      'DETCOEX cats ${_describe(before.first)} '
      'maxBox=${golden.maxBoxPx.toStringAsFixed(2)}px '
      'maxDScore=${golden.maxScoreDelta.toStringAsFixed(4)} '
      'run=${(before.last.runMicros / 1000).toStringAsFixed(1)}ms',
    );
    expect(golden.classes, [
      for (final g in kCatsGolden) g.cls,
    ], reason: 'golden classes in score order');
    expect(golden.maxBoxPx, lessThanOrEqualTo(kCatsMaxBoxPx));
    expect(golden.maxScoreDelta, lessThanOrEqualTo(kCatsMaxScoreDelta));
    expect(before.every((f) => f.sameBoxes(before.first)), isTrue);

    // 3. Gemma next to the loaded detector, on exactly the requested backend.
    final (backendOut, tokps, chunks) = (await tester.runAsync(() async {
      await FlutterEdgeAi.initialize(
        inferenceEngines: [const LiteRtLmEngine()],
      );
      await FlutterEdgeAi.installModel(
        modelType: ModelType.gemma4,
        fileType: ModelFileType.litertlm,
      ).fromFile(kGemmaModelPath).install();
      final model = await FlutterEdgeAi.getActiveModel(
        maxTokens: 1024,
        preferredBackend: gemmaBackend,
      );
      try {
        expect(
          model.activeBackend,
          gemmaBackend,
          reason: 'requested $gemmaBackend, loaded ${model.activeBackend}',
        );
        final session = await model.createSession(maxOutputTokens: 64);
        var n = 0;
        final watch = Stopwatch()..start();
        Duration? first;
        try {
          await session.addQueryChunk(
            const Message(
              text: 'Tell me a short story about a lighthouse keeper.',
              isUser: true,
            ),
          );
          await for (final _ in session.getResponseAsync().timeout(
            const Duration(minutes: 5),
          )) {
            first ??= watch.elapsed;
            n++;
          }
        } finally {
          await session.close();
        }
        final decode = watch.elapsed - (first ?? Duration.zero);
        final rate = n > 1 ? (n - 1) / (decode.inMicroseconds / 1e6) : 0.0;
        return (model.activeBackend, rate, n);
      } finally {
        await model.close();
      }
    }))!;

    // 4. The detector afterwards: bit-identical.
    final after = await detectCats(10, 100);
    final identical = after.every((f) => f.sameBoxes(before.first));
    await tester.runAsync(detector.close);

    debugPrint(
      'DETCOEX os="${Platform.operatingSystemVersion}" '
      'det=${info.label} vs=${info.verifyReference.label} '
      'verify=${info.verifyRelative.toStringAsExponential(1)} '
      'box=${golden.maxBoxPx.toStringAsFixed(2)}px '
      'det_run=${(before.last.runMicros / 1000).toStringAsFixed(1)}ms '
      'gemma=$backendOut chunks=$chunks tokps=${tokps.toStringAsFixed(1)} '
      'identical=$identical',
    );
    expect(chunks, greaterThan(8));
    expect(identical, isTrue);
  }, timeout: const Timeout(Duration(minutes: 15)));
}
