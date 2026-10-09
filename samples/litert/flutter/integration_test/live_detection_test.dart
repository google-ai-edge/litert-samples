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

// Live detection on the real app (macOS): Gemma 4 E2B on the GPU plus the
// YOLO26n raw-head detector in its worker isolate, fed by the fixture
// slideshow.
//
//   flutter test integration_test/live_detection_test.dart -d macos \
//     --dart-define=GEMMA_MODEL_PATH=<path/to/gemma-4-E2B-it.litertlm> \
//     --dart-define=FRAME_SOURCE=fixture \
//     --dart-define=FIXTURE_DIR=<path/to/a/folder/of/images> \
//     --dart-define=ARM_DETECTOR_PATH=<path/to/yolo26n_conv2d_f16_weights.tflite>
//
// The detector is the one built into the app. FIXTURE_DIR is a folder of
// still images (e.g. 30 COCO val2017 photos); the cats image is the app's
// test_assets/cats.jpg, copied where the sandboxed app may read it. Step 4
// needs Arm's original yolo26n_conv2d_f16_weights.tflite (tool/fetch_models.sh
// keeps it in build/fetch_models/yolo26n/) and is skipped without
// ARM_DETECTOR_PATH.
//
// 1. setup → home → Demo 3: the FIXTURE_DIR slideshow runs with boxes;
//    processed fps and p50 pre/run/post are measured over ~6 s.
// 2. The cats image through the same fixture path (engine JPEG decode) gives
//    the golden classes: box ≤ 3 px, |Δscore| ≤ 0.03 (PIL decoded the golden).
// 3. COEX: 20 detections before and 20 after a 64-token Gemma generation are
//    bit-identical; during the generation the state and the overlay show
//    `paused (<the chat model's name>)` and no new frame is detected.
// 4. Negative control: Arm's original file is rejected (strict GPU throws in
//    LiteRT; the app's load names the file).
//
// Prints `LIVE src=fixture det=… full=… fps=… pre/run/post=… COEX identical=…`.

import 'dart:convert';
import 'dart:io';
import 'dart:ui' as ui;

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_litert/native.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/app.dart';
import 'package:litert_edge_demos/config/demos.dart';
import 'package:litert_edge_demos/config/dependencies.dart';
import 'package:litert_edge_demos/config/env.dart';
import 'package:litert_edge_demos/config/live_camera_config.dart'
    show kLiveDetectFps;
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/services/detector/detector_service.dart';
import 'package:litert_edge_demos/domain/models/assistant_event.dart';
import 'package:litert_edge_demos/domain/models/detection.dart';
import 'package:litert_edge_demos/domain/models/frame_source_info.dart';
import 'package:litert_edge_demos/domain/models/frame_source_spec.dart';
import 'package:litert_edge_demos/domain/models/live_state.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/domain/vision/coco_vocabulary.dart';
import 'package:litert_edge_demos/ui/features/home/views/home_screen.dart';
import 'package:litert_edge_demos/ui/features/live_camera/view_models/live_camera_view_model.dart';
import 'package:litert_edge_demos/ui/features/live_camera/views/live_camera_screen.dart';
import 'package:litert_edge_demos/utils/result.dart';
import 'package:provider/provider.dart';

import 'support/app_window.dart';
import 'support/demo3_settings.dart';
import 'support/pump.dart';

/// Arm's original YOLO26n file for the negative control (step 4); empty
/// skips it.
const _armDetectorPath = String.fromEnvironment('ARM_DETECTOR_PATH');

/// The golden from `test_assets/yolo26n/cats_golden.json` (frame px, PIL).
const _golden = [
  (cls: 15, score: 0.912, box: [344.0, 24.5, 640.0, 374.8]),
  (cls: 15, score: 0.899, box: [6.9, 55.1, 317.3, 466.1]),
  (cls: 65, score: 0.857, box: [40.4, 74.0, 175.9, 118.6]),
  (cls: 57, score: 0.297, box: [0.8, 0.6, 640.0, 480.0]),
];

final _screenKey = GlobalKey();

Future<String> saveScreenshot(String name) async {
  final boundary =
      _screenKey.currentContext?.findRenderObject() as RenderRepaintBoundary?;
  if (boundary == null) fail('No RepaintBoundary to capture');
  final image = await boundary.toImage();
  final png = await image.toByteData(format: ui.ImageByteFormat.png);
  image.dispose();
  if (png == null) fail('PNG encoding failed');
  final file = File('${Directory.systemTemp.path}/$name.png');
  await file.writeAsBytes(png.buffer.asUint8List());
  return file.path;
}

/// Records every published detection with its arrival time.
final class FrameLog {
  FrameLog(this._frames) {
    _frames.addListener(_onFrame);
  }

  final ValueListenable<DetectionFrame?> _frames;
  final Stopwatch clock = Stopwatch()..start();
  final List<(Duration, DetectionFrame)> entries = [];

  void _onFrame() {
    final frame = _frames.value;
    if (frame != null) entries.add((clock.elapsed, frame));
  }

  int get length => entries.length;

  /// Processed frames per second between entry [from] and the last one.
  double fpsSince(int from) {
    final n = entries.length - from;
    if (n < 2) return 0;
    final span = entries.last.$1 - entries[from].$1;
    return (n - 1) / (span.inMicroseconds / 1e6);
  }

  void dispose() => _frames.removeListener(_onFrame);
}

/// For each golden detection, the same-class box closest to it: the worst
/// coordinate error and score error over all four.
({double box, double score, String detail}) goldenError(DetectionFrame f) {
  var worstBox = 0.0;
  var worstScore = 0.0;
  final parts = <String>[];
  for (final g in _golden) {
    var best = double.infinity;
    var bestScore = 0.0;
    for (var i = 0; i < f.count; i++) {
      if (f.classId(i) != g.cls) continue;
      final err = [
        (f.x1(i) - g.box[0]).abs(),
        (f.y1(i) - g.box[1]).abs(),
        (f.x2(i) - g.box[2]).abs(),
        (f.y2(i) - g.box[3]).abs(),
      ].reduce((a, b) => a > b ? a : b);
      if (err < best) {
        best = err;
        bestScore = (f.score(i) - g.score).abs();
      }
    }
    if (best > worstBox) worstBox = best;
    if (bestScore > worstScore) worstScore = bestScore;
    parts.add('${cocoName(g.cls)}:${best.toStringAsFixed(2)}px');
  }
  return (box: worstBox, score: worstScore, detail: parts.join(' '));
}

String describeFrame(DetectionFrame f) => [
  for (var i = 0; i < f.count; i++)
    '${cocoName(f.classId(i))} ${f.score(i).toStringAsFixed(3)} '
        '[${f.x1(i).toStringAsFixed(1)}, ${f.y1(i).toStringAsFixed(1)}, '
        '${f.x2(i).toStringAsFixed(1)}, ${f.y2(i).toStringAsFixed(1)}]',
].join('; ');

/// What the detector's pause names (`GpuArbiter`): the loaded chat model.
String _pauseReason(AppDependencies deps) =>
    switch (deps.models.states.value[ModelId.chat]) {
      ModelReady(:final info) => info.chat?.name ?? kDefineChatModel.name,
      _ => 'the chat model',
    };

void main() {
  initIntegrationTest();

  testWidgets('live detection: fixture slideshow, cats golden, GPU '
      'coexistence with Gemma', (tester) async {
    for (final (name, value) in [
      ('GEMMA_MODEL_PATH', kGemmaModelPath),
      ('FIXTURE_DIR', kFixtureDir),
    ]) {
      if (value.isEmpty) fail('Pass --dart-define=$name=…');
    }
    expect(kFrameSource, 'fixture', reason: 'pass FRAME_SOURCE=fixture');
    // The cats image, where the sandboxed app may read it.
    final catsPath = '${Directory.systemTemp.path}/coco_39769_cats.jpg';
    final catsJpg = await rootBundle.load('test_assets/cats.jpg');
    await File(catsPath).writeAsBytes(catsJpg.buffer.asUint8List());

    await resetDemo3Settings();
    final deps = await AppDependencies.create();
    await tester.pumpWidget(
      RepaintBoundary(
        key: _screenKey,
        child: App(dependencies: deps),
      ),
    );

    // Setup: Gemma, then the detector (strict GPU unless DETECTOR_BACKEND=cpu).
    await pumpUntil(
      tester,
      () => find.byType(HomeScreen).evaluate().isNotEmpty,
      timeout: const Duration(minutes: 6),
      reason: 'model setup',
      onPoll: () {
        if (deps.models.states.value[ModelId.chat] case ModelFailed(
          :final message,
        )) {
          fail('Gemma setup failed: $message');
        }
      },
    );
    await tester.pump(const Duration(milliseconds: 400));
    final detectorState = deps.models.states.value[ModelId.yolo26n];
    if (detectorState is! ModelReady) {
      fail(
        'Detector not ready: $detectorState '
        '${switch (detectorState) {
          ModelFailed(:final message) => message,
          ModelUnavailable(:final reason) => reason,
          _ => '',
        }}',
      );
    }
    final detector = deps.live.detectorInfo!;
    // What the detector loaded on (the define, or Demo 3's setting, which
    // the test resets to the GPU at its start).
    final explicitCpu = deps.live.detectorInfo?.backend == DetectorBackend.cpu;
    expect(
      detector.backend,
      explicitCpu ? DetectorBackend.cpu : DetectorBackend.gpu,
    );
    if (!explicitCpu) expect(detector.fullyAccelerated, isTrue);
    expect(detectorState.info.explicitCpu, explicitCpu);
    debugPrint('DETECTOR $detector');

    // 1. Demo 3 with the FIXTURE_DIR slideshow.
    final tile = find.byKey(HomeKeys.tile(Demo.liveCamera));
    expect(tester.widget<ListTile>(tile).enabled, isTrue);
    await tester.tap(tile);
    await pumpUntil(
      tester,
      () => find.byType(LiveCameraScreen).evaluate().isNotEmpty,
      timeout: const Duration(seconds: 5),
      reason: 'the camera screen to open',
    );
    final vm = Provider.of<LiveCameraViewModel>(
      tester.element(find.byType(LiveCameraScreen)),
      listen: false,
    );
    await pumpUntil(
      tester,
      () => vm.chatReady && deps.live.state.value is LiveRunning,
      timeout: const Duration(seconds: 60),
      reason: 'the chat and the slideshow to start',
      describe: () =>
          'chatReady=${vm.chatReady} chatError=${vm.chatError} '
          'startError=${vm.startError} state=${deps.live.state.value}',
    );
    final log = FrameLog(deps.live.frames);
    addTearDown(log.dispose);
    await pumpFor(tester, const Duration(seconds: 1)); // warm the JIT
    final from = log.length;
    await pumpFor(tester, const Duration(seconds: 6));
    final fps = log.fpsSince(from);
    final stats = deps.live.stats.value;
    final classesSeen = {
      for (final (_, f) in log.entries.skip(from))
        for (var i = 0; i < f.count; i++)
          if (f.score(i) >= 0.35) cocoName(f.classId(i)),
    };
    final label = explicitCpu ? 'det CPU (chosen) ·' : 'det GPU fp32 full ·';
    expect(find.textContaining(label), findsOneWidget, reason: 'overlay line');
    final slideshotPath = await saveScreenshot('live_detection_slideshow');
    debugPrint(
      'SLIDESHOW frames=${log.length - from} fps=${fps.toStringAsFixed(2)} '
      'stats_fps=${stats.fps.toStringAsFixed(2)} '
      'source_frames=${stats.sourceFrames} busy=${stats.droppedBusy} '
      'rate=${stats.droppedRate} p50 copy=${stats.copyMs} pre=${stats.preMs} '
      'run=${stats.runMs} post=${stats.postMs} lat=${stats.latencyMs} '
      'classes=${classesSeen.length} (${classesSeen.take(12).join(', ')}) '
      'screenshot=$slideshotPath',
    );

    // 2. Cats through the fixture path, owned by this test (a take-over).
    final owner = Object();
    final started = await deps.live.start(
      FixtureSourceSpec([catsPath]),
      owner: owner,
    );
    expect(started, isA<Ok<FrameSourceInfo>>());
    final catsFrom = log.length;
    await pumpUntil(
      tester,
      () => log.length - catsFrom >= 20,
      timeout: const Duration(seconds: 10),
      reason: '20 cats detections',
    );
    final before = [
      for (final (_, f) in log.entries.skip(catsFrom).take(20)) f,
    ];
    final golden = goldenError(before.first);
    debugPrint(
      'CATS ${describeFrame(before.first)} | maxBox=${golden.box.toStringAsFixed(2)}px '
      'maxDScore=${golden.score.toStringAsFixed(4)} (${golden.detail})',
    );
    expect(
      [for (var i = 0; i < before.first.count; i++) before.first.classId(i)],
      [for (final g in _golden) g.cls],
      reason: 'golden classes in score order',
    );
    expect(golden.box, lessThanOrEqualTo(3));
    expect(golden.score, lessThanOrEqualTo(0.03));
    final steadyBefore = before.every((f) => f.sameBoxes(before.first));
    await tester.pump(const Duration(milliseconds: 300));
    final catsShot = await saveScreenshot('live_detection_cats');

    // 3. COEX: a 64-token generation; detection pauses, then resumes with
    //    bit-identical output.
    expect(
      await deps.conversation.open(
        const ConversationProfile(
          name: 'coex-64',
          systemInstruction: 'You are a storyteller.',
          maxOutputTokens: 64,
        ),
      ),
      isA<Ok<void>>(),
    );
    var chunks = 0;
    GenerationMetrics? metrics;
    Object? failure;
    var sawPausedOverlay = false;
    Duration? pausedAt;
    Duration? lastPausedAt;
    final generation = deps.conversation
        .ask('Tell me a long story about a lighthouse keeper and a storm.')
        .listen((event) {
          switch (event) {
            case AssistantTextDelta():
              chunks++;
            case AssistantContextReset():
              break;
            case AssistantDone(metrics: final m):
              metrics = m;
            case AssistantFailed(:final error):
              failure = error;
          }
        });
    final genWatch = Stopwatch()..start();
    final pauseReason = _pauseReason(deps);
    await pumpUntil(
      tester,
      () => metrics != null || failure != null,
      timeout: const Duration(seconds: 120),
      reason: 'the 64-token generation',
      onPoll: () {
        if (deps.live.state.value case LivePaused(:final reason)
            when reason == pauseReason) {
          pausedAt ??= log.clock.elapsed;
          lastPausedAt = log.clock.elapsed;
        }
        if (find
            .textContaining('det paused ($pauseReason)')
            .evaluate()
            .isNotEmpty) {
          sawPausedOverlay = true;
        }
      },
    );
    final genMs = genWatch.elapsedMilliseconds;
    final genEnd = log.clock.elapsed;
    final sawPausedState = pausedAt != null;
    // The frame in flight at the pause may still land (about 10 ms); any
    // later frame while still paused would have been detected during the
    // generation.
    final framesDuringGeneration = sawPausedState
        ? log.entries
              .where(
                (e) =>
                    e.$1 > pausedAt! + const Duration(milliseconds: 150) &&
                    e.$1 <= lastPausedAt!,
              )
              .length
        : -1;
    await generation.cancel();
    expect(failure, isNull, reason: 'generation failed: $failure');
    final firstAfter = log.entries.indexWhere((e) => e.$1 > genEnd);
    final resumeFrom = firstAfter < 0 ? log.length : firstAfter;
    await pumpUntil(
      tester,
      () => log.length - resumeFrom >= 20,
      timeout: const Duration(seconds: 10),
      reason: '20 detections after the generation',
      describe: () => 'state=${deps.live.state.value}',
    );
    final recoverMs = (log.entries[resumeFrom].$1 - genEnd).inMilliseconds;
    final resumedFps = log.fpsSince(resumeFrom);
    final after = [
      for (final (_, f) in log.entries.skip(resumeFrom).take(20)) f,
    ];
    final identical =
        steadyBefore && after.every((f) => f.sameBoxes(before.first));
    debugPrint(
      'COEX chunks=$chunks gen_ms=$genMs '
      'tokps=${metrics?.tokensPerSecond?.toStringAsFixed(1)} '
      'paused_state=$sawPausedState paused_overlay=$sawPausedOverlay '
      'frames_during_generation=$framesDuringGeneration '
      'recover_ms=$recoverMs resumed_fps=${resumedFps.toStringAsFixed(1)} '
      'steady_before=$steadyBefore '
      'identical=$identical cats_screenshot=$catsShot',
    );
    expect(chunks, inInclusiveRange(32, 64), reason: 'a 64-token cap');
    expect(sawPausedState, isTrue);
    expect(sawPausedOverlay, isTrue);
    expect(framesDuringGeneration, 0);
    expect(identical, isTrue);

    // 4. Negative control: Arm's original file.
    const arm = _armDetectorPath;
    if (arm.isNotEmpty && File(arm).existsSync()) {
      String strict;
      try {
        CompiledModel.fromFile(
          arm,
          accelerators: {Accelerator.gpu},
          precision: Precision.fp32,
        ).close();
        strict = 'BUILT (the runtime changed: it now places the whole graph)';
      } catch (e) {
        strict = 'threw ${e.toString().split('\n').first}';
      }
      final service = DetectorService();
      final loaded = await service.load(
        source: const DetectorFile(arm),
        backend: DetectorBackend.gpu,
      );
      await service.close();
      debugPrint('NEG arm_strict_gpu=$strict');
      debugPrint('NEG arm_app_load=$loaded');
      expect(loaded, isA<Error<DetectorInfo>>());
      expect(strict, startsWith('threw'));
    } else {
      debugPrint(
        'NEG skipped: ${arm.isEmpty ? 'no ARM_DETECTOR_PATH' : '$arm not found'}',
      );
    }

    final backendTag = '${detector.backend.name}-fp32';
    debugPrint(
      'LIVE src=fixture det=$backendTag full=${detector.fullyAccelerated} '
      'fps=${fps.toStringAsFixed(1)} '
      'pre/run/post=${stats.preMs?.toStringAsFixed(1)}/'
      '${stats.runMs?.toStringAsFixed(1)}/${stats.postMs?.toStringAsFixed(1)}ms '
      'COEX identical=$identical',
    );
    debugPrint(
      'LIVE_JSON ${jsonEncode({'fps': fps, 'stats_fps': stats.fps, 'busy_drops': stats.droppedBusy, 'source_frames': stats.sourceFrames, 'latency_ms': stats.latencyMs, 'verify_abs': detector.verifyAbsolute, 'create_ms': detector.createTime.inMilliseconds, 'verify_ms': detector.verifyTime.inMilliseconds, 'first_run_ms': detector.firstRunTime.inMilliseconds})}',
    );
    // The gate holds detection at kLiveDetectFps (15) and never above, and
    // `fps` counts arrivals on the wall clock over ~6 s, so load on the
    // machine (a late source tick, a slower frame) only ever lowers it:
    // 14.33 was seen on a busy Mac, under a 14.5 bound with 3 % headroom.
    // 90 % of the cap leaves room for that and still fails a detector that
    // cannot keep up (one that misses every other frame shows 7.5).
    expect(
      fps,
      greaterThanOrEqualTo(kLiveDetectFps * 0.9),
      reason: 'the 15 fps cap, held',
    );

    await deps.live.stop(owner: owner);
    await tester.pageBack();
    await tester.pump(const Duration(milliseconds: 400));
    await tester.pumpWidget(const SizedBox.shrink());
    await deps.dispose();
  }, timeout: const Timeout(Duration(minutes: 10)));
}
