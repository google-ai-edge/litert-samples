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

// On the real app (macOS, camera_desktop): Demo 3 with the default
// FRAME_SOURCE=camera, the built-in camera, the YOLO26n worker on the GPU.
//
//   flutter test integration_test/camera_source_test.dart -d macos \
//     --dart-define=GEMMA_MODEL_PATH=<path/to/gemma-4-E2B-it.litertlm>
//
// The first run may show macOS's camera prompt for the app
// (com.google.ai.edge.examples.litertEdgeDemos); the test fails with that hint if the
// camera does not start.
//
// 1. Demo 3 starts the camera; ≥30 frames are detected. Prints the source's
//    `CAMERA src=camera_desktop 1280x720 bgra stride=… mirrored=true` line,
//    then `CAMERA_TEST fps=… luma_later=… black_frames=…` and a screenshot
//    path. The luma of a later frame (the first is often dark during
//    auto-exposure) is logged, never asserted: on some Macs the camera
//    delivers zeroed frames when access is attributed to the terminal,
//    which the screen shows as a warning chip, not a failure.
// 2. Leaving Demo 3 stops the camera (stream stopped, controller disposed,
//    which stops the AVCaptureSession) within 1 s: `STOP ms=…`.
// 3. Re-entering works: ≥30 more frames, then leave again.

import 'dart:io';
import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/app.dart';
import 'package:litert_edge_demos/config/demos.dart';
import 'package:litert_edge_demos/config/dependencies.dart';
import 'package:litert_edge_demos/config/env.dart';
import 'package:litert_edge_demos/domain/models/detection.dart';
import 'package:litert_edge_demos/domain/models/live_state.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/domain/models/preview_source.dart';
import 'package:litert_edge_demos/domain/vision/coco_vocabulary.dart';
import 'package:litert_edge_demos/ui/core/detection_painter.dart';
import 'package:litert_edge_demos/ui/features/home/views/home_screen.dart';
import 'package:litert_edge_demos/ui/features/live_camera/view_models/live_camera_view_model.dart';
import 'package:litert_edge_demos/ui/features/live_camera/views/live_camera_screen.dart';
import 'package:provider/provider.dart';

import 'support/app_window.dart';
import 'support/demo3_settings.dart';
import 'support/pump.dart';

/// The waits poll every 10 ms: the stop time after leaving Demo 3 is
/// measured to that resolution.
const _poll = Duration(milliseconds: 10);

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

/// Every published detection with its arrival time.
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

  double fpsSince(int from) {
    final n = entries.length - from;
    if (n < 2) return 0;
    final span = entries.last.$1 - entries[from].$1;
    return (n - 1) / (span.inMicroseconds / 1e6);
  }

  void dispose() => _frames.removeListener(_onFrame);
}

void main() {
  initIntegrationTest();

  testWidgets('camera source: frames, boxes, stop within 1 s, re-entry', (
    tester,
  ) async {
    if (kGemmaModelPath.isEmpty) fail('Pass GEMMA_MODEL_PATH');
    expect(kFrameSource, 'camera', reason: 'FRAME_SOURCE must be camera');
    final logs = <String>[];
    final originalDebugPrint = debugPrint;
    debugPrint = (message, {wrapWidth}) {
      logs.add(message ?? '');
      originalDebugPrint(message, wrapWidth: wrapWidth);
    };
    try {
      await resetDemo3Settings();
      final deps = await AppDependencies.create();
      await tester.pumpWidget(
        RepaintBoundary(
          key: _screenKey,
          child: App(dependencies: deps),
        ),
      );
      await pumpUntil(
        tester,
        () => find.byType(HomeScreen).evaluate().isNotEmpty,
        timeout: const Duration(minutes: 6),
        reason: 'model setup',
        step: _poll,
      );
      await tester.pump(const Duration(milliseconds: 400));
      final detector = deps.models.states.value[ModelId.yolo26n];
      expect(detector, isA<ModelReady>(), reason: 'detector: $detector');
      final log = FrameLog(deps.live.frames);
      addTearDown(log.dispose);

      Future<LiveCameraViewModel> enterDemo3() async {
        final tile = find.byKey(HomeKeys.tile(Demo.liveCamera));
        await pumpUntil(
          tester,
          () => tester.widget<ListTile>(tile).enabled,
          timeout: const Duration(seconds: 5),
          reason: 'the Demo 3 tile to be enabled',
          step: _poll,
        );
        await tester.tap(tile);
        await pumpUntil(
          tester,
          () => find.byType(LiveCameraScreen).evaluate().isNotEmpty,
          timeout: const Duration(seconds: 5),
          reason: 'the camera screen to open',
          step: _poll,
        );
        final vm = Provider.of<LiveCameraViewModel>(
          tester.element(find.byType(LiveCameraScreen)),
          listen: false,
        );
        await pumpUntil(
          tester,
          () =>
              deps.live.state.value is LiveRunning ||
              deps.live.state.value is LiveFailed,
          timeout: const Duration(seconds: 30),
          reason:
              'the camera to start (if macOS shows a camera prompt for '
              'com.google.ai.edge.examples.litertEdgeDemos, allow it once)',
          describe: () =>
              'state=${deps.live.state.value} startError=${vm.startError}',
          step: _poll,
        );
        if (deps.live.state.value case LiveFailed(:final message)) {
          fail('Camera failed: $message');
        }
        return vm;
      }

      /// Leaves Demo 3; returns ms until the source is stopped (stream
      /// stopped, controller disposed).
      Future<int> leaveDemo3() async {
        final watch = Stopwatch()..start();
        await tester.pageBack();
        await pumpUntil(
          tester,
          () => deps.live.state.value is LiveStopped,
          timeout: const Duration(seconds: 3),
          reason: 'the camera to stop after leaving Demo 3',
          describe: () => 'state=${deps.live.state.value}',
          step: _poll,
        );
        final ms = watch.elapsedMilliseconds;
        await tester.pump(const Duration(milliseconds: 400));
        return ms;
      }

      // 1. Frames and boxes.
      final vm = await enterDemo3();
      final info = deps.live.sourceInfo!;
      final from = log.length;
      await pumpUntil(
        tester,
        () => log.length - from >= 30,
        timeout: const Duration(seconds: 20),
        reason: '30 detected camera frames',
        step: _poll,
      );
      final firstBatch = log.length - from;
      final fpsFrom = log.length;
      await pumpFor(tester, const Duration(seconds: 4));
      final fps = log.fpsSince(fpsFrom);
      final stats = deps.live.stats.value;
      final classes = <String, double>{};
      for (final (_, f) in log.entries.skip(from)) {
        for (var i = 0; i < f.count; i++) {
          final name = cocoName(f.classId(i));
          if (f.score(i) > (classes[name] ?? 0)) classes[name] = f.score(i);
        }
      }
      final screenshot = await saveScreenshot('camera_source');
      final cameraLine = logs.firstWhere(
        (l) => l.startsWith('CAMERA src='),
        orElse: () => '(no CAMERA line)',
      );
      final last = log.entries.last.$2;
      debugPrint(
        'CAMERA_TEST source="${info.label}" frame=${last.width}x${last.height} '
        'mirrored=${info.mirrored} preview_mirrored=${info.previewMirrored} '
        'overlay_mirror=${vm.mirrorBoxes} first_batch=$firstBatch '
        'fps=${fps.toStringAsFixed(1)} busy=${stats.droppedBusy} '
        'rate=${stats.droppedRate} source_frames=${stats.sourceFrames} '
        'p50 copy=${stats.copyMs} pre=${stats.preMs} run=${stats.runMs} '
        'post=${stats.postMs} lat=${stats.latencyMs} '
        'luma_later=${deps.live.luma?.toStringAsFixed(1)} '
        'black_frames=${deps.live.blackFrames.value} '
        'classes=${classes.entries.map((e) => '${e.key}:${e.value.toStringAsFixed(2)}').join(',')} '
        'screenshot=$screenshot',
      );
      expect(
        cameraLine,
        startsWith('CAMERA src=camera_desktop 1280x720 bgra stride='),
      );
      expect(cameraLine, contains('mirrored=true'));
      expect(info.mirrored, isTrue);
      expect(info.previewMirrored, isTrue);
      // Behaviour, not the table: camera_desktop draws the preview texture
      // and streams the frames from one mirrored pixel buffer
      // (CameraSession.swift, isVideoMirrored = true), so a box the detector
      // reports at the frame's left must be drawn over the same pixels in the
      // cover-fitted preview, with the painter the screen actually uses.
      final boxesLayer = find.byKey(LiveCameraKeys.boxes);
      final painter =
          tester.widget<CustomPaint>(boxesLayer).painter! as DetectionPainter;
      final layer = tester.getSize(boxesLayer);
      final probeX = last.width * 0.2; // a cup left of centre in the frame
      final probe = DetectionFrame(
        frameId: 0,
        width: last.width,
        height: last.height,
        boxes: Float32List.fromList([
          probeX - 40,
          last.height / 2 - 40,
          probeX + 40,
          last.height / 2 + 40,
          0.9,
          41,
        ]),
        preMicros: 0,
        runMicros: 0,
        postMicros: 0,
        backend: DetectorBackend.gpu,
      );
      final drawnX = layoutDetectionBoxes(
        probe,
        layer,
        mirror: painter.mirror,
      ).single.rect.center.dx;
      final scale = math.max(
        layer.width / last.width,
        layer.height / last.height,
      );
      final shownX = (probeX - last.width / 2) * scale + layer.width / 2;
      expect(
        drawnX,
        closeTo(shownX, 1),
        reason: 'the box must cover the pixels the preview shows there',
      );
      expect(vm.mirrorBoxes, painter.mirror, reason: 'the screen binds the VM');
      expect((last.width, last.height), (1280, 720));
      expect(deps.live.preview.value, isA<CameraPreviewSource>());
      expect(fps, greaterThanOrEqualTo(12), reason: 'capped at 15');

      // 2. Leave: the camera stops within 1 s.
      final stopMs = await leaveDemo3();
      debugPrint(
        'STOP ms=$stopMs (leave → stream stopped + controller disposed)',
      );
      expect(stopMs, lessThanOrEqualTo(1000));
      expect(deps.live.preview.value, isNull);

      // 3. Re-enter, frames again, leave again.
      await enterDemo3();
      final again = log.length;
      await pumpUntil(
        tester,
        () => log.length - again >= 30,
        timeout: const Duration(seconds: 20),
        reason: '30 frames after re-entering',
        step: _poll,
      );
      final reFps = log.fpsSince(again);
      final stopMs2 = await leaveDemo3();
      debugPrint(
        'REENTER frames=${log.length - again} fps=${reFps.toStringAsFixed(1)} '
        'stop_ms=$stopMs2',
      );
      expect(stopMs2, lessThanOrEqualTo(1000));

      await tester.pumpWidget(const SizedBox.shrink());
      await deps.dispose();
    } finally {
      debugPrint = originalDebugPrint;
    }
  }, timeout: const Timeout(Duration(minutes: 10)));
}
