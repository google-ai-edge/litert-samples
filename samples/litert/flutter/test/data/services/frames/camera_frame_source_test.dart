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
import 'dart:math' as math;
import 'dart:ui' show Size;

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart'
    show DeviceOrientation, PlatformException;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/repositories/live_detection_repository.dart';
import 'package:litert_edge_demos/data/services/detector/detector_codec.dart';
import 'package:litert_edge_demos/data/services/detector/frame_message.dart';
import 'package:litert_edge_demos/data/services/frames/camera_frame_source.dart';
import 'package:litert_edge_demos/data/services/frames/frame_source.dart';
import 'package:litert_edge_demos/domain/models/detection.dart';
import 'package:litert_edge_demos/domain/models/frame_source_info.dart';
import 'package:litert_edge_demos/domain/models/frame_source_spec.dart';
import 'package:litert_edge_demos/domain/models/live_state.dart';
import 'package:litert_edge_demos/domain/models/preview_source.dart';
import 'package:litert_edge_demos/ui/core/detection_painter.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../../fakes/fake_camera_controller.dart';
import '../../../fakes/fake_detector.dart';
import '../../../support/yolo26n_reference.dart' as ref;

Future<void> settle() => Future<void>.delayed(Duration.zero);

const _back = CameraDescription(
  name: 'back',
  lensDirection: CameraLensDirection.back,
  sensorOrientation: 90,
);
const _front = CameraDescription(
  name: 'front',
  lensDirection: CameraLensDirection.front,
  sensorOrientation: 270,
);

void main() {
  group('CameraFrameView', () {
    test('wraps camera_desktop BGRA keeping the padded row stride', () {
      final image = bgraImage(width: 1280, height: 720, bytesPerRow: 5184);

      final view = CameraFrameView.wrap(image, rotationDeg: 0)!;

      expect((view.width, view.height), (1280, 720));
      expect(view.format, FramePixelFormat.bgra8888);
      expect(view.rotationDeg, 0);
      expect(view.planes.single.bytesPerRow, 5184, reason: 'not width × 4');
      expect(view.planes.single.bytesPerPixel, 4);
      expect(view.planes.single.bytes, same(image.planes.single.bytes));
    });

    test('the padded stride reaches the gather: same tensor as the reference '
        'reading the same padded buffer', () {
      const w = 1280;
      const h = 720;
      const stride = 5184;
      final rnd = math.Random(4);
      final bytes = Uint8List.fromList(
        List.generate(stride * h, (_) => rnd.nextInt(256)),
      );
      final view = CameraFrameView.wrap(
        bgraImage(width: w, height: h, bytesPerRow: stride, bytes: bytes),
        rotationDeg: 0,
      )!;
      final message = FrameMessage.copyOf(view, frameId: 1);
      final got = Float32List(3 * 640 * 640);
      FrameGatherer().gather(message, message.materialize(), got);
      final want = Float32List(3 * 640 * 640);
      ref.preprocessBgraNchw(bytes, stride, ref.RefLetterbox(w, h, 0), want);

      var maxDiff = 0.0;
      for (var i = 0; i < got.length; i++) {
        maxDiff = math.max(maxDiff, (got[i] - want[i]).abs());
      }
      expect(maxDiff, lessThan(1e-6));
    });

    test('CameraX NV21 maps to nv21 with bytesPerPixel 1; JPEG is refused', () {
      final view = CameraFrameView.wrap(nv21Image(), rotationDeg: 90)!;
      expect(view.format, FramePixelFormat.nv21);
      expect(view.rotationDeg, 90);
      expect(view.planes.single.bytesPerPixel, 1);

      expect(CameraFrameView.wrap(jpegImage(), rotationDeg: 0), isNull);
    });
  });

  test('meanLuma: black is 0, white is 255 (BGRA with padded rows)', () {
    final black = CameraFrameView.wrap(
      bgraImage(width: 64, height: 32, bytesPerRow: 300),
      rotationDeg: 0,
    )!;
    expect(meanLuma(black), 0);
    final white = CameraFrameView.wrap(
      bgraImage(
        width: 64,
        height: 32,
        bytesPerRow: 300,
        bytes: Uint8List(300 * 32)..fillRange(0, 300 * 32, 255),
      ),
      rotationDeg: 0,
    )!;
    expect(meanLuma(white), closeTo(255, 1e-9));
  });

  group('rotation', () {
    tearDown(() => debugDefaultTargetPlatformOverride = null);

    int rotate(
      TargetPlatform platform,
      CameraDescription camera,
      DeviceOrientation orientation, {
      int w = 1280,
      int h = 720,
    }) {
      debugDefaultTargetPlatformOverride = platform;
      return cameraFrameRotation(
        width: w,
        height: h,
        camera: camera,
        deviceOrientation: orientation,
      );
    }

    test('desktop frames are upright: 0', () {
      expect(
        rotate(
          TargetPlatform.macOS,
          kMacCamera,
          DeviceOrientation.landscapeLeft,
        ),
        0,
      );
    });

    test('phones rotate by sensor and device orientation', () {
      expect(
        rotate(TargetPlatform.android, _back, DeviceOrientation.portraitUp),
        90,
      );
      expect(
        rotate(TargetPlatform.android, _back, DeviceOrientation.landscapeLeft),
        0,
      );
      expect(
        rotate(TargetPlatform.iOS, _back, DeviceOrientation.portraitUp),
        90,
      );
      expect(
        rotate(TargetPlatform.iOS, _back, DeviceOrientation.landscapeLeft),
        0,
      );
    });
  });

  group('mirroring: the box is drawn over the object where the preview '
      'shows it', () {
    const w = 1280.0;
    const h = 720.0;
    const view = Size(800, 600); // cover fit: scale 5/6, 133 px cut per side
    const sensorX = 400.0; // a cup left of centre in the sensor image

    /// For a camera whose preview and frames are made from the sensor image
    /// as [previewMirrored] and [framesMirrored] say: where the preview shows
    /// the cup on screen, and the centre of the box the overlay draws for the
    /// detector's box (in frame coordinates) using [info].
    ({double shown, double drawn}) place(
      FrameSourceInfo info, {
      required bool previewMirrored,
      required bool framesMirrored,
    }) {
      final scale = math.max(view.width / w, view.height / h);
      double onScreen(double x) => (x - w / 2) * scale + view.width / 2;
      final frameX = framesMirrored ? w - sensorX : sensorX;
      final detection = DetectionFrame(
        frameId: 1,
        width: w.toInt(),
        height: h.toInt(),
        boxes: Float32List.fromList([
          frameX - 50,
          300,
          frameX + 50,
          400,
          0.9,
          41,
        ]),
        preMicros: 0,
        runMicros: 0,
        postMicros: 0,
        backend: DetectorBackend.gpu,
      );
      final box = layoutDetectionBoxes(
        detection,
        view,
        mirror: info.overlayMirrored,
      ).single;
      return (
        shown: onScreen(previewMirrored ? w - sensorX : sensorX),
        drawn: box.rect.center.dx,
      );
    }

    test('every preview/frames combination: the box lands on the cup', () {
      for (final (previewMirrored, framesMirrored) in [
        (false, false),
        (true, true),
        (true, false),
        (false, true),
      ]) {
        final info = FrameSourceInfo(
          label: 'x',
          width: 1280,
          height: 720,
          format: FramePixelFormat.bgra8888,
          mirrored: framesMirrored,
          previewMirrored: previewMirrored,
        );

        final (:shown, :drawn) = place(
          info,
          previewMirrored: previewMirrored,
          framesMirrored: framesMirrored,
        );

        expect(
          drawn,
          closeTo(shown, 1e-3),
          reason: 'preview mirrored=$previewMirrored, frames=$framesMirrored',
        );
      }
    });

    test('per plugin, from what each one does with the sensor image: the '
        'box CameraFrameSource reports lands on the cup', () async {
      // (platform, camera, preview mirrored, frames mirrored, source fact)
      final plugins = [
        (
          TargetPlatform.macOS,
          kMacCamera,
          true,
          true,
          'camera_desktop 2.0.0 CameraSession.swift: isVideoMirrored = true '
              'on the connection feeding texture and stream',
        ),
        (
          TargetPlatform.linux,
          kMacCamera,
          true,
          true,
          'camera_desktop on Linux mirrors at capture',
        ),
        (
          TargetPlatform.windows,
          kMacCamera,
          true,
          false,
          'camera_desktop on Windows flips only the preview widget',
        ),
        (
          TargetPlatform.iOS,
          _front,
          true,
          true,
          'camera_avfoundation mirrors the front capture connection',
        ),
        (TargetPlatform.iOS, _back, false, false, 'back camera: no mirroring'),
        (
          TargetPlatform.android,
          _front,
          true,
          false,
          'camera_android_camerax flips only the front preview widget',
        ),
        (
          TargetPlatform.android,
          _back,
          false,
          false,
          'back camera: no mirroring',
        ),
      ];
      for (final (platform, camera, previewMirrored, framesMirrored, fact)
          in plugins) {
        final source = CameraFrameSource(
          listCameras: () async => [camera],
          createController: FakeCameraController.new,
          platform: platform,
        );
        final info = switch (await source.start((_) {})) {
          Ok(:final value) => value,
          Error(:final error) => fail('$platform: $error'),
        };
        await source.stop();

        final (:shown, :drawn) = place(
          info,
          previewMirrored: previewMirrored,
          framesMirrored: framesMirrored,
        );

        expect(
          drawn,
          closeTo(shown, 1e-3),
          reason: '$platform ${camera.lensDirection.name}: $fact',
        );
      }
    });
  });

  group('errors', () {
    test('permission denied says how to grant access, per platform', () {
      final mac = cameraFailure(
        CameraException('permission_denied', 'Camera permission was denied'),
        TargetPlatform.macOS,
      );
      expect(
        mac.message,
        contains('System Settings > Privacy & Security > Camera'),
      );
      expect(mac.message, contains('LiteRT Demos'));
      expect(mac.message, contains('Retry'));

      final ios = cameraFailure(
        CameraException('CameraAccessDenied', 'denied'),
        TargetPlatform.iOS,
      );
      expect(ios.message, contains('Settings > Privacy & Security > Camera'));
      final android = cameraFailure(
        CameraException('CameraAccessDeniedWithoutPrompt', 'denied'),
        TargetPlatform.android,
      );
      expect(android.message, contains('Permissions > Camera'));
    });

    test('other camera errors keep their code and description', () {
      final error = cameraFailure(
        CameraException('initialization_timeout', 'no frames received'),
        TargetPlatform.macOS,
      );
      expect(
        error.message,
        'Camera error (initialization_timeout): no frames received',
      );
    });
  });

  test('pickCamera: back camera on phones, the first one elsewhere', () {
    expect(pickCamera([_front, _back], TargetPlatform.iOS), same(_back));
    expect(pickCamera([_front, _back], TargetPlatform.android), same(_back));
    expect(pickCamera([_front], TargetPlatform.android), same(_front));
    expect(
      pickCamera([kMacCamera, _back], TargetPlatform.macOS),
      same(kMacCamera),
    );
    expect(pickCamera([], TargetPlatform.macOS), isNull);
  });

  group('CameraFrameSource (fake controller)', () {
    late List<FakeCameraController> controllers;
    late List<CameraDescription> cameras;
    Object? initError;
    Exception? disposeError;
    Completer<void>? initGate;

    CameraFrameSource source({
      TargetPlatform platform = TargetPlatform.macOS,
    }) => CameraFrameSource(
      listCameras: () async => cameras,
      createController: (camera) {
        final c = FakeCameraController(
          camera,
          initError: initError,
          initGate: initGate,
          disposeError: disposeError,
        );
        controllers.add(c);
        return c;
      },
      platform: platform,
    );

    setUp(() {
      controllers = [];
      cameras = [kMacCamera];
      initError = null;
      disposeError = null;
      initGate = null;
    });

    test('start opens the first camera, streams frames as FrameViews, logs the '
        'CAMERA line once, and stop releases the camera', () async {
      final logs = <String>[];
      final original = debugPrint;
      debugPrint = (message, {wrapWidth}) => logs.add(message ?? '');
      addTearDown(() => debugPrint = original);
      final frames = <FrameView>[];
      final camera = source();

      final started = await camera.start(frames.add);

      final info = (started as Ok<FrameSourceInfo>).value;
      expect(info.label, 'camera_desktop 1280x720');
      expect((info.width, info.height), (1280, 720));
      expect(info.mirrored, isTrue);
      expect(info.previewMirrored, isTrue);
      expect(info.overlayMirrored, isFalse);
      final controller = controllers.single;
      expect(controller.description, same(kMacCamera));
      expect(controller.enableAudio, isFalse);
      expect(
        (camera.preview as CameraPreviewSource).controller,
        same(controller),
      );

      controller
        ..emit(bgraImage(bytesPerRow: 5184))
        ..emit(bgraImage(bytesPerRow: 5184));
      expect(frames, hasLength(2));
      expect(frames.first.planes.single.bytesPerRow, 5184);
      final cameraLines = logs.where((l) => l.startsWith('CAMERA ')).toList();
      expect(cameraLines, hasLength(1));
      expect(
        cameraLines.single,
        startsWith(
          'CAMERA src=camera_desktop 1280x720 bgra stride=5184 mirrored=true',
        ),
      );

      await camera.stop();
      expect(controller.stopStreamCalls, 1);
      expect(controller.disposed, isTrue);
      controller.emit(bgraImage());
      expect(frames, hasLength(2), reason: 'nothing after stop');
    });

    test('permission denied: an error that says how to grant access, and the '
        'controller is released', () async {
      initError = CameraException(
        'permission_denied',
        'Camera permission was denied',
      );

      final started = await source().start((_) {});

      expect((started as Error).error.toString(), contains('System Settings'));
      expect(controllers.single.disposed, isTrue);
      expect(controllers.single.startStreamCalls, 0);
    });

    test('camera_desktop\'s availableCameras throws a raw PlatformException: '
        'start returns an error with its code instead of throwing', () async {
      final camera = CameraFrameSource(
        listCameras: () async => throw PlatformException(
          code: 'enumeration_failed',
          message: 'AVCaptureDevice discovery failed',
        ),
        createController: (_) => fail('no controller without a camera'),
        platform: TargetPlatform.macOS,
      );

      final started = await camera.start((_) {});

      final message = (started as Error).error.toString();
      expect(message, contains('enumeration_failed'));
      expect(message, contains('AVCaptureDevice discovery failed'));
    });

    test('a non-camera error from initialize is an error result and the '
        'controller is released', () async {
      initError = StateError('plugin not registered');

      final started = await source().start((_) {});

      expect(
        (started as Error).error.toString(),
        contains('plugin not registered'),
      );
      expect(controllers.single.disposed, isTrue);
      expect(controllers.single.startStreamCalls, 0);
    });

    test('a release that throws (camera_avfoundation\'s unwrapped Pigeon '
        'dispose) is logged; stop completes', () async {
      final logs = <String>[];
      final original = debugPrint;
      debugPrint = (message, {wrapWidth}) => logs.add(message ?? '');
      addTearDown(() => debugPrint = original);
      disposeError = PlatformException(code: 'channel-error');
      final camera = source();
      await camera.start((_) {});

      await camera.stop();

      expect(controllers.single.disposed, isTrue);
      expect(logs, contains(contains('channel-error')));
    });

    test('camera_desktop\'s native startImageStream failure (inside the '
        'stream\'s onListen, which no future reports) reaches onError instead '
        'of being lost', () async {
      final errors = <Exception>[];
      final camera = CameraFrameSource(
        listCameras: () async => [kMacCamera],
        createController: (description) {
          final c = FakeCameraController(description)
            ..nativeStreamError = PlatformException(
              code: 'stream_failed',
              message: 'AVCaptureSession could not start',
            );
          controllers.add(c);
          return c;
        },
        platform: TargetPlatform.macOS,
      );

      await camera.start((_) {}, onError: errors.add);
      await settle();
      await settle();

      expect(errors, hasLength(1));
      expect(errors.single.toString(), contains('stream_failed'));
      expect(
        errors.single.toString(),
        contains('AVCaptureSession could not start'),
      );
      await camera.stop();
    });

    test('an error from startImageStream\'s future is a start error and the '
        'controller is released', () async {
      final camera = CameraFrameSource(
        listCameras: () async => [kMacCamera],
        createController: (description) {
          final c = FakeCameraController(description)
            ..startStreamError = PlatformException(code: 'stream_busy');
          controllers.add(c);
          return c;
        },
        platform: TargetPlatform.macOS,
      );

      final started = await camera.start((_) {});

      expect((started as Error).error.toString(), contains('stream_busy'));
      expect(controllers.single.disposed, isTrue);
    });

    test('no camera is an error', () async {
      cameras = [];

      final started = await source().start((_) {});

      expect((started as Error).error.toString(), contains('No camera found'));
      expect(controllers, isEmpty);
    });

    // failAtRuntime plays a CameraErrorEvent, which camera_desktop 2.0.0 on
    // macOS never delivers; see the native-stream and watchdog tests for the
    // real-world path there.
    test(
      'a runtime camera error and an unreadable format are reported once',
      () async {
        final errors = <Exception>[];
        final camera = source();
        await camera.start((_) {}, onError: errors.add);

        controllers.single
          ..failAtRuntime('device disconnected')
          ..failAtRuntime('again');
        expect(errors, hasLength(1));
        expect(errors.single.toString(), contains('device disconnected'));
        await camera.stop();

        final other = source();
        final otherErrors = <Exception>[];
        await other.start((_) {}, onError: otherErrors.add);
        controllers.last
          ..emit(jpegImage())
          ..emit(jpegImage());
        expect(otherErrors, hasLength(1));
        expect(otherErrors.single.toString(), contains('jpeg'));
        await other.stop();
      },
    );

    test('stop while the camera is initializing: start fails, the controller '
        'is disposed once and never streams', () async {
      initGate = Completer<void>();
      final camera = source();
      final starting = camera.start((_) {});
      await settle();

      final stopping = camera.stop();
      initGate!.complete();
      await stopping;
      final started = await starting;

      expect(started, isA<Error<FrameSourceInfo>>());
      expect(controllers.single.disposed, isTrue);
      expect(controllers.single.startStreamCalls, 0);
    });

    test('a phone opens the back camera with its rotation', () async {
      debugDefaultTargetPlatformOverride = TargetPlatform.android;
      addTearDown(() => debugDefaultTargetPlatformOverride = null);
      cameras = [_front, _back];
      final frames = <FrameView>[];
      final camera = source(platform: TargetPlatform.android);

      final started = await camera.start(frames.add);
      controllers.single.emit(nv21Image());

      expect(controllers.single.description, same(_back));
      expect((started as Ok<FrameSourceInfo>).value.overlayMirrored, isFalse);
      expect(frames.single.rotationDeg, 90, reason: 'portrait, sensor 90');
      await camera.stop();
    });
  });

  group('through LiveDetectionRepository', () {
    late List<FakeCameraController> controllers;
    late LiveDetectionRepository live;
    late FakeDetector detector;
    CameraException? initError;
    Exception? disposeError;
    Exception? nativeStreamError;
    final owner = Object();

    setUp(() {
      controllers = [];
      initError = null;
      disposeError = null;
      nativeStreamError = null;
      detector = FakeDetector(autoComplete: true);
      live = LiveDetectionRepository(
        detector: detector,
        createSource: (spec) => Result.ok(
          CameraFrameSource(
            listCameras: () async => [kMacCamera],
            createController: (camera) {
              final c = FakeCameraController(
                camera,
                initError: initError,
                disposeError: disposeError,
              )..nativeStreamError = nativeStreamError;
              controllers.add(c);
              return c;
            },
            platform: TargetPlatform.macOS,
          ),
        ),
      );
    });

    tearDown(() => live.close());

    test('start, detect, stop, restart: each start opens and each stop '
        'disposes its own controller', () async {
      expect(
        await live.start(const CameraSourceSpec(), owner: owner),
        isA<Ok<FrameSourceInfo>>(),
      );
      expect(live.state.value, isA<LiveRunning>());
      expect(
        (live.state.value as LiveRunning).source,
        'camera_desktop 1280x720',
      );
      expect(
        (live.preview.value! as CameraPreviewSource).controller,
        same(controllers.first),
      );
      controllers.first.emit(bgraImage(bytesPerRow: 5184));
      await settle();
      expect(detector.calls.single.planes.single.bytesPerRow, 5184);
      expect(live.frames.value, isNotNull);

      await live.stop(owner: owner);
      expect(controllers.first.disposed, isTrue);
      expect(live.preview.value, isNull);
      expect(live.state.value, isA<LiveStopped>());

      expect(
        await live.start(const CameraSourceSpec(), owner: owner),
        isA<Ok<FrameSourceInfo>>(),
      );
      expect(controllers, hasLength(2));
      controllers.last.emit(bgraImage());
      await settle();
      expect(detector.calls, hasLength(2));
      await live.stop(owner: owner);
      expect(controllers.last.disposed, isTrue);
    });

    test('permission denied ends in LiveFailed with the System Settings '
        'message, and a Retry start works once access is granted', () async {
      initError = CameraException(
        'permission_denied',
        'Camera permission was denied',
      );

      final started = await live.start(const CameraSourceSpec(), owner: owner);

      expect(started, isA<Error<FrameSourceInfo>>());
      final state = live.state.value;
      expect(state, isA<LiveFailed>());
      expect(
        (state as LiveFailed).message,
        contains('System Settings > Privacy & Security > Camera'),
      );
      expect(live.preview.value, isNull);

      initError = null;
      expect(
        await live.start(const CameraSourceSpec(), owner: owner),
        isA<Ok<FrameSourceInfo>>(),
      );
      expect(live.state.value, isA<LiveRunning>());
    });

    // Only for plugins that deliver a CameraErrorEvent: camera_desktop 2.0.0
    // on macOS never does (its `cameraError` is dropped), which the source
    // watchdog covers (live_detection_repository_test).
    test(
      'a runtime camera error fails the pipeline and releases the camera',
      () async {
        await live.start(const CameraSourceSpec(), owner: owner);

        controllers.single.failAtRuntime('device disconnected');
        await settle();
        await settle();

        final state = live.state.value;
        expect(state, isA<LiveFailed>());
        expect((state as LiveFailed).message, contains('device disconnected'));
        expect(controllers.single.disposed, isTrue);
      },
    );

    test('camera_desktop\'s failed native stream start fails the pipeline '
        'at once (no wait for the watchdog) and releases the camera', () async {
      nativeStreamError = PlatformException(
        code: 'stream_failed',
        message: 'AVCaptureSession could not start',
      );

      await live.start(const CameraSourceSpec(), owner: owner);
      await settle();
      await settle();

      final state = live.state.value;
      expect(state, isA<LiveFailed>());
      expect((state as LiveFailed).message, contains('stream_failed'));
      expect(controllers.single.disposed, isTrue);
    });

    test('a camera dispose that throws (camera_avfoundation) still ends '
        'Stopped, and the camera can be opened again', () async {
      disposeError = PlatformException(code: 'channel-error');
      await live.start(const CameraSourceSpec(), owner: owner);

      await live.stop(owner: owner);

      expect(live.state.value, isA<LiveStopped>());
      expect(live.preview.value, isNull);
      disposeError = null;
      expect(
        await live.start(const CameraSourceSpec(), owner: owner),
        isA<Ok<FrameSourceInfo>>(),
      );
    });
  });

  // The camera's "access is off" hint names the settings path of the
  // platform it runs on (Android has no "Privacy & Security" page).
  test('camera denied: macOS, iOS and Android each get their own path', () {
    String hint(TargetPlatform p) =>
        cameraFailure(CameraException('CameraAccessDenied', null), p).message;
    expect(hint(TargetPlatform.macOS), contains('System Settings'));
    expect(hint(TargetPlatform.iOS), contains('Settings > Privacy & Security'));
    expect(hint(TargetPlatform.android), isNot(contains('Privacy & Security')));
    expect(hint(TargetPlatform.android), contains('Permissions > Camera'));
  });
}
