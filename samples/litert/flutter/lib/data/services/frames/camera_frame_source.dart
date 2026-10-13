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

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart'
    show DeviceOrientation, PlatformException;
import 'package:flutter_litert/flutter_litert.dart'
    show CameraFrameRotation, rotationForFrame;

import '../../../config/live_camera_config.dart';
import '../../../domain/models/frame_source_info.dart';
import '../../../domain/models/frame_source_spec.dart';
import '../../../domain/models/preview_source.dart';
import '../../../utils/result.dart';
import 'frame_source.dart';

/// The `camera` platform implementation in use, for labels and logs.
String cameraPluginName(TargetPlatform platform) => switch (platform) {
  TargetPlatform.macOS ||
  TargetPlatform.linux ||
  TargetPlatform.windows => 'camera_desktop',
  TargetPlatform.iOS => 'camera_avfoundation',
  TargetPlatform.android => 'camera_android_camerax',
  TargetPlatform.fuchsia => 'camera',
};

/// The frame format to request. camera_desktop ignores it and always
/// streams BGRA; camera_android_camerax offers only `yuv420` and `nv21`.
ImageFormatGroup cameraFormatGroup(TargetPlatform platform) =>
    platform == TargetPlatform.android
    ? ImageFormatGroup.nv21
    : ImageFormatGroup.bgra8888;

/// Whether each plugin mirrors the preview and the streamed frames, read
/// from the resolved sources:
/// - camera_desktop 2.0.0 on macOS and Linux mirrors at capture
///   (`CameraSession.swift`: `connection.isVideoMirrored = true`), so preview
///   and frames are both mirrored; on Windows only the preview widget is
///   flipped (README, `buildPreview`).
/// - camera_avfoundation 0.10.3+1 mirrors the front camera's capture
///   connection (`DefaultCamera.swift`), which feeds preview and stream.
/// - camera_android_camerax 0.7.5+1 flips only the front camera's preview
///   widget (`image_reader_rotated_preview.dart`).
///
/// The phone rows are unverified on a device.
({bool preview, bool frames}) cameraMirroring(
  TargetPlatform platform,
  CameraLensDirection lens,
) => switch (platform) {
  TargetPlatform.macOS || TargetPlatform.linux => (preview: true, frames: true),
  TargetPlatform.windows => (preview: true, frames: false),
  TargetPlatform.iOS => (
    preview: lens == CameraLensDirection.front,
    frames: lens == CameraLensDirection.front,
  ),
  TargetPlatform.android => (
    preview: lens == CameraLensDirection.front,
    frames: false,
  ),
  TargetPlatform.fuchsia => (preview: false, frames: false),
};

/// The camera to open: the back camera on phones, the first one elsewhere. Null
/// when there is none.
CameraDescription? pickCamera(
  List<CameraDescription> cameras,
  TargetPlatform platform,
) {
  if (cameras.isEmpty) return null;
  if (platform == TargetPlatform.iOS || platform == TargetPlatform.android) {
    for (final camera in cameras) {
      if (camera.lensDirection == CameraLensDirection.back) return camera;
    }
  }
  return cameras.first;
}

/// Clockwise degrees that turn a frame upright: `rotationForFrame` from
/// flutter_litert, which is null on desktop (camera_desktop frames are
/// already upright), so 0 there.
int cameraFrameRotation({
  required int width,
  required int height,
  required CameraDescription camera,
  required DeviceOrientation deviceOrientation,
}) => switch (rotationForFrame(
  width: width,
  height: height,
  sensorOrientation: camera.sensorOrientation,
  isFrontCamera: camera.lensDirection == CameraLensDirection.front,
  deviceOrientation: deviceOrientation,
)) {
  null => 0,
  CameraFrameRotation.cw90 => 90,
  CameraFrameRotation.cw180 => 180,
  CameraFrameRotation.cw270 => 270,
};

/// The `CameraException` codes for "the user or a policy denied the camera"
/// (camera_desktop: `permission_denied`; camera_avfoundation and
/// camera_android_camerax: `CameraAccess*`).
const _accessDeniedCodes = {
  'permission_denied',
  'CameraAccessDenied',
  'CameraAccessDeniedWithoutPrompt',
  'CameraAccessRestricted',
};

/// A camera failure as the message the user sees. Access denied says how to
/// grant it; Retry is on the screen.
FrameSourceUnavailableException cameraFailure(
  CameraException error,
  TargetPlatform platform,
) {
  if (_accessDeniedCodes.contains(error.code)) {
    return FrameSourceUnavailableException(switch (platform) {
      TargetPlatform.macOS =>
        'Camera access is off for this app. Open System Settings > Privacy & '
            'Security > Camera, turn on LiteRT Demos, then press Retry.',
      TargetPlatform.iOS =>
        'Camera access is off for this app. Open Settings > Privacy & '
            'Security > Camera, turn it on, then press Retry.',
      TargetPlatform.android =>
        'Camera access is off for this app. Open Settings > Apps > '
            'LiteRT Demos > Permissions > Camera, allow it, then press '
            'Retry.',
      _ =>
        'Camera access was denied (${error.code}). Allow it, then press '
            'Retry.',
    });
  }
  return FrameSourceUnavailableException(
    'Camera error (${error.code}): ${error.description ?? 'no details'}',
  );
}

/// Any camera plugin failure as the message the user sees. Not every plugin
/// wraps its errors: camera_desktop's `availableCameras` throws a raw
/// `PlatformException`, camera_avfoundation's Pigeon calls do too, so those
/// are read by code like a `CameraException`; anything else by its text.
FrameSourceUnavailableException cameraFailureOf(
  Object error,
  TargetPlatform platform,
) => switch (error) {
  CameraException() => cameraFailure(error, platform),
  PlatformException(:final code, :final message) => cameraFailure(
    CameraException(code, message),
    platform,
  ),
  _ => FrameSourceUnavailableException('Camera error: $error'),
};

/// A [CameraImage] as a [FrameView]. Valid only during the stream callback:
/// the plugin may reuse the bytes.
final class CameraFrameView implements FrameView {
  CameraFrameView._(this._image, this.format, this.rotationDeg, this.planes);

  /// Wraps [image]; null when its format has no gather (e.g. JPEG, or an
  /// unknown group).
  static CameraFrameView? wrap(CameraImage image, {required int rotationDeg}) {
    final format = switch (image.format.group) {
      ImageFormatGroup.bgra8888 => FramePixelFormat.bgra8888,
      ImageFormatGroup.nv21 => FramePixelFormat.nv21,
      ImageFormatGroup.yuv420 => FramePixelFormat.yuv420,
      ImageFormatGroup.jpeg || ImageFormatGroup.unknown => null,
    };
    if (format == null) return null;
    final defaultBytesPerPixel = format == FramePixelFormat.bgra8888 ? 4 : 1;
    return CameraFrameView._(image, format, rotationDeg, [
      for (final plane in image.planes)
        FramePlane(
          bytes: plane.bytes,
          // Rows can be padded (macOS in particular): always the plane's.
          bytesPerRow: plane.bytesPerRow,
          bytesPerPixel: plane.bytesPerPixel ?? defaultBytesPerPixel,
        ),
    ]);
  }

  final CameraImage _image;

  @override
  final FramePixelFormat format;

  @override
  final int rotationDeg;

  @override
  final List<FramePlane> planes;

  @override
  int get width => _image.width;

  @override
  int get height => _image.height;
}

/// Builds the controller for [camera]; the app's is [createCameraController],
/// tests pass a fake.
typedef CameraControllerFactory = CameraController Function(
  CameraDescription camera,
);

/// `CameraController(camera, high, enableAudio: false, fps: 30, BGRA on
/// Apple / NV21 on Android)`.
CameraController createCameraController(CameraDescription camera) =>
    CameraController(
      camera,
      kCameraPreset,
      enableAudio: false,
      fps: kCameraFps,
      imageFormatGroup: cameraFormatGroup(defaultTargetPlatform),
    );

/// The device camera as a [FrameSource]: `CameraController` +
/// `startImageStream` (camera_desktop on macOS). [stop] stops the stream and
/// disposes the controller, which turns the camera off.
final class CameraFrameSource with SingleUseStart {
  CameraFrameSource({
    this._listCameras = availableCameras,
    this._createController = createCameraController,
    TargetPlatform? platform,
  }) : _platform = platform ?? defaultTargetPlatform;

  final Future<List<CameraDescription>> Function() _listCameras;
  final CameraControllerFactory _createController;
  final TargetPlatform _platform;

  CameraController? _controller;
  CameraDescription? _camera;
  CameraPreviewSource? _preview;
  ({bool preview, bool frames}) _mirroring = (preview: false, frames: false);
  bool _logged = false;

  @override
  PreviewSource get preview =>
      _preview ??
      (throw StateError('CameraFrameSource.preview before a successful start'));

  @override
  String get sourceKind => 'camera';

  @override
  FrameSourceUnavailableException get stoppedWhileStarting =>
      const FrameSourceUnavailableException(
        'Stopped while the camera was starting',
      );

  /// Lists the cameras, opens the chosen one and starts its image stream.
  @override
  Future<Result<FrameSourceInfo>> acquire(
    void Function(FrameView) onFrame, {
    void Function(Exception error)? onError,
  }) async {
    final plugin = cameraPluginName(_platform);
    final List<CameraDescription> cameras;
    try {
      cameras = await _listCameras();
    } catch (e, st) {
      debugPrint('[CameraFrameSource] availableCameras failed: $e\n$st');
      return Result.error(cameraFailureOf(e, _platform));
    }
    if (stopped) return Result.error(stoppedWhileStarting);
    final camera = pickCamera(cameras, _platform);
    if (camera == null) {
      return Result.error(
        FrameSourceUnavailableException('No camera found ($plugin)'),
      );
    }
    _camera = camera;
    _mirroring = cameraMirroring(_platform, camera.lensDirection);
    final controller = _controller = _createController(camera);
    try {
      await controller.initialize();
      // stop() released the controller.
      if (stopped) return Result.error(stoppedWhileStarting);
      attach(onFrame, onError: onError);
      controller.addListener(_onControllerValue);
      await _startStream(controller);
    } catch (e, st) {
      debugPrint('[CameraFrameSource] starting the camera failed: $e\n$st');
      await release();
      return Result.error(cameraFailureOf(e, _platform));
    }
    if (stopped) return Result.error(stoppedWhileStarting);
    final size = controller.value.previewSize;
    final width = size?.width.round() ?? 0;
    final height = size?.height.round() ?? 0;
    _preview = CameraPreviewSource(controller);
    return Result.ok(
      FrameSourceInfo(
        label: '$plugin ${width}x$height',
        width: width,
        height: height,
        format: switch (cameraFormatGroup(_platform)) {
          ImageFormatGroup.nv21 => FramePixelFormat.nv21,
          _ => FramePixelFormat.bgra8888,
        },
        mirrored: _mirroring.frames,
        previewMirrored: _mirroring.preview,
      ),
    );
  }

  void _onImage(CameraImage image) {
    final onFrame = frameCallback;
    final controller = _controller;
    final camera = _camera;
    if (onFrame == null || controller == null || camera == null) return;
    final rotation = cameraFrameRotation(
      width: image.width,
      height: image.height,
      camera: camera,
      deviceOrientation: controller.value.deviceOrientation,
    );
    final view = CameraFrameView.wrap(image, rotationDeg: rotation);
    if (view == null) {
      _reportError(
        FrameSourceUnavailableException(
          'The camera delivers ${image.format.group.name} frames, which the '
          'detector cannot read',
        ),
      );
      return;
    }
    if (!_logged) {
      _logged = true;
      final format = switch (view.format) {
        FramePixelFormat.bgra8888 => 'bgra',
        FramePixelFormat.rgba8888 => 'rgba',
        FramePixelFormat.nv21 => 'nv21',
        FramePixelFormat.yuv420 => 'yuv420',
      };
      debugPrint(
        'CAMERA src=${cameraPluginName(_platform)} '
        '${image.width}x${image.height} $format '
        'stride=${image.planes.first.bytesPerRow} '
        'mirrored=${_mirroring.frames} preview_mirrored=${_mirroring.preview} '
        'rotation=$rotation luma=${meanLuma(view).round()} '
        'camera="${camera.name}"',
      );
    }
    onFrame(view);
  }

  /// `startImageStream` in a guarded zone. camera_desktop 2.0.0 runs the
  /// native `startImageStream` inside the frame stream's async `onListen`
  /// (`camera_desktop_plugin.dart`, `onStreamedFrameAvailable`): its failure
  /// completes no future we hold and would be an uncaught error in the
  /// calling zone. Here it, and any later error in the stream's callbacks or
  /// the plugin's frame poller (both run in this zone), goes to `onError`.
  /// Errors of the returned future itself fail [start] as usual.
  Future<void> _startStream(CameraController controller) {
    final started = Completer<void>();
    runZonedGuarded(
      // Both outcomes go to [started]; the chained future cannot fail.
      () => unawaited(
        controller
            .startImageStream(_onImage)
            .then(started.complete, onError: started.completeError),
      ),
      _onStreamZoneError,
    );
    return started.future;
  }

  void _onStreamZoneError(Object error, StackTrace stack) {
    debugPrint('[CameraFrameSource] camera stream error: $error\n$stack');
    _reportError(cameraFailureOf(error, _platform));
  }

  /// A runtime camera error (unplugged, interrupted) ends the stream. Not
  /// every plugin delivers one (camera_desktop 2.0.0 on macOS never does):
  /// the repository's source watchdog covers the rest.
  void _onControllerValue() {
    final controller = _controller;
    if (controller == null || !controller.value.hasError) return;
    _reportError(
      FrameSourceUnavailableException(
        'Camera error: ${controller.value.errorDescription}',
      ),
    );
  }

  void _reportError(Exception error) {
    if (!markFailed()) return;
    debugPrint('[CameraFrameSource] $error');
    errorCallback?.call(error);
  }

  /// Stops the stream and disposes the controller (the camera turns off):
  /// on stop, and when start fails. Never throws: a plugin failure here
  /// (camera_avfoundation's `dispose` is an unwrapped Pigeon call) is
  /// logged, so a stop always completes.
  @override
  Future<void> release() async {
    final controller = _controller;
    _controller = null;
    _preview = null;
    if (controller == null) return;
    controller.removeListener(_onControllerValue);
    try {
      if (controller.value.isStreamingImages) {
        await controller.stopImageStream();
      }
    } catch (e, st) {
      debugPrint('[CameraFrameSource] stopImageStream failed: $e\n$st');
    }
    try {
      await controller.dispose();
    } catch (e, st) {
      debugPrint('[CameraFrameSource] dispose failed: $e\n$st');
    }
  }
}
