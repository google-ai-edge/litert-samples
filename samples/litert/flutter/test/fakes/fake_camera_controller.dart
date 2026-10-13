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
import 'dart:typed_data';
import 'dart:ui' show Size;

import 'package:camera/camera.dart';
import 'package:camera_platform_interface/camera_platform_interface.dart'
    show CameraImageData, CameraImageFormat, CameraImagePlane;

/// The built-in MacBook camera as camera_desktop enumerates it.
const kMacCamera = CameraDescription(
  name: 'MacBook Pro Camera',
  lensDirection: CameraLensDirection.front,
  sensorOrientation: 0,
);

/// A `CameraController` without a platform: the test plays initialize, the
/// image stream and errors. Never touches `CameraPlatform` (its `initialize`
/// is replaced, so `dispose` skips the platform call).
class FakeCameraController extends CameraController {
  FakeCameraController(
    CameraDescription description, {
    this.initError,
    this.initGate,
    this.disposeError,
    this.previewSize = const Size(1280, 720),
  }) : super(description, ResolutionPreset.high, enableAudio: false);

  /// Thrown by [initialize]: a `CameraException`, or a raw
  /// `PlatformException` as some plugins leak.
  Object? initError;
  Completer<void>? initGate;

  /// Thrown by [dispose] after the controller is released, like
  /// camera_avfoundation's unwrapped Pigeon `PlatformException`.
  Exception? disposeError;

  /// Thrown by [startImageStream]'s future.
  Exception? startStreamError;

  /// Plays camera_desktop 2.0.0's stream start (`camera_desktop_plugin.dart`
  /// `onStreamedFrameAvailable`): the native `startImageStream` runs inside
  /// the frame stream's async `onListen`, so its failure is an uncaught error
  /// in the zone that called [startImageStream], whose own future succeeds.
  Exception? nativeStreamError;
  StreamController<CameraImage>? _nativeFrames;
  StreamSubscription<CameraImage>? _nativeStream;
  final Size previewSize;
  onLatestImageAvailable? _onImage;
  int initializeCalls = 0;
  int startStreamCalls = 0;
  int stopStreamCalls = 0;
  int disposeCalls = 0;
  bool disposed = false;

  bool get streaming => _onImage != null;

  @override
  Future<void> initialize() async {
    initializeCalls++;
    await initGate?.future;
    // A plugin may leak any throwable here (CameraException,
    // PlatformException, StateError); the tests inject each.
    // ignore: only_throw_errors
    if (initError case final error?) throw error;
    if (disposed) return;
    value = value.copyWith(isInitialized: true, previewSize: previewSize);
  }

  @override
  Future<void> startImageStream(onLatestImageAvailable onAvailable) async {
    startStreamCalls++;
    if (startStreamError case final error?) throw error;
    _onImage = onAvailable;
    value = value.copyWith(isStreamingImages: true);
    if (nativeStreamError case final error?) {
      _nativeFrames = StreamController<CameraImage>(
        onListen: () async {
          await Future<void>.delayed(Duration.zero); // the platform call
          throw error;
        },
      );
      _nativeStream = _nativeFrames?.stream.listen(onAvailable);
    }
  }

  @override
  Future<void> stopImageStream() async {
    stopStreamCalls++;
    _onImage = null;
    value = value.copyWith(isStreamingImages: false);
    await _closeNativeStream();
  }

  @override
  Future<void> dispose() async {
    disposeCalls++;
    if (disposed) return;
    disposed = true;
    _onImage = null;
    await _closeNativeStream();
    await super.dispose();
    if (disposeError case final error?) throw error;
  }

  Future<void> _closeNativeStream() async {
    await _nativeStream?.cancel();
    _nativeStream = null;
    await _nativeFrames?.close();
    _nativeFrames = null;
  }

  /// Delivers one frame to the stream, if it runs.
  void emit(CameraImage image) => _onImage?.call(image);

  /// Plays a `CameraErrorEvent` (the controller records its description).
  /// camera_desktop 2.0.0 on macOS never delivers one (its native
  /// `cameraError` sends `message`, the Dart side reads `description` and
  /// throws), so this path is real only for the other plugins.
  /// [nativeStreamError] and a source that stops emitting play the macOS
  /// failures.
  void failAtRuntime(String description) =>
      value = value.copyWith(errorDescription: description);
}

/// A BGRA frame as camera_desktop delivers it: one plane, rows padded to
/// [bytesPerRow].
CameraImage bgraImage({
  int width = 1280,
  int height = 720,
  int? bytesPerRow,
  Uint8List? bytes,
}) {
  final stride = bytesPerRow ?? width * 4;
  return CameraImage.fromPlatformInterface(
    CameraImageData(
      format: const CameraImageFormat(ImageFormatGroup.bgra8888, raw: 'BGRA'),
      width: width,
      height: height,
      planes: [
        CameraImagePlane(
          bytes: bytes ?? Uint8List(stride * height),
          bytesPerRow: stride,
          bytesPerPixel: 4,
          width: width,
          height: height,
        ),
      ],
    ),
  );
}

/// A CameraX NV21 frame: one plane (Y then V U), bytesPerPixel 1.
CameraImage nv21Image({int width = 640, int height = 480}) =>
    CameraImage.fromPlatformInterface(
      CameraImageData(
        format: const CameraImageFormat(ImageFormatGroup.nv21, raw: 17),
        width: width,
        height: height,
        planes: [
          CameraImagePlane(
            bytes: Uint8List(width * height * 3 ~/ 2),
            bytesPerRow: width,
            bytesPerPixel: 1,
          ),
        ],
      ),
    );

/// A frame format the detector cannot read.
CameraImage jpegImage() => CameraImage.fromPlatformInterface(
  CameraImageData(
    format: const CameraImageFormat(ImageFormatGroup.jpeg, raw: 256),
    width: 4,
    height: 4,
    planes: [CameraImagePlane(bytes: Uint8List(16), bytesPerRow: 16)],
  ),
);
