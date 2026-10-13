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

/// Pixel layouts the detector's gather understands.
enum FramePixelFormat {
  /// Apple camera frames: one plane, B G R A.
  bgra8888,

  /// Fixture frames (decoded by the engine): one plane, R G B A.
  rgba8888,

  /// Android CameraX `nv21`: one plane (Y, then interleaved V U, both with
  /// the plane's row stride) or two planes (Y; V U).
  nv21,

  /// Android `YUV_420_888`: three planes Y, U, V; U and V share row and pixel
  /// stride.
  yuv420,
}

/// What a started source reports.
final class const FrameSourceInfo({
  /// Shown in the overlay and the log, e.g. `fixture (35 images)`.
  required final String label,
  required final int width,
  required final int height,
  required final FramePixelFormat format,

  /// The streamed frames are mirrored (camera_desktop on macOS mirrors at
  /// capture, so detections come out in mirrored coordinates).
  required final bool mirrored,

  /// The preview the user sees is mirrored.
  final bool previewMirrored = false,

  /// How long the live repository waits for a frame before it fails the
  /// pipeline; null is its default (`kSourceStallTimeout`). A network
  /// camera reports its own stall first and sets a longer backstop here.
  final Duration? stallTimeout,

  /// The network camera's JPEG decoder (`TurboJPEG (worker isolate…)`,
  /// `engine codec`); null for sources that decode nothing.
  final String? decoder,
}) {
  /// Whether the box overlay must be mirrored to line up with the preview:
  /// exactly when one of preview and frames is mirrored and the other is not.
  /// camera_desktop on macOS mirrors both, so boxes are drawn unmirrored.
  bool get overlayMirrored => previewMirrored != mirrored;
}
