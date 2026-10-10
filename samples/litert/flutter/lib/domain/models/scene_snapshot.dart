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

import 'dart:typed_data';

import 'detection.dart';
import 'detection_summary.dart';

/// An upright frame as tightly packed R G B A bytes (`width × height × 4`),
/// in the same orientation and mirroring as its detections.
final class RgbaPixels {
  RgbaPixels({required this.width, required this.height, required this.bytes})
    : assert(width > 0 && height > 0),
      assert(bytes.length == width * height * 4);

  final int width;
  final int height;
  final Uint8List bytes;
}

/// The frame a camera question is about: the next frame after
/// the mic was released, detected on its own, plus the summary of the
/// recent window ending with it, and the frame's pixels (for the frozen
/// view and the PNG sent to Gemma).
final class const SceneSnapshot({
  required final int frameId,

  /// This frame's own detections (the boxes a frozen view would show).
  required final DetectionFrame detections,

  /// The fast path's basis: the median counts over the last frames,
  /// including this one.
  required final DetectionSummary summary,

  /// The frame itself, upright, as the detector saw it (mirrored when
  /// [mirrored]); the same size as [detections].
  required final RgbaPixels pixels,

  /// The source mirrors its frames (camera_desktop on macOS mirrors at
  /// capture): Gemma must get the frame mirrored back.
  required final bool mirrored,

  /// The live preview the user saw is mirrored. A frozen view flips the
  /// frame when this differs from [mirrored], so it looks like the preview.
  final bool previewMirrored = false,

  /// From the capture request to the detection result.
  required final Duration latency,
});

/// The PNG of one snapshot as sent to Gemma: at most
/// `kLlmImageMaxSide` on the long side, mirrored back when the source
/// mirrors.
final class const EncodedSnapshot({
  required final Uint8List png,

  /// The snapshot this was encoded from (the frozen frame's id).
  required final int frameId,
  required final int width,
  required final int height,

  /// The PNG was mirrored back (the source mirrors its frames).
  required final bool unmirrored,

  /// Raw pixels → scaled image → (flip) → PNG.
  required final Duration encodeTime,
});

/// Why a question's frame could not be captured; each is said differently.
enum CaptureFailure {
  /// No source runs (or it stopped before a frame came).
  notRunning,

  /// The source runs, but no frame was detected within the capture timeout.
  timedOut,

  /// The pipeline failed (the detector, a frame copy, the camera).
  failed,
}

/// `LiveDetectionRepository.capture` could not deliver a snapshot.
final class CaptureUnavailableException implements Exception {
  const CaptureUnavailableException(
    this.message, {
    this.kind = CaptureFailure.notRunning,
  });

  final String message;
  final CaptureFailure kind;

  @override
  String toString() => message;
}
