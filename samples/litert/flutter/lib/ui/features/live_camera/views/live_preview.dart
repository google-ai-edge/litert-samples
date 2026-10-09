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

import 'dart:ui' as ui;

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show DeviceOrientation;

import '../../../../domain/models/preview_source.dart';

/// What the live view shows under the boxes, cover-fitted and centred — the
/// same fit the detection painter maps boxes with (`CoverFitTransform`).
/// Rebuilds only when the source, its image or the camera value changes,
/// never per detected frame.
class LivePreview extends StatelessWidget {
  const LivePreview({super.key, required this.preview});

  final ValueListenable<PreviewSource?> preview;

  @override
  Widget build(BuildContext context) {
    return ValueListenableBuilder<PreviewSource?>(
      valueListenable: preview,
      builder: (context, source, _) => switch (source) {
        null => const ColoredBox(color: Colors.black),
        ImagePreviewSource(:final image) => ValueListenableBuilder<ui.Image?>(
          valueListenable: image,
          builder: (context, frame, _) => frame == null
              ? const ColoredBox(color: Colors.black)
              : RawImage(image: frame, fit: BoxFit.cover),
        ),
        CameraPreviewSource(:final controller) => CameraCoverPreview(
          controller: controller,
        ),
      },
    );
  }
}

/// The camera texture with `BoxFit.cover`, centred. `CameraPreview` itself
/// letterboxes (an `AspectRatio`), so it is put in a box of the preview's
/// upright size and that box is cover-fitted.
class CameraCoverPreview extends StatelessWidget {
  const CameraCoverPreview({super.key, required this.controller});

  final CameraController controller;

  @override
  Widget build(BuildContext context) {
    return ValueListenableBuilder<CameraValue>(
      valueListenable: controller,
      builder: (context, value, _) {
        final size = uprightPreviewSize(value);
        if (size == null) return const ColoredBox(color: Colors.black);
        return ClipRect(
          child: FittedBox(
            fit: BoxFit.cover,
            child: SizedBox(
              width: size.width,
              height: size.height,
              child: CameraPreview(controller),
            ),
          ),
        );
      },
    );
  }
}

/// The preview's size as the user sees it: `previewSize` is reported in
/// landscape, so it is swapped when the applicable orientation is portrait —
/// the same rule `CameraPreview` uses for its aspect ratio. Null before
/// initialization.
@visibleForTesting
Size? uprightPreviewSize(CameraValue value) {
  final size = value.previewSize;
  if (!value.isInitialized || size == null) return null;
  final orientation = value.isRecordingVideo
      ? value.recordingOrientation
      : (value.previewPauseOrientation ??
            value.lockedCaptureOrientation ??
            value.deviceOrientation);
  final landscape =
      orientation == DeviceOrientation.landscapeLeft ||
      orientation == DeviceOrientation.landscapeRight;
  return landscape ? size : Size(size.height, size.width);
}
