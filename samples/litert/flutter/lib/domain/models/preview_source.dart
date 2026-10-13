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

import 'package:camera/camera.dart' show CameraController;
import 'package:flutter/foundation.dart';

/// What the live view shows under the boxes: a source's preview, which the
/// live camera screen draws (`LivePreview`).
///
/// The camera variant carries the camera plugin's [CameraController]: its
/// texture can only be drawn through it (`CameraPreview`), so this model
/// holds one plugin type, and the view never reaches into a service for it.
sealed class const PreviewSource();

/// The live camera texture of an initialized [controller].
final class const CameraPreviewSource(final CameraController controller)
    extends PreviewSource;

/// A still image that changes now and then (the fixture slideshow, a
/// network camera's decoded frames).
final class const ImagePreviewSource(final ValueListenable<ui.Image?> image)
    extends PreviewSource;
