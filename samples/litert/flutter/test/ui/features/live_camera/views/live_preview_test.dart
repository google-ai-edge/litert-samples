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

import 'dart:ui' show Size;

import 'package:camera/camera.dart';
import 'package:flutter/services.dart' show DeviceOrientation;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/ui/features/live_camera/views/live_preview.dart';

import '../../../../fakes/fake_camera_controller.dart';

void main() {
  CameraValue initialized(DeviceOrientation orientation) =>
      const CameraValue.uninitialized(kMacCamera).copyWith(
        isInitialized: true,
        previewSize: const Size(1280, 720),
        deviceOrientation: orientation,
      );

  test('the cover box has the preview\'s upright size, like CameraPreview\'s '
      'aspect ratio: landscape as reported, portrait swapped', () {
    expect(
      uprightPreviewSize(initialized(DeviceOrientation.landscapeLeft)),
      const Size(1280, 720),
    );
    expect(
      uprightPreviewSize(initialized(DeviceOrientation.portraitUp)),
      const Size(720, 1280),
    );
    expect(
      uprightPreviewSize(
        initialized(DeviceOrientation.portraitUp).copyWith(
          lockedCaptureOrientation: const Optional.of(
            DeviceOrientation.landscapeRight,
          ),
        ),
      ),
      const Size(1280, 720),
      reason: 'a locked orientation wins',
    );
    expect(
      uprightPreviewSize(const CameraValue.uninitialized(kMacCamera)),
      isNull,
    );
  });
}
