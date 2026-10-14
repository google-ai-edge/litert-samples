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

import '../../../config/live_camera_config.dart';
import '../../../domain/models/scene_snapshot.dart';
import '../../../utils/result.dart';
import '../detector/memory_shot.dart';
import '../frames/engine_image_decoder.dart';
import 'png_encoder.dart';

/// An encoded photo (JPEG, PNG…) as the visual memory would store it: decoded
/// as the fixture source decodes a picture (long side at most
/// [kFixtureMaxSide]), then the detector worker's own steps — the area
/// downscale to at most [maxSide] ([downscaleRgb]) and the filter-0 PNG
/// ([encodePngRgb]). Self-test step 7c embeds this, so it checks the image
/// path the memory uses, not the photo as it is. Error when the bytes do
/// not decode. The decode runs on the engine's threads.
Future<Result<Uint8List>> memoryFramePng(
  Uint8List encoded, {
  int maxSide = kMemoryThumbMaxSide,
}) async {
  try {
    final decoded = await decodeEncodedImage(encoded, maxSide: kFixtureMaxSide);
    final width = decoded.image.width;
    final height = decoded.image.height;
    decoded.image.dispose();
    final (w, h) = fitLongSide(width, height, maxSide);
    final rgb = downscaleRgb(
      RgbaPixels(width: width, height: height, bytes: decoded.rgba),
      w,
      h,
    );
    return Result.ok(encodePngRgb(w, h, rgb));
  } catch (e) {
    return Result.error(asException(e));
  }
}
