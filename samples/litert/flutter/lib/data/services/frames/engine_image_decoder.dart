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

import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:flutter/foundation.dart';

/// An encoded image decoded by the engine: the `ui.Image` for a preview and
/// its packed RGBA bytes for the detector.
final class const DecodedImage({
  /// Owned by the caller, who must dispose it.
  required final ui.Image image,

  /// `width × height × 4`, packed rows, R G B A.
  required final Uint8List rgba,
});

/// Decodes [encoded] (JPEG, PNG, …) with the engine's codecs, downscaled so
/// the long side is at most [maxSide]. The decode and the RGBA read-back run
/// on the engine's threads, not on the UI thread; the Dart side only awaits.
/// Throws when the bytes do not decode.
Future<DecodedImage> decodeEncodedImage(
  Uint8List encoded, {
  required int maxSide,
}) async {
  final buffer = await ui.ImmutableBuffer.fromUint8List(encoded);
  final ui.ImageDescriptor descriptor;
  try {
    descriptor = await ui.ImageDescriptor.encoded(buffer);
  } finally {
    buffer.dispose();
  }
  ui.Codec? codec;
  try {
    final longSide = math.max(descriptor.width, descriptor.height);
    final scale = longSide > maxSide ? maxSide / longSide : 1.0;
    codec = await descriptor.instantiateCodec(
      targetWidth: scale < 1 ? (descriptor.width * scale).round() : null,
      targetHeight: scale < 1 ? (descriptor.height * scale).round() : null,
    );
    final image = (await codec.getNextFrame()).image;
    final ByteData? rgba;
    try {
      rgba = await image.toByteData();
    } catch (_) {
      image.dispose();
      rethrow;
    }
    if (rgba == null) {
      image.dispose();
      throw StateError('the engine returned no RGBA bytes');
    }
    return DecodedImage(
      image: image,
      rgba: rgba.buffer.asUint8List(rgba.offsetInBytes, rgba.lengthInBytes),
    );
  } finally {
    codec?.dispose();
    descriptor.dispose();
  }
}

/// Packed RGBA ([width]×[height]) as an image: the mirrored fixture's
/// preview, and the network camera's preview when a CPU decoder made the
/// pixels. Copies [rgba] into an engine buffer; the caller owns the image.
Future<ui.Image> imageFromRgba(Uint8List rgba, int width, int height) async {
  final buffer = await ui.ImmutableBuffer.fromUint8List(rgba);
  final descriptor = ui.ImageDescriptor.raw(
    buffer,
    width: width,
    height: height,
    pixelFormat: ui.PixelFormat.rgba8888,
  );
  try {
    final codec = await descriptor.instantiateCodec();
    try {
      return (await codec.getNextFrame()).image;
    } finally {
      codec.dispose();
    }
  } finally {
    descriptor.dispose();
    buffer.dispose();
  }
}
