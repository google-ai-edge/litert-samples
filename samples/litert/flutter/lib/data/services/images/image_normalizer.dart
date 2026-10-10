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

import '../../../config/model_catalog.dart';

/// What [normalizeForLlm] produced.
final class const NormalizedImage({
  required final Uint8List png,
  required final int width,
  required final int height,

  /// The input's size with its EXIF orientation applied.
  required final int sourceWidth,
  required final int sourceHeight,
});

/// The engine could not decode the picked bytes.
final class UndecodableImageException implements Exception {
  const UndecodableImageException(this.detail);

  final String detail;

  @override
  String toString() =>
      'This image could not be read (JPEG, PNG, HEIC or WebP expected): '
      '$detail';
}

/// Decodes [encoded] with the engine's codecs (JPEG, PNG, GIF, WebP, BMP, and
/// HEIC on Apple platforms), applies its EXIF orientation, scales it so the
/// long side is at most [maxSide], and re-encodes it as PNG — the format
/// LiteRT-LM's stb_image decoder always reads (it has no HEIC or WebP).
///
/// Runs on the engine's IO and raster threads; the main isolate only awaits.
/// Throws [UndecodableImageException] for bytes no codec accepts.
Future<NormalizedImage> normalizeForLlm(
  Uint8List encoded, {
  int maxSide = kLlmImageMaxSide,
}) async {
  final buffer = await ui.ImmutableBuffer.fromUint8List(encoded);
  ui.ImageDescriptor? descriptor;
  try {
    try {
      // Width and height already have the EXIF orientation applied.
      descriptor = await ui.ImageDescriptor.encoded(buffer);
    } catch (e) {
      throw UndecodableImageException('$e');
    }
    final width = descriptor.width;
    final height = descriptor.height;
    final scale = math.min(1.0, maxSide / math.max(width, height));
    final codec = await descriptor.instantiateCodec(
      targetWidth: math.max(1, (width * scale).round()),
      targetHeight: math.max(1, (height * scale).round()),
    );
    try {
      final image = (await codec.getNextFrame()).image;
      try {
        final data = await image.toByteData(format: ui.ImageByteFormat.png);
        if (data == null) {
          throw const UndecodableImageException(
            'PNG encoding returned nothing',
          );
        }
        return NormalizedImage(
          png: data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes),
          width: image.width,
          height: image.height,
          sourceWidth: width,
          sourceHeight: height,
        );
      } finally {
        image.dispose();
      }
    } finally {
      codec.dispose();
    }
  } finally {
    descriptor?.dispose();
    buffer.dispose();
  }
}

/// A [side]×[side] single-colour PNG: the warm-up image that makes the vision
/// encoder's first use happen at setup.
Future<Uint8List> warmUpPng({int side = 32}) async {
  final recorder = ui.PictureRecorder();
  ui.Canvas(recorder).drawColor(const ui.Color(0xFF808080), ui.BlendMode.src);
  final picture = recorder.endRecording();
  try {
    final image = await picture.toImage(side, side);
    try {
      final data = await image.toByteData(format: ui.ImageByteFormat.png);
      if (data == null) throw StateError('PNG encoding returned nothing');
      return data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
    } finally {
      image.dispose();
    }
  } finally {
    picture.dispose();
  }
}
