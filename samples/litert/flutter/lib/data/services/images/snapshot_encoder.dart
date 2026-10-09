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

import '../../../domain/models/scene_snapshot.dart';

/// Turns a question's snapshot pixels into what the UI and Gemma need, with the
/// engine's codecs on its IO and raster threads; the main isolate only awaits.
///
/// - [toImage]: the frame at full size for the frozen view (`RawImage`).
/// - [toPng]: the frame for Gemma — at most `maxSide` on the long side,
///   mirrored back when the source mirrors (LiteRT-LM's stb_image reads PNG).
final class SnapshotEncoder {
  const SnapshotEncoder();

  /// [pixels] as an image, scaled so the long side is at most [maxSide]
  /// (never up). The caller owns it and disposes it.
  Future<ui.Image> toImage(RgbaPixels pixels, {int? maxSide}) async {
    final buffer = await ui.ImmutableBuffer.fromUint8List(pixels.bytes);
    final descriptor = ui.ImageDescriptor.raw(
      buffer,
      width: pixels.width,
      height: pixels.height,
      pixelFormat: ui.PixelFormat.rgba8888,
    );
    try {
      final (width, height) = fitWithin(
        pixels.width,
        pixels.height,
        maxSide ?? math.max(pixels.width, pixels.height),
      );
      final codec = await descriptor.instantiateCodec(
        targetWidth: width,
        targetHeight: height,
      );
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

  /// The PNG Gemma gets for snapshot [frameId]: scaled to at most [maxSide],
  /// flipped horizontally when [unmirror] (drawn on a canvas with
  /// `scale(-1, 1)`), then encoded.
  Future<EncodedSnapshot> toPng(
    RgbaPixels pixels, {
    required int frameId,
    required int maxSide,
    required bool unmirror,
  }) async {
    final watch = Stopwatch()..start();
    final scaled = await toImage(pixels, maxSide: maxSide);
    ui.Image? flipped;
    try {
      final image = unmirror ? flipped = await _flipped(scaled) : scaled;
      final data = await image.toByteData(format: ui.ImageByteFormat.png);
      if (data == null) throw StateError('PNG encoding returned nothing');
      return EncodedSnapshot(
        png: data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes),
        frameId: frameId,
        width: image.width,
        height: image.height,
        unmirrored: unmirror,
        encodeTime: watch.elapsed,
      );
    } finally {
      flipped?.dispose();
      scaled.dispose();
    }
  }

  static Future<ui.Image> _flipped(ui.Image image) async {
    final recorder = ui.PictureRecorder();
    ui.Canvas(recorder)
      ..translate(image.width.toDouble(), 0)
      ..scale(-1, 1)
      ..drawImage(image, ui.Offset.zero, ui.Paint());
    final picture = recorder.endRecording();
    try {
      return await picture.toImage(image.width, image.height);
    } finally {
      picture.dispose();
    }
  }
}

/// [width]×[height] scaled so the long side is at most [maxSide], keeping
/// the aspect ratio; never scaled up, never below 1 px.
@visibleForTesting
(int, int) fitWithin(int width, int height, int maxSide) {
  final scale = math.min(1.0, maxSide / math.max(width, height));
  return (
    math.max(1, (width * scale).round()),
    math.max(1, (height * scale).round()),
  );
}
