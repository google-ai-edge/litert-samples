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
import 'dart:typed_data';

import 'package:litert_edge_demos/data/services/frames/frame_source.dart';
import 'package:litert_edge_demos/domain/models/frame_source_info.dart';

import 'frames.dart';

/// Things put on (or taken off) the synthetic desk, at the sizes a 720p
/// camera about 60 cm away sees them (≈ 15 px per cm).
enum DeskObject {
  /// A key ring (r 20 px) with a silver and a brass key: ≈ 95 × 45 px.
  keys,

  /// A black phone face down: 80 × 160 px.
  phone,

  /// A white mug with its handle: ≈ 135 × 130 px.
  mug,

  /// A red book lying flat, its light spine on the left: 260 × 180 px.
  book,

  /// An open dark-grey laptop: 520 × 420 px, screen and base.
  laptop,
}

/// A synthetic 1280×720 RGBA desk scene for the scene-change tests: a beige
/// wall with a dark poster above a wooden desk with grain, [objects] on it,
/// per-channel sensor noise of up to ±[noise] grey levels ([seed]), every
/// channel then scaled by [brightness] (clipped at 255). [shiftX] and
/// [shiftY] move the camera: the whole scene appears that many pixels to
/// the left and up (a hand shaking the phone).
TestFrame deskScene({
  Set<DeskObject> objects = const {},
  int noise = 3,
  int seed = 1,
  double brightness = 1,
  int shiftX = 0,
  int shiftY = 0,
}) {
  const w = 1280;
  const h = 720;
  final random = math.Random(seed);
  final bytes = Uint8List(w * h * 4);
  for (var y = 0; y < h; y++) {
    for (var x = 0; x < w; x++) {
      final (r, g, b) = _pixel(x + shiftX, y + shiftY, objects);
      final o = (y * w + x) * 4;
      bytes[o] = _channel(r, brightness, noise, random);
      bytes[o + 1] = _channel(g, brightness, noise, random);
      bytes[o + 2] = _channel(b, brightness, noise, random);
      bytes[o + 3] = 255;
    }
  }
  return TestFrame(w, h, FramePixelFormat.rgba8888, 0, [
    FramePlane(bytes: bytes, bytesPerRow: w * 4, bytesPerPixel: 4),
  ]);
}

/// A synthetic 1280×720 RGBA scene of the highest contrast a desk shows: a
/// black phone (grey 25) face down on white paper (grey 225), 80 × 160 px,
/// with per-channel sensor noise of up to ±[noise] ([seed]); [phone] false
/// leaves the paper empty. [shiftX] and [shiftY] move the camera as in
/// [deskScene].
TestFrame phoneOnPaper({
  bool phone = true,
  int noise = 3,
  int seed = 1,
  int shiftX = 0,
  int shiftY = 0,
}) {
  const w = 1280;
  const h = 720;
  final random = math.Random(seed);
  final bytes = Uint8List(w * h * 4);
  for (var y = 0; y < h; y++) {
    for (var x = 0; x < w; x++) {
      final dark = phone && _inRect(x + shiftX, y + shiftY, 560, 280, 640, 440);
      final v = dark ? 25.0 : 225.0;
      final o = (y * w + x) * 4;
      bytes[o] = _channel(v, 1, noise, random);
      bytes[o + 1] = _channel(v, 1, noise, random);
      bytes[o + 2] = _channel(v, 1, noise, random);
      bytes[o + 3] = 255;
    }
  }
  return TestFrame(w, h, FramePixelFormat.rgba8888, 0, [
    FramePlane(bytes: bytes, bytesPerRow: w * 4, bytesPerPixel: 4),
  ]);
}

int _channel(double value, double brightness, int noise, math.Random random) {
  final noisy = noise == 0
      ? value
      : value + random.nextInt(2 * noise + 1) - noise;
  return (noisy * brightness).round().clamp(0, 255);
}

bool _inRect(int x, int y, int x0, int y0, int x1, int y1) =>
    x >= x0 && x < x1 && y >= y0 && y < y1;

/// The scene's colour at ([x], [y]), the topmost object first.
(double, double, double) _pixel(int x, int y, Set<DeskObject> objects) {
  if (objects.contains(DeskObject.keys)) {
    final dx = x - 610;
    final dy = y - 470;
    final ring = (math.sqrt(dx * dx + dy * dy) - 20).abs() <= 2.5;
    if (ring || _inRect(x, y, 628, 463, 700, 476)) return (195, 195, 200);
    if (_inRect(x, y, 618, 481, 680, 492)) return (200, 170, 90);
  }
  if (objects.contains(DeskObject.phone) && _inRect(x, y, 300, 380, 380, 540)) {
    return (25, 25, 30);
  }
  if (objects.contains(DeskObject.mug)) {
    if (_inRect(x, y, 850, 360, 960, 490)) return (235, 232, 228);
    if (_inRect(x, y, 960, 390, 985, 450) &&
        !_inRect(x, y, 960, 402, 973, 438)) {
      return (235, 232, 228);
    }
  }
  if (objects.contains(DeskObject.book)) {
    if (_inRect(x, y, 420, 520, 440, 700)) return (220, 210, 190);
    if (_inRect(x, y, 440, 520, 680, 700)) return (150, 35, 45);
  }
  if (objects.contains(DeskObject.laptop)) {
    if (_inRect(x, y, 415, 155, 865, 415)) return (90, 110, 140);
    if (_inRect(x, y, 400, 140, 880, 430)) return (40, 45, 55);
    if (_inRect(x, y, 380, 430, 900, 560)) return (70, 70, 75);
  }
  if (y < 288) {
    if (_inRect(x, y, 900, 50, 1100, 230)) return (60, 70, 90);
    final shade = 0.92 + 0.08 * y / 288;
    return (205 * shade, 195 * shade, 175 * shade);
  }
  final grain = 12 * math.sin(y / 7 + 0.3 * math.sin(x / 90));
  return (150 + grain, 105 + grain, 65 + grain);
}
