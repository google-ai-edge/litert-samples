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

/// A tangent-space normal map of small water ripples, RGBA8, [size]×[size].
///
/// Built from a sum of sines whose wave vectors are whole numbers of cycles
/// across the texture, so it tiles without a seam: the material repeats it
/// across a stream and scrolls it with the current. Pure Dart, so the curve
/// is unit-tested and no asset ships with the app.
Uint8List rippleNormalMap(int size, {int seed = 7, double strength = 0.015}) {
  final rng = math.Random(seed);
  // (cycles across u, cycles across v, amplitude, phase)
  final waves = [
    for (final (m, n, a) in const [(3, 1, 1.0), (-2, 3, 0.8), (5, -2, 0.5), (1, 6, 0.4), (7, 4, 0.25), (-6, 5, 0.2)])
      (m, n, a, rng.nextDouble() * 2 * math.pi),
  ];

  final out = Uint8List(size * size * 4);
  for (var y = 0; y < size; y++) {
    for (var x = 0; x < size; x++) {
      final u = x / size;
      final v = y / size;
      var du = 0.0;
      var dv = 0.0;
      for (final (m, n, a, phase) in waves) {
        final c = math.cos(2 * math.pi * (m * u + n * v) + phase) * a * 2 * math.pi;
        du += c * m;
        dv += c * n;
      }
      var nx = -du * strength;
      var ny = -dv * strength;
      var nz = 1.0;
      final len = math.sqrt(nx * nx + ny * ny + nz * nz);
      nx /= len;
      ny /= len;
      nz /= len;
      final i = (y * size + x) * 4;
      out[i] = ((nx * 0.5 + 0.5) * 255).round();
      out[i + 1] = ((ny * 0.5 + 0.5) * 255).round();
      out[i + 2] = ((nz * 0.5 + 0.5) * 255).round();
      out[i + 3] = 255;
    }
  }
  return out;
}
