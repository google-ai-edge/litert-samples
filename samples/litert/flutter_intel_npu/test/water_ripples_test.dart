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

import 'package:flutter_test/flutter_test.dart';
import 'package:intel_npu_demo/core/world/water_ripples.dart';

void main() {
  const size = 64;
  final pixels = rippleNormalMap(size);

  (double, double, double) normalAt(Uint8List px, int x, int y) {
    final i = (y * size + x) * 4;
    return (px[i] / 255 * 2 - 1, px[i + 1] / 255 * 2 - 1, px[i + 2] / 255 * 2 - 1);
  }

  double distance((double, double, double) a, (double, double, double) b) {
    final dx = a.$1 - b.$1, dy = a.$2 - b.$2, dz = a.$3 - b.$3;
    return math.sqrt(dx * dx + dy * dy + dz * dz);
  }

  test('is an RGBA image of the requested size', () {
    expect(pixels.length, size * size * 4);
    for (var i = 3; i < pixels.length; i += 4) {
      expect(pixels[i], 255);
    }
  });

  test('encodes unit normals that point out of the surface', () {
    for (var y = 0; y < size; y += 3) {
      for (var x = 0; x < size; x += 3) {
        final (nx, ny, nz) = normalAt(pixels, x, y);
        expect(math.sqrt(nx * nx + ny * ny + nz * nz), closeTo(1, 0.03));
        expect(nz, greaterThan(0.5), reason: 'ripples this steep read as noise');
      }
    }
  });

  test('actually ripples', () {
    var spread = 0.0;
    final first = normalAt(pixels, 0, 0);
    for (var x = 0; x < size; x++) {
      spread = math.max(spread, distance(first, normalAt(pixels, x, size ~/ 2)));
    }
    expect(spread, greaterThan(0.15));
  });

  test('tiles without a seam', () {
    // The texture repeats across a stream and scrolls with the current, so
    // its last column must lead into its first as smoothly as any two
    // neighbouring columns inside it.
    var interior = 0.0;
    for (var y = 0; y < size - 1; y++) {
      for (var x = 0; x < size - 1; x++) {
        final here = normalAt(pixels, x, y);
        interior = math.max(interior, distance(here, normalAt(pixels, x + 1, y)));
        interior = math.max(interior, distance(here, normalAt(pixels, x, y + 1)));
      }
    }
    for (var i = 0; i < size; i++) {
      expect(distance(normalAt(pixels, size - 1, i), normalAt(pixels, 0, i)), lessThanOrEqualTo(interior + 0.02),
          reason: 'seam between the last and first column, row $i');
      expect(distance(normalAt(pixels, i, size - 1), normalAt(pixels, i, 0)), lessThanOrEqualTo(interior + 0.02),
          reason: 'seam between the last and first row, column $i');
    }
  });
}
