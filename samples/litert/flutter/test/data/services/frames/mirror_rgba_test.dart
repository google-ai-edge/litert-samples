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

import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/services/frames/fixture_frame_source.dart';

/// [width]×[height] packed RGBA, pixel (x, y) = [x, y, 7, 255].
Uint8List grid(int width, int height) => Uint8List.fromList([
  for (var y = 0; y < height; y++)
    for (var x = 0; x < width; x++) ...[x, y, 7, 255],
]);

/// The pixel at ([x], [y]) of a [width]-wide RGBA buffer.
List<int> pixel(Uint8List rgba, int width, int x, int y) =>
    rgba.sublist((y * width + x) * 4, (y * width + x) * 4 + 4);

void main() {
  test('each row flipped left to right, whole pixels kept, rows in '
      'place', () {
    final rgba = grid(3, 2);

    final out = mirrorRgba(rgba, 3, 2);

    expect(out, hasLength(3 * 2 * 4));
    for (var y = 0; y < 2; y++) {
      for (var x = 0; x < 3; x++) {
        expect(pixel(out, 3, x, y), [2 - x, y, 7, 255], reason: '($x, $y)');
      }
    }
  });

  test('a new buffer: the source is left as it was', () {
    final rgba = grid(4, 3);
    final before = Uint8List.fromList(rgba);

    final out = mirrorRgba(rgba, 4, 3);

    expect(rgba, before);
    expect(identical(out.buffer, rgba.buffer), isFalse);
  });

  test('twice is the original', () {
    final rgba = grid(5, 4);

    expect(mirrorRgba(mirrorRgba(rgba, 5, 4), 5, 4), rgba);
  });

  test('a view into a larger buffer: only its own pixels are read', () {
    final whole = Uint8List(4 + 2 * 1 * 4 + 4)
      ..fillRange(0, 4, 99)
      ..setRange(4, 12, [1, 2, 3, 4, 5, 6, 7, 8])
      ..fillRange(12, 16, 99);
    final view = Uint8List.sublistView(whole, 4, 12);

    expect(mirrorRgba(view, 2, 1), [5, 6, 7, 8, 1, 2, 3, 4]);
  });

  test('one pixel wide: unchanged', () {
    final rgba = grid(1, 3);

    expect(mirrorRgba(rgba, 1, 3), rgba);
  });
}
