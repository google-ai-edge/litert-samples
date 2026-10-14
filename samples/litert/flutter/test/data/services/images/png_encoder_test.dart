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
import 'dart:ui' as ui;

import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/services/images/png_encoder.dart';

/// [w]×[h] RGB where every pixel differs: r = x, g = y, b = x ^ y (mod 256).
Uint8List gradient(int w, int h) {
  final rgb = Uint8List(w * h * 3);
  for (var y = 0; y < h; y++) {
    for (var x = 0; x < w; x++) {
      final o = (y * w + x) * 3;
      rgb[o] = x & 0xFF;
      rgb[o + 1] = y & 0xFF;
      rgb[o + 2] = (x ^ y) & 0xFF;
    }
  }
  return rgb;
}

/// The engine's decode of [png]: size and RGBA.
Future<(int, int, Uint8List)> decode(Uint8List png) async {
  final codec = await ui.instantiateImageCodec(png);
  final image = (await codec.getNextFrame()).image;
  final data = (await image.toByteData())!;
  final out = (image.width, image.height, data.buffer.asUint8List());
  image.dispose();
  codec.dispose();
  return out;
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test(
    'the engine decodes it to exactly the pixels written (alpha 255)',
    () async {
      for (final (w, h) in [(1, 1), (7, 3), (320, 180), (257, 2)]) {
        final rgb = gradient(w, h);
        final png = encodePngRgb(w, h, rgb);
        final (dw, dh, rgba) = await decode(png);
        expect((dw, dh), (w, h));
        for (var i = 0; i < w * h; i++) {
          expect(
            [rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2], rgba[i * 4 + 3]],
            [rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2], 255],
            reason: '${w}x$h pixel $i',
          );
        }
      }
    },
  );

  test('a PNG: the signature, IHDR (truecolour, 8 bits, no interlace) '
      'first, IEND last, every chunk with its CRC', () {
    final png = encodePngRgb(5, 4, gradient(5, 4));
    expect(png.sublist(0, 8), [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
    final data = ByteData.sublistView(png);
    var offset = 8;
    final types = <String>[];
    while (offset < png.length) {
      final length = data.getUint32(offset);
      final type = String.fromCharCodes(png.sublist(offset + 4, offset + 8));
      final body = png.sublist(offset + 8, offset + 8 + length);
      final crc = data.getUint32(offset + 8 + length);
      expect(crc, crc32([...type.codeUnits, ...body]), reason: type);
      if (type == 'IHDR') {
        final header = ByteData.sublistView(body);
        expect(header.getUint32(0), 5);
        expect(header.getUint32(4), 4);
        expect(body.sublist(8), [8, 2, 0, 0, 0]);
      }
      types.add(type);
      offset += 12 + length;
    }
    expect(types, ['IHDR', 'IDAT', 'IEND']);
  });

  test('CRC-32 matches the standard check value and the PNG IEND CRC', () {
    expect(crc32('123456789'.codeUnits), 0xCBF43926);
    expect(crc32('IEND'.codeUnits), 0xAE426082);
  });

  test('a flat image compresses far below its raw size', () {
    final rgb = Uint8List(320 * 180 * 3)..fillRange(0, 320 * 180 * 3, 128);
    expect(encodePngRgb(320, 180, rgb).length, lessThan(2000));
  });

  test('a wrong byte count or an empty size is refused', () {
    expect(() => encodePngRgb(2, 2, Uint8List(11)), throwsArgumentError);
    expect(() => encodePngRgb(0, 2, Uint8List(0)), throwsArgumentError);
  });
}
