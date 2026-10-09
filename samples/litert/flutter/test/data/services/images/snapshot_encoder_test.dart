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
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/services/images/snapshot_encoder.dart';
import 'package:litert_edge_demos/domain/models/scene_snapshot.dart';

/// [w]×[h], left half red, right half blue (opaque).
RgbaPixels halves(int w, int h) {
  final bytes = Uint8List(w * h * 4);
  for (var y = 0; y < h; y++) {
    for (var x = 0; x < w; x++) {
      final o = (y * w + x) * 4;
      final left = x < w ~/ 2;
      bytes[o] = left ? 255 : 0;
      bytes[o + 2] = left ? 0 : 255;
      bytes[o + 3] = 255;
    }
  }
  return RgbaPixels(width: w, height: h, bytes: bytes);
}

Future<({int width, int height, ByteData rgba})> decodePng(
  Uint8List png,
) async {
  final codec = await ui.instantiateImageCodec(png);
  final image = (await codec.getNextFrame()).image;
  final rgba = (await image.toByteData())!;
  final result = (width: image.width, height: image.height, rgba: rgba);
  image.dispose();
  codec.dispose();
  return result;
}

/// (r, b) of the pixel at ([x], [y]).
(int, int) rb(({int width, int height, ByteData rgba}) d, int x, int y) {
  final o = (y * d.width + x) * 4;
  return (d.rgba.getUint8(o), d.rgba.getUint8(o + 2));
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  const encoder = SnapshotEncoder();

  test('a 16:9 1280×720 frame becomes a 1024×576 PNG, not mirrored back '
      'unless asked', () async {
    final out = await encoder.toPng(
      halves(1280, 720),
      frameId: 7,
      maxSide: kLlmImageMaxSide,
      unmirror: false,
    );
    expect((out.width, out.height), (1024, 576));
    expect(out.frameId, 7);
    expect(out.unmirrored, isFalse);
    final d = await decodePng(out.png);
    expect((d.width, d.height), (1024, 576));
    expect(rb(d, 10, 300), (255, 0), reason: 'left stays red');
    expect(rb(d, 1010, 300), (0, 255), reason: 'right stays blue');
  });

  test('unmirror flips left and right (a mirroring source)', () async {
    final out = await encoder.toPng(
      halves(1280, 720),
      frameId: 1,
      maxSide: kLlmImageMaxSide,
      unmirror: true,
    );
    expect(out.unmirrored, isTrue);
    final d = await decodePng(out.png);
    expect((d.width, d.height), (1024, 576));
    expect(rb(d, 10, 300), (0, 255), reason: 'blue is on the left now');
    expect(rb(d, 1010, 300), (255, 0));
  });

  test('a frame within the limit keeps its size and pixels', () async {
    final pixels = halves(640, 480);
    final out = await encoder.toPng(
      pixels,
      frameId: 2,
      maxSide: kLlmImageMaxSide,
      unmirror: false,
    );
    final d = await decodePng(out.png);
    expect((d.width, d.height), (640, 480));
    expect(d.rgba.buffer.asUint8List(), pixels.bytes, reason: 'lossless');
  });

  test('toImage: the frame at full size for the frozen view', () async {
    final image = await encoder.toImage(halves(1280, 720));
    expect((image.width, image.height), (1280, 720));
    image.dispose();
  });

  group('fitWithin', () {
    test('the long side brought to the limit, the aspect kept', () {
      expect(fitWithin(1280, 720, 1024), (1024, 576));
      expect(fitWithin(720, 1280, 1024), (576, 1024), reason: 'portrait');
      expect(fitWithin(4032, 3024, 896), (896, 672));
    });

    test('never scaled up: a frame within the limit keeps its size', () {
      expect(fitWithin(640, 480, 1024), (640, 480));
      expect(fitWithin(1024, 768, 1024), (1024, 768), reason: 'at the limit');
    });

    test('rounded to whole pixels, never below one', () {
      expect(fitWithin(1000, 333, 500), (500, 167), reason: '166.5 rounds up');
      expect(fitWithin(4000, 1, 100), (100, 1), reason: 'not 0 px high');
    });
  });
}
