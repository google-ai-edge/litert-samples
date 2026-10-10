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

import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/services/images/image_normalizer.dart';

/// Fixtures made with PIL:
/// - `big_2000x1500.jpg`: a 2000×1500 grey gradient, quality 40.
/// - `exif6_80x40.jpg`: stored 80×40, left half red, right half blue, EXIF
///   Orientation=6 (shown rotated 90° clockwise: 40×80, red on top).
Uint8List fixture(String name) =>
    File('test_assets/images/$name').readAsBytesSync();

/// The PNG's size and RGBA pixels, decoded by the engine.
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

bool isPng(Uint8List bytes) =>
    bytes.length > 8 &&
    bytes[0] == 0x89 &&
    bytes[1] == 0x50 &&
    bytes[2] == 0x4E &&
    bytes[3] == 0x47;

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test('a 2000×1500 JPEG becomes a 1024×768 PNG', () async {
    final out = await normalizeForLlm(fixture('big_2000x1500.jpg'));

    expect(isPng(out.png), isTrue);
    expect((out.sourceWidth, out.sourceHeight), (2000, 1500));
    expect((out.width, out.height), (kLlmImageMaxSide, 768));
    final decoded = await decodePng(out.png);
    expect((decoded.width, decoded.height), (1024, 768));
  });

  test('the cats photo (640×480) keeps its size and becomes PNG', () async {
    final jpeg = File('test_assets/cats.jpg').readAsBytesSync();

    final out = await normalizeForLlm(jpeg);

    expect(isPng(out.png), isTrue);
    expect((out.width, out.height), (640, 480));
    expect((out.sourceWidth, out.sourceHeight), (640, 480));
  });

  test('EXIF Orientation=6 is applied: width and height swap and the pixels '
      'are rotated', () async {
    final out = await normalizeForLlm(fixture('exif6_80x40.jpg'));

    expect((out.sourceWidth, out.sourceHeight), (40, 80));
    expect((out.width, out.height), (40, 80));
    final decoded = await decodePng(out.png);
    expect((decoded.width, decoded.height), (40, 80));
    // Stored left half (red) is on top once rotated 90° clockwise.
    int red(int x, int y) => decoded.rgba.getUint8((y * 40 + x) * 4);
    int blue(int x, int y) => decoded.rgba.getUint8((y * 40 + x) * 4 + 2);
    expect(red(20, 10), greaterThan(200));
    expect(blue(20, 10), lessThan(60));
    expect(red(20, 70), lessThan(60));
    expect(blue(20, 70), greaterThan(200));
  });

  test('a long side under the limit is never upscaled; maxSide applies to '
      'portrait images too', () async {
    final out = await normalizeForLlm(fixture('exif6_80x40.jpg'), maxSide: 20);

    expect((out.width, out.height), (10, 20));
  });

  test('bytes no codec reads fail with UndecodableImageException', () async {
    await expectLater(
      normalizeForLlm(Uint8List.fromList(List.filled(64, 7))),
      throwsA(isA<UndecodableImageException>()),
    );
  });

  test('the warm-up image is a 32×32 PNG', () async {
    final png = await warmUpPng();

    expect(isPng(png), isTrue);
    final decoded = await decodePng(png);
    expect((decoded.width, decoded.height), (32, 32));
  });
}
