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
import 'package:litert_edge_demos/config/live_camera_config.dart';
import 'package:litert_edge_demos/data/services/frames/frame_source.dart';
import 'package:litert_edge_demos/domain/models/frame_source_info.dart';

import '../../../support/frames.dart';

/// An NV21 frame whose Y plane is [y] at each pixel.
TestFrame nv21(int Function(int x, int y) y, {int width = 640, int h = 480}) {
  final bytes = Uint8List(width * h * 3 ~/ 2);
  for (var row = 0; row < h; row++) {
    for (var x = 0; x < width; x++) {
      bytes[row * width + x] = y(x, row);
    }
  }
  return TestFrame(width, h, FramePixelFormat.nv21, 90, [
    FramePlane(bytes: bytes, bytesPerRow: width, bytesPerPixel: 1),
  ]);
}

bool black(FrameView frame) {
  final s = lumaStats(frame);
  return s.mean < kBlackFrameLuma && s.spread < kBlackFrameSpread;
}

void main() {
  test('limited-range video black (Y 16) with sensor noise is a black frame '
      '(a phone camera in a dark box)', () {
    final frame = nv21((x, y) => 16 + (x * 7 + y * 13) % 5);
    final s = lumaStats(frame);
    expect(s.mean, closeTo(18, 1));
    expect(s.spread, lessThan(2));
    expect(black(frame), isTrue);
  });

  test('zeros (macOS camera access attributed to the terminal) are black', () {
    expect(black(TestFrame.rgba()), isTrue);
  });

  test('a dim scene with edges is not black, nor a lit one', () {
    final dim = nv21((x, y) => x ~/ 32 % 2 == 0 ? 4 : 30);
    expect(lumaStats(dim).mean, lessThan(kBlackFrameLuma));
    expect(black(dim), isFalse, reason: 'its edges spread the luma');
    expect(black(TestFrame.rgba(fill: 128)), isFalse);
  });
}
