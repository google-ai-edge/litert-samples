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
import 'package:litert_edge_demos/data/services/detector/detector_codec.dart';
import 'package:litert_edge_demos/data/services/detector/frame_message.dart';
import 'package:litert_edge_demos/data/services/frames/frame_source.dart';
import 'package:litert_edge_demos/domain/models/frame_source_info.dart';
import 'package:litert_edge_demos/domain/models/scene_snapshot.dart';

import '../../../support/frames.dart';

Uint8List randomBytes(int n, int seed) {
  final rnd = math.Random(seed);
  return Uint8List.fromList(List.generate(n, (_) => rnd.nextInt(256)));
}

/// The app path: copy into a FrameMessage, materialize, convert.
RgbaPixels upright(FrameView view) {
  final message = FrameMessage.copyOf(view, frameId: 1);
  return uprightRgba(message, message.materialize());
}

/// [up] (w×h RGBA, upright) as the buffer a camera delivers for a clockwise
/// turn of [rot] (explicit formulas, independent of the codec's index map;
/// the same construction as gather_plan_test).
(Uint8List, int, int) sourceFor(Uint8List up, int w, int h, int rot) {
  final (sw, sh) = rot % 180 == 0 ? (w, h) : (h, w);
  final src = Uint8List(sw * sh * 4);
  for (var y = 0; y < sh; y++) {
    for (var x = 0; x < sw; x++) {
      final (ux, uy) = switch (rot) {
        90 => (w - 1 - y, x),
        180 => (w - 1 - x, h - 1 - y),
        270 => (y, h - 1 - x),
        _ => (x, y),
      };
      final s = (y * sw + x) * 4;
      final u = (uy * w + ux) * 4;
      src.setRange(s, s + 4, up, u);
    }
  }
  return (src, sw, sh);
}

/// [px] with every alpha byte set to 255.
Uint8List opaque(Uint8List px) {
  final out = Uint8List.fromList(px);
  for (var i = 3; i < out.length; i += 4) {
    out[i] = 255;
  }
  return out;
}

void main() {
  test('RGBA, upright, padded rows: the pixels without the padding', () {
    const w = 33;
    const h = 7;
    const bpr = w * 4 + 12;
    final px = randomBytes(bpr * h, 1);
    final out = upright(
      TestFrame(w, h, FramePixelFormat.rgba8888, 0, [
        FramePlane(bytes: px, bytesPerRow: bpr, bytesPerPixel: 4),
      ]),
    );
    expect((out.width, out.height), (w, h));
    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        final o = (y * w + x) * 4;
        final i = y * bpr + x * 4;
        expect(out.bytes.sublist(o, o + 3), px.sublist(i, i + 3));
        expect(out.bytes[o + 3], 255, reason: 'opaque');
      }
    }
  });

  test('BGRA (Apple camera) swaps to RGBA, opaque', () {
    final out = upright(
      TestFrame(2, 1, FramePixelFormat.bgra8888, 0, [
        FramePlane(
          bytes: Uint8List.fromList([10, 20, 30, 0, 40, 50, 60, 7]),
          bytesPerRow: 8,
          bytesPerPixel: 4,
        ),
      ]),
    );
    expect(out.bytes, [30, 20, 10, 255, 60, 50, 40, 255]);
  });

  for (final (w, h) in [(5, 3), (64, 48), (33, 17)]) {
    test(
      'rotation r of the source buffer gives the upright frame, ${w}x$h',
      () {
        final up = opaque(randomBytes(w * h * 4, w + h));
        for (final rot in [0, 90, 180, 270]) {
          final (src, sw, sh) = sourceFor(up, w, h, rot);
          for (final format in [
            FramePixelFormat.rgba8888,
            FramePixelFormat.bgra8888,
          ]) {
            final bytes = format == FramePixelFormat.bgra8888
                ? swapRB(src)
                : src;
            final out = upright(
              TestFrame(sw, sh, format, rot, [
                FramePlane(bytes: bytes, bytesPerRow: sw * 4, bytesPerPixel: 4),
              ]),
            );
            expect((out.width, out.height), (w, h), reason: '$format rot $rot');
            expect(out.bytes, up, reason: '$format rot $rot');
          }
        }
      },
    );
  }

  test('NV21: grey stays grey (integer BT.601), and every rotation equals the '
      'rotated upright conversion', () {
    const w = 8;
    const h = 6;
    final grey = upright(
      TestFrame(w, h, FramePixelFormat.nv21, 0, [
        FramePlane(
          bytes: Uint8List(w * h + w * h ~/ 2)
            ..fillRange(0, w * h * 3 ~/ 2, 128),
          bytesPerRow: w,
          bytesPerPixel: 1,
        ),
      ]),
    );
    expect(grey.bytes.toSet(), {128, 255});

    final y = randomBytes(w * h, 3);
    final vu = randomBytes(w * h ~/ 2, 4);
    final nv21 = Uint8List.fromList([...y, ...vu]);
    final at0 = upright(
      TestFrame(w, h, FramePixelFormat.nv21, 0, [
        FramePlane(bytes: nv21, bytesPerRow: w, bytesPerPixel: 1),
      ]),
    );
    for (final rot in [90, 180, 270]) {
      final rotated = upright(
        TestFrame(w, h, FramePixelFormat.nv21, rot, [
          FramePlane(bytes: nv21, bytesPerRow: w, bytesPerPixel: 1),
        ]),
      );
      // Rotating the rot-0 picture by r must give the same pixels.
      final (uw, uh) = rot % 180 == 0 ? (w, h) : (h, w);
      expect((rotated.width, rotated.height), (uw, uh));
      final (src, _, _) = sourceFor(rotated.bytes, uw, uh, rot);
      expect(src, at0.bytes, reason: 'rot $rot');
    }
  });

  test('a buffer smaller than the frame is an error, not a silent read', () {
    expect(
      () => upright(
        TestFrame(4, 4, FramePixelFormat.rgba8888, 0, [
          FramePlane(bytes: Uint8List(10), bytesPerRow: 16, bytesPerPixel: 4),
        ]),
      ),
      throwsArgumentError,
    );
  });
}

/// R and B swapped (RGBA ↔ BGRA).
Uint8List swapRB(Uint8List px) {
  final out = Uint8List.fromList(px);
  for (var i = 0; i + 3 < out.length; i += 4) {
    out[i] = px[i + 2];
    out[i + 2] = px[i];
  }
  return out;
}
