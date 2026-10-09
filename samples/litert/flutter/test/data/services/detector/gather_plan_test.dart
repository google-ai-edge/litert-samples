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

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/services/detector/detector_codec.dart';
import 'package:litert_edge_demos/data/services/detector/frame_message.dart';
import 'package:litert_edge_demos/data/services/frames/frame_source.dart';
import 'package:litert_edge_demos/domain/models/detection.dart';
import 'package:litert_edge_demos/domain/models/frame_source_info.dart';

import '../../../support/frames.dart';
import '../../../support/yolo26n_reference.dart' as ref;

const _rotations = [0, 90, 180, 270];

/// Integer vs float BT.601: the reference's integer path differs from the
/// float conversion by up to 0.0042 (1.08 LSB).
const _floatYuvTolerance = 0.0043;

Uint8List randomBytes(int n, int seed) {
  final rnd = math.Random(seed);
  return Uint8List.fromList(List.generate(n, (_) => rnd.nextInt(256)));
}

/// Runs the app path: copy into a FrameMessage, materialize, gather.
Float32List gatherApp(FrameView view, {FrameGatherer? gatherer}) {
  final message = FrameMessage.copyOf(view, frameId: 1);
  final dst = Float32List(3 * 640 * 640);
  (gatherer ?? FrameGatherer()).gather(message, message.materialize(), dst);
  return dst;
}

double maxAbsDiff(Float32List a, Float32List b) {
  expect(a.length, b.length);
  var m = 0.0;
  for (var i = 0; i < a.length; i++) {
    final d = (a[i] - b[i]).abs();
    if (d > m) m = d;
  }
  return m;
}

/// BGRA bytes with R and B swapped (RGBA ↔ BGRA).
Uint8List swapRB(Uint8List px) {
  final out = Uint8List.fromList(px);
  for (var i = 0; i + 3 < out.length; i += 4) {
    out[i] = px[i + 2];
    out[i + 2] = px[i];
  }
  return out;
}

void main() {
  group('Letterbox', () {
    test('matches the reference for camera, fixture and odd sizes', () {
      for (final (w, h) in [(640, 480), (1280, 720), (480, 640), (333, 251)]) {
        for (final rot in _rotations) {
          final a = Letterbox(w, h, rot);
          final b = ref.RefLetterbox(w, h, rot);
          expect(
            (a.uprightW, a.uprightH, a.ratio, a.newW, a.newH, a.padX, a.padY),
            (b.uprightW, b.uprightH, b.ratio, b.newW, b.newH, b.padX, b.padY),
            reason: '${w}x$h rot $rot',
          );
        }
      }
    });

    test('720p letterboxes to 640×360 with 140 px bars; 640×480 maps 1:1', () {
      final hd = Letterbox(1280, 720, 0);
      expect(
        (hd.ratio, hd.newW, hd.newH, hd.padX, hd.padY),
        (0.5, 640, 360, 0, 140),
      );
      final vga = Letterbox(640, 480, 0);
      expect((vga.ratio, vga.padX, vga.padY), (1.0, 0, 80));
      final portrait = Letterbox(640, 480, 90);
      expect(
        (portrait.uprightW, portrait.uprightH, portrait.padX),
        (480, 640, 80),
      );
    });

    test('rejects a rotation that is not a quarter turn', () {
      expect(() => Letterbox(640, 480, 45), throwsArgumentError);
    });
  });

  group('rotation, independent of the reference', () {
    /// [up] (w×h RGBA, upright) as the buffer a camera would deliver for a
    /// clockwise turn of [rot]: the image rotated the other way, built here
    /// with explicit formulas rather than the codec's index map.
    (Uint8List, int, int) sourceFor(Uint8List up, int w, int h, int rot) {
      final (sw, sh) = rot % 180 == 0 ? (w, h) : (h, w);
      final src = Uint8List(sw * sh * 4);
      for (var y = 0; y < sh; y++) {
        for (var x = 0; x < sw; x++) {
          // Upright pixel shown at source (x, y).
          final (ux, uy) = switch (rot) {
            90 => (w - 1 - y, x), // source = upright turned counter-clockwise
            180 => (w - 1 - x, h - 1 - y),
            270 => (y, h - 1 - x), // source = upright turned clockwise
            _ => (x, y),
          };
          final s = (y * sw + x) * 4;
          final u = (uy * w + ux) * 4;
          src.setRange(s, s + 4, up, u);
        }
      }
      return (src, sw, sh);
    }

    for (final (w, h) in [(320, 240), (1280, 720), (333, 251)]) {
      test('gather(source, rot r) == gather(upright, rot 0), ${w}x$h', () {
        final up = randomBytes(w * h * 4, w * 7 + h);
        final want = gatherApp(
          TestFrame(w, h, FramePixelFormat.rgba8888, 0, [
            FramePlane(bytes: up, bytesPerRow: w * 4, bytesPerPixel: 4),
          ]),
        );
        for (final rot in [90, 180, 270]) {
          final (src, sw, sh) = sourceFor(up, w, h, rot);
          final got = gatherApp(
            TestFrame(sw, sh, FramePixelFormat.rgba8888, rot, [
              FramePlane(bytes: src, bytesPerRow: sw * 4, bytesPerPixel: 4),
            ]),
          );
          expect(maxAbsDiff(got, want), 0, reason: 'rot $rot');
        }
      });
    }
  });

  group('GatherPlan vs the reference implementation', () {
    for (final (w, h, pad) in [
      (640, 480, 64),
      (1280, 720, 0),
      (333, 251, 12),
    ]) {
      final bpr = w * 4 + pad;
      final px = randomBytes(bpr * h, w + h);
      for (final rot in _rotations) {
        test('BGRA ${w}x$h stride $bpr rot $rot', () {
          final want = Float32List(3 * 640 * 640);
          ref.preprocessBgraNchw(px, bpr, ref.RefLetterbox(w, h, rot), want);
          final got = gatherApp(
            TestFrame(w, h, FramePixelFormat.bgra8888, rot, [
              FramePlane(bytes: px, bytesPerRow: bpr, bytesPerPixel: 4),
            ]),
          );
          expect(maxAbsDiff(got, want), lessThan(1e-6));
        });

        test('RGBA ${w}x$h stride $bpr rot $rot', () {
          final want = Float32List(3 * 640 * 640);
          ref.preprocessBgraNchw(
            swapRB(px),
            bpr,
            ref.RefLetterbox(w, h, rot),
            want,
          );
          final got = gatherApp(
            TestFrame(w, h, FramePixelFormat.rgba8888, rot, [
              FramePlane(bytes: px, bytesPerRow: bpr, bytesPerPixel: 4),
            ]),
          );
          expect(maxAbsDiff(got, want), lessThan(1e-6));
        });
      }
    }

    for (final (w, h) in [(640, 480), (720, 480), (1280, 720)]) {
      for (final rot in _rotations) {
        test('NV21 (CameraX single plane) ${w}x$h rot $rot', () {
          // CameraX: bytesPerRow = width, Y then V U interleaved.
          final y = randomBytes(w * h, 7 + rot);
          final vu = randomBytes(w * (h ~/ 2), 11 + rot);
          final want = Float32List(3 * 640 * 640);
          ref.preprocessYuv420Nchw(
            y,
            Uint8List.sublistView(vu, 1),
            vu,
            w,
            w,
            2,
            ref.RefLetterbox(w, h, rot),
            want,
          );
          final wantInt = Float32List(3 * 640 * 640);
          ref.refGatherYuv(
            y,
            Uint8List.sublistView(vu, 1),
            vu,
            w,
            w,
            2,
            ref.RefLetterbox(w, h, rot),
            wantInt,
          );
          final got = gatherApp(
            TestFrame(w, h, FramePixelFormat.nv21, rot, [
              FramePlane(
                bytes: Uint8List.fromList([...y, ...vu]),
                bytesPerRow: w,
                bytesPerPixel: 1,
              ),
            ]),
          );
          expect(maxAbsDiff(got, wantInt), 0, reason: 'reference integer path');
          expect(maxAbsDiff(got, want), lessThanOrEqualTo(_floatYuvTolerance));
        });
      }
    }

    test('NV21 as two planes and YUV420 as three planes, padded strides', () {
      const w = 640;
      const h = 480;
      const yStride = 704;
      const uvStride = 704;
      final y = randomBytes(yStride * h, 3);
      final vu = randomBytes(uvStride * (h ~/ 2), 5);
      final want = Float32List(3 * 640 * 640);
      ref.preprocessYuv420Nchw(
        y,
        Uint8List.sublistView(vu, 1),
        vu,
        yStride,
        uvStride,
        2,
        ref.RefLetterbox(w, h, 90),
        want,
      );
      final twoPlanes = gatherApp(
        TestFrame(w, h, FramePixelFormat.nv21, 90, [
          FramePlane(bytes: y, bytesPerRow: yStride, bytesPerPixel: 1),
          FramePlane(bytes: vu, bytesPerRow: uvStride, bytesPerPixel: 2),
        ]),
      );
      expect(
        maxAbsDiff(twoPlanes, want),
        lessThanOrEqualTo(_floatYuvTolerance),
      );

      // The same chroma as separate U and V planes with pixel stride 2.
      final u = Uint8List.sublistView(vu, 1);
      final v = Uint8List.sublistView(vu, 0, vu.length - 1);
      final threePlanes = gatherApp(
        TestFrame(w, h, FramePixelFormat.yuv420, 90, [
          FramePlane(bytes: y, bytesPerRow: yStride, bytesPerPixel: 1),
          FramePlane(bytes: u, bytesPerRow: uvStride, bytesPerPixel: 2),
          FramePlane(bytes: v, bytesPerRow: uvStride, bytesPerPixel: 2),
        ]),
      );
      expect(
        maxAbsDiff(threePlanes, want),
        lessThanOrEqualTo(_floatYuvTolerance),
      );
    });

    test('cats RGBA fixture equals the reference', () {
      final rgba = File('test_assets/yolo26n/cats_640x480_rgba.u8')
          .readAsBytesSync();
      final want = Float32List(3 * 640 * 640);
      ref.preprocessBgraNchw(
        swapRB(rgba),
        640 * 4,
        ref.RefLetterbox(640, 480, 0),
        want,
      );
      final got = gatherApp(
        TestFrame(640, 480, FramePixelFormat.rgba8888, 0, [
          FramePlane(bytes: rgba, bytesPerRow: 640 * 4, bytesPerPixel: 4),
        ]),
      );
      expect(maxAbsDiff(got, want), lessThan(1e-6));
    });

    test('plans are cached per layout and rebuilt when the layout changes', () {
      final gatherer = FrameGatherer();
      FrameView frame(int w, int h) =>
          TestFrame(w, h, FramePixelFormat.rgba8888, 0, [
            FramePlane(
              bytes: Uint8List(w * h * 4),
              bytesPerRow: w * 4,
              bytesPerPixel: 4,
            ),
          ]);
      gatherApp(frame(640, 480), gatherer: gatherer);
      gatherApp(frame(640, 480), gatherer: gatherer);
      expect(gatherer.plansBuilt, 1);
      gatherApp(frame(480, 640), gatherer: gatherer);
      gatherApp(frame(640, 480), gatherer: gatherer);
      expect(gatherer.plansBuilt, 2);
    });

    test('a buffer smaller than its plan is an error, not a silent read', () {
      expect(
        () => gatherApp(
          TestFrame(640, 480, FramePixelFormat.bgra8888, 0, [
            FramePlane(
              bytes: Uint8List(1000),
              bytesPerRow: 2560,
              bytesPerPixel: 4,
            ),
          ]),
        ),
        throwsArgumentError,
      );
      expect(
        () => gatherApp(
          TestFrame(640, 480, FramePixelFormat.yuv420, 0, [
            FramePlane(
              bytes: Uint8List(640 * 480),
              bytesPerRow: 640,
              bytesPerPixel: 1,
            ),
          ]),
        ),
        throwsArgumentError,
        reason: 'yuv420 needs three planes',
      );
    });
  });

  group('decodeDetections', () {
    final golden =
        (jsonDecode(
              File('test_assets/yolo26n/cats_golden.json').readAsStringSync(),
            ) as Map<String, Object?>)['detections']!
            as List<Object?>;

    test('maps raw-head rows back through the letterbox, sorted by score', () {
      // A 1280×960 frame: ratio 0.5, padY 80. Put each golden box (frame
      // px of a 640×480 image, scaled ×2) at a different anchor.
      final lb = Letterbox(1280, 960, 0);
      final raw = Float32List(8400 * 84);
      final anchors = [8000, 12, 4000, 300];
      for (var k = 0; k < golden.length; k++) {
        final d = golden[k]! as Map<String, Object?>;
        final box = (d['box']! as List<Object?>).cast<num>();
        final base = anchors[k] * 84;
        raw[base] = box[0] * 2 * 0.5 + lb.padX;
        raw[base + 1] = box[1] * 2 * 0.5 + lb.padY;
        raw[base + 2] = box[2] * 2 * 0.5 + lb.padX;
        raw[base + 3] = box[3] * 2 * 0.5 + lb.padY;
        raw[base + 4 + (d['id']! as int)] = (d['score']! as num).toDouble();
      }
      raw[50 * 84 + 4 + 3] = 0.2499; // below the 0.25 floor

      final boxes = decodeDetections(raw, lb);

      expect(boxes.length, golden.length * kBoxStride);
      for (var k = 0; k < golden.length; k++) {
        final d = golden[k]! as Map<String, Object?>;
        final box = (d['box']! as List<Object?>).cast<num>();
        final o = k * kBoxStride;
        for (var j = 0; j < 4; j++) {
          expect(boxes[o + j], closeTo(box[j] * 2, 1e-3));
        }
        expect(boxes[o + 4], closeTo((d['score']! as num).toDouble(), 1e-6));
        expect(boxes[o + 5], d['id']);
      }
    });

    test('clamps overshooting coordinates to the frame', () {
      final lb = Letterbox(640, 480, 0);
      final raw = Float32List(8400 * 84);
      raw
        ..[0] = -9.6
        ..[1] = 70
        ..[2] = 651
        ..[3] = 600
        ..[4 + 15] = 0.9;
      final boxes = decodeDetections(raw, lb);
      expect(boxes.sublist(0, 4), [0, 0, 640, 480]);
    });

    test('equals the reference selectRaw + unletterbox on a dense raw '
        'output', () {
      // 150 distinct scores at or above the 0.25 floor (more than the 100
      // kept), the rest of the 8400 × 80 below it, and coordinates that
      // overshoot the letterbox on every side.
      final rnd = math.Random(42);
      final raw = Float32List(8400 * 84);
      for (var a = 0; a < 8400; a++) {
        final base = a * 84;
        for (var j = 0; j < 4; j++) {
          raw[base + j] = rnd.nextDouble() * 680 - 20;
        }
        for (var c = 0; c < 80; c++) {
          raw[base + 4 + c] = rnd.nextDouble() * 0.24;
        }
      }
      final picked = <int>{};
      while (picked.length < 150) {
        picked.add(rnd.nextInt(8400 * 80));
      }
      var i = 0;
      for (final p in picked) {
        raw[(p ~/ 80) * 84 + 4 + p % 80] = 0.25 + 0.004 * i++;
      }
      final lb = Letterbox(640, 480, 0);
      final want = [
        for (final d in ref.selectRaw(raw, 0.25, maxDet: 100))
          ref.unletterbox(d, ref.RefLetterbox(640, 480, 0)),
      ];

      final got = decodeDetections(raw, lb);

      expect(got.length, want.length * kBoxStride);
      for (var k = 0; k < want.length; k++) {
        final o = k * kBoxStride;
        expect(got[o], closeTo(want[k].x1, 1e-3));
        expect(got[o + 1], closeTo(want[k].y1, 1e-3));
        expect(got[o + 2], closeTo(want[k].x2, 1e-3));
        expect(got[o + 3], closeTo(want[k].y2, 1e-3));
        expect(got[o + 4], want[k].score);
        expect(got[o + 5], want[k].cls);
      }
    });
  });
}
