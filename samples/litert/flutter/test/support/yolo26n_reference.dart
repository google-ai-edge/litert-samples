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

// A standalone reference implementation of YOLO26n's pre- and
// post-processing (letterbox, normalize, NCHW gather; raw-head decode),
// written and verified before the app's codec and kept here unchanged in
// behaviour (reformatted for the lints) so the app's GatherPlan and decode are
// tested against it. Its BGRA path matched the Python pipeline's input tensor
// with maxAbsDiff 0.0.
import 'dart:math' as math;
import 'dart:typed_data';

const int kIn = 640;
const int kAnchors = 8400;
const int kClasses = 80;
const int kRawStride = 4 + kClasses;
const int kPad = 114;

class RefLetterbox {
  RefLetterbox(this.srcW, this.srcH, this.rotation, {this.size = kIn})
    : uprightW = (rotation % 180 == 0) ? srcW : srcH,
      uprightH = (rotation % 180 == 0) ? srcH : srcW {
    ratio = math.min(size / uprightW, size / uprightH);
    newW = (uprightW * ratio).round();
    newH = (uprightH * ratio).round();
    padX = (size - newW) ~/ 2;
    padY = (size - newH) ~/ 2;
  }
  final int srcW;
  final int srcH;
  final int rotation;
  final int size;
  final int uprightW;
  final int uprightH;
  late final double ratio;
  late final int newW;
  late final int newH;
  late final int padX;
  late final int padY;
}

int _srcIndex(int ux, int uy, int srcW, int srcH, int rotation) {
  switch (rotation) {
    case 90:
      return (srcH - 1 - ux) * srcW + uy;
    case 180:
      return (srcH - 1 - uy) * srcW + (srcW - 1 - ux);
    case 270:
      return ux * srcW + (srcW - 1 - uy);
    default:
      return uy * srcW + ux;
  }
}

int _srcIndexStride(
  int ux,
  int uy,
  int srcW,
  int srcH,
  int rotation,
  int rowPx,
) {
  final i = _srcIndex(ux, uy, srcW, srcH, rotation);
  if (rowPx == srcW) return i;
  final sy = i ~/ srcW;
  final sx = i - sy * srcW;
  return sy * rowPx + sx;
}

/// BGRA8888 → NCHW float32 RGB in [0,1], letterboxed, pad 114.
void preprocessBgraNchw(
  Uint8List bgra,
  int bytesPerRow,
  RefLetterbox lb,
  Float32List dst,
) {
  final s = lb.size;
  final plane = s * s;
  const padV = kPad / 255.0;
  dst.fillRange(0, 3 * plane, padV);
  final inv = 1.0 / lb.ratio;
  final xs = Int32List(lb.newW);
  for (var x = 0; x < lb.newW; x++) {
    xs[x] = math.min(lb.uprightW - 1, ((x + 0.5) * inv).floor());
  }
  final rowPx = bytesPerRow >> 2;
  for (var y = 0; y < lb.newH; y++) {
    final uy = math.min(lb.uprightH - 1, ((y + 0.5) * inv).floor());
    var o = (y + lb.padY) * s + lb.padX;
    for (var x = 0; x < lb.newW; x++, o++) {
      final si =
          _srcIndexStride(xs[x], uy, lb.srcW, lb.srcH, lb.rotation, rowPx) << 2;
      dst[o] = bgra[si + 2] * (1 / 255.0);
      dst[plane + o] = bgra[si + 1] * (1 / 255.0);
      dst[2 * plane + o] = bgra[si] * (1 / 255.0);
    }
  }
}

/// YUV420 (3 planes) → NCHW float32 RGB [0,1], letterboxed. Float BT.601.
void preprocessYuv420Nchw(
  Uint8List yP,
  Uint8List uP,
  Uint8List vP,
  int yRowStride,
  int uvRowStride,
  int uvPixelStride,
  RefLetterbox lb,
  Float32List dst,
) {
  final s = lb.size;
  final plane = s * s;
  const padV = kPad / 255.0;
  dst.fillRange(0, 3 * plane, padV);
  final inv = 1.0 / lb.ratio;
  for (var y = 0; y < lb.newH; y++) {
    final uy = math.min(lb.uprightH - 1, ((y + 0.5) * inv).floor());
    var o = (y + lb.padY) * s + lb.padX;
    for (var x = 0; x < lb.newW; x++, o++) {
      final ux = math.min(lb.uprightW - 1, ((x + 0.5) * inv).floor());
      final i = _srcIndex(ux, uy, lb.srcW, lb.srcH, lb.rotation);
      final sy = i ~/ lb.srcW;
      final sx = i - sy * lb.srcW;
      final yy = yP[sy * yRowStride + sx];
      final uvi = (sy >> 1) * uvRowStride + (sx >> 1) * uvPixelStride;
      final u = uP[uvi] - 128;
      final v = vP[uvi] - 128;
      final r = yy + 1.402 * v;
      final g = yy - 0.344136 * u - 0.714136 * v;
      final b = yy + 1.772 * u;
      dst[o] = (r < 0 ? 0 : (r > 255 ? 255 : r)) * (1 / 255.0);
      dst[plane + o] = (g < 0 ? 0 : (g > 255 ? 255 : g)) * (1 / 255.0);
      dst[2 * plane + o] = (b < 0 ? 0 : (b > 255 ? 255 : b)) * (1 / 255.0);
    }
  }
}

class Det {
  Det(this.x1, this.y1, this.x2, this.y2, this.score, this.cls);
  double x1;
  double y1;
  double x2;
  double y2;
  final double score;
  final int cls;
}

/// Raw head [1,8400,84] → every (anchor, class) ≥ thr, sorted desc, capped.
List<Det> selectRaw(Float32List raw, double thr, {int maxDet = 300}) {
  final dets = <Det>[];
  for (var a = 0; a < kAnchors; a++) {
    final base = a * kRawStride;
    for (var c = 0; c < kClasses; c++) {
      final s = raw[base + 4 + c];
      if (s >= thr) {
        dets.add(
          Det(raw[base], raw[base + 1], raw[base + 2], raw[base + 3], s, c),
        );
      }
    }
  }
  dets.sort((p, q) => q.score.compareTo(p.score));
  return dets.length > maxDet ? dets.sublist(0, maxDet) : dets;
}

/// 640-input px → upright-image px (undo letterbox), clamped.
Det unletterbox(Det d, RefLetterbox lb) {
  double fx(double v) =>
      ((v - lb.padX) / lb.ratio).clamp(0.0, lb.uprightW.toDouble());
  double fy(double v) =>
      ((v - lb.padY) / lb.ratio).clamp(0.0, lb.uprightH.toDouble());
  return Det(fx(d.x1), fy(d.y1), fx(d.x2), fy(d.y2), d.score, d.cls);
}

final Float32List _u8ToUnit = Float32List.fromList(
  List<double>.generate(256, (i) => i / 255.0),
);

/// The probe's integer BT.601 YUV gather (`GatherPlan.yuv` + `yuvToNchw`).
/// It differs from the float [preprocessYuv420Nchw] by up to 0.0042
/// (1.08 LSB: truncating >> 10 and the 1436/1024-style coefficients), which
/// is what the probe's own bench prints.
void refGatherYuv(
  Uint8List yP,
  Uint8List uP,
  Uint8List vP,
  int yRowStride,
  int uvRowStride,
  int uvPixelStride,
  RefLetterbox lb,
  Float32List dst,
) {
  final plane = lb.size * lb.size;
  final lut = _u8ToUnit;
  dst.fillRange(0, 3 * plane, kPad / 255.0);
  final inv = 1.0 / lb.ratio;
  for (var y = 0; y < lb.newH; y++) {
    final uy = math.min(lb.uprightH - 1, ((y + 0.5) * inv).floor());
    for (var x = 0; x < lb.newW; x++) {
      final ux = math.min(lb.uprightW - 1, ((x + 0.5) * inv).floor());
      final si = _srcIndex(ux, uy, lb.srcW, lb.srcH, lb.rotation);
      final sy = si ~/ lb.srcW;
      final sx = si - sy * lb.srcW;
      final yy = yP[sy * yRowStride + sx] << 10;
      final uvi = (sy >> 1) * uvRowStride + (sx >> 1) * uvPixelStride;
      final u = uP[uvi] - 128;
      final v = vP[uvi] - 128;
      final r = ((yy + 1436 * v) >> 10).clamp(0, 255);
      final g = ((yy - 352 * u - 731 * v) >> 10).clamp(0, 255);
      final b = ((yy + 1815 * u) >> 10).clamp(0, 255);
      final o = (y + lb.padY) * lb.size + x + lb.padX;
      dst[o] = lut[r];
      dst[plane + o] = lut[g];
      dst[2 * plane + o] = lut[b];
    }
  }
}
