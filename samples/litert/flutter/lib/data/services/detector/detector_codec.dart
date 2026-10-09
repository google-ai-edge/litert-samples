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

/// YOLO26n pre/post-processing (`Letterbox`, `GatherPlan`, `selectRaw`,
/// `unletterbox`), ported from a standalone reference implementation verified
/// against the Python pipeline (kept as test/support/yolo26n_reference.dart,
/// which the tests compare against), plus the RGBA gather and plane resolution
/// for every [FramePixelFormat]. Pure Dart: runs in the detector worker isolate
/// and in unit tests.
library;

import 'dart:collection';
import 'dart:math' as math;
import 'dart:typed_data';

import '../../../domain/models/detection.dart';
import '../../../domain/models/detector_spec.dart';
import '../../../domain/models/frame_source_info.dart';
import '../../../domain/models/scene_snapshot.dart';
import 'frame_message.dart';

/// value / 255 for every byte.
final Float32List _unit = Float32List.fromList([
  for (var i = 0; i < 256; i++) i / 255.0,
]);

/// The pad value as the model sees it (114 / 255).
final double kDetPadUnit = _unit[kDetPadByte];

/// Frame → model-input geometry for one frame layout.
final class Letterbox {
  Letterbox._(
    this.srcW,
    this.srcH,
    this.rotation,
    this.size,
    this.uprightW,
    this.uprightH,
    this.ratio,
    this.newW,
    this.newH,
  ) : padX = (size - newW) ~/ 2,
      padY = (size - newH) ~/ 2;

  /// [rotation] is the clockwise turn (0, 90, 180, 270) that makes the
  /// [srcW] × [srcH] buffer upright; the letterbox is computed on the upright
  /// frame.
  factory Letterbox(int srcW, int srcH, int rotation, {int size = kDetInput}) {
    if (srcW <= 0 || srcH <= 0) {
      throw ArgumentError('Frame size must be positive, got ${srcW}x$srcH');
    }
    if (rotation != 0 && rotation != 90 && rotation != 180 && rotation != 270) {
      throw ArgumentError.value(rotation, 'rotation', 'must be 0/90/180/270');
    }
    final uprightW = rotation % 180 == 0 ? srcW : srcH;
    final uprightH = rotation % 180 == 0 ? srcH : srcW;
    final ratio = math.min(size / uprightW, size / uprightH);
    return Letterbox._(
      srcW,
      srcH,
      rotation,
      size,
      uprightW,
      uprightH,
      ratio,
      (uprightW * ratio).round(),
      (uprightH * ratio).round(),
    );
  }

  final int srcW;
  final int srcH;
  final int rotation;
  final int size;
  final int uprightW;
  final int uprightH;
  final double ratio;
  final int newW;
  final int newH;
  final int padX;
  final int padY;

  /// Model-input x (640 px) → upright-frame x, clamped.
  double unX(double x) =>
      ((x - padX) / ratio).clamp(0.0, uprightW.toDouble()).toDouble();

  /// Model-input y (640 px) → upright-frame y, clamped.
  double unY(double y) =>
      ((y - padY) / ratio).clamp(0.0, uprightH.toDouble()).toDouble();
}

/// Upright pixel (ux, uy) → source pixel, for a clockwise [rot].
@pragma('vm:prefer-inline')
int _srcX(int ux, int uy, int srcW, int rot) => switch (rot) {
  90 => uy,
  180 => srcW - 1 - ux,
  270 => srcW - 1 - uy,
  _ => ux,
};

@pragma('vm:prefer-inline')
int _srcY(int ux, int uy, int srcH, int rot) => switch (rot) {
  90 => srcH - 1 - ux,
  180 => srcH - 1 - uy,
  270 => ux,
  _ => uy,
};

/// The gather tables for one (format, size, strides, rotation), built once
/// per stream (about 2 ms) and reused for every frame: per output pixel, its
/// offset in one 640×640 plane and its source byte offsets (nearest
/// neighbour). Rotation and letterbox are folded into the tables.
final class GatherPlan {
  GatherPlan._(this.letterbox, this._dstOffset, this._srcIdx, this._uvIdx);

  /// BGRA or RGBA: one plane with [bytesPerRow].
  factory GatherPlan.packed(Letterbox lb, {required int bytesPerRow}) {
    final n = lb.newW * lb.newH;
    final plan = GatherPlan._(lb, Int32List(n), Int32List(n), Int32List(0));
    plan._build((sx, sy, i) => plan._srcIdx[i] = sy * bytesPerRow + sx * 4);
    return plan;
  }

  /// Y plane with [yRowStride]; U and V with [uvRowStride] and
  /// [uvPixelStride] (2 for NV21's interleaved V U).
  factory GatherPlan.yuv(
    Letterbox lb, {
    required int yRowStride,
    required int uvRowStride,
    required int uvPixelStride,
  }) {
    final n = lb.newW * lb.newH;
    final plan = GatherPlan._(lb, Int32List(n), Int32List(n), Int32List(n));
    plan._build((sx, sy, i) {
      plan._srcIdx[i] = sy * yRowStride + sx;
      plan._uvIdx[i] = (sy >> 1) * uvRowStride + (sx >> 1) * uvPixelStride;
    });
    return plan;
  }

  final Letterbox letterbox;
  final Int32List _dstOffset;

  /// Packed: the pixel's byte offset. YUV: the Y byte offset.
  final Int32List _srcIdx;
  final Int32List _uvIdx;
  int _maxSrc = 0;
  int _maxUv = 0;

  void _build(void Function(int sx, int sy, int i) set) {
    final lb = letterbox;
    final inv = 1.0 / lb.ratio;
    var i = 0;
    for (var y = 0; y < lb.newH; y++) {
      final uy = math.min(lb.uprightH - 1, ((y + 0.5) * inv).floor());
      for (var x = 0; x < lb.newW; x++, i++) {
        final ux = math.min(lb.uprightW - 1, ((x + 0.5) * inv).floor());
        set(
          _srcX(ux, uy, lb.srcW, lb.rotation),
          _srcY(ux, uy, lb.srcH, lb.rotation),
          i,
        );
        _dstOffset[i] = (y + lb.padY) * lb.size + x + lb.padX;
      }
    }
    for (final v in _srcIdx) {
      if (v > _maxSrc) _maxSrc = v;
    }
    for (final v in _uvIdx) {
      if (v > _maxUv) _maxUv = v;
    }
  }

  /// BGRA ([rgba] false) or RGBA → NCHW RGB / 255 into [dst], pad 114 / 255.
  void packedToNchw(Uint8List px, Float32List dst, {required bool rgba}) {
    if (_maxSrc + 3 >= px.length) {
      throw ArgumentError(
        'Frame buffer too small: ${px.length} bytes, the plan reads up to '
        'byte ${_maxSrc + 3}',
      );
    }
    final plane = letterbox.size * letterbox.size;
    final n = _dstOffset.length;
    final lut = _unit;
    final dstOffset = _dstOffset;
    final srcIdx = _srcIdx;
    final rOff = rgba ? 0 : 2;
    final bOff = rgba ? 2 : 0;
    dst.fillRange(0, 3 * plane, kDetPadUnit);
    for (var i = 0; i < n; i++) {
      final si = srcIdx[i];
      final o = dstOffset[i];
      dst[o] = lut[px[si + rOff]];
      dst[plane + o] = lut[px[si + 1]];
      dst[2 * plane + o] = lut[px[si + bOff]];
    }
  }

  /// YUV (integer BT.601) → NCHW RGB / 255 into [dst].
  void yuvToNchw(Uint8List yP, Uint8List uP, Uint8List vP, Float32List dst) {
    if (_maxSrc >= yP.length || _maxUv >= uP.length || _maxUv >= vP.length) {
      throw ArgumentError(
        'YUV planes too small: Y ${yP.length} (needs ${_maxSrc + 1}), '
        'U ${uP.length} / V ${vP.length} (need ${_maxUv + 1})',
      );
    }
    final plane = letterbox.size * letterbox.size;
    final n = _dstOffset.length;
    final lut = _unit;
    final dstOffset = _dstOffset;
    final yIdx = _srcIdx;
    final uvIdx = _uvIdx;
    dst.fillRange(0, 3 * plane, kDetPadUnit);
    for (var i = 0; i < n; i++) {
      final yy = yP[yIdx[i]] << 10;
      final uvi = uvIdx[i];
      final u = uP[uvi] - 128;
      final v = vP[uvi] - 128;
      var r = (yy + 1436 * v) >> 10;
      var g = (yy - 352 * u - 731 * v) >> 10;
      var b = (yy + 1815 * u) >> 10;
      if (r < 0) {
        r = 0;
      } else if (r > 255) {
        r = 255;
      }
      if (g < 0) {
        g = 0;
      } else if (g > 255) {
        g = 255;
      }
      if (b < 0) {
        b = 0;
      } else if (b > 255) {
        b = 255;
      }
      final o = dstOffset[i];
      dst[o] = lut[r];
      dst[plane + o] = lut[g];
      dst[2 * plane + o] = lut[b];
    }
  }
}

/// Cache key: everything a [GatherPlan] depends on.
typedef GatherKey = (FramePixelFormat, int, int, int, int, int, int);

/// Resolves a [FrameMessage]'s planes per format and keeps the last few
/// [GatherPlan]s (a slideshow changes size every slide; a camera never does).
final class FrameGatherer {
  FrameGatherer({this.maxPlans = 8});

  final int maxPlans;
  final LinkedHashMap<GatherKey, GatherPlan> _plans = LinkedHashMap();

  /// Number of plans built so far (each costs about 2 ms).
  int plansBuilt = 0;

  /// Gathers [frame] (whose packed planes are [data]) into [dst]; returns the
  /// plan, whose letterbox maps the detections back.
  GatherPlan gather(FrameMessage frame, Uint8List data, Float32List dst) {
    final planes = frame.planes;
    switch (frame.format) {
      case FramePixelFormat.bgra8888 || FramePixelFormat.rgba8888:
        _require(planes.length == 1, frame, 'one plane');
        final p = planes.single;
        final plan = _plan(
          (
            frame.format,
            frame.width,
            frame.height,
            frame.rotationDeg,
            p.bytesPerRow,
            0,
            0,
          ),
          () =>
              GatherPlan.packed(_letterbox(frame), bytesPerRow: p.bytesPerRow),
        );
        plan.packedToNchw(
          Uint8List.sublistView(data, p.offset, p.offset + p.length),
          dst,
          rgba: frame.format == FramePixelFormat.rgba8888,
        );
        return plan;
      case FramePixelFormat.nv21 || FramePixelFormat.yuv420:
        final yuv = _yuvPlanes(frame, data);
        final plan = _plan(
          (
            frame.format,
            frame.width,
            frame.height,
            frame.rotationDeg,
            yuv.yRowStride,
            yuv.uvRowStride,
            yuv.uvPixelStride,
          ),
          () => GatherPlan.yuv(
            _letterbox(frame),
            yRowStride: yuv.yRowStride,
            uvRowStride: yuv.uvRowStride,
            uvPixelStride: yuv.uvPixelStride,
          ),
        );
        plan.yuvToNchw(yuv.y, yuv.u, yuv.v, dst);
        return plan;
    }
  }

  GatherPlan _plan(GatherKey key, GatherPlan Function() build) {
    final cached = _plans.remove(key);
    if (cached != null) return _plans[key] = cached;
    final plan = build();
    plansBuilt++;
    _plans[key] = plan;
    while (_plans.length > maxPlans) {
      _plans.remove(_plans.keys.first);
    }
    return plan;
  }

  static Letterbox _letterbox(FrameMessage f) =>
      Letterbox(f.width, f.height, f.rotationDeg);

  static ({
    Uint8List y,
    Uint8List u,
    Uint8List v,
    int yRowStride,
    int uvRowStride,
    int uvPixelStride,
  })
  _yuvPlanes(FrameMessage f, Uint8List data) {
    Uint8List view(int offset, int end) =>
        Uint8List.sublistView(data, offset, end);
    final planes = f.planes;
    switch (f.format) {
      case FramePixelFormat.nv21 when planes.length == 1:
        // CameraX: Y (rowStride × height), then V U interleaved.
        final p = planes.single;
        final vu = p.offset + p.bytesPerRow * f.height;
        final end = p.offset + p.length;
        _require(vu < end, f, 'a V U part after the Y plane');
        return (
          y: view(p.offset, vu),
          u: view(vu + 1, end),
          v: view(vu, end),
          yRowStride: p.bytesPerRow,
          uvRowStride: p.bytesPerRow,
          uvPixelStride: 2,
        );
      case FramePixelFormat.nv21 when planes.length == 2:
        final (y, vu) = (planes[0], planes[1]);
        return (
          y: view(y.offset, y.offset + y.length),
          u: view(vu.offset + 1, vu.offset + vu.length),
          v: view(vu.offset, vu.offset + vu.length),
          yRowStride: y.bytesPerRow,
          uvRowStride: vu.bytesPerRow,
          uvPixelStride: 2,
        );
      case FramePixelFormat.yuv420 when planes.length == 3:
        final (y, u, v) = (planes[0], planes[1], planes[2]);
        _require(
          u.bytesPerRow == v.bytesPerRow && u.bytesPerPixel == v.bytesPerPixel,
          f,
          'U and V with the same strides',
        );
        return (
          y: view(y.offset, y.offset + y.length),
          u: view(u.offset, u.offset + u.length),
          v: view(v.offset, v.offset + v.length),
          yRowStride: y.bytesPerRow,
          uvRowStride: u.bytesPerRow,
          uvPixelStride: u.bytesPerPixel,
        );
      default:
        throw ArgumentError(
          '${f.format.name} frame with ${planes.length} plane(s) is not '
          'supported (nv21: 1 or 2 planes; yuv420: 3)',
        );
    }
  }

  static void _require(bool ok, FrameMessage f, String what) {
    if (!ok) {
      throw ArgumentError('${f.format.name} frame needs $what');
    }
  }
}

/// [frame] (whose packed planes are [data]) as upright, tightly packed RGBA
/// with alpha 255 (a question's snapshot): rotation applied, the same
/// orientation and mirroring as its detections. Runs in the detector worker on
/// the frame the detector just took, once per question (about 1–3 ms per 720p
/// frame in AOT). YUV uses the gather's integer BT.601.
RgbaPixels uprightRgba(FrameMessage frame, Uint8List data) {
  final rot = frame.rotationDeg;
  if (rot != 0 && rot != 90 && rot != 180 && rot != 270) {
    throw ArgumentError.value(rot, 'rotationDeg', 'must be 0/90/180/270');
  }
  final srcW = frame.width;
  final srcH = frame.height;
  final upW = rot % 180 == 0 ? srcW : srcH;
  final upH = rot % 180 == 0 ? srcH : srcW;
  final out = Uint8List(upW * upH * 4);
  switch (frame.format) {
    case FramePixelFormat.bgra8888 || FramePixelFormat.rgba8888:
      final planes = frame.planes;
      FrameGatherer._require(planes.length == 1, frame, 'one plane');
      final p = planes.single;
      final px = Uint8List.sublistView(data, p.offset, p.offset + p.length);
      final bpr = p.bytesPerRow;
      final last = (srcH - 1) * bpr + (srcW - 1) * 4 + 3;
      if (last >= px.length) {
        throw ArgumentError(
          'Frame buffer too small: ${px.length} bytes for ${srcW}x$srcH '
          'with $bpr bytes per row',
        );
      }
      final rgba = frame.format == FramePixelFormat.rgba8888;
      if (rot == 0 && (px.offsetInBytes | bpr) % 4 == 0) {
        _packedUpright32(px, bpr, srcW, srcH, out, rgba: rgba);
        break;
      }
      final rOff = rgba ? 0 : 2;
      final bOff = rgba ? 2 : 0;
      var o = 0;
      for (var uy = 0; uy < upH; uy++) {
        for (var ux = 0; ux < upW; ux++, o += 4) {
          final si =
              _srcY(ux, uy, srcH, rot) * bpr + _srcX(ux, uy, srcW, rot) * 4;
          out[o] = px[si + rOff];
          out[o + 1] = px[si + 1];
          out[o + 2] = px[si + bOff];
          out[o + 3] = 255;
        }
      }
    case FramePixelFormat.nv21 || FramePixelFormat.yuv420:
      final yuv = FrameGatherer._yuvPlanes(frame, data);
      final yP = yuv.y;
      final uP = yuv.u;
      final vP = yuv.v;
      var o = 0;
      for (var uy = 0; uy < upH; uy++) {
        for (var ux = 0; ux < upW; ux++, o += 4) {
          final sx = _srcX(ux, uy, srcW, rot);
          final sy = _srcY(ux, uy, srcH, rot);
          final yi = sy * yuv.yRowStride + sx;
          final uvi =
              (sy >> 1) * yuv.uvRowStride + (sx >> 1) * yuv.uvPixelStride;
          if (yi >= yP.length || uvi >= uP.length || uvi >= vP.length) {
            throw ArgumentError(
              'YUV planes too small for ${srcW}x$srcH: Y ${yP.length}, '
              'U ${uP.length}, V ${vP.length}',
            );
          }
          final yy = yP[yi] << 10;
          final u = uP[uvi] - 128;
          final v = vP[uvi] - 128;
          out[o] = ((yy + 1436 * v) >> 10).clamp(0, 255);
          out[o + 1] = ((yy - 352 * u - 731 * v) >> 10).clamp(0, 255);
          out[o + 2] = ((yy + 1815 * u) >> 10).clamp(0, 255);
          out[o + 3] = 255;
        }
      }
  }
  return RgbaPixels(width: upW, height: upH, bytes: out);
}

/// The unrotated packed case one 32-bit word per pixel (little-endian):
/// RGBA gets alpha 255; BGRA (A R G B as a word) swaps R and B too.
void _packedUpright32(
  Uint8List px,
  int bpr,
  int w,
  int h,
  Uint8List out, {
  required bool rgba,
}) {
  final rowWords = bpr >> 2;
  final src = Uint32List.view(
    px.buffer,
    px.offsetInBytes,
    ((h - 1) * bpr + w * 4) >> 2,
  );
  final dst = Uint32List.view(out.buffer, out.offsetInBytes, w * h);
  var o = 0;
  for (var y = 0; y < h; y++) {
    var i = y * rowWords;
    final end = i + w;
    if (rgba) {
      for (; i < end; i++, o++) {
        dst[o] = src[i] | 0xFF000000;
      }
    } else {
      for (; i < end; i++, o++) {
        final v = src[i];
        dst[o] =
            0xFF000000 | ((v & 0xFF) << 16) | (v & 0xFF00) | ((v >> 16) & 0xFF);
      }
    }
  }
}

/// Top-k decode of the raw head: every (anchor, class) with score ≥
/// [threshold], sorted by score (descending, the reference's order), at most
/// [maxDet], mapped back to upright frame pixels through [lb]. Returns
/// `count × 6` floats: x1, y1, x2, y2, score, class. No NMS: YOLO26 is trained
/// one-to-one.
Float32List decodeDetections(
  Float32List raw,
  Letterbox lb, {
  double threshold = kDetDecodeScore,
  int maxDet = kDetMaxDet,
}) {
  if (raw.length != kDetAnchors * kDetRawStride) {
    throw ArgumentError(
      'Raw output has ${raw.length} floats, expected '
      '${kDetAnchors * kDetRawStride}',
    );
  }
  final candidates = <_Candidate>[];
  for (var a = 0; a < kDetAnchors; a++) {
    final base = a * kDetRawStride;
    for (var c = 0; c < kDetClasses; c++) {
      final s = raw[base + 4 + c];
      if (s >= threshold) candidates.add(_Candidate(base, c, s));
    }
  }
  candidates.sort((p, q) => q.score.compareTo(p.score));
  final n = math.min(candidates.length, maxDet);
  final out = Float32List(n * kBoxStride);
  for (var i = 0; i < n; i++) {
    final d = candidates[i];
    final o = i * kBoxStride;
    out[o] = lb.unX(raw[d.base]);
    out[o + 1] = lb.unY(raw[d.base + 1]);
    out[o + 2] = lb.unX(raw[d.base + 2]);
    out[o + 3] = lb.unY(raw[d.base + 3]);
    out[o + 4] = d.score;
    out[o + 5] = d.cls.toDouble();
  }
  return out;
}

final class _Candidate {
  _Candidate(this.base, this.cls, this.score);

  final int base;
  final int cls;
  final double score;
}
