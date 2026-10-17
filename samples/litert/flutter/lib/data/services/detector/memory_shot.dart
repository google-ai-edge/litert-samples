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

/// The visual memory's frame, made in the detector worker from the frame the
/// detector just read: an upright downscale (≤ `maxSide`, mirrored back when
/// the source mirrors), its grey thumbnail for the scene-change test (32×32
/// cells of 4×4 pixels), and — only when the scene changed — the PNG.
/// Nothing of this runs on the UI isolate.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import '../../../domain/models/detection.dart';
import '../../../domain/models/scene_snapshot.dart';
import '../../../domain/models/visual_memory.dart';
import '../images/png_encoder.dart';
import 'detector_codec.dart';
import 'frame_message.dart';

/// What the worker is asked for with a memory frame (main → worker).
final class const MemoryShotSpec({
  /// The PNG's long side, at most (never scaled up).
  required final int maxSide,

  /// The scene-change test's side, in cells.
  required final int greySide,

  /// Each cell is this many thumbnail pixels a side: the grey thumbnail is
  /// [greySide] × [cellPixels] pixels square, fine enough to re-align a
  /// camera that moved a few pixels before it is averaged into cells.
  required final int cellPixels,

  /// The reference is re-aligned by up to this many thumbnail pixels in x
  /// and y ([sceneChange]'s `maxShift`).
  required final int maxShift,

  /// Mirror the frame back (the source mirrors its frames): text reads the
  /// right way round for the embedder, the card and the chat model.
  required final bool unmirror,

  /// The last stored frame's grey thumbnail; null when none is stored.
  final Uint8List? reference,

  /// The mean absolute grey difference that counts as a scene change (after
  /// the brightness match, [sceneChange]).
  required final double changeThreshold,

  /// A cell that differs by at least this (after the brightness match)…
  required final double cellChange,

  /// …counts towards a scene change, which this many such cells make.
  required final int changedCells,
});

/// Worker → main for a memory capture's frame: the frame's [detections],
/// and its memory frame ([shot]) or why it could not be made ([error]). A
/// failed memory frame fails only the memory capture: the detections are
/// published like any other frame's and the live pipeline goes on.
final class const MemoryShotReply({
  required final DetectionFrame detections,
  final MemoryShot? shot,
  final String? error,
});

/// The memory frame of [frame] (whose packed planes are [data]) and its
/// detections, as [MemoryShot]. The PNG is made only when there is no
/// [MemoryShotSpec.reference] or the scene changed ([sceneChange]: the mean
/// difference reaches [MemoryShotSpec.changeThreshold], or
/// [MemoryShotSpec.changedCells] cells differ by
/// [MemoryShotSpec.cellChange]): an unchanged scene is not stored again
/// (the memory refreshes its newest frame's time).
MemoryShot memoryShot(
  FrameMessage frame,
  Uint8List data,
  DetectionFrame detections,
  MemoryShotSpec spec,
) {
  final watch = Stopwatch()..start();
  final upright = uprightRgba(frame, data);
  final (width, height) = fitLongSide(
    upright.width,
    upright.height,
    spec.maxSide,
  );
  final rgb = downscaleRgb(upright, width, height, mirror: spec.unmirror);
  final grey = greyThumbnail(
    rgb,
    width,
    height,
    spec.greySide * spec.cellPixels,
  );
  final reference = spec.reference;
  final change = reference == null
      ? null
      : sceneChange(
          reference,
          grey,
          cellThreshold: spec.cellChange,
          cellPixels: spec.cellPixels,
          maxShift: spec.maxShift,
        );
  final changed =
      change == null ||
      isSceneChange(
        change,
        meanThreshold: spec.changeThreshold,
        changedCells: spec.changedCells,
      );
  final png = changed ? encodePngRgb(width, height, rgb) : null;
  return MemoryShot(
    frameId: frame.frameId,
    detections: detections,
    grey: grey,
    difference: change?.mean,
    changedCells: change?.cells,
    png: png,
    width: width,
    height: height,
    unmirrored: spec.unmirror,
    workerTime: watch.elapsed,
  );
}

/// [width]×[height] scaled so the long side is at most [maxSide], keeping
/// the aspect ratio; never up, never below 1 px.
(int, int) fitLongSide(int width, int height, int maxSide) {
  final scale = math.min(1.0, maxSide / math.max(width, height));
  return (
    math.max(1, (width * scale).round()),
    math.max(1, (height * scale).round()),
  );
}

/// [pixels] averaged down to [width]×[height] tightly packed RGB: each output
/// pixel is the mean of the source block it covers (area averaging, so a
/// downscale does not alias). [mirror] flips left to right on the way.
Uint8List downscaleRgb(
  RgbaPixels pixels,
  int width,
  int height, {
  bool mirror = false,
}) {
  final srcW = pixels.width;
  final srcH = pixels.height;
  if (width <= 0 || height <= 0 || width > srcW || height > srcH) {
    throw ArgumentError(
      'downscale ${srcW}x$srcH to ${width}x$height: the target must be '
      'positive and not larger',
    );
  }
  final src = pixels.bytes;
  final x0 = Int32List(width);
  final x1 = Int32List(width);
  for (var x = 0; x < width; x++) {
    x0[x] = x * srcW ~/ width;
    x1[x] = math.max(x0[x] + 1, (x + 1) * srcW ~/ width);
  }
  final out = Uint8List(width * height * 3);
  final sums = Int32List(width * 3);
  for (var y = 0; y < height; y++) {
    final y0 = y * srcH ~/ height;
    final y1 = math.max(y0 + 1, (y + 1) * srcH ~/ height);
    sums.fillRange(0, sums.length, 0);
    for (var sy = y0; sy < y1; sy++) {
      final row = sy * srcW * 4;
      for (var x = 0; x < width; x++) {
        var r = 0;
        var g = 0;
        var b = 0;
        for (var i = row + x0[x] * 4, end = row + x1[x] * 4; i < end; i += 4) {
          r += src[i];
          g += src[i + 1];
          b += src[i + 2];
        }
        final s = x * 3;
        sums[s] += r;
        sums[s + 1] += g;
        sums[s + 2] += b;
      }
    }
    final rows = y1 - y0;
    for (var x = 0; x < width; x++) {
      final n = rows * (x1[x] - x0[x]);
      final s = x * 3;
      final o = (y * width + (mirror ? width - 1 - x : x)) * 3;
      out[o] = (sums[s] + n ~/ 2) ~/ n;
      out[o + 1] = (sums[s + 1] + n ~/ 2) ~/ n;
      out[o + 2] = (sums[s + 2] + n ~/ 2) ~/ n;
    }
  }
  return out;
}

/// The [side]×[side] grey (BT.601 luma) thumbnail of tightly packed RGB
/// [rgb] ([width]×[height]): each pixel is the mean over exactly the area it
/// covers (source pixels on its edges count by the fraction inside), so the
/// thumbnail's pixels sit on a uniform grid and a shift by one of them is
/// the same distance everywhere. The aspect ratio is not kept (it only has
/// to compare two frames of one source). An image smaller than [side]
/// spreads its pixels.
Uint8List greyThumbnail(Uint8List rgb, int width, int height, int side) {
  if (rgb.length != width * height * 3 || width <= 0 || height <= 0) {
    throw ArgumentError('${rgb.length} bytes for ${width}x$height RGB');
  }
  final luma = Float64List(width * height);
  for (var i = 0, p = 0; i < luma.length; i++, p += 3) {
    luma[i] = 0.299 * rgb[p] + 0.587 * rgb[p + 1] + 0.114 * rgb[p + 2];
  }
  // Across: each row of the source into [side] columns.
  final xs = _coverage(width, side);
  final across = Float64List(side * height);
  for (var y = 0; y < height; y++) {
    final row = y * width;
    for (var cx = 0; cx < side; cx++) {
      var sum = 0.0;
      for (final (x, weight) in xs[cx]) {
        sum += luma[row + x] * weight;
      }
      across[y * side + cx] = sum;
    }
  }
  // Down: each column of that into [side] rows.
  final ys = _coverage(height, side);
  final out = Uint8List(side * side);
  for (var cy = 0; cy < side; cy++) {
    for (var cx = 0; cx < side; cx++) {
      var sum = 0.0;
      for (final (y, weight) in ys[cy]) {
        sum += across[y * side + cx] * weight;
      }
      out[cy * side + cx] = sum.round().clamp(0, 255);
    }
  }
  return out;
}

/// For each of [n] equal parts of [length] source pixels, the source pixels
/// it covers with the fraction of the part each one makes up (they sum to
/// 1).
List<List<(int, double)>> _coverage(int length, int n) {
  final step = length / n;
  return [
    for (var i = 0; i < n; i++)
      () {
        final start = i * step;
        final end = (i + 1) * step;
        return [
          for (var p = start.floor(); p < end && p < length; p++)
            if (math.min(end, p + 1.0) - math.max(start, p.toDouble())
                case final inside when inside > 1e-9)
              (p, inside / step),
        ];
      }(),
  ];
}

/// What [sceneChange] measured between two grey thumbnails, after matching
/// their brightness and re-aligning the reference: the mean absolute cell
/// difference and the largest one (on 0–255), how many cells differ by at
/// least the threshold, the gain applied to the current thumbnail, and the
/// shift of the reference (thumbnail pixels, in halves) that matched best.
typedef SceneChange = ({
  double mean,
  double max,
  int cells,
  double gain,
  double shiftX,
  double shiftY,
});

/// [change] is a scene change: a mean difference of at least
/// [meanThreshold], or at least [changedCells] changed cells.
bool isSceneChange(
  SceneChange change, {
  required double meanThreshold,
  required int changedCells,
}) => change.mean >= meanThreshold || change.cells >= changedCells;

/// Cells darker than this, on either side, do not vote on the brightness
/// match (their ratio is mostly noise).
const _minGainLuma = 8;

/// The brightness match is at most this factor either way: a bigger jump is
/// a change of light the memory stores anew.
const _maxGain = 2.0;

/// How far grey thumbnail [current] is from [reference] (both square, the
/// same size, a whole number of cells of [cellPixels]×[cellPixels] pixels),
/// cell by cell:
///
/// - **Camera motion.** A hand-held camera moves a few pixels between two
///   samples, and every cell on an edge changes with it. The reference is
///   re-aligned by every global shift of up to [maxShift] thumbnail pixels
///   in x and y, in steps of half a pixel (a half step averages the two or
///   four whole-pixel alignments around it; edge pixels repeated), and the
///   alignment with the smallest mean difference counts (no shift on a tie).
///   Whole pixels alone would leave a quarter of a pixel at worst, which a
///   high-contrast edge (a black phone on white paper) turns into a changed
///   cell. A global shift cannot hide an object put down or taken away: its
///   cells differ under every alignment.
/// - **Brightness.** [current] is scaled by the median of the per-cell
///   ratios reference / current at no shift (cells under [_minGainLuma] on
///   either side do not vote; 1 when none votes; at most [_maxGain] either
///   way). An exposure or lighting change of the whole view scales every
///   cell alike and drops out; an object covering less than half the view
///   leaves the median alone and keeps its cells' full difference.
///
/// [SceneChange.cells] counts the cells that differ by at least
/// [cellThreshold].
SceneChange sceneChange(
  Uint8List reference,
  Uint8List current, {
  required double cellThreshold,
  int cellPixels = 1,
  int maxShift = 0,
}) {
  final side = math.sqrt(reference.length).round();
  if (reference.length != current.length ||
      reference.isEmpty ||
      side * side != reference.length ||
      cellPixels < 1 ||
      side % cellPixels != 0 ||
      maxShift < 0) {
    throw ArgumentError(
      'thumbnails of ${reference.length} and ${current.length} pixels, '
      'cells of $cellPixels px, shifts up to $maxShift px',
    );
  }
  final cells = side ~/ cellPixels;
  final area = cellPixels * cellPixels;
  // The current thumbnail's cells.
  final now = Float64List(cells * cells);
  for (var y = 0; y < side; y++) {
    final row = (y ~/ cellPixels) * cells;
    for (var x = 0; x < side; x++) {
      now[row + x ~/ cellPixels] += current[y * side + x];
    }
  }
  for (var i = 0; i < now.length; i++) {
    now[i] /= area;
  }
  // The reference padded by [maxShift] on every side (its edge pixels
  // repeated), as an integral image: a shifted cell's sum in four reads.
  final padded = side + 2 * maxShift;
  final stride = padded + 1;
  final integral = Int32List(stride * stride);
  for (var py = 0; py < padded; py++) {
    final sy = (py - maxShift).clamp(0, side - 1);
    var rowSum = 0;
    for (var px = 0; px < padded; px++) {
      rowSum += reference[sy * side + (px - maxShift).clamp(0, side - 1)];
      integral[(py + 1) * stride + px + 1] =
          integral[py * stride + px + 1] + rowSum;
    }
  }
  double referenceCell(int cell, int dx, int dy) {
    final x0 = (cell % cells) * cellPixels + dx + maxShift;
    final y0 = (cell ~/ cells) * cellPixels + dy + maxShift;
    final x1 = x0 + cellPixels;
    final y1 = y0 + cellPixels;
    return (integral[y1 * stride + x1] -
            integral[y0 * stride + x1] -
            integral[y1 * stride + x0] +
            integral[y0 * stride + x0]) /
        area;
  }

  final ratios = <double>[
    for (var i = 0; i < now.length; i++)
      if (referenceCell(i, 0, 0) case final r
          when r >= _minGainLuma && now[i] >= _minGainLuma)
        r / now[i],
  ]..sort();
  final gain = ratios.isEmpty
      ? 1.0
      : (ratios.length.isOdd
                ? ratios[ratios.length ~/ 2]
                : (ratios[ratios.length ~/ 2 - 1] +
                          ratios[ratios.length ~/ 2]) /
                      2)
            .clamp(1 / _maxGain, _maxGain);
  // The whole pixels around a shift of [half] half-pixels: one, or two for
  // an odd one (each read with half the weight).
  List<int> around(int half) =>
      half.isEven ? [half ~/ 2] : [(half - 1) ~/ 2, (half + 1) ~/ 2];
  // [halfX], [halfY]: the shift in half thumbnail pixels.
  SceneChange at(int halfX, int halfY) {
    final xs = around(halfX);
    final ys = around(halfY);
    final weight = 1 / (xs.length * ys.length);
    var sum = 0.0;
    var max = 0.0;
    var changed = 0;
    for (var i = 0; i < now.length; i++) {
      var shifted = 0.0;
      for (final dy in ys) {
        for (final dx in xs) {
          shifted += referenceCell(i, dx, dy);
        }
      }
      final d = (shifted * weight - now[i] * gain).abs();
      sum += d;
      if (d > max) max = d;
      if (d >= cellThreshold) changed++;
    }
    return (
      mean: sum / now.length,
      max: max,
      cells: changed,
      gain: gain,
      shiftX: halfX / 2,
      shiftY: halfY / 2,
    );
  }

  var best = at(0, 0);
  for (var halfY = -2 * maxShift; halfY <= 2 * maxShift; halfY++) {
    for (var halfX = -2 * maxShift; halfX <= 2 * maxShift; halfX++) {
      if (halfX == 0 && halfY == 0) continue;
      final aligned = at(halfX, halfY);
      if (aligned.mean < best.mean) best = aligned;
    }
  }
  return best;
}

/// The mean absolute difference of two grey thumbnails of the same size, on
/// 0–255, as they are (no brightness match): 0 for the same scene.
double greyDifference(Uint8List a, Uint8List b) {
  if (a.length != b.length || a.isEmpty) {
    throw ArgumentError('thumbnails of ${a.length} and ${b.length} cells');
  }
  var sum = 0;
  for (var i = 0; i < a.length; i++) {
    sum += (a[i] - b[i]).abs();
  }
  return sum / a.length;
}
