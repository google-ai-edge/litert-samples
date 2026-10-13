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

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_litert/flutter_litert.dart' show CoverFitTransform;

import '../../config/live_camera_config.dart';
import '../../domain/models/detection.dart';
import '../../domain/models/detector_spec.dart';
import '../../domain/vision/coco_vocabulary.dart';

/// One box as drawn: view coordinates and its label.
final class const PaintedBox({
  required final Rect rect,
  required final int classId,
  required final double score,
}) {
  String get label => '${cocoName(classId)} ${score.toStringAsFixed(2)}';
}

/// Maps [frame]'s boxes (upright frame pixels) onto a [view] that shows the
/// frame with `BoxFit.cover`, centred — the same fit as the preview
/// (`CoverFitTransform` from flutter_litert). [mirror] reflects x first (a
/// mirrored preview of unmirrored frames). Boxes come sorted by score; those
/// below [minScore] and beyond [maxBoxes] are left out.
List<PaintedBox> layoutDetectionBoxes(
  DetectionFrame frame,
  Size view, {
  required bool mirror,
  double minScore = kDetDisplayScore,
  int maxBoxes = kMaxPaintedBoxes,
}) {
  if (view.isEmpty || frame.count == 0 || frame.width <= 0) return const [];
  final fit = CoverFitTransform.cover(
    sourceWidth: frame.width.toDouble(),
    sourceHeight: frame.height.toDouble(),
    viewWidth: view.width,
    viewHeight: view.height,
    mirror: mirror,
  );
  final boxes = <PaintedBox>[];
  for (var i = 0; i < frame.count && boxes.length < maxBoxes; i++) {
    final score = frame.score(i);
    if (score < minScore) break; // sorted, descending
    boxes.add(
      PaintedBox(
        rect: Rect.fromPoints(
          fit.map(frame.x1(i), frame.y1(i)),
          fit.map(frame.x2(i), frame.y2(i)),
        ),
        classId: frame.classId(i),
        score: score,
      ),
    );
  }
  return boxes;
}

/// Draws the newest detections over the live preview. Repaints when
/// [frames] changes (≤15 Hz) without rebuilding any widget; put it in its
/// own `RepaintBoundary` so the preview under it is not repainted.
class DetectionPainter extends CustomPainter {
  DetectionPainter({
    required this.frames,
    required this.mirror,
    this.minScore = kDetDisplayScore,
    this.maxBoxes = kMaxPaintedBoxes,
  }) : super(repaint: frames);

  final ValueListenable<DetectionFrame?> frames;
  final bool mirror;
  final double minScore;
  final int maxBoxes;

  @override
  void paint(Canvas canvas, Size size) {
    final frame = frames.value;
    if (frame == null) return;
    final boxes = layoutDetectionBoxes(
      frame,
      size,
      mirror: mirror,
      minScore: minScore,
      maxBoxes: maxBoxes,
    );
    final stroke = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.5;
    final fill = Paint();
    for (final box in boxes) {
      final color = _colorFor(box.classId);
      canvas.drawRect(box.rect, stroke..color = color);
      final text = TextPainter(
        text: TextSpan(
          text: box.label,
          style: const TextStyle(
            color: Colors.black,
            fontSize: 12,
            fontWeight: FontWeight.w600,
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout();
      final top = box.rect.top - text.height - 2 >= 0
          ? box.rect.top - text.height - 2
          : box.rect.top;
      final at = Offset(box.rect.left.clamp(0, size.width).toDouble(), top);
      canvas.drawRect(
        Rect.fromLTWH(at.dx, at.dy, text.width + 6, text.height + 2),
        fill..color = color,
      );
      text
        ..paint(canvas, at.translate(3, 1))
        ..dispose();
    }
  }

  /// A stable, distinct colour per class.
  static Color _colorFor(int cls) =>
      HSVColor.fromAHSV(1, (cls * 47 % 360).toDouble(), 0.75, 1).toColor();

  @override
  bool shouldRepaint(DetectionPainter oldDelegate) =>
      oldDelegate.frames != frames ||
      oldDelegate.mirror != mirror ||
      oldDelegate.minScore != minScore ||
      oldDelegate.maxBoxes != maxBoxes;
}
