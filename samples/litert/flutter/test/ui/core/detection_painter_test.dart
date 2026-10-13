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

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/domain/models/detection.dart';
import 'package:litert_edge_demos/ui/core/detection_painter.dart';

DetectionFrame frame(List<double> boxes, {int width = 640, int height = 480}) =>
    DetectionFrame(
      frameId: 1,
      width: width,
      height: height,
      boxes: Float32List.fromList(boxes),
      preMicros: 0,
      runMicros: 0,
      postMicros: 0,
      backend: DetectorBackend.gpu,
    );

/// One cat box (100, 100)–(300, 300) in a 640×480 frame.
final _cat = frame([100, 100, 300, 300, 0.9, 15]);

Rect only(DetectionFrame f, Size view, {bool mirror = false}) =>
    layoutDetectionBoxes(f, view, mirror: mirror).single.rect;

void main() {
  group('layoutDetectionBoxes (cover fit, like the preview)', () {
    test('same aspect: plain scale', () {
      expect(
        only(_cat, const Size(320, 240)),
        const Rect.fromLTRB(50, 50, 150, 150),
      );
    });

    test('a narrower view crops the sides: scale by height, shift left', () {
      // 640×480 into 480×480: scale 1, 80 px cut on each side.
      expect(
        only(_cat, const Size(480, 480)),
        const Rect.fromLTRB(20, 100, 220, 300),
      );
    });

    test('a taller view crops the sides more: scale 2, shift 320', () {
      expect(
        only(_cat, const Size(640, 960)),
        const Rect.fromLTRB(-120, 200, 280, 600),
      );
    });

    test('a wider view crops top and bottom', () {
      // 640×480 into 640×240: scale 1, 120 px cut top and bottom.
      expect(
        only(_cat, const Size(640, 240)),
        const Rect.fromLTRB(100, -20, 300, 180),
      );
    });

    test('mirror reflects x about the frame width before the fit', () {
      expect(
        only(_cat, const Size(320, 240), mirror: true),
        const Rect.fromLTRB(170, 50, 270, 150),
      );
    });

    test(
      'keeps boxes at or above minScore, at most maxBoxes, in score order',
      () {
        final f = frame([
          0, 0, 10, 10, 0.9, 0, //
          0, 0, 20, 20, 0.5, 1, //
          0, 0, 30, 30, 0.36, 2, //
          0, 0, 40, 40, 0.3, 3,
        ]);
        final all = layoutDetectionBoxes(
          f,
          const Size(640, 480),
          mirror: false,
        );
        expect([for (final b in all) b.classId], [0, 1, 2]);
        final capped = layoutDetectionBoxes(
          f,
          const Size(640, 480),
          mirror: false,
          maxBoxes: 2,
        );
        expect([for (final b in capped) b.classId], [0, 1]);
        expect(capped.first.label, 'person 0.90');
      },
    );

    test('an empty view or frame draws nothing', () {
      expect(layoutDetectionBoxes(_cat, Size.zero, mirror: false), isEmpty);
      expect(
        layoutDetectionBoxes(frame([]), const Size(320, 240), mirror: false),
        isEmpty,
      );
    });
  });

  testWidgets('repaints on every new frame without rebuilding the widget', (
    tester,
  ) async {
    final frames = ValueNotifier<DetectionFrame?>(null);
    addTearDown(frames.dispose);
    final painter = _CountingPainter(frames);
    var builds = 0;

    await tester.pumpWidget(
      StatefulBuilder(
        builder: (context, _) {
          builds++;
          return Center(
            child: RepaintBoundary(
              child: CustomPaint(size: const Size(320, 240), painter: painter),
            ),
          );
        },
      ),
    );
    final paintsBefore = painter.paints;

    frames.value = _cat;
    await tester.pump();
    frames.value = frame([0, 0, 640, 480, 0.5, 57]);
    await tester.pump();

    expect(painter.paints, paintsBefore + 2);
    expect(builds, 1);
    expect(painter.shouldRepaint(_CountingPainter(frames)), isFalse);
    expect(
      painter.shouldRepaint(DetectionPainter(frames: frames, mirror: true)),
      isTrue,
    );
  });
}

class _CountingPainter extends DetectionPainter {
  _CountingPainter(ValueNotifier<DetectionFrame?> frames)
    : super(frames: frames, mirror: false);

  int paints = 0;

  @override
  void paint(Canvas canvas, Size size) {
    paints++;
    super.paint(canvas, size);
  }
}
