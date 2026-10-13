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

import '../domain/models/detection.dart';

/// The bundled cats image (`test_assets/cats.jpg`): COCO 2017 val image
/// 000000039769, 640×480, from Flickr via the COCO dataset (cocodataset.org;
/// Creative Commons, author and licence in the COCO annotations).
const kCatsAsset = 'test_assets/cats.jpg';

/// YOLO26n's strict-GPU fp32 detections on the cats image, in score order
/// (`test_assets/yolo26n/cats_golden.json`; frame pixels, PIL decode). The
/// same golden as integration_test/detector_coex_test.dart.
const kCatsGolden = [
  (cls: 15, score: 0.912, box: [344.0, 24.5, 640.0, 374.8]),
  (cls: 15, score: 0.899, box: [6.9, 55.1, 317.3, 466.1]),
  (cls: 65, score: 0.857, box: [40.4, 74.0, 175.9, 118.6]),
  (cls: 57, score: 0.297, box: [0.8, 0.6, 640.0, 480.0]),
];

/// Largest box corner error allowed, in frame pixels (the coex test's).
const kCatsMaxBoxPx = 3.0;

/// Largest score difference allowed.
const kCatsMaxScoreDelta = 0.03;

/// How one detected frame compares with [kCatsGolden].
final class const CatsGoldenCheck({
  /// The frame's class ids, in score order.
  required final List<int> classes,

  /// Worst box corner error over the golden detections, best match each.
  required final double maxBoxPx,
  required final double maxScoreDelta,
}) {
  bool get classesMatch =>
      classes.length == kCatsGolden.length &&
      [
        for (var i = 0; i < classes.length; i++)
          classes[i] == kCatsGolden[i].cls,
      ].every((ok) => ok);

  bool get passed =>
      classesMatch &&
      maxBoxPx <= kCatsMaxBoxPx &&
      maxScoreDelta <= kCatsMaxScoreDelta;
}

/// [frame] against [kCatsGolden]: for each golden detection the same-class
/// box with the smallest corner error.
CatsGoldenCheck checkCatsGolden(DetectionFrame frame) {
  var worstBox = 0.0;
  var worstScore = 0.0;
  for (final g in kCatsGolden) {
    var best = double.infinity;
    var bestScore = 0.0;
    for (var i = 0; i < frame.count; i++) {
      if (frame.classId(i) != g.cls) continue;
      final err = [
        (frame.x1(i) - g.box[0]).abs(),
        (frame.y1(i) - g.box[1]).abs(),
        (frame.x2(i) - g.box[2]).abs(),
        (frame.y2(i) - g.box[3]).abs(),
      ].reduce((a, b) => a > b ? a : b);
      if (err < best) {
        best = err;
        bestScore = (frame.score(i) - g.score).abs();
      }
    }
    if (best > worstBox) worstBox = best;
    if (bestScore > worstScore) worstScore = bestScore;
  }
  return CatsGoldenCheck(
    classes: [for (var i = 0; i < frame.count; i++) frame.classId(i)],
    maxBoxPx: worstBox,
    maxScoreDelta: worstScore,
  );
}
