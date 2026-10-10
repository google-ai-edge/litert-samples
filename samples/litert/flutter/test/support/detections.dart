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

import 'package:litert_edge_demos/domain/models/detection.dart';

/// A detected 640×480 frame with [boxes] as (class, score), sorted by score
/// like the detector's output.
DetectionFrame detections(int frameId, [List<(int, double)> boxes = const []]) {
  final sorted = [...boxes]..sort((a, b) => b.$2.compareTo(a.$2));
  return DetectionFrame(
    frameId: frameId,
    width: 640,
    height: 480,
    boxes: Float32List.fromList([
      for (final (cls, score) in sorted) ...[
        10,
        20,
        110,
        220,
        score,
        cls.toDouble(),
      ],
    ]),
    preMicros: 1000,
    runMicros: 4000,
    postMicros: 1000,
    backend: DetectorBackend.gpu,
  );
}
