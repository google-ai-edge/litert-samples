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

import '../models/detection.dart';
import '../vision/coco_vocabulary.dart';

/// Detections a camera prompt lists at most.
const kCameraPromptMaxDetections = 8;

/// Only detections at least this confident are listed: below it the hint
/// misleads more than it helps (the cats fixture's sofa is 0.30).
const kCameraPromptMinScore = 0.5;

/// The prompt for a detailed camera question: the
/// question, preceded by at most [kCameraPromptMaxDetections] detections
/// scoring at least [kCameraPromptMinScore], each with a coarse position
/// (thirds: left/center/right × top/middle/bottom), labelled as possibly
/// incomplete. The hint helps spatial answers; the image stays the source of
/// truth.
///
/// Positions are those of the image Gemma gets: a [mirrored] frame is sent
/// mirrored back, so its x positions are mirrored too. Without a confident
/// detection the question goes alone.
String buildCameraPrompt(
  String question,
  DetectionFrame frame, {
  required bool mirrored,
  CocoVocabulary vocabulary = kCocoVocabulary,
}) {
  final hints = <String>[];
  for (var i = 0; i < frame.count; i++) {
    if (hints.length == kCameraPromptMaxDetections) break;
    if (frame.score(i) < kCameraPromptMinScore) break; // sorted, descending
    var cx = (frame.x1(i) + frame.x2(i)) / 2 / frame.width;
    if (mirrored) cx = 1 - cx;
    final cy = (frame.y1(i) + frame.y2(i)) / 2 / frame.height;
    final name = vocabulary.spokenName(frame.classId(i), 1);
    hints.add('$name (${_position(cx, cy)})');
  }
  final asked = question.trim();
  if (hints.isEmpty) return asked;
  return 'Objects a detector found in this frame (may be incomplete): '
      '${hints.join(', ')}.\n\n'
      'Question: $asked';
}

/// "top left", "left", "center", "bottom"… for a centre in 0–1 coordinates.
String _position(double cx, double cy) {
  final h = cx < 1 / 3
      ? 'left'
      : cx > 2 / 3
      ? 'right'
      : 'center';
  final v = cy < 1 / 3
      ? 'top'
      : cy > 2 / 3
      ? 'bottom'
      : 'middle';
  return switch ((v, h)) {
    ('middle', _) => h,
    (_, 'center') => v,
    _ => '$v $h',
  };
}
