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

import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/domain/models/detection.dart';
import 'package:litert_edge_demos/domain/use_cases/camera_prompt.dart';
import 'package:litert_edge_demos/domain/vision/coco_vocabulary.dart';

int cls(String name) => kCocoNames.indexOf(name);

DetectionFrame frameOf(List<double> boxes, {int w = 640, int h = 480}) =>
    DetectionFrame(
      frameId: 1,
      width: w,
      height: h,
      boxes: Float32List.fromList(boxes),
      preMicros: 0,
      runMicros: 0,
      postMicros: 0,
      backend: DetectorBackend.gpu,
    );

/// The cats fixture's golden detections (sorted by score).
DetectionFrame cats() => frameOf([
  344, 24.5, 640, 374.8, 0.912, cls('cat').toDouble(), //
  6.9, 55.1, 317.3, 466.1, 0.899, cls('cat').toDouble(),
  40.4, 74, 175.9, 118.6, 0.857, cls('remote').toDouble(),
  0.8, 0.6, 640, 480, 0.297, cls('sofa').toDouble(),
]);

void main() {
  test('the question plus the confident detections with coarse positions, '
      'labelled as possibly incomplete', () {
    final prompt = buildCameraPrompt(
      ' Describe the scene. ',
      cats(),
      mirrored: false,
    );
    expect(
      prompt,
      'Objects a detector found in this frame (may be incomplete): '
      'cat (right), cat (left), remote (top left).\n\n'
      'Question: Describe the scene.',
    );
    expect(prompt, isNot(contains('sofa')), reason: '0.297 < 0.5');
  });

  test('a mirroring source: positions are those of the un-mirrored image '
      'Gemma gets', () {
    final prompt = buildCameraPrompt(
      'Where is the remote?',
      cats(),
      mirrored: true,
    );
    expect(prompt, contains('cat (left), cat (right), remote (top right)'));
  });

  test('spoken names, not Darknet spellings; at most 8', () {
    final prompt = buildCameraPrompt(
      'What is this?',
      frameOf([
        for (var i = 0; i < 12; i++) ...[
          300.0,
          200,
          340,
          280,
          0.9 - i * 0.01,
          cls(i.isEven ? 'tvmonitor' : 'pottedplant').toDouble(),
        ],
      ]),
      mirrored: false,
    );
    expect(RegExp(r'\(center\)').allMatches(prompt), hasLength(8));
    expect(prompt, contains('TV (center), potted plant (center)'));
    expect(prompt, isNot(contains('tvmonitor')));
  });

  test('nothing confident: the question alone (no empty hint to mislead)', () {
    expect(
      buildCameraPrompt(
        ' What does the sign say?',
        frameOf([0, 0, 10, 10, 0.3, 1]),
        mirrored: false,
      ),
      'What does the sign say?',
    );
  });
}
