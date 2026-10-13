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
import 'package:litert_edge_demos/domain/models/detection_summary.dart';
import 'package:litert_edge_demos/domain/models/route_decision.dart';
import 'package:litert_edge_demos/domain/use_cases/fast_answer_composer.dart';
import 'package:litert_edge_demos/domain/vision/coco_vocabulary.dart';

int cls(String name) => kCocoNames.indexOf(name);

/// A frame with one box per (class name, score).
DetectionFrame frame(List<(String, double)> boxes, {int id = 1}) =>
    DetectionFrame(
      frameId: id,
      width: 640,
      height: 480,
      boxes: Float32List.fromList([
        for (final (name, score) in boxes) ...[
          0,
          0,
          10,
          10,
          score,
          cls(name).toDouble(),
        ],
      ]),
      preMicros: 0,
      runMicros: 0,
      postMicros: 0,
      backend: DetectorBackend.gpu,
    );

/// The cats fixture's detections (test_assets/yolo26n/cats_golden.json).
DetectionFrame cats() =>
    frame([('cat', 0.912), ('cat', 0.899), ('remote', 0.857), ('sofa', 0.297)]);

void main() {
  group('DetectionSummary (median over the window at ≥0.4)', () {
    test('the cats frame: two cats and a remote, the sofa is under 0.4', () {
      final s = DetectionSummary.fromWindow([
        for (var i = 0; i < 5; i++) cats(),
      ], minScore: 0.4);
      expect(s.counts, {cls('cat'): 2, cls('remote'): 1});
      expect(s.frames, 5);
      expect(s.label(cocoName), 'cat ×2 · remote');
    });

    test('one-frame flicker is absorbed by the median', () {
      final s = DetectionSummary.fromWindow([
        frame([('cat', 0.9), ('cat', 0.9)]),
        frame([('cat', 0.9), ('cat', 0.9), ('cat', 0.5)]), // a flicker
        frame([('cat', 0.9), ('cat', 0.9)]),
        frame([('cat', 0.9)]), // a miss
        frame([('cat', 0.9), ('cat', 0.9), ('dog', 0.8)]), // a one-off dog
      ], minScore: 0.4);
      expect(s.counts, {cls('cat'): 2});
    });

    test('an even window takes the lower median; empty is empty', () {
      final s = DetectionSummary.fromWindow([
        frame([]),
        frame([('cup', 0.9)]),
      ], minScore: 0.4);
      expect(s.isEmpty, isTrue);
      expect(DetectionSummary.fromWindow([], minScore: 0.4).isEmpty, isTrue);
    });

    test('most numerous first', () {
      final s = DetectionSummary.fromWindow([
        frame([('cup', 0.9), ('person', 0.9), ('person', 0.9)]),
      ], minScore: 0.4);
      expect(s.counts.keys, [cls('person'), cls('cup')]);
    });
  });

  group('FastAnswerComposer', () {
    const composer = FastAnswerComposer();
    final catsSummary = DetectionSummary.fromWindow([
      for (var i = 0; i < 5; i++) cats(),
    ], minScore: 0.4);

    test('"how many cats" on the cats frame', () {
      expect(
        composer.compose(
          FastRoute(FastIntent.count, 'count', cls: cls('cat')),
          catsSummary,
        ),
        'I count two cats.',
      );
    });

    test('counts: one, none (hedged), people', () {
      expect(
        composer.compose(
          FastRoute(FastIntent.count, 'count', cls: cls('remote')),
          catsSummary,
        ),
        'I count one remote.',
      );
      expect(
        composer.compose(
          FastRoute(FastIntent.count, 'count', cls: cls('dog')),
          catsSummary,
        ),
        "I don't see any dogs right now.",
      );
      final people = DetectionSummary.fromWindow([
        frame([('person', 0.9), ('person', 0.8)]),
      ], minScore: 0.4);
      expect(
        composer.compose(
          FastRoute(FastIntent.count, 'count', cls: cls('person')),
          people,
        ),
        'I count two people.',
      );
    });

    test('presence: yes, several, hedged no', () {
      expect(
        composer.compose(
          FastRoute(FastIntent.presence, 'presence', cls: cls('remote')),
          catsSummary,
        ),
        'Yes, I see a remote.',
      );
      expect(
        composer.compose(
          FastRoute(FastIntent.presence, 'presence', cls: cls('cat')),
          catsSummary,
        ),
        'Yes, I see two cats.',
      );
      expect(
        composer.compose(
          FastRoute(FastIntent.presence, 'presence', cls: cls('giraffe')),
          catsSummary,
        ),
        "I don't see a giraffe right now.",
      );
      expect(
        composer.compose(
          FastRoute(FastIntent.presence, 'presence', cls: cls('tvmonitor')),
          catsSummary,
        ),
        "I don't see a TV right now.",
      );
    });

    test('inventory lists the summary; nothing is hedged', () {
      expect(
        composer.compose(
          const FastRoute(FastIntent.inventory, 'inventory'),
          catsSummary,
        ),
        'I see two cats and a remote.',
      );
      final many = DetectionSummary.fromWindow([
        frame([
          ('person', 0.9),
          ('person', 0.9),
          ('cup', 0.9),
          ('laptop', 0.9),
          ('tvmonitor', 0.9),
        ]),
      ], minScore: 0.4);
      expect(
        composer.compose(
          const FastRoute(FastIntent.inventory, 'inventory'),
          many,
        ),
        'I see two people, a cup, a TV and a laptop.',
      );
      expect(
        composer.compose(
          const FastRoute(FastIntent.inventory, 'inventory'),
          DetectionSummary.fromWindow([], minScore: 0.4),
        ),
        "I don't see anything I recognize right now.",
      );
    });

    test('basis names classes as spoken', () {
      expect(composer.basis(catsSummary), 'cat ×2 · remote');
    });
  });
}
