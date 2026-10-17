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

import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/live_camera_config.dart';
import 'package:litert_edge_demos/domain/use_cases/memory_acceptance.dart';

const _s = 1000000;

/// Every frame a different scene from every other.
double different(int i, int best) => i == best ? 1 : 0.5;

MemoryVerdict judge(
  List<double> cosines,
  List<int> seconds, {
  SceneCosineOf scene = different,
  ViewChangedOf? viewChanged,
}) => judgeMemory(
  cosines: cosines,
  atMicros: [for (final s in seconds) s * _s],
  sceneCosine: scene,
  viewChanged: viewChanged,
  floor: 0.62,
  margin: 0.025,
  separation: const Duration(seconds: 5),
  sameScene: 0.88,
  tie: 0.01,
)!;

void main() {
  test('the constants are the calibrated ones', () {
    expect(kMemoryMinCosine, 0.62);
    expect(kMemoryMinMargin, 0.025);
    expect(kMemoryMarginSeparation, const Duration(seconds: 5));
    expect(kMemorySameSceneCosine, 0.88);
    expect(kMemoryTieCosine, 0.01);
  });

  test('a clear winner above the floor is accepted, with its rival', () {
    final v = judge([0.55, 0.76, 0.67, 0.58], [0, 10, 20, 30]);
    expect(v.accepted, isTrue);
    expect(v.outcome, MemoryOutcome.accepted);
    expect(v.best, 1);
    expect(v.cosine, 0.76);
    expect(v.rival, 0.67);
    expect(v.reason, 'accepted');
  });

  test('the best under the floor is rejected', () {
    final v = judge([0.55, 0.612, 0.50], [0, 10, 20]);
    expect(v.accepted, isFalse);
    expect(v.outcome, MemoryOutcome.rejected);
    expect(v.reason, 'below the floor 0.62');
  });

  test('a best that does not beat a different scene more than 5 s away by '
      'the margin is ambiguous: the newest competing frame is reported', () {
    final v = judge([0.70, 0.68], [0, 10]);
    expect(v.accepted, isFalse);
    expect(v.outcome, MemoryOutcome.ambiguous);
    expect(v.rival, 0.68);
    expect(v.cosine, 0.70);
    expect(v.best, 1, reason: 'the rival is the newer of the two');
    expect(v.reason, startsWith('not distinct: 0.020 over a rival at 0.680'));
  });

  test('ambiguous: the competing frames are those at or above the floor and '
      'within the margin of the best, whatever their scene', () {
    // 0 best (0.70, 0 s); 1 a rival (0.69, 20 s); 2 within the margin but
    // under the floor (0.61, 40 s); 3 above the floor but out of the margin
    // (0.66, 60 s).
    final v = judge([0.70, 0.69, 0.61, 0.66], [0, 20, 40, 60]);
    expect(v.outcome, MemoryOutcome.ambiguous);
    expect(v.best, 1);

    // The best's own scene can be the newest competitor.
    final own = judge(
      [0.70, 0.69, 0.695],
      [0, 20, 40],
      scene: (i, best) => i == best || i == 2 ? 0.99 : 0.5,
    );
    expect(own.outcome, MemoryOutcome.ambiguous);
    expect(own.best, 2);
  });

  test('frames within 5 s of the best are its neighbours, not rivals', () {
    final v = judge([0.70, 0.69, 0.60], [10, 14, 30]);
    expect(v.accepted, isTrue);
    expect(v.rival, 0.60);
  });

  test('a frame of the same scene (image cosine ≥ 0.88) far away in time is '
      'one long sighting, not a rival', () {
    final v = judge(
      [0.75, 0.74, 0.60],
      [0, 40, 80],
      scene: (i, best) => i == best || i == 1 ? 0.99 : 0.6,
    );
    expect(v.accepted, isTrue);
    expect(v.rival, 0.60);
  });

  test('accepted: the newest near-identical frame within the tie of the best '
      'is reported (its latest sighting)', () {
    double sameAsBest(int i, int best) => i == best || i == 2 ? 0.99 : 0.5;
    // 0 the best (0.750, 0 s); 2 the same scene 40 s later at 0.742.
    final v = judge([0.75, 0.60, 0.742], [0, 20, 40], scene: sameAsBest);
    expect(v.accepted, isTrue);
    expect(v.best, 2);
    expect(v.cosine, 0.75, reason: 'the scene is judged on its best');

    // Out of the tie (0.735): the best itself.
    expect(judge([0.75, 0.60, 0.735], [0, 20, 40], scene: sameAsBest).best, 0);
    // Within the tie but a different scene: never reported for the best.
    expect(
      judge([0.75, 0.60, 0.745], [0, 20, 40]).outcome,
      MemoryOutcome.ambiguous,
    );
  });

  test('the tie leaves out a frame whose thumbnail shows a change from the '
      'best (sceneChange): keys put down in E2 and taken away in E3 (E1\'s '
      'scene again) report E2, not E3', () {
    double sameScene(int i, int best) => 0.99;
    // E1 0.695 at 0 s, E2 0.700 at 20 s (the keys), E3 0.695 at 40 s.
    const cosines = [0.695, 0.70, 0.695];
    const seconds = [0, 20, 40];
    expect(
      judge(cosines, seconds, scene: sameScene).best,
      2,
      reason: 'by the image vectors alone E3 ties with E2',
    );
    final v = judge(
      cosines,
      seconds,
      scene: sameScene,
      viewChanged: (i, best) => i != best,
    );
    expect(v.accepted, isTrue);
    expect(v.best, 1);
    // A frame of the same view still counts for the tie.
    expect(
      judge(
        cosines,
        seconds,
        scene: sameScene,
        viewChanged: (i, best) => i == 0,
      ).best,
      2,
    );
  });

  test('ambiguous needs the best at the floor plus the margin: a faint best '
      'that does not stand out from its rival is rejected', () {
    // The measured ambiguous cases (visual_memory_test.dart, 2026-10-14):
    // "Where was the sign?" (street sign 0.712, GATE 42 0.694) and "Did you
    // see some text?" (Wi-Fi 0.648, street sign 0.637).
    for (final (best, rival) in [(0.712, 0.694), (0.648, 0.637)]) {
      final v = judge([best, rival], [0, 10]);
      expect(v.outcome, MemoryOutcome.ambiguous, reason: '$best/$rival');
      expect(v.best, 1, reason: 'the newer competing frame');
    }
    // Exactly at floor + margin, a rival within the margin: ambiguous.
    expect(judge([0.645, 0.63], [0, 10]).outcome, MemoryOutcome.ambiguous);

    final one = judge([0.625], [0]);
    expect(one.outcome, MemoryOutcome.rejected);
    expect(one.reason, 'no rival and below 0.645 (floor + margin)');

    final two = judge([0.625, 0.61], [0, 10]);
    expect(two.outcome, MemoryOutcome.rejected);
    expect(two.rival, 0.61);
    expect(two.best, 0, reason: 'a rejection reports the best');
    expect(
      two.reason,
      'not distinct and below 0.645 (floor + margin): 0.015 over a rival at '
      '0.610 (margin 0.025)',
    );

    final three = judge([0.63, 0.615, 0.60], [0, 10, 20]);
    expect(three.outcome, MemoryOutcome.rejected);
    expect(three.reason, startsWith('not distinct and below 0.645'));
    expect(three.best, 0);

    // A faint best that does beat its rival by the margin is accepted.
    expect(judge([0.63, 0.55], [0, 10]).accepted, isTrue);
  });

  test('with no rival the floor plus the margin decides', () {
    final faint = judge([0.64], [0]);
    expect(faint.outcome, MemoryOutcome.rejected);
    expect(faint.reason, 'no rival and below 0.645 (floor + margin)');
    expect(judge([0.645], [0]).accepted, isTrue);
    expect(judge([0.70, 0.69], [0, 3]).rival, isNull);
    expect(judge([0.70, 0.69], [0, 3]).accepted, isTrue);
  });

  test('a NaN score is never the best, a rival or the reported frame', () {
    // First: it used to stay the best (every comparison with NaN is false)
    // and pass every test, an accepted NaN.
    final first = judge([double.nan, 0.76, 0.67], [0, 10, 20]);
    expect(first.accepted, isTrue);
    expect(first.best, 1);
    expect(first.cosine, 0.76);
    expect(first.rival, 0.67);

    // Later (an infinite score too): not a rival, and never the newest
    // near-identical frame, though its scene would say it is one.
    final asked = <int>[];
    final later = judge(
      [0.76, double.nan, double.infinity],
      [0, 10, 20],
      scene: (i, best) {
        asked.add(i);
        return 0.99;
      },
    );
    expect(later.accepted, isTrue);
    expect(later.best, 0, reason: 'the newer frames have no finite score');
    expect(later.cosine, 0.76);
    expect(later.rival, isNull);
    expect(asked, isEmpty, reason: 'no image cosine asked for them');
  });

  test('no finite score at all: rejected, never accepted', () {
    final v = judge([double.nan, double.nan], [0, 10]);
    expect(v.outcome, MemoryOutcome.rejected);
    expect(v.reason, 'no score is a number');
    expect(v.cosine, isNaN);
    expect(v.rival, isNull);
  });

  test('newest is the latest sighting (the store\'s order), not the largest '
      'time: a wall clock set back an hour between two sightings', () {
    // Keys on the desk at 3600 s; the clock goes back an hour; keys on the
    // shelf at 10 s. Two places alike: the shelf, the later sighting.
    final ambiguous = judge([0.76, 0.75], [3600, 10]);
    expect(ambiguous.outcome, MemoryOutcome.ambiguous);
    expect(ambiguous.rival, 0.75, reason: 'still 3590 s apart on the clock');
    expect(ambiguous.best, 1, reason: 'the shelf');

    // One scene seen again after the clock went back: its later frame.
    final accepted = judge(
      [0.76, 0.40, 0.755],
      [3600, 3610, 10],
      scene: (i, best) => i == 2 ? 0.99 : 0.5,
    );
    expect(accepted.accepted, isTrue);
    expect(accepted.best, 2);
  });

  test('of equal best scores the earliest frame is the best', () {
    // Rejected (under the floor), so the best is the frame reported.
    final v = judge([0.50, 0.60, 0.60], [0, 10, 20]);
    expect(v.outcome, MemoryOutcome.rejected);
    expect(v.best, 1);
  });

  test('an empty memory has no verdict; mismatched lists are refused', () {
    expect(
      judgeMemory(
        cosines: const [],
        atMicros: const [],
        sceneCosine: different,
      ),
      isNull,
    );
    expect(
      () => judgeMemory(
        cosines: const [0.7],
        atMicros: const [],
        sceneCosine: different,
      ),
      throwsArgumentError,
    );
  });
}
