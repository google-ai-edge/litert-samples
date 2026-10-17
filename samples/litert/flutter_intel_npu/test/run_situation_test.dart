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
import 'package:intel_npu_demo/features/act3_scene/runner/run_situation.dart';

void main() {
  group('RunSituation', () {
    test('says nothing when there is nothing to say', () {
      expect(const RunSituation().asText, '');
    });

    test('names the next two obstacles, nearest first', () {
      const s = RunSituation(ahead: [
        AheadItem('fallen log', 12.4),
        AheadItem('boulder', 27),
        AheadItem('stream', 38),
      ]);
      expect(s.lines().first, 'Ahead: a fallen log in 12 m, then a boulder in 27 m.');
    });

    test('hints at a fumble without announcing it', () {
      final hinted = const RunSituation(foreshadow: true).asText;
      expect(hinted, contains('too relaxed'));
      for (final spoiler in ['stumble', 'fall', 'crash', 'miss', 'fumble']) {
        expect(hinted.toLowerCase(), isNot(contains(spoiler)));
      }
      expect(const RunSituation().asText, isNot(contains('relaxed')));
    });

    test('reports form only when there is a story in it', () {
      expect(const RunSituation(streak: 2, stumblesLastMinute: 1).asText, '');
      final s = const RunSituation(streak: 7, bestStreak: 12, stumblesLastMinute: 2).asText;
      expect(s, contains('7 clean in a row (best 12)'));
      expect(s, contains('2 stumbles in the last minute'));
    });

    test('tells the narrator what it runs on and how fast, as percentages', () {
      const s = RunSituation(backend: 'NPU', speedShare: 0.96, frameBudget: 0.72, gpuLoad: 0.81, npuLoad: 0.23);
      final ls = s.lines();
      expect(ls, contains('You are running on the NPU at 96% of your own speed; the scene is using 72% of its frame budget.'));
      expect(ls, contains('Chip load: GPU 81%, NPU 23%.'));
    });

    test('keeps the hardware lines when it has to cut', () {
      const s = RunSituation(
        ahead: [AheadItem('boulder', 9)],
        riverIn: 55,
        rise: 3,
        nextBiome: 'forest',
        nextBiomeIn: 70,
        foreshadow: true,
        presenterDriving: true,
        streak: 5,
        bestStreak: 5,
        backend: 'GPU',
        speedShare: 0.4,
        frameBudget: 1.3,
        cpuLoad: 0.1,
        gpuLoad: 0.99,
      );
      final ls = s.lines();
      expect(ls.length, RunSituation.maxLines);
      expect(ls[ls.length - 2], startsWith('You are running on the GPU at 40%'));
      expect(ls.last, startsWith('Chip load:'));
    });

    test('mentions climbs and drops only when they are real', () {
      expect(const RunSituation(rise: 0.8).asText, '');
      expect(const RunSituation(rise: 2.2).asText, contains('climbs'));
      expect(const RunSituation(rise: -2.2).asText, contains('drops away'));
    });
  });
}
