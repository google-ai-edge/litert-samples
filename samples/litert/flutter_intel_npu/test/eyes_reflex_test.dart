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
import 'package:intel_npu_demo/features/act3_scene/runner/dash_pilot.dart';
import 'package:intel_npu_demo/features/act3_scene/runner/vision_pilot.dart';

void main() {
  // One look on the trail, the answers the reflex gets, and what it did.
  String look(EyesReflex r, String near, [String centre = 'trail', bool airborne = false]) =>
      r.look(near: near, centre: centre, airborne: airborne);

  group('jumping', () {
    test('two jump looks in a row jump, once', () {
      final r = EyesReflex();
      expect(look(r, 'jump'), '');
      expect(look(r, 'jump'), 'jump');
      expect(r.intent(lateral: 0, halfWidth: 2).jump, isTrue);
      expect(r.intent(lateral: 0, halfWidth: 2).jump, isFalse, reason: 'the game sees a jump once');
      expect(look(r, 'jump'), '', reason: 'no second jump until the near part says run');
    });

    test('two run looks rearm the jump', () {
      final r = EyesReflex();
      for (final n in ['jump', 'jump', 'run', 'run', 'jump', 'jump']) {
        look(r, n);
      }
      expect(r.intent(lateral: 0, halfWidth: 2).jump, isTrue);
    });

    test('no jump in the air, and the next jump look on the ground jumps', () {
      final r = EyesReflex();
      look(r, 'jump');
      expect(look(r, 'jump', 'trail', true), '');
      expect(look(r, 'jump'), 'jump');
    });

    test('a reset forgets a jump nobody took', () {
      final r = EyesReflex();
      look(r, 'jump');
      look(r, 'jump');
      r.reset(); // the presenter took over before the game asked
      expect(r.intent(lateral: 0, halfWidth: 2).jump, isFalse);
      look(r, 'jump');
      expect(look(r, 'jump'), 'jump', reason: 'the latch is reset too');
    });
  });

  group('rocks', () {
    test('a rock seen twice asks the band, and its side sends Dash to that edge', () {
      final r = EyesReflex();
      expect(look(r, 'run', 'rock'), '');
      final act = look(r, 'run', 'rock');
      expect(act, 'pick');
      expect(r.wantsSteer(act), isTrue);
      expect(r.steer('left', act: act, lateral: 0.3), 'left');
      expect(r.intent(lateral: 0.3, halfWidth: 2).lateral, closeTo(-1.6, 1e-9));
    });

    test('a rock seen once does not ask the band', () {
      final r = EyesReflex();
      for (final c in ['trail', 'rock', 'trail']) {
        final act = look(r, 'run', c);
        expect(r.wantsSteer(act), isFalse);
      }
    });

    test('straight holds the line it was on', () {
      final r = EyesReflex();
      look(r, 'run', 'rock');
      final act = look(r, 'run', 'rock');
      r.steer('straight', act: act, lateral: 0.5);
      r.steer('straight', act: '', lateral: 1.2);
      expect(r.intent(lateral: 1.2, halfWidth: 2).lateral, 0.5);
    });

    test('three looks without a rock let Dash go back; fewer do not', () {
      final r = EyesReflex();
      look(r, 'run', 'rock');
      r.steer('right', act: look(r, 'run', 'rock'), lateral: 0);
      look(r, 'run', 'trail');
      look(r, 'run', 'trail');
      expect(r.intent(lateral: 0, halfWidth: 2).lateral, closeTo(1.6, 1e-9));
      look(r, 'run', 'trail');
      expect(r.intent(lateral: 1.0, halfWidth: 2).lateral, closeTo(0.98, 1e-9));
      expect(r.wantsSteer(''), isFalse);
    });

    test('a reset drops the side being passed', () {
      final r = EyesReflex();
      look(r, 'run', 'rock');
      r.steer('left', act: look(r, 'run', 'rock'), lateral: 0);
      r.reset();
      expect(r.intent(lateral: 1.0, halfWidth: 2).lateral, closeTo(0.98, 1e-9));
    });
  });

  test("the presenter's hold times out, so the eyes drive again", () {
    final pilot = DashPilot()..takeOver(holdSeconds: 5);
    PilotIntent? step(double dt) =>
        pilot.update(dt: dt, z: 0, lateral: 0, halfWidth: 2, speed: 10, airborne: false, next: null);
    expect(step(4.9), isNull);
    expect(pilot.manual, isTrue);
    expect(step(0.2), isNotNull);
    expect(pilot.manual, isFalse);
  });
}
