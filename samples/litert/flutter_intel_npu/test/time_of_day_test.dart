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
import 'package:intel_npu_demo/core/world/time_of_day.dart';

void main() {
  group('TimeOfDay', () {
    test('the light stays low and above the horizon all day and night', () {
      // A steep sun hides every shadow under its caster; a light below the
      // horizon leaves the scene unlit. Both read as broken on stage.
      for (var t = 0.0; t < 1.0; t += 0.001) {
        final el = TimeOfDay.lookAt(t).elevationDegrees;
        expect(el, greaterThanOrEqualTo(5.0), reason: 'light too low at t=$t');
        expect(el, lessThanOrEqualTo(33.0), reason: 'light too steep at t=$t');
      }
    });

    test('the light always comes from the trail\'s right', () {
      // Its side is a GPU-load dial (see the key table): a sunset from the
      // left ran 5-10% cheaper than the day and would hand frames back.
      for (var t = 0.0; t < 1.0; t += 0.001) {
        final toward = TimeOfDay.lookAt(t).towardLight;
        expect(toward.x, greaterThan(0.5), reason: 'light not from the right at t=$t');
      }
    });

    test('there is always enough light to read the scene', () {
      for (var t = 0.0; t < 1.0; t += 0.001) {
        expect(TimeOfDay.lookAt(t).lightIntensity, greaterThan(0.7), reason: 't=$t');
      }
    });

    test('changes smoothly, including across the wrap from dawn to morning', () {
      const step = 0.0005;
      for (var t = 0.0; t < 1.0; t += step) {
        final a = TimeOfDay.lookAt(t);
        final b = TimeOfDay.lookAt((t + step) % 1.0);
        expect((a.towardLight - b.towardLight).length, lessThan(0.02), reason: 'light jumps at t=$t');
        expect((a.lightIntensity - b.lightIntensity).abs(), lessThan(0.05), reason: 'intensity jumps at t=$t');
        expect((a.zenith - b.zenith).length, lessThan(0.02), reason: 'sky jumps at t=$t');
        expect((a.exposure - b.exposure).abs(), lessThan(0.01), reason: 'exposure jumps at t=$t');
      }
    });

    test('phases run in order and come back round', () {
      final seen = <DayPhase>[];
      for (var t = 0.0; t < 1.0; t += 0.001) {
        final p = TimeOfDay.phaseAt(t);
        if (seen.isEmpty || seen.last != p) seen.add(p);
      }
      expect(seen, [DayPhase.morning, DayPhase.day, DayPhase.sunset, DayPhase.night, DayPhase.dawn]);
    });

    test('every peak lies inside its own phase', () {
      for (final phase in DayPhase.values) {
        expect(TimeOfDay.phaseAt(TimeOfDay.phasePeak(phase)), phase);
      }
    });

    test('night is darker than day', () {
      final day = TimeOfDay.lookAt(TimeOfDay.phasePeak(DayPhase.day));
      final night = TimeOfDay.lookAt(TimeOfDay.phasePeak(DayPhase.night));
      expect(night.lightIntensity, lessThan(day.lightIntensity));
      expect(night.zenith.length, lessThan(day.zenith.length * 0.3));
    });

    test('advance moves through the cycle and reports a phase change once', () {
      final clock = TimeOfDay(cycleSeconds: 100, start: DayPhase.day);
      expect(clock.phase, DayPhase.day);
      final changes = <DayPhase>[];
      // One cycle plus a step: a thousand float additions of 0.1 may land a
      // hair short of the start of the day and report it as morning.
      for (var i = 0; i < 1001; i++) {
        final changed = clock.advance(0.1);
        if (changed != null) changes.add(changed);
      }
      expect(changes, [DayPhase.sunset, DayPhase.night, DayPhase.dawn, DayPhase.morning, DayPhase.day]);
      expect(clock.phase, DayPhase.day);
    });

    test('a zero-length cycle stands still', () {
      final clock = TimeOfDay(cycleSeconds: 0, start: DayPhase.sunset);
      final before = clock.fraction;
      expect(clock.advance(5), isNull);
      expect(clock.fraction, before);
    });

    test('a run starts in the heart of its phase, not on its edge', () {
      // A phase begins mid-transition; starting there would show dusk and
      // call it night.
      for (final phase in DayPhase.values) {
        final clock = TimeOfDay(cycleSeconds: 240, start: phase);
        expect(clock.phase, phase);
        expect(clock.fraction, TimeOfDay.phasePeak(phase));
      }
    });

    test('skipping fast-forwards to the heart of the next phase, then runs on', () {
      final clock = TimeOfDay(cycleSeconds: 240, start: DayPhase.day);
      expect(clock.skipToNextPhase(), DayPhase.sunset);
      final changes = <DayPhase>[];
      var seconds = 0.0;
      while (clock.fraction != TimeOfDay.phasePeak(DayPhase.sunset)) {
        final changed = clock.advance(1 / 60);
        if (changed != null) changes.add(changed);
        seconds += 1 / 60;
        expect(seconds, lessThan(6), reason: 'a skip should take a few seconds, not a minute');
      }
      expect(seconds, greaterThan(1), reason: 'a skip should be seen, not cut');
      expect(changes, [DayPhase.sunset]);
      // Back to the normal pace afterwards.
      final before = clock.fraction;
      clock.advance(1);
      expect(clock.fraction - before, closeTo(1 / 240, 1e-9));
    });

    test('a frozen clock still fast-forwards on a skip, then stands still', () {
      final clock = TimeOfDay(cycleSeconds: 0, start: DayPhase.night);
      expect(clock.skipToNextPhase(), DayPhase.dawn);
      for (var i = 0; i < 600; i++) {
        clock.advance(1 / 60);
      }
      expect(clock.fraction, TimeOfDay.phasePeak(DayPhase.dawn));
      expect(clock.advance(5), isNull);
      expect(clock.fraction, TimeOfDay.phasePeak(DayPhase.dawn));
    });

    test('skipping from dawn wraps round to the morning', () {
      final clock = TimeOfDay(cycleSeconds: 240, start: DayPhase.dawn);
      expect(clock.skipToNextPhase(), DayPhase.morning);
      var seconds = 0.0;
      while (clock.fraction != TimeOfDay.phasePeak(DayPhase.morning)) {
        clock.advance(1 / 60);
        seconds += 1 / 60;
        expect(seconds, lessThan(6), reason: 'the skip overshot the wrap');
      }
      expect(clock.phase, DayPhase.morning);
    });

    test('the sky lighting re-bakes during transitions, not all day', () {
      // Every bake costs a couple of missed frames under load; on a fixed
      // interval that was ~30 missed frames per 20 s. Bake on change instead.
      final clock = TimeOfDay(cycleSeconds: 240, start: DayPhase.day);
      final gate = SkyBakeGate();
      var bakes = 0;
      var bakesInDay = 0;
      for (var i = 0; i < 240 * 60; i++) {
        clock.advance(1 / 60);
        if (gate.shouldBake(clock.look)) {
          bakes++;
          final f = clock.fraction;
          if (f > 0.15 && f < 0.42) bakesInDay++;
        }
      }
      expect(bakes, inInclusiveRange(10, 60), reason: 'bakes per 4-minute day');
      expect(bakesInDay, lessThanOrEqualTo(3), reason: 'the midday hold barely changes');
    });

    test('the bake gate fires at once for a new look and again after a skip', () {
      final clock = TimeOfDay(cycleSeconds: 240, start: DayPhase.day);
      final gate = SkyBakeGate();
      expect(gate.shouldBake(clock.look), isTrue);
      expect(gate.shouldBake(clock.look), isFalse);
      clock.skipToNextPhase();
      var fired = 0;
      for (var i = 0; i < 4 * 60; i++) {
        clock.advance(1 / 60);
        if (gate.shouldBake(clock.look)) fired++;
      }
      expect(fired, greaterThan(0));
    });

    test('light directions are unit vectors', () {
      for (var t = 0.0; t < 1.0; t += 0.01) {
        expect(TimeOfDay.lookAt(t).towardLight.length, closeTo(1, 1e-5));
      }
    });
  });
}
