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

import 'dart:math' as math;

import 'package:vector_math/vector_math.dart' as vm;

enum DayPhase { morning, day, sunset, night, dawn }

/// The lighting of one moment of the day. Colours are linear RGB.
///
/// One light serves as the sun by day and the moon by night, so the shadow
/// pass never switches off: a night that rendered fewer shadows would hand
/// frames back to the GPU mid-act, and Act 3 measures exactly that.
class SkyLook {
  SkyLook({
    required this.towardLight,
    required this.lightColor,
    required this.lightIntensity,
    required this.zenith,
    required this.horizon,
    required this.ground,
    required this.fog,
    required this.disc,
    required this.skyIntensity,
    required this.exposure,
  });

  /// Unit vector from the scene towards the sun or moon.
  final vm.Vector3 towardLight;
  final vm.Vector3 lightColor;
  final double lightIntensity;
  final vm.Vector3 zenith;
  final vm.Vector3 horizon;
  final vm.Vector3 ground;
  final vm.Vector3 fog;

  /// Colour of the sun or moon disc drawn in the sky; black hides it while
  /// one hands over to the other.
  final vm.Vector3 disc;
  final double skyIntensity;
  final double exposure;

  double get elevationDegrees => math.asin(towardLight.y.clamp(-1.0, 1.0)) * 180 / math.pi;

  /// How different the sky's lighting contribution is from [other]: the
  /// colours and the sun that a sky bake captures.
  double distanceTo(SkyLook other) =>
      (zenith - other.zenith).length +
      (horizon - other.horizon).length +
      (ground - other.ground).length +
      (disc - other.disc).length * 0.5 +
      (towardLight - other.towardLight).length * 0.5;
}

/// Decides when the sky has drifted far enough from its last bake to bake
/// its lighting again.
///
/// A bake is time-sliced, but each slice still lands on a frame: under load a
/// bake costs a couple of missed frames. On a fixed 1.5 s interval that came
/// to ~30 missed frames per 20 s on an M4 Pro, and the midday hold changes
/// nothing worth baking. The sky drifts about 12 units of [SkyLook.distanceTo]
/// over a day, so the default 0.3 means ~40 bakes per cycle, nearly all of
/// them in the transitions.
class SkyBakeGate {
  SkyBakeGate({this.threshold = 0.3});

  final double threshold;
  SkyLook? _baked;

  bool shouldBake(SkyLook look) {
    final baked = _baked;
    if (baked != null && look.distanceTo(baked) <= threshold) return false;
    _baked = look;
    return true;
  }
}

/// A day-night cycle for the runner, as a fraction of the cycle in 0..1.
///
/// Pure Dart: no engine imports, so the curve is unit-tested and the render
/// side only copies [SkyLook] values onto the scene.
class TimeOfDay {
  TimeOfDay({required this.cycleSeconds, DayPhase start = DayPhase.day})
      : _fraction = phasePeak(start),
        _phase = start;

  /// Length of a whole day in seconds; 0 or less stops the clock.
  final double cycleSeconds;

  /// How long a presenter skip takes: long enough to read as time passing,
  /// short enough not to stall the act.
  static const skipSeconds = 3.0;

  double _fraction;
  DayPhase _phase;
  double? _target;
  double _skipRate = 0;

  double get fraction => _fraction;
  DayPhase get phase => _phase;
  SkyLook get look => lookAt(_fraction);

  /// Moves the clock on; returns the phase it entered, or null.
  DayPhase? advance(double seconds) {
    if (seconds <= 0) return null;
    final target = _target;
    if (target != null) {
      final remaining = (target - _fraction) % 1.0;
      final step = _skipRate * seconds;
      if (step >= remaining) {
        _fraction = target;
        _target = null;
      } else {
        _fraction = (_fraction + step) % 1.0;
      }
    } else {
      if (cycleSeconds <= 0) return null;
      _fraction = (_fraction + seconds / cycleSeconds) % 1.0;
    }
    final now = phaseAt(_fraction);
    if (now == _phase) return null;
    _phase = now;
    return now;
  }

  /// Fast-forwards to the heart of the next phase over [skipSeconds] (a
  /// presenter key on stage). The phase change itself is reported by
  /// [advance] when the clock gets there. Returns the phase it heads for.
  DayPhase skipToNextPhase() {
    final from = _target != null ? phaseAt(_target!) : _phase;
    final next = DayPhase.values[(from.index + 1) % DayPhase.values.length];
    final target = _target = phasePeak(next);
    _skipRate = ((target - _fraction) % 1.0) / skipSeconds;
    return next;
  }

  static double phaseStart(DayPhase phase) => _phaseStarts[phase]!;

  /// The moment that looks most like [phase]: a phase begins mid-transition,
  /// so its start would show dusk and call it night.
  static double phasePeak(DayPhase phase) => _phasePeaks[phase]!;

  static DayPhase phaseAt(double t) {
    final f = t % 1.0;
    var current = DayPhase.morning;
    for (final phase in DayPhase.values) {
      if (f >= _phaseStarts[phase]!) current = phase;
    }
    return current;
  }

  /// A sentence for the narrator when [phase] begins.
  static String narration(DayPhase phase) => switch (phase) {
        DayPhase.morning => 'the morning sun is up',
        DayPhase.day => 'it is full daylight now',
        DayPhase.sunset => 'the sun is going down',
        DayPhase.night => 'night has fallen, only the moon lights the trail',
        DayPhase.dawn => 'dawn is breaking',
      };

  static const _phaseStarts = {
    DayPhase.morning: 0.0,
    DayPhase.day: 0.08,
    DayPhase.sunset: 0.50,
    DayPhase.night: 0.62,
    DayPhase.dawn: 0.90,
  };

  // Each one sits on the key that defines its phase's look (see [_keys]).
  static const _phasePeaks = {
    DayPhase.morning: 0.0,
    DayPhase.day: 0.10,
    DayPhase.sunset: 0.53,
    DayPhase.night: 0.66,
    DayPhase.dawn: 0.97,
  };

  static SkyLook lookAt(double t) {
    final f = t % 1.0;
    var i = 0;
    while (i < _keys.length - 1 && _keys[i + 1].at <= f) {
      i++;
    }
    final a = _keys[i];
    // The last key hands over to the first one, a whole cycle later.
    final b = i + 1 < _keys.length ? _keys[i + 1] : _keys.first;
    final bAt = i + 1 < _keys.length ? b.at : 1.0;
    final raw = ((f - a.at) / (bAt - a.at)).clamp(0.0, 1.0);
    final s = raw * raw * (3 - 2 * raw); // eases in and out of every key
    double mix(double x, double y) => x + (y - x) * s;
    vm.Vector3 mixV(vm.Vector3 x, vm.Vector3 y) => x + (y - x) * s;

    // Interpolate the angles, not the vectors: the elevation then never
    // leaves the range the keys span.
    final az = mix(a.azimuth, b.azimuth) * math.pi / 180;
    final el = mix(a.elevation, b.elevation) * math.pi / 180;
    return SkyLook(
      towardLight: vm.Vector3(math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)),
      lightColor: mixV(a.light, b.light),
      lightIntensity: mix(a.intensity, b.intensity),
      zenith: mixV(a.zenith, b.zenith),
      horizon: mixV(a.horizon, b.horizon),
      ground: mixV(a.ground, b.ground),
      fog: mixV(a.fog, b.fog),
      disc: mixV(a.disc, b.disc),
      skyIntensity: mix(a.skyIntensity, b.skyIntensity),
      exposure: mix(a.exposure, b.exposure),
    );
  }

  // Azimuth is measured from the trail's forward direction (+Z) towards its
  // right (+X). Elevations stay between 6° and 32°: a steep light hides
  // every shadow under its caster.
  //
  // Every key keeps the light on the trail's right, between 45° and 100°.
  // The shadow pass costs what the light-space frustum takes in, so the
  // light's side is a GPU-load dial: on an M4 Pro at render scale 3 a sunset
  // from straight ahead (-12°) ran ~10% cheaper than the day, from the left
  // (-45°, -80°) 5-7% cheaper, from the right (95°) the same as the day.
  // A phase that hands frames back mid-act looks like the effect Act 3
  // measures, so the side does not change.
  static final _keys = <_Key>[
    _Key(0.00, 'morning', azimuth: 88, elevation: 12, light: _v(1.0, 0.80, 0.58), intensity: 2.5,
        zenith: _v(0.18, 0.38, 0.74), horizon: _v(0.98, 0.78, 0.60), ground: _v(0.20, 0.22, 0.20),
        fog: _v(0.62, 0.56, 0.56), disc: _v(1.0, 0.85, 0.60), skyIntensity: 1.6, exposure: 0.64),
    // The look the scene was tuned with: a low raking sun ahead and right.
    _Key(0.10, 'day', azimuth: 66, elevation: 31.5, light: _v(1.0, 1.0, 1.0), intensity: 3.2,
        zenith: _v(0.10, 0.34, 0.78), horizon: _v(0.62, 0.80, 0.95), ground: _v(0.20, 0.24, 0.22),
        fog: _v(0.36, 0.46, 0.62), disc: _v(1.0, 0.94, 0.78), skyIntensity: 1.6, exposure: 0.62),
    _Key(0.46, 'late day', azimuth: 80, elevation: 29, light: _v(1.0, 0.97, 0.90), intensity: 3.1,
        zenith: _v(0.10, 0.32, 0.74), horizon: _v(0.70, 0.80, 0.90), ground: _v(0.20, 0.24, 0.22),
        fog: _v(0.40, 0.46, 0.60), disc: _v(1.0, 0.92, 0.74), skyIntensity: 1.6, exposure: 0.62),
    // Sets on the right (see above for why not ahead). Warm haze, not pink:
    // a pinkish fog over green hills reads as mauve.
    _Key(0.53, 'sunset', azimuth: 95, elevation: 10, light: _v(1.0, 0.55, 0.25), intensity: 2.6,
        zenith: _v(0.20, 0.26, 0.52), horizon: _v(1.0, 0.60, 0.26), ground: _v(0.18, 0.15, 0.12),
        fog: _v(0.72, 0.50, 0.32), disc: _v(1.4, 0.70, 0.28), skyIntensity: 1.5, exposure: 0.66),
    _Key(0.59, 'dusk', azimuth: 70, elevation: 16, light: _v(0.55, 0.46, 0.72), intensity: 0.95,
        zenith: _v(0.06, 0.08, 0.22), horizon: _v(0.40, 0.26, 0.32), ground: _v(0.08, 0.08, 0.09),
        fog: _v(0.24, 0.20, 0.26), disc: _v(0, 0, 0), skyIntensity: 1.4, exposure: 0.85),
    // Moonlight: blue and dim, but bright enough to read from the back of a
    // hall, and it still casts shadows.
    _Key(0.66, 'night', azimuth: 45, elevation: 30, light: _v(0.42, 0.55, 0.95), intensity: 1.1,
        zenith: _v(0.015, 0.03, 0.09), horizon: _v(0.07, 0.12, 0.26), ground: _v(0.05, 0.06, 0.08),
        fog: _v(0.06, 0.08, 0.15), disc: _v(0.85, 0.90, 1.0), skyIntensity: 1.3, exposure: 1.05),
    _Key(0.86, 'late night', azimuth: 72, elevation: 27, light: _v(0.42, 0.55, 0.95), intensity: 1.1,
        zenith: _v(0.015, 0.03, 0.09), horizon: _v(0.07, 0.12, 0.26), ground: _v(0.05, 0.06, 0.08),
        fog: _v(0.06, 0.08, 0.15), disc: _v(0.85, 0.90, 1.0), skyIntensity: 1.3, exposure: 1.05),
    _Key(0.92, 'before dawn', azimuth: 100, elevation: 12, light: _v(0.60, 0.50, 0.72), intensity: 0.95,
        zenith: _v(0.08, 0.10, 0.26), horizon: _v(0.55, 0.38, 0.45), ground: _v(0.08, 0.08, 0.09),
        fog: _v(0.30, 0.26, 0.32), disc: _v(0, 0, 0), skyIntensity: 1.4, exposure: 0.85),
    _Key(0.97, 'dawn', azimuth: 92, elevation: 6, light: _v(1.0, 0.62, 0.55), intensity: 1.8,
        zenith: _v(0.16, 0.26, 0.55), horizon: _v(0.98, 0.64, 0.60), ground: _v(0.18, 0.18, 0.18),
        fog: _v(0.60, 0.50, 0.54), disc: _v(1.0, 0.60, 0.50), skyIntensity: 1.5, exposure: 0.70),
  ];

  static vm.Vector3 _v(double r, double g, double b) => vm.Vector3(r, g, b);
}

class _Key {
  _Key(
    this.at,
    this.name, {
    required this.azimuth,
    required this.elevation,
    required this.light,
    required this.intensity,
    required this.zenith,
    required this.horizon,
    required this.ground,
    required this.fog,
    required this.disc,
    required this.skyIntensity,
    required this.exposure,
  });

  final double at;
  final String name;
  final double azimuth;
  final double elevation;
  final vm.Vector3 light;
  final double intensity;
  final vm.Vector3 zenith;
  final vm.Vector3 horizon;
  final vm.Vector3 ground;
  final vm.Vector3 fog;
  final vm.Vector3 disc;
  final double skyIntensity;
  final double exposure;
}
