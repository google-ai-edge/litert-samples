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

/// The path Dash runs along, parameterised by `z` — metres along +Z, the run
/// direction. The trail only wanders in x, which keeps every query O(1) and
/// closed-form: no arc-length integration, no spline table, and a chunk is
/// just a z-range.
///
/// Distance actually travelled is longer than z (the path is not straight);
/// the game accumulates real displacement for the HUD instead of integrating
/// here.
class Trail {
  const Trail({
    this.halfWidth = 3.2,
    this.amplitude1 = 15.0,
    this.wavelength1 = 210.0,
    this.amplitude2 = 5.5,
    this.wavelength2 = 71.0,
    this.phase = 1.7,
  });

  /// Runnable corridor, metres from the centre line to either edge.
  final double halfWidth;

  final double amplitude1;
  final double wavelength1;
  final double amplitude2;
  final double wavelength2;
  final double phase;

  /// x of the centre line at [z].
  double centerX(double z) =>
      amplitude1 * math.sin(2 * math.pi * z / wavelength1) +
      amplitude2 * math.sin(2 * math.pi * z / wavelength2 + phase);

  /// dx/dz — the trail's slope in the XZ plane. |slope| stays below ~0.95 for
  /// the defaults, so the path never doubles back and z remains monotonic.
  double slopeAt(double z) =>
      amplitude1 * (2 * math.pi / wavelength1) * math.cos(2 * math.pi * z / wavelength1) +
      amplitude2 * (2 * math.pi / wavelength2) * math.cos(2 * math.pi * z / wavelength2 + phase);

  /// Unit direction of travel in XZ.
  vm.Vector2 tangentAt(double z) {
    final s = slopeAt(z);
    final len = math.sqrt(1 + s * s);
    return vm.Vector2(s / len, 1 / len);
  }

  /// Unit vector pointing to Dash's right (up × forward in a Y-up,
  /// right-handed world: forward +Z gives right +X).
  vm.Vector2 rightAt(double z) {
    final t = tangentAt(z);
    return vm.Vector2(t.y, -t.x);
  }

  /// Yaw (radians about +Y) that faces a model along the trail.
  double headingAt(double z) => math.atan2(slopeAt(z), 1);

  /// Signed perpendicular distance from the centre line: negative left of the
  /// trail, positive right. Exact for a straight trail and within a few
  /// centimetres of true perpendicular distance at the curvature used here.
  double lateralOffset(double x, double z) {
    final s = slopeAt(z);
    return (x - centerX(z)) / math.sqrt(1 + s * s);
  }

  /// World XZ of a point on the trail at [z], offset [lateral] metres to the
  /// right of the centre line.
  vm.Vector2 pointAt(double z, {double lateral = 0}) {
    final c = vm.Vector2(centerX(z), z);
    if (lateral == 0) return c;
    return c + rightAt(z) * lateral;
  }

  /// The inverse of [pointAt]: the trail `z` and lateral offset that land on
  /// world ([x], [z]).
  ///
  /// [lateralOffset] reads the trail at the point's own z, which is off by
  /// tens of metres at the edge of the terrain ribbon, where "right" is far
  /// from +X. Features laid out in trail space (rivers) need the exact pair.
  ///
  /// Not unique everywhere: the tight bend of the short wave has a radius of
  /// ~20 m, so on its inside the terrain ribbon (46 m wide) folds over itself
  /// and two trail positions reach the same point. Newton's method from
  /// [guess] picks the one nearest it; callers that know where a point came
  /// from (the mesher, Dash, props) pass its trail z and land on that branch.
  (double, double) toTrail(double x, double z, {double? guess}) {
    // The point lies on the normal through the trail at zt exactly when its
    // offset along the tangent there is zero.
    double along(double zt) {
      final t = tangentAt(zt);
      return (x - centerX(zt)) * t.x + (z - zt) * t.y;
    }

    var zt = guess ?? z;
    for (var i = 0; i < 12; i++) {
      final f = along(zt);
      // 1 mm. The maths is Float32 (vector_math), whose noise at z ~ 3 km is
      // ~3e-4 m: a tolerance below it never stops, and near a fold, where the
      // slope is small, dividing that noise by it threw the point onto the
      // other branch of the trail.
      if (f.abs() < 1e-3) break;
      const h = 0.01;
      final slope = (along(zt + h) - along(zt - h)) / (2 * h);
      if (slope.abs() < 1e-9) break;
      zt -= (f / slope).clamp(-20.0, 20.0);
    }
    final r = rightAt(zt);
    return (zt, (x - centerX(zt)) * r.x + (z - zt) * r.y);
  }
}
