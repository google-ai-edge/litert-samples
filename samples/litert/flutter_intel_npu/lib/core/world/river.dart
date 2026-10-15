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

/// A river crossing the landscape, laid out in trail space: its centre line
/// is a trail `z` for every lateral offset, so it runs across the whole
/// terrain ribbon and crosses the trail exactly once, at [zCross].
///
/// Geometry only. [TerrainField] carves the channel and decides the water
/// level; the chunk mesher lays a water ribbon along [centerZ].
class River {
  const River({
    required this.id,
    required this.zCross,
    required this.amplitude,
    required this.wavelength,
    required this.phase,
    required this.skew,
  });

  /// Half the width of the water ribbon, metres. Wider than the channel at
  /// water level on purpose: the banks rise over its edges and draw the
  /// shoreline, so the water has no edge of its own to show.
  static const ribbonHalfWidth = 4.5;

  final int id;

  /// Trail z where the river crosses the centre line.
  final double zCross;

  /// Meander: amplitude and wavelength in metres along the river's course.
  final double amplitude;
  final double wavelength;
  final double phase;

  /// Overall lean of the course: dz per metre of lateral.
  final double skew;

  double get _k => 2 * math.pi / wavelength;

  /// Trail z of the centre line at [lateral] metres right of the trail.
  double centerZ(double lateral) =>
      zCross + skew * lateral + amplitude * (math.sin(_k * lateral + phase) - math.sin(phase));

  /// d(centerZ)/d(lateral).
  double slopeAt(double lateral) => skew + amplitude * _k * math.cos(_k * lateral + phase);

  /// Unit vector across the flow in trail space, as (dLateral, dZ), pointing
  /// to the far bank (+z side).
  (double, double) acrossAt(double lateral) {
    final s = slopeAt(lateral);
    final len = math.sqrt(1 + s * s);
    return (-s / len, 1 / len);
  }

  /// Signed distance from the centre line, measured across the flow, of a
  /// point given in trail space.
  double distanceFrom(double zTrail, double lateral) {
    final s = slopeAt(lateral);
    return (zTrail - centerZ(lateral)) / math.sqrt(1 + s * s);
  }

  /// How far the course strays from [zCross] in z over [halfWidth] metres of
  /// lateral either side: the window in which it can touch a point.
  double reach(double halfWidth) => skew.abs() * halfWidth + 2 * amplitude.abs();
}
