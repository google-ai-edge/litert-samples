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

import 'package:flutter_scene/noise.dart';

import 'river.dart';
import 'trail.dart';

/// Stretches of landscape Dash runs through. Each one is a palette plus a set
/// of terrain and vegetation parameters; the switch between them is also a
/// narration beat ("Dash just broke out of the pines").
enum Biome { meadow, forest, canyon, snow }

/// Per-biome knobs. Colours are linear RGB (the engine's vertex colours are
/// linear, not sRGB).
class BiomeSpec {
  const BiomeSpec({
    required this.biome,
    required this.label,
    required this.hillAmplitude,
    required this.mountainAmplitude,
    required this.groundColor,
    required this.slopeColor,
    required this.peakColor,
    required this.treeDensity,
    required this.rockDensity,
    required this.grassDensity,
  });

  final Biome biome;
  final String label;

  /// Rolling-hill height away from the trail, metres.
  final double hillAmplitude;

  /// Extra ridged height for the far background, metres.
  final double mountainAmplitude;

  /// Flat ground, steep slopes, and high ground respectively.
  final (double, double, double) groundColor;
  final (double, double, double) slopeColor;
  final (double, double, double) peakColor;

  /// Instances per 100 m² of eligible ground.
  ///
  /// The four biomes are deliberately balanced to a similar TOTAL: the scene
  /// is calibrated once to sit on the frame budget, and a biome that carries
  /// a third of the instances would hand the frame rate back half a minute
  /// later — which looks exactly like the effect Act 3 is trying to show.
  /// Character comes from the mix (canyon is rock, forest is trees), not from
  /// the count.
  final double treeDensity;
  final double rockDensity;
  final double grassDensity;

  /// Roughly what this biome costs to draw, for the balance test.
  double get totalDensity => treeDensity + rockDensity + grassDensity;

  static const meadow = BiomeSpec(
    biome: Biome.meadow,
    label: 'meadow',
    hillAmplitude: 4.5,
    mountainAmplitude: 14,
    groundColor: (0.16, 0.42, 0.12),
    slopeColor: (0.32, 0.28, 0.16),
    peakColor: (0.42, 0.48, 0.24),
    treeDensity: 0.9,
    rockDensity: 0.9,
    grassDensity: 8.4,
  );

  static const forest = BiomeSpec(
    biome: Biome.forest,
    label: 'forest',
    hillAmplitude: 6.5,
    mountainAmplitude: 18,
    groundColor: (0.08, 0.26, 0.09),
    slopeColor: (0.20, 0.17, 0.11),
    peakColor: (0.16, 0.30, 0.14),
    treeDensity: 3.6,
    rockDensity: 0.9,
    grassDensity: 5.7,
  );

  static const canyon = BiomeSpec(
    biome: Biome.canyon,
    label: 'canyon',
    hillAmplitude: 9.0,
    mountainAmplitude: 26,
    groundColor: (0.52, 0.28, 0.14),
    slopeColor: (0.40, 0.20, 0.11),
    peakColor: (0.62, 0.40, 0.22),
    treeDensity: 0.3,
    rockDensity: 5.4,
    grassDensity: 4.5,
  );

  static const snow = BiomeSpec(
    biome: Biome.snow,
    label: 'snowfield',
    hillAmplitude: 7.0,
    mountainAmplitude: 30,
    groundColor: (0.78, 0.82, 0.88),
    slopeColor: (0.38, 0.38, 0.42),
    peakColor: (0.92, 0.94, 0.98),
    treeDensity: 1.1,
    rockDensity: 3.4,
    grassDensity: 5.7,
  );

  static const order = [meadow, forest, canyon, snow];
}

/// The landscape itself: one deterministic height function, shared by the
/// mesher, by prop placement and by Dash's feet. Everything that needs to
/// know where the ground is asks this, so nothing can disagree with the
/// geometry the player sees.
///
/// Pure Dart on purpose (only the engine's noise library, which has no engine
/// imports): it runs on a background isolate and in unit tests.
class TerrainField {
  TerrainField({this.trail = const Trail(), this.seed = 1337, this.biomeLength = 520.0})
      : _hills = FastNoiseLite(seed: seed)
          ..noiseType = NoiseType.openSimplex2
          ..fractalType = FractalType.fbm
          ..octaves = 4
          ..frequency = 0.0065,
        _mountains = FastNoiseLite(seed: seed + 101)
          ..noiseType = NoiseType.openSimplex2
          ..fractalType = FractalType.ridged
          ..octaves = 3
          ..frequency = 0.0028,
        _detail = FastNoiseLite(seed: seed + 202)
          ..noiseType = NoiseType.openSimplex2
          ..fractalType = FractalType.fbm
          ..octaves = 2
          ..frequency = 0.05;

  final Trail trail;
  final int seed;

  /// Metres of trail per biome, with the last [blendLength] blended into the
  /// next one so the seam is a slope, not a step.
  final double biomeLength;
  static const blendLength = 90.0;

  final FastNoiseLite _hills;
  final FastNoiseLite _mountains;
  final FastNoiseLite _detail;

  /// Trench depth below the surrounding ground, metres. A real path is the
  /// lowest line across terrain, scoured by water and feet.
  static const trenchDepth = 0.55;

  BiomeSpec biomeAt(double z) => BiomeSpec.order[_biomeIndex(z) % BiomeSpec.order.length];

  /// The biome Dash is blending into, and how far along (0..1). 0 means the
  /// current biome is unchallenged.
  (BiomeSpec, double) blendAt(double z) {
    final i = _biomeIndex(z);
    final local = z - i * biomeLength;
    final t = ((local - (biomeLength - blendLength)) / blendLength).clamp(0.0, 1.0);
    return (BiomeSpec.order[(i + 1) % BiomeSpec.order.length], t.toDouble());
  }

  int _biomeIndex(double z) => (z / biomeLength).floor();

  double _lerp(double a, double b, double t) => a + (b - a) * t;

  /// Ground height at a world XZ. [zHint] is the trail z the point was laid
  /// out from, when the caller knows it (see [Trail.toTrail]).
  double height(double x, double z, {double? zHint}) {
    final cur = biomeAt(z);
    final (next, t) = blendAt(z);
    final hillAmp = _lerp(cur.hillAmplitude, next.hillAmplitude, t);
    final mountainAmp = _lerp(cur.mountainAmplitude, next.mountainAmplitude, t);

    final lateral = trail.lateralOffset(x, z).abs();

    var h = _hills.getNoise2(x, z) * hillAmp;

    // Mountains only well away from the trail, so the corridor stays runnable
    // and the horizon still has something to look at.
    final far = ((lateral - 34) / 55).clamp(0.0, 1.0);
    h += _mountains.getNoise2(x, z) * mountainAmp * far * far;

    h += _detail.getNoise2(x, z) * 0.3;

    // Carve the trail: blend the surface towards a smoothed elevation of the
    // centre line, minus the trench, with banks rising just outside it.
    final w = trail.halfWidth * 1.25;
    final g = math.exp(-(lateral * lateral) / (2 * w * w));
    h = h * (1 - g) + (_trailElevation(z) - trenchDepth) * g;

    final bankD = lateral - w * 1.7;
    final bank = math.exp(-(bankD * bankD) / (2 * (w * 0.55) * (w * 0.55)));
    h += bank * 0.65 * (1 - g);

    return _carveRivers(x, z, h, zHint);
  }

  // --- rivers ----------------------------------------------------------------

  /// One possible river per this many metres of trail.
  static const riverSpacing = 150.0;
  static const _riverChance = 0.8;

  /// Channel floor and water surface below a river's level line, metres.
  static const _riverBed = 0.9;
  static const _waterDrop = 0.4;

  /// Widths (gaussian sigma) of the channel and of the valley around it.
  /// The valley is what keeps both banks over the water: hills slope across
  /// a river by up to a metre and a half over its ribbon, and a channel cut
  /// into that alone leaves the low bank under the water line.
  static const _channelSigma = 2.4;
  static const _valleySigma = 7.0;

  /// Window around a point's z in which a river can touch it: the trail z of
  /// the terrain's far edge is up to ~32 m off the point's own z, a course
  /// strays ~22 m from its crossing, and the valley reaches ~28 m.
  static const _riverSearch = 90.0;

  final Map<int, River?> _riverSlots = {};
  final Map<int, double> _riverFloors = {};

  River? _riverInSlot(int slot) => _riverSlots.putIfAbsent(slot, () {
        // The first stretch stays calm, like the obstacles.
        if (slot < 1) return null;
        final rng = math.Random(seed * 7919 + slot * 104729);
        if (rng.nextDouble() > _riverChance) return null;
        return River(
          id: slot,
          zCross: slot * riverSpacing + 25 + rng.nextDouble() * (riverSpacing - 50),
          amplitude: 2.5 + rng.nextDouble() * 2.5,
          wavelength: 30 + rng.nextDouble() * 20,
          // Near a quarter wave at the trail, so the course crosses it
          // steeply instead of running along it.
          phase: math.pi / 2 + (rng.nextDouble() - 0.5) * 0.6,
          skew: (rng.nextDouble() - 0.5) * 0.5,
        );
      });

  /// Rivers whose crossing lies in [zStart, zEnd), in order.
  List<River> riversBetween(double zStart, double zEnd) => [
        for (var s = (zStart / riverSpacing).floor(); s <= (zEnd / riverSpacing).floor(); s++)
          if (_riverInSlot(s) case final r? when r.zCross >= zStart && r.zCross < zEnd) r,
      ];

  List<River> _riversNear(double z) => [
        for (var s = ((z - _riverSearch) / riverSpacing).floor(); s <= ((z + _riverSearch) / riverSpacing).floor(); s++)
          ?_riverInSlot(s),
      ];

  double _carveRivers(double x, double z, double h, double? zHint) {
    final near = _riversNear(z);
    if (near.isEmpty) return h;
    final (zt, lateral) = trail.toTrail(x, z, guess: zHint);
    for (final r in near) {
      final d = r.distanceFrom(zt, lateral);
      if (d.abs() > 4 * _valleySigma) continue;
      final level = riverLevel(r, lateral);
      // A valley floor rising gently from the water, then the channel in it.
      final v = math.exp(-(d * d) / (2 * _valleySigma * _valleySigma));
      h = h * (1 - v) + (level + math.min(0.03 * d * d, 1.2)) * v;
      final c = math.exp(-(d * d) / (2 * _channelSigma * _channelSigma));
      h = h * (1 - c) + (level - _riverBed) * c;
    }
    return h;
  }

  /// A river's level line at [lateral]: the lie of the land along its course,
  /// smoothed like the trail's own elevation, blending into the trail's
  /// floor where it crosses the trail, so the crossing is a dip in the path
  /// rather than a step.
  double riverLevel(River r, double lateral) {
    const weights = [0.25, 0.5, 0.25];
    const offsets = [-3.5, 0.0, 3.5];
    var land = 0.0;
    for (var i = 0; i < 3; i++) {
      final l = lateral + offsets[i];
      final p = trail.pointAt(r.centerZ(l), lateral: l);
      final cur = biomeAt(p.y);
      final (next, t) = blendAt(p.y);
      land += weights[i] * _hills.getNoise2(p.x, p.y) * _lerp(cur.hillAmplitude, next.hillAmplitude, t);
    }
    final floor = _riverFloors.putIfAbsent(r.id, () => _trailElevation(r.zCross) - trenchDepth);
    final w = trail.halfWidth * 1.25;
    final g = math.exp(-(lateral * lateral) / (2 * w * w));
    return land * (1 - g) + floor * g;
  }

  /// Height of the water surface of [r] at [lateral].
  double waterLevel(River r, double lateral) => riverLevel(r, lateral) - _waterDrop;

  /// 0..1: how much of a river channel a point is in (1 = mid-stream).
  /// Props stay out where this is above ~0.1.
  double riverInfluence(double x, double z, {double? zHint}) {
    final near = _riversNear(z);
    if (near.isEmpty) return 0;
    final (zt, lateral) = trail.toTrail(x, z, guess: zHint);
    var most = 0.0;
    for (final r in near) {
      final d = r.distanceFrom(zt, lateral);
      most = math.max(most, math.exp(-(d * d) / (2 * _channelSigma * _channelSigma)));
    }
    return most;
  }

  /// Hill height along the centre line, smoothed over a few metres: the trail
  /// follows the lie of the land instead of cutting a level shelf through it.
  ///
  /// The window has to stay well under the hills' feature size (~150 m at the
  /// frequency above). A wide window averages away real terrain, the trail
  /// then floats metres off the local ground, and the trench disappears into
  /// that error instead of reading as a scoured path.
  double _trailElevation(double z) {
    const weights = [0.25, 0.5, 0.25];
    const offsets = [-3.5, 0.0, 3.5];
    var sum = 0.0;
    for (var i = 0; i < 3; i++) {
      final zz = z + offsets[i];
      final cur = biomeAt(zz);
      final (next, t) = blendAt(zz);
      sum += weights[i] *
          _hills.getNoise2(trail.centerX(zz), zz) *
          _lerp(cur.hillAmplitude, next.hillAmplitude, t);
    }
    return sum;
  }

  /// Surface normal from central differences of [height].
  (double, double, double) normalAt(double x, double z, {double eps = 0.6, double? zHint}) {
    final dx = height(x + eps, z, zHint: zHint) - height(x - eps, z, zHint: zHint);
    final dz = height(x, z + eps, zHint: zHint) - height(x, z - eps, zHint: zHint);
    final nx = -dx / (2 * eps);
    final nz = -dz / (2 * eps);
    final len = math.sqrt(nx * nx + 1 + nz * nz);
    return (nx / len, 1 / len, nz / len);
  }

  /// Steepness 0..1 (0 flat, 1 vertical), used for colouring and for deciding
  /// where props may stand.
  double slopeAt(double x, double z, {double? zHint}) {
    final (_, ny, _) = normalAt(x, z, zHint: zHint);
    return (1 - ny).clamp(0.0, 1.0);
  }

  /// Vertex colour for a point: ground, rock on steep faces, biome peak colour
  /// high up, and a trodden strip along the trail itself. Linear RGB.
  (double, double, double) colorAt(double x, double z, double y, {double? zHint}) {
    final cur = biomeAt(z);
    final (next, t) = blendAt(z);
    (double, double, double) mix((double, double, double) a, (double, double, double) b, double k) =>
        (_lerp(a.$1, b.$1, k), _lerp(a.$2, b.$2, k), _lerp(a.$3, b.$3, k));

    var ground = mix(cur.groundColor, next.groundColor, t);
    final slopeCol = mix(cur.slopeColor, next.slopeColor, t);
    final peakCol = mix(cur.peakColor, next.peakColor, t);

    final steep = ((slopeAt(x, z, zHint: zHint) - 0.25) / 0.35).clamp(0.0, 1.0);
    ground = mix(ground, slopeCol, steep);

    final high = ((y - 6) / 14).clamp(0.0, 1.0);
    ground = mix(ground, peakCol, high * 0.8);

    // The trail has to read as a path from a camera 8 m back, so it is bare
    // earth across the full corridor with only a short blend into the verge.
    final lateral = trail.lateralOffset(x, z).abs();
    final onTrail = 1 - ((lateral - trail.halfWidth * 0.75) / (trail.halfWidth * 0.6)).clamp(0.0, 1.0);
    return mix(ground, (0.34, 0.25, 0.16), onTrail * 0.95);
  }
}
