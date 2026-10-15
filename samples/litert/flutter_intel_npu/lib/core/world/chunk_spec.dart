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

import 'terrain_field.dart';

/// Something on the trail that Dash has to deal with. Gameplay data first:
/// the pilot reads these, the meshes that represent them are built from the
/// same numbers.
enum ObstacleKind {
  /// Fallen trunk right across the corridor: there is no way around it, so
  /// it is always a jump. That is the point — a jump reads from the back of
  /// the room, a sidestep does not.
  log,

  /// Sits to one side — run around it.
  boulder,

  /// Where a river crosses the trail — jump it or get wet (a stumble). Placed
  /// by the [TerrainField]'s rivers, not rolled per chunk: the water is part
  /// of the landscape, the obstacle is only where the path meets it.
  stream,
}

class Obstacle {
  const Obstacle({
    required this.id,
    required this.z,
    required this.lateral,
    required this.halfSpan,
    required this.height,
    required this.kind,
  });

  /// Stable across runs: chunk index and position in the chunk.
  final int id;

  /// Where along the trail it sits.
  final double z;

  /// Centre of the obstacle, metres right of the trail centre line.
  final double lateral;

  /// Half its lateral extent.
  final double halfSpan;

  /// How tall it stands above the trail surface.
  final double height;

  final ObstacleKind kind;

  /// Can Dash pass at this lateral offset without jumping?
  bool clearsAt(double offset, {double dashHalfWidth = 0.45}) =>
      (offset - lateral).abs() > halfSpan + dashHalfWidth;

  String get label => switch (kind) {
        ObstacleKind.log => 'fallen log',
        ObstacleKind.boulder => 'boulder',
        ObstacleKind.stream => 'stream',
      };
}

enum PropKind { tree, bush, rock, grass }

/// One scattered instance. Placed at spec time so the render isolate only
/// uploads transforms.
class PropInstance {
  const PropInstance({
    required this.kind,
    required this.x,
    required this.y,
    required this.z,
    required this.yaw,
    required this.scale,
    required this.tint,
  });

  final PropKind kind;
  final double x;
  final double y;
  final double z;
  final double yaw;
  final double scale;

  /// Per-instance colour multiplier, linear RGB, so a forest is not one flat
  /// green.
  final (double, double, double) tint;
}

/// Dials that change how much world there is — the GPU load of the act.
class WorldConfig {
  const WorldConfig({
    this.seed = 1337,
    this.chunkLength = 64,
    this.terrainHalfWidth = 46,
    this.cols = 72,
    this.rows = 56,
    this.propDensity = 1.0,
    this.obstacleSpacing = 16,
  });

  final int seed;

  /// Metres of trail per chunk.
  final double chunkLength;

  /// Half width of the generated ribbon, metres either side of the trail.
  final double terrainHalfWidth;

  /// Mesh resolution per chunk (vertices = (cols + 1) * (rows + 1)).
  final int cols;
  final int rows;

  /// Multiplier on every biome's prop density.
  final double propDensity;

  /// Average metres between obstacles.
  final double obstacleSpacing;

  WorldConfig copyWith({double? propDensity, int? cols, int? rows}) => WorldConfig(
        seed: seed,
        chunkLength: chunkLength,
        terrainHalfWidth: terrainHalfWidth,
        cols: cols ?? this.cols,
        rows: rows ?? this.rows,
        propDensity: propDensity ?? this.propDensity,
        obstacleSpacing: obstacleSpacing,
      );
}

/// Everything in one 64 m slice of world, derived from its index alone: two
/// runs with the same seed generate the same landscape, which is what makes a
/// measured run repeatable.
class ChunkSpec {
  const ChunkSpec({
    required this.index,
    required this.zStart,
    required this.zEnd,
    required this.obstacles,
    required this.props,
    required this.biomeLabel,
  });

  final int index;
  final double zStart;
  final double zEnd;
  final List<Obstacle> obstacles;
  final List<PropInstance> props;
  final String biomeLabel;
}

/// Builds a chunk's gameplay content. Pure and deterministic; safe to call on
/// a background isolate.
ChunkSpec buildChunkSpec(int index, TerrainField field, WorldConfig config) {
  final rng = math.Random(config.seed ^ (index * 0x9E3779B1));
  final zStart = index * config.chunkLength;
  final zEnd = zStart + config.chunkLength;
  final trail = field.trail;

  final obstacles = <Obstacle>[];
  // Rivers just past either end count too: a log at the end of this chunk
  // would otherwise lie in a river that crosses at the start of the next.
  final crossings = field.riversBetween(zStart - 8, zEnd + 8);
  bool inRiver(double z) => crossings.any((r) => (r.zCross - z).abs() <= 8);

  // The first stretch is deliberately empty so a run starts calmly.
  var z = zStart + rng.nextDouble() * config.obstacleSpacing;
  var n = 0;
  while (z < zEnd) {
    // Rolled logs and boulders; streams come from the rivers below.
    if (index > 0 && !inRiver(z)) {
      obstacles.add(rng.nextDouble() < 0.53
          ? Obstacle(
              id: index * 100 + n,
              z: z,
              lateral: (rng.nextDouble() - 0.5) * 0.6,
              // Wider than the corridor: dodging is not an option.
              halfSpan: trail.halfWidth * (1.15 + rng.nextDouble() * 0.2),
              height: 0.8,
              kind: ObstacleKind.log,
            )
          : Obstacle(
              id: index * 100 + n,
              z: z,
              lateral: (rng.nextBool() ? 1 : -1) * trail.halfWidth * (0.3 + rng.nextDouble() * 0.45),
              halfSpan: 0.8 + rng.nextDouble() * 0.7,
              height: 1.4,
              kind: ObstacleKind.boulder,
            ));
      n++;
    }
    z += config.obstacleSpacing * (0.6 + rng.nextDouble() * 0.9);
  }
  final owned = field.riversBetween(zStart, zEnd);
  for (var i = 0; i < owned.length; i++) {
    obstacles.add(Obstacle(
      id: index * 100 + 90 + i,
      z: owned[i].zCross,
      lateral: 0,
      // Wider than the corridor: the water runs right across it.
      halfSpan: trail.halfWidth * 1.4,
      height: 0.0,
      kind: ObstacleKind.stream,
    ));
  }
  obstacles.sort((a, b) => a.z.compareTo(b.z));

  final props = <PropInstance>[];
  final area = config.chunkLength * config.terrainHalfWidth * 2 / 100; // per 100 m²
  final biome = field.biomeAt((zStart + zEnd) / 2);
  void scatter(PropKind kind, double density, {required double minLateral, double maxSlope = 0.45}) {
    final count = (density * config.propDensity * area).round();
    for (var i = 0; i < count; i++) {
      final lateral = (rng.nextDouble() * 2 - 1) * config.terrainHalfWidth;
      if (lateral.abs() < minLateral) continue;
      final pz = zStart + rng.nextDouble() * config.chunkLength;
      final p = trail.pointAt(pz, lateral: lateral);
      if (field.slopeAt(p.x, p.y, zHint: pz) > maxSlope) continue;
      if (field.riverInfluence(p.x, p.y, zHint: pz) > 0.1) continue; // nothing grows in the water
      final y = field.height(p.x, p.y, zHint: pz);
      final shade = 0.8 + rng.nextDouble() * 0.4;
      props.add(PropInstance(
        kind: kind,
        x: p.x,
        y: y,
        z: p.y,
        yaw: rng.nextDouble() * math.pi * 2,
        scale: switch (kind) {
          PropKind.tree => 0.8 + rng.nextDouble() * 0.7,
          PropKind.bush => 0.6 + rng.nextDouble() * 0.5,
          PropKind.rock => 0.5 + rng.nextDouble() * 0.9,
          PropKind.grass => 0.7 + rng.nextDouble() * 0.6,
        },
        tint: (shade, shade * (0.95 + rng.nextDouble() * 0.1), shade * 0.95),
      ));
    }
  }

  scatter(PropKind.tree, biome.treeDensity, minLateral: trail.halfWidth + 2.5, maxSlope: 0.4);
  scatter(PropKind.bush, biome.treeDensity * 0.6 + 0.4, minLateral: trail.halfWidth + 1.2);
  scatter(PropKind.rock, biome.rockDensity, minLateral: trail.halfWidth + 0.8, maxSlope: 0.75);
  // Grass stops at the verge: covering the corridor hides the one piece of
  // ground the audience needs to read as a path.
  scatter(PropKind.grass, biome.grassDensity, minLateral: trail.halfWidth + 0.6, maxSlope: 0.5);

  return ChunkSpec(
    index: index,
    zStart: zStart,
    zEnd: zEnd,
    obstacles: obstacles,
    props: props,
    biomeLabel: biome.label,
  );
}
