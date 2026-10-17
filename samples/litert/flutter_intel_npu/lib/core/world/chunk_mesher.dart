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
import 'dart:typed_data';

import 'chunk_spec.dart';
import 'river.dart';
import 'terrain_field.dart';
import 'trail.dart';

/// The water surface of one river: a flat-topped ribbon along its course,
/// wider than the channel so the banks draw the shoreline. UVs are in units
/// of 3 m (u along the flow, v across) for the tiling ripple map.
class WaterRibbon {
  const WaterRibbon({
    required this.riverId,
    required this.positions,
    required this.normals,
    required this.texCoords,
    required this.indices,
  });

  final int riverId;
  final Float32List positions;
  final Float32List normals;
  final Float32List texCoords;
  final Uint16List indices;
}

/// Raw vertex arrays for one chunk. Deliberately plain typed data rather than
/// the engine's `MeshData`: this half of the world runs on a background
/// isolate and must not import flutter_scene's render types. The render
/// isolate wraps these in `MeshData.build` without copying.
class ChunkMesh {
  const ChunkMesh({
    required this.index,
    required this.positions,
    required this.normals,
    required this.texCoords,
    required this.colors,
    required this.indices,
    this.water = const [],
  });

  final int index;
  final Float32List positions;
  final Float32List normals;
  final Float32List texCoords;
  final Float32List colors;
  final Uint16List indices;

  /// Rivers whose crossing lies in this chunk. A river's course strays up to
  /// ~22 m either side, so its ribbon can overhang the neighbouring chunks;
  /// it lives and dies with the chunk that owns the crossing.
  final List<WaterRibbon> water;

  int get vertexCount => positions.length ~/ 3;
  int get triangleCount => indices.length ~/ 3;
}

/// What a background isolate needs to build a chunk: small, sendable, and
/// enough to rebuild the field identically.
class ChunkRequest {
  const ChunkRequest({
    required this.index,
    required this.config,
    this.biomeLength = 520.0,
    this.trail = const Trail(),
  });

  final int index;
  final WorldConfig config;
  final double biomeLength;
  final Trail trail;
}

/// Gameplay content plus geometry for one chunk.
class ChunkBuild {
  const ChunkBuild({required this.spec, required this.mesh});
  final ChunkSpec spec;
  final ChunkMesh mesh;
}

/// Entry point for `compute`: top-level, pure, deterministic.
ChunkBuild buildChunk(ChunkRequest request) {
  final field = TerrainField(
    trail: request.trail,
    seed: request.config.seed,
    biomeLength: request.biomeLength,
  );
  return ChunkBuild(
    spec: buildChunkSpec(request.index, field, request.config),
    mesh: buildChunkMesh(request.index, field, request.config),
  );
}

/// A ribbon of terrain that follows the trail: the grid is sampled in
/// (z, lateral) rather than (x, z), so the mesh spends its vertices where
/// Dash actually runs, and neighbouring chunks share their edge row exactly —
/// same z, same lateral offsets, same height function, so there are no cracks
/// at the seam.
ChunkMesh buildChunkMesh(int index, TerrainField field, WorldConfig config) {
  final trail = field.trail;
  final cols = config.cols;
  final rows = config.rows;
  final zStart = index * config.chunkLength;
  final dz = config.chunkLength / rows;
  final dLateral = config.terrainHalfWidth * 2 / cols;

  final vertexCount = (cols + 1) * (rows + 1);
  final positions = Float32List(vertexCount * 3);
  final normals = Float32List(vertexCount * 3);
  final texCoords = Float32List(vertexCount * 2);
  final colors = Float32List(vertexCount * 4);

  var v = 0;
  for (var j = 0; j <= rows; j++) {
    final z = zStart + j * dz;
    final right = trail.rightAt(z);
    final centerX = trail.centerX(z);
    for (var i = 0; i <= cols; i++) {
      final lateral = -config.terrainHalfWidth + i * dLateral;
      final x = centerX + right.x * lateral;
      final zz = z + right.y * lateral;
      final y = field.height(x, zz, zHint: z);
      final (nx, ny, nz) = field.normalAt(x, zz, zHint: z);
      final (r, g, b) = field.colorAt(x, zz, y, zHint: z);

      positions[v * 3] = x;
      positions[v * 3 + 1] = y;
      positions[v * 3 + 2] = zz;
      normals[v * 3] = nx;
      normals[v * 3 + 1] = ny;
      normals[v * 3 + 2] = nz;
      texCoords[v * 2] = lateral / 4;
      texCoords[v * 2 + 1] = z / 4;
      colors[v * 4] = r;
      colors[v * 4 + 1] = g;
      colors[v * 4 + 2] = b;
      colors[v * 4 + 3] = 1;
      v++;
    }
  }

  // Two triangles per cell, wound so the lit face points up (+Y): with x
  // increasing along i and z along j, (i,j) → (i,j+1) → (i+1,j) has its
  // normal up, and the mirrored order would light the underside instead.
  final indices = Uint16List(cols * rows * 6);
  var t = 0;
  int idx(int i, int j) => j * (cols + 1) + i;
  for (var j = 0; j < rows; j++) {
    for (var i = 0; i < cols; i++) {
      indices[t++] = idx(i, j);
      indices[t++] = idx(i, j + 1);
      indices[t++] = idx(i + 1, j);
      indices[t++] = idx(i, j + 1);
      indices[t++] = idx(i + 1, j + 1);
      indices[t++] = idx(i + 1, j);
    }
  }

  return ChunkMesh(
    index: index,
    positions: positions,
    normals: normals,
    texCoords: texCoords,
    colors: colors,
    indices: indices,
    water: [
      for (final r in field.riversBetween(zStart, zStart + config.chunkLength)) buildWaterRibbon(r, field, config),
    ],
  );
}

/// Lays [river]'s water surface across the whole terrain ribbon: one pair of
/// vertices per [step] metres of lateral, at the water level there, offset
/// across the flow by [River.ribbonHalfWidth] to either side.
WaterRibbon buildWaterRibbon(River river, TerrainField field, WorldConfig config, {double step = 1.5}) {
  final trail = field.trail;
  final n = (config.terrainHalfWidth * 2 / step).round();
  final positions = Float32List((n + 1) * 2 * 3);
  final normals = Float32List((n + 1) * 2 * 3);
  final texCoords = Float32List((n + 1) * 2 * 2);
  const hw = River.ribbonHalfWidth;

  var along = 0.0;
  double? lastX;
  double? lastZ;
  for (var i = 0; i <= n; i++) {
    final lateral = -config.terrainHalfWidth + i * step;
    final zc = river.centerZ(lateral);
    final (aLat, aZ) = river.acrossAt(lateral);
    final y = field.waterLevel(river, lateral);
    final centre = trail.pointAt(zc, lateral: lateral);
    if (lastX != null) {
      final dx = centre.x - lastX, dz = centre.y - lastZ!;
      along += math.sqrt(dx * dx + dz * dz);
    }
    lastX = centre.x;
    lastZ = centre.y;
    for (var side = 0; side < 2; side++) {
      final s = side == 0 ? -1.0 : 1.0;
      final p = trail.pointAt(zc + aZ * s * hw, lateral: lateral + aLat * s * hw);
      final v = i * 2 + side;
      positions[v * 3] = p.x;
      positions[v * 3 + 1] = y;
      positions[v * 3 + 2] = p.y;
      normals[v * 3 + 1] = 1;
      texCoords[v * 2] = along / 3;
      texCoords[v * 2 + 1] = (s * hw + hw) / 3;
    }
  }

  // Lateral runs along +X and the far bank (side 1) towards +Z, the same
  // layout as the terrain grid, so the same winding puts the lit face up.
  final indices = Uint16List(n * 6);
  var t = 0;
  for (var i = 0; i < n; i++) {
    final a = i * 2, b = i * 2 + 1, c = i * 2 + 2, d = i * 2 + 3;
    indices[t++] = a;
    indices[t++] = b;
    indices[t++] = c;
    indices[t++] = b;
    indices[t++] = d;
    indices[t++] = c;
  }
  return WaterRibbon(riverId: river.id, positions: positions, normals: normals, texCoords: texCoords, indices: indices);
}
