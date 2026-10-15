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

import 'package:flutter_test/flutter_test.dart';
import 'package:intel_npu_demo/core/world/chunk_mesher.dart';
import 'package:intel_npu_demo/core/world/chunk_spec.dart';
import 'package:intel_npu_demo/core/world/terrain_field.dart';
import 'package:intel_npu_demo/core/world/trail.dart';

void main() {
  group('Trail', () {
    const trail = Trail();

    test('tangent is a unit vector and points forward', () {
      for (final z in [0.0, 37.5, 210.0, -64.0, 1234.5]) {
        final t = trail.tangentAt(z);
        expect(t.length, closeTo(1.0, 1e-6)); // vector_math stores float32
        expect(t.y, greaterThan(0), reason: 'z must stay monotonic along the trail');
      }
    });

    test('right is perpendicular to the tangent, and +X on a straight stretch', () {
      for (final z in [0.0, 52.5, 300.0]) {
        final t = trail.tangentAt(z);
        final r = trail.rightAt(z);
        expect(t.dot(r), closeTo(0, 1e-6));
        expect(r.length, closeTo(1.0, 1e-6));
      }
      // Where the trail runs straight along +Z, right must be +X (up × forward).
      const straight = Trail(amplitude1: 0, amplitude2: 0);
      expect(straight.rightAt(10).x, closeTo(1.0, 1e-6));
    });

    test('the centre line has zero lateral offset', () {
      for (final z in [0.0, 19.0, 143.0, 777.0]) {
        expect(trail.lateralOffset(trail.centerX(z), z), closeTo(0, 1e-9));
      }
    });

    test('a point offset to the right reports that offset back', () {
      for (final z in [5.0, 88.0, 421.0]) {
        final p = trail.pointAt(z, lateral: 2.5);
        // pointAt moves along the perpendicular, so the z of the point shifts
        // too; measuring at the point's own z is what the game does.
        expect(trail.lateralOffset(p.x, p.y), closeTo(2.5, 0.1));
      }
    });
  });

  group('TerrainField', () {
    final field = TerrainField();

    test('is deterministic for a seed and differs between seeds', () {
      final again = TerrainField();
      final other = TerrainField(seed: 99);
      expect(field.height(12.5, 40), again.height(12.5, 40));
      expect(field.height(12.5, 40), isNot(other.height(12.5, 40)));
    });

    test('produces finite heights across a long run', () {
      for (var z = 0.0; z < 2200; z += 37) {
        for (final lateral in [-40.0, -3.0, 0.0, 3.0, 40.0]) {
          final p = field.trail.pointAt(z, lateral: lateral);
          expect(field.height(p.x, p.y).isFinite, isTrue);
        }
      }
    });

    test('the trail is a trench between its own banks', () {
      // Measured against the banks at ±8 m, which is what the carve acts on.
      // Farther out (±14 m and beyond) natural hill variation is larger than
      // the trench, so a comparison there says nothing about the carving.
      final depths = <double>[];
      for (var z = 0.0; z < 1600; z += 23) {
        final on = field.trail.pointAt(z);
        final left = field.trail.pointAt(z, lateral: -8);
        final right = field.trail.pointAt(z, lateral: 8);
        final banks = (field.height(left.x, left.y) + field.height(right.x, right.y)) / 2;
        depths.add(banks - field.height(on.x, on.y));
      }
      depths.sort();
      expect(depths.where((d) => d > 0).length / depths.length, greaterThan(0.95));
      expect(depths[depths.length ~/ 2], greaterThan(0.3), reason: 'trench too shallow to read');
    });

    test('the corridor stays runnable — no cliffs across the trail', () {
      for (var z = 0.0; z < 1200; z += 4) {
        final a = field.trail.pointAt(z);
        final b = field.trail.pointAt(z + 4);
        final rise = (field.height(b.x, b.y) - field.height(a.x, a.y)).abs();
        expect(rise, lessThan(2.5), reason: 'trail slope at z=$z is a wall');
      }
    });

    test('biomes carry a comparable amount of scenery', () {
      // Act 3 calibrates the scene to the frame budget once. If one biome
      // drew a third of the instances of another, the frame rate would move
      // on its own — and that movement is the very thing the act measures.
      final totals = [for (final b in BiomeSpec.order) b.totalDensity];
      final lo = totals.reduce((a, b) => a < b ? a : b);
      final hi = totals.reduce((a, b) => a > b ? a : b);
      expect(hi / lo, lessThan(1.25), reason: 'biome density spread: $totals');
    });

    test('biomes cycle and blend', () {
      expect(field.biomeAt(10).biome, Biome.meadow);
      expect(field.biomeAt(600).biome, Biome.forest);
      expect(field.biomeAt(4 * 520 + 10).biome, Biome.meadow);
      final (next, t) = field.blendAt(10);
      expect(t, 0.0);
      expect(next.biome, Biome.forest);
      final (_, late) = field.blendAt(519);
      expect(late, greaterThan(0.9));
    });

    test('normals point upwards and are unit length', () {
      for (var z = 0.0; z < 800; z += 61) {
        final p = field.trail.pointAt(z, lateral: 7);
        final (nx, ny, nz) = field.normalAt(p.x, p.y);
        expect(math.sqrt(nx * nx + ny * ny + nz * nz), closeTo(1.0, 1e-9));
        expect(ny, greaterThan(0));
      }
    });
  });

  group('chunks', () {
    const config = WorldConfig(cols: 24, rows: 18);
    final field = TerrainField();

    test('mesh indices stay in range and triangles face up', () {
      final mesh = buildChunkMesh(3, field, config);
      expect(mesh.vertexCount, (config.cols + 1) * (config.rows + 1));
      expect(mesh.triangleCount, config.cols * config.rows * 2);
      for (final i in mesh.indices) {
        expect(i, lessThan(mesh.vertexCount));
      }
      // Sample the first triangle's geometric normal: cross(b-a, c-a).y > 0.
      double px(int v, int c) => mesh.positions[v * 3 + c];
      final a = mesh.indices[0], b = mesh.indices[1], c = mesh.indices[2];
      final e1 = [px(b, 0) - px(a, 0), px(b, 1) - px(a, 1), px(b, 2) - px(a, 2)];
      final e2 = [px(c, 0) - px(a, 0), px(c, 1) - px(a, 1), px(c, 2) - px(a, 2)];
      final ny = e1[2] * e2[0] - e1[0] * e2[2];
      expect(ny, greaterThan(0), reason: 'winding puts the lit face underground');
    });

    test('neighbouring chunks share their seam row exactly', () {
      final a = buildChunkMesh(5, field, config);
      final b = buildChunkMesh(6, field, config);
      final lastRow = config.rows * (config.cols + 1);
      for (var i = 0; i <= config.cols; i++) {
        for (var c = 0; c < 3; c++) {
          expect(
            a.positions[(lastRow + i) * 3 + c],
            b.positions[i * 3 + c],
            reason: 'crack at the chunk seam, column $i, component $c',
          );
        }
      }
    });

    test('a chunk is the same every time it is built', () {
      final a = buildChunkSpec(7, field, config);
      final b = buildChunkSpec(7, field, config);
      expect(a.obstacles.map((o) => (o.id, o.z, o.lateral, o.kind)).toList(),
          b.obstacles.map((o) => (o.id, o.z, o.lateral, o.kind)).toList());
      expect(a.props.length, b.props.length);
    });

    test('obstacles sit inside the chunk and leave a way through', () {
      final spec = buildChunkSpec(9, field, WorldConfig());
      expect(spec.obstacles, isNotEmpty);
      for (final o in spec.obstacles) {
        expect(o.z, inInclusiveRange(spec.zStart, spec.zEnd));
        if (o.kind == ObstacleKind.boulder) {
          // A boulder must always be dodgeable on one side or the other.
          final half = const Trail().halfWidth;
          expect(o.clearsAt(-half) || o.clearsAt(half), isTrue);
        }
      }
    });

    test('logs span the corridor, so they can only be jumped', () {
      const trail = Trail();
      var logs = 0;
      for (var i = 1; i < 30; i++) {
        for (final o in buildChunkSpec(i, field, const WorldConfig()).obstacles) {
          if (o.kind != ObstacleKind.log) continue;
          logs++;
          for (var x = -trail.halfWidth; x <= trail.halfWidth; x += 0.25) {
            expect(o.clearsAt(x), isFalse, reason: 'log ${o.id} can be dodged at $x');
          }
        }
      }
      expect(logs, greaterThan(10), reason: 'no logs generated to check');
    });

    test('props keep off the trail corridor', () {
      final spec = buildChunkSpec(11, field, WorldConfig());
      const trail = Trail();
      for (final p in spec.props.where((p) => p.kind == PropKind.tree)) {
        expect(trail.lateralOffset(p.x, p.z).abs(), greaterThan(trail.halfWidth));
      }
    });

    test('the first chunk starts clear', () {
      expect(buildChunkSpec(0, field, WorldConfig()).obstacles, isEmpty);
    });
  });
}
