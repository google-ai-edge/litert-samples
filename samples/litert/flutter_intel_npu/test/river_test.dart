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
import 'package:intel_npu_demo/core/world/chunk_mesher.dart';
import 'package:intel_npu_demo/core/world/chunk_spec.dart';
import 'package:intel_npu_demo/core/world/river.dart';
import 'package:intel_npu_demo/core/world/terrain_field.dart';
import 'package:intel_npu_demo/core/world/trail.dart';

void main() {
  group('Trail.toTrail', () {
    const trail = Trail();

    test('undoes pointAt across the whole ribbon, given where the point came from', () {
      for (final z in [0.0, 40.0, 133.0, 260.0, 777.0]) {
        for (final lateral in [-44.0, -12.0, 0.0, 7.5, 44.0]) {
          final p = trail.pointAt(z, lateral: lateral);
          final (zt, lat) = trail.toTrail(p.x, p.y, guess: z + 3);
          expect(zt, closeTo(z, 0.02), reason: 'z at ($z, $lateral)');
          expect(lat, closeTo(lateral, 0.02), reason: 'lateral at ($z, $lateral)');
        }
      }
    });

    test('needs no hint near the trail, where the ribbon cannot fold', () {
      // The tightest bend has a radius of about 20 m; inside that the
      // mapping is one-to-one.
      for (var z = 0.0; z < 800; z += 37) {
        for (final lateral in [-15.0, -6.0, 0.0, 4.0, 15.0]) {
          final p = trail.pointAt(z, lateral: lateral);
          final (zt, lat) = trail.toTrail(p.x, p.y);
          expect(zt, closeTo(z, 0.02), reason: 'z at ($z, $lateral)');
          expect(lat, closeTo(lateral, 0.02), reason: 'lateral at ($z, $lateral)');
        }
      }
    });
  });

  group('rivers', () {
    final field = TerrainField();
    final trail = field.trail;
    final rivers = field.riversBetween(0, 3000);

    test('are deterministic for a seed and differ between seeds', () {
      final again = TerrainField().riversBetween(0, 3000);
      expect(rivers.map((r) => r.zCross), again.map((r) => r.zCross));
      final other = TerrainField(seed: 99).riversBetween(0, 3000);
      expect(rivers.map((r) => r.zCross), isNot(other.map((r) => r.zCross)));
    });

    test('come every couple of hundred metres, not in the first stretch', () {
      expect(rivers.length, inInclusiveRange(10, 22));
      expect(rivers.first.zCross, greaterThan(TerrainField.riverSpacing));
      for (var i = 1; i < rivers.length; i++) {
        expect(rivers[i].zCross - rivers[i - 1].zCross, greaterThan(40), reason: 'two rivers on top of each other');
      }
    });

    test('cross the trail where they say, at a steep angle', () {
      for (final r in rivers) {
        expect(r.centerZ(0), r.zCross);
        // Shallow crossings would put metres of water along the corridor.
        expect(r.slopeAt(0).abs(), lessThan(0.6));
      }
    });

    test('run in a channel: water over the bed, banks over the water', () {
      // The ribbon is wider than the channel; the banks must rise over its
      // edges on both sides, or the water floats above the ground.
      var samples = 0;
      var floating = 0;
      for (final r in rivers.take(12)) {
        for (var lateral = -40.0; lateral <= 40; lateral += 4) {
          final level = field.waterLevel(r, lateral);
          final zc = r.centerZ(lateral);
          final centre = trail.pointAt(zc, lateral: lateral);
          expect(field.height(centre.x, centre.y, zHint: zc), lessThan(level - 0.25), reason: 'no water depth at $lateral');
          final (aLat, aZ) = r.acrossAt(lateral);
          for (final side in [-1.0, 1.0]) {
            final bankZ = r.centerZ(lateral) + aZ * side * River.ribbonHalfWidth;
            final bank = trail.pointAt(bankZ, lateral: lateral + aLat * side * River.ribbonHalfWidth);
            samples++;
            if (field.height(bank.x, bank.y, zHint: bankZ) < level) floating++;
          }
        }
      }
      expect(floating / samples, lessThan(0.02), reason: '$floating of $samples ribbon edges float');
    });

    test('the trail dips into the water where a river crosses it', () {
      for (final r in rivers) {
        final crossing = trail.pointAt(r.zCross);
        expect(field.height(crossing.x, crossing.y, zHint: r.zCross), lessThan(field.waterLevel(r, 0)));
      }
    });
  });

  group('chunks with rivers', () {
    final field = TerrainField();
    const config = WorldConfig();

    test('every crossing is one stream obstacle, with nothing else on top of it', () {
      for (var i = 1; i < 30; i++) {
        final spec = buildChunkSpec(i, field, config);
        final crossings = field.riversBetween(spec.zStart, spec.zEnd).map((r) => r.zCross).toList();
        final streams = spec.obstacles.where((o) => o.kind == ObstacleKind.stream).map((o) => o.z).toList();
        expect(streams, crossings, reason: 'chunk $i');
        for (final o in spec.obstacles.where((o) => o.kind != ObstacleKind.stream)) {
          for (final c in crossings) {
            expect((o.z - c).abs(), greaterThan(8), reason: '${o.label} in the river at chunk $i');
          }
        }
      }
    });

    test('nothing grows in the water', () {
      // Within 18 m of the trail, where a world point has one trail position.
      // Farther out, on the inside of the tightest bends, the terrain ribbon
      // folds over itself (it did before rivers too): a prop stands on its own
      // layer, and ~1% of them sit over another layer's river.
      for (var i = 1; i < 16; i++) {
        for (final p in buildChunkSpec(i, field, config).props) {
          final (_, lateral) = field.trail.toTrail(p.x, p.z);
          if (lateral.abs() > 18) continue;
          expect(field.riverInfluence(p.x, p.z), lessThanOrEqualTo(0.1), reason: '${p.kind.name} in a river');
        }
      }
    });

    test('a chunk carries a water ribbon for each river it owns', () {
      for (var i = 1; i < 12; i++) {
        final mesh = buildChunkMesh(i, field, config);
        final owned = field.riversBetween(i * config.chunkLength, (i + 1) * config.chunkLength);
        expect(mesh.water.map((w) => w.riverId), owned.map((r) => r.id));
        for (final w in mesh.water) {
          final vertices = w.positions.length ~/ 3;
          for (final index in w.indices) {
            expect(index, lessThan(vertices));
          }
          // Lit face up, like the terrain. Checked mid-ribbon, at the trail:
          // the far ends can lie where the terrain ribbon folds.
          double px(int v, int c) => w.positions[v * 3 + c];
          final mid = (w.indices.length ~/ 12) * 6;
          final a = w.indices[mid], b = w.indices[mid + 1], c = w.indices[mid + 2];
          final e1 = [px(b, 0) - px(a, 0), px(b, 1) - px(a, 1), px(b, 2) - px(a, 2)];
          final e2 = [px(c, 0) - px(a, 0), px(c, 1) - px(a, 1), px(c, 2) - px(a, 2)];
          expect(e1[2] * e2[0] - e1[0] * e2[2], greaterThan(0), reason: "water lit from below");
        }
      }
    });
  });
}
