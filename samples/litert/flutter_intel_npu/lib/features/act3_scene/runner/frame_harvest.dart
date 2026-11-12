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

import 'dart:convert';
import 'dart:io';
import 'dart:ui' as ui;

import 'package:flutter/rendering.dart';
import 'package:flutter/widgets.dart';
import 'package:flutter_scene/scene.dart';
import 'package:vector_math/vector_math.dart' as vm;

import '../../../core/world/chunk_spec.dart';
import 'runner_game.dart';

/// Training frames for a pilot that drives from what it sees
/// (`ACT3_HARVEST`): every [interval] seconds the scene alone — no HUD — is
/// written with what was in front of the camera, in world space: the camera,
/// the trail's edges and every obstacle nearby. Labels for parts of the
/// screen are worked out from that afterwards; the app never reads them.
class FrameHarvest {
  FrameHarvest({required this.dir, required this.sceneKey, this.pixelRatio = 1.0, this.interval = 0.3}) {
    Directory(dir).createSync(recursive: true);
    _index = File('$dir/frames.jsonl').openWrite(mode: FileMode.append);
  }

  final String dir;
  final GlobalKey sceneKey;
  final double pixelRatio;
  final double interval;

  late final IOSink _index;
  double _next = 0;
  bool _busy = false;
  int frames = 0;

  /// Call once per frame after the game ticked, with the camera the frame
  /// was drawn with.
  void observe(RunnerGame game, PerspectiveCamera? camera) {
    if (camera == null || _busy || game.elapsed < _next) return;
    _next = game.elapsed + interval;
    final size = sceneKey.currentContext?.size;
    if (size == null) return;
    final trail = game.trail;
    List<double> v3(vm.Vector3 v) => [for (final c in v.storage) double.parse(c.toStringAsFixed(3))];
    List<double> ground(double z, double lateral, double lift) {
      final p = trail.pointAt(z, lateral: lateral);
      return v3(vm.Vector3(p.x, game.field.height(p.x, p.y, zHint: z) + lift, p.y));
    }

    final record = {
      'at': double.parse(game.elapsed.toStringAsFixed(2)),
      'z': double.parse(game.z.toStringAsFixed(2)),
      'dashLateral': double.parse(game.lateral.toStringAsFixed(2)),
      'airborne': game.airborne,
      'speed': double.parse(game.speed.toStringAsFixed(2)),
      'teacher': game.teacherDriving,
      'view': [size.width, size.height],
      'camera': {'position': v3(camera.position), 'target': v3(camera.target), 'up': v3(camera.up), 'fovY': camera.fovRadiansY},
      'dash': v3(game.dashPosition),
      // Both edges every metre from 2 to 30 m ahead.
      'trail': [
        for (var d = 2.0; d <= 30; d += 1)
          {'z': double.parse((game.z + d).toStringAsFixed(2)), 'left': ground(game.z + d, -trail.halfWidth, 0), 'right': ground(game.z + d, trail.halfWidth, 0)},
      ],
      'obstacles': [
        for (final o in game.obstaclesAround())
          {
            'id': o.id,
            'kind': o.kind.name,
            'z': o.z,
            'lateral': o.lateral,
            'halfSpan': o.halfSpan,
            // Across the trail through the obstacle: its left and right ends
            // on the ground, and its top at the middle.
            'left': ground(o.z, o.lateral - o.halfSpan, 0),
            'right': ground(o.z, o.lateral + o.halfSpan, 0),
            'top': ground(o.z, o.lateral, switch (o.kind) {
              ObstacleKind.boulder => o.halfSpan * 1.4,
              ObstacleKind.log => 0.68,
              ObstacleKind.stream => 0.05,
            }),
          },
      ],
    };
    _busy = true;
    _grab('${frames.toString().padLeft(5, '0')}.png', record).whenComplete(() => _busy = false);
  }

  Future<void> _grab(String name, Map<String, Object?> record) async {
    final boundary = sceneKey.currentContext?.findRenderObject() as RenderRepaintBoundary?;
    if (boundary == null) return;
    final image = await boundary.toImage(pixelRatio: pixelRatio);
    final bytes = await image.toByteData(format: ui.ImageByteFormat.png);
    image.dispose();
    if (bytes == null) return;
    await File('$dir/$name').writeAsBytes(bytes.buffer.asUint8List());
    _index.writeln(jsonEncode({'file': name, ...record}));
    frames++;
  }

  Future<void> close() => _index.close();
}
