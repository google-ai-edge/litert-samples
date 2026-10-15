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

import 'dart:async';
import 'dart:math';

import 'package:flutter/foundation.dart';
import 'package:flutter_scene/physics.dart';
import 'package:flutter_scene/scene.dart';
import 'package:flutter_scene_rapier/flutter_scene_rapier.dart';
import 'package:vector_math/vector_math.dart' as vm;

import '../../core/scene/scene_events.dart';

/// The 3D half of Act 3: a walled arena into which crates and balls keep
/// dropping, colliding, and settling. Owns the [Scene]; the widget only
/// displays it. Physics events become [SceneEvent]s for the narrator.
///
/// "Heavy" is a dial, not a fixed scene: [bodyCount] (CPU + draw calls),
/// [renderScale] and the post stack (raster) are adjustable live, so the
/// presenter can push the GPU until the frame budget is at the edge.
class ArenaGame extends ChangeNotifier {
  ArenaGame({required this.events, int? seed}) : _rng = Random(seed ?? 7);

  final SceneEventLog events;
  final Random _rng;

  final Scene scene = Scene();
  late final PhysicsWorld world;
  final Node _root = Node(name: 'arena');
  final List<_Body> _bodies = [];
  StreamSubscription<CollisionEvent>? _collisions;

  static const arenaHalf = 6.0;
  static const spawnHeight = 9.0;

  int bodyCount = 60;
  double spawnInterval = 0.35;
  double _spawnClock = 0;
  double elapsed = 0;
  int _spawned = 0;
  bool paused = false;

  // Shared GPU resources: one geometry + one material per kind.
  late final Geometry _crateGeometry;
  late final Geometry _ballGeometry;
  late final List<PhysicallyBasedMaterial> _palette;

  bool _built = false;
  bool get isBuilt => _built;
  int get liveBodies => _bodies.length;

  /// Call once after `Scene.initializeStaticResources()`.
  void build() {
    if (_built) return;
    _built = true;

    // Scene._stepPhysics only looks for the world on scene.root, so it must
    // live there, not on a child node (a child world never steps).
    world = PhysicsWorld(RapierWorld(gravity: vm.Vector3(0, -9.81, 0)));
    scene.root.addComponent(world);
    scene.add(_root);

    _crateGeometry = CuboidGeometry(vm.Vector3(0.8, 0.8, 0.8));
    _ballGeometry = IcosphereGeometry(radius: 0.45, subdivisions: 3); // high-ish poly on purpose
    _palette = [
      for (final c in const [
        (0.90, 0.30, 0.25),
        (0.25, 0.60, 0.95),
        (0.95, 0.80, 0.20),
        (0.30, 0.85, 0.45),
        (0.85, 0.40, 0.90),
      ])
        PhysicallyBasedMaterial()
          ..baseColorFactor = vm.Vector4(c.$1, c.$2, c.$3, 1)
          ..metallicFactor = 0.1
          ..roughnessFactor = 0.45,
    ];

    // Visual slab is 0.5 m thick; the collider is 4 m so fast bodies cannot
    // tunnel through it (one did, at -1214 m, before CCD + this).
    _addStatic(
      name: 'floor',
      size: vm.Vector3(arenaHalf * 2, 0.5, arenaHalf * 2),
      position: vm.Vector3(0, -0.25, 0),
      color: vm.Vector4(0.55, 0.55, 0.58, 1),
      colliderSize: vm.Vector3(arenaHalf * 2 + 2, 4, arenaHalf * 2 + 2),
      colliderOffset: vm.Vector3(0, -1.75, 0),
    );
    // Visible walls are 3 m; their colliders reach 12 m and an invisible lid
    // closes the box, so bouncing balls cannot escape over the rim (they
    // did: "fell out at x=10.8" with 3 m colliders and restitution 0.55).
    for (final (name, pos, size) in [
      ('wall-north', vm.Vector3(0, 1.5, arenaHalf), vm.Vector3(arenaHalf * 2, 3, 0.4)),
      ('wall-south', vm.Vector3(0, 1.5, -arenaHalf), vm.Vector3(arenaHalf * 2, 3, 0.4)),
      ('wall-east', vm.Vector3(arenaHalf, 1.5, 0), vm.Vector3(0.4, 3, arenaHalf * 2)),
      ('wall-west', vm.Vector3(-arenaHalf, 1.5, 0), vm.Vector3(0.4, 3, arenaHalf * 2)),
    ]) {
      _addStatic(
        name: name,
        size: size,
        position: pos,
        color: vm.Vector4(0.35, 0.36, 0.40, 1),
        colliderSize: vm.Vector3(size.x, 12, size.z),
        colliderOffset: vm.Vector3(0, 4.5, 0),
      );
    }
    final lid = Node(name: 'lid')..position = vm.Vector3(0, 13, 0);
    lid.addComponent(RigidBody(type: BodyType.fixed));
    lid.addComponent(Collider(shape: BoxShape(halfExtents: vm.Vector3(arenaHalf, 0.5, arenaHalf))));
    _root.add(lid);

    _collisions = world.collisions.listen(_onCollision);
    scene.environmentSettings = EnvironmentSettings(
      ambientOcclusionEnabled: true,
      screenSpaceReflectionsEnabled: true,
      bloomEnabled: true,
    );
  }

  void _addStatic({
    required String name,
    required vm.Vector3 size,
    required vm.Vector3 position,
    required vm.Vector4 color,
    vm.Vector3? colliderSize,
    vm.Vector3? colliderOffset,
  }) {
    final node = Node(
      name: name,
      mesh: Mesh(CuboidGeometry(size), PhysicallyBasedMaterial()..baseColorFactor = color..roughnessFactor = 0.8),
    )..position = position;
    if (colliderSize == null) {
      node.addComponent(RigidBody(type: BodyType.fixed));
      node.addComponent(Collider(shape: BoxShape(halfExtents: size / 2)));
      _root.add(node);
      return;
    }
    // Separate invisible node so the visual slab keeps its real size.
    final body = Node(name: name)..position = position + (colliderOffset ?? vm.Vector3.zero());
    body.addComponent(RigidBody(type: BodyType.fixed));
    body.addComponent(Collider(shape: BoxShape(halfExtents: colliderSize / 2)));
    _root.add(body);
    _root.add(node);
  }

  /// Per rendered frame (from SceneView.onTick). Physics itself is stepped
  /// by the scene's fixed-step accumulator; this only drives spawning.
  void tick(double dt) {
    if (paused) return;
    elapsed += dt;
    _spawnClock += dt;
    if (_spawnClock >= spawnInterval) {
      _spawnClock = 0;
      spawnOne();
    }
    while (_bodies.length > bodyCount) {
      _despawn(_bodies.first);
    }
    // Registration check one frame after spawn, and an out-of-world guard.
    for (final b in List.of(_bodies)) {
      if (!b.checked) {
        b.checked = true;
        if (b.body.handle == null) {
          events.add(SceneEvent(at: elapsed, kind: 'warning', subject: b.node.name, object: 'physics', detail: 'body not registered'));
        }
      }
      if (b.node.position.y < -5) {
        events.add(SceneEvent(
          at: elapsed,
          kind: 'warning',
          subject: b.node.name,
          object: 'world',
          detail: 'fell out at ${_fmt(b.node.position)} v=${_fmt(b.body.linearVelocity)}; spawned ${b.spawnInfo}',
        ));
        _despawn(b);
      }
    }
  }

  void setBodyCount(int n) {
    bodyCount = n;
    notifyListeners();
  }

  void setRenderScale(double v) {
    scene.renderScale = v;
    notifyListeners();
  }

  void togglePause() {
    paused = !paused;
    notifyListeners();
  }

  void spawnBurst(int n) {
    for (var i = 0; i < n; i++) {
      spawnOne();
    }
  }

  void spawnOne() {
    final isBall = _rng.nextBool();
    final id = ++_spawned;
    final name = '${isBall ? 'ball' : 'crate'}-$id';
    final node = Node(
      name: name,
      mesh: Mesh(isBall ? _ballGeometry : _crateGeometry, _palette[id % _palette.length]),
    )
      ..position = vm.Vector3(
        (_rng.nextDouble() * 2 - 1) * (arenaHalf - 1.5),
        spawnHeight + _rng.nextDouble() * 2,
        (_rng.nextDouble() * 2 - 1) * (arenaHalf - 1.5),
      )
      ..rotation = vm.Quaternion.euler(_rng.nextDouble() * 3, _rng.nextDouble() * 3, 0);
    final body = RigidBody(
      mass: isBall ? 1.0 : 2.0,
      linearVelocity: vm.Vector3((_rng.nextDouble() - 0.5) * 4, 0, (_rng.nextDouble() - 0.5) * 4),
      angularVelocity: vm.Vector3(_rng.nextDouble() * 2, _rng.nextDouble() * 2, 0),
      linearDamping: 0.05,
      angularDamping: 0.05,
      ccdEnabled: true,
    );
    node.addComponent(body);
    node.addComponent(Collider(
      shape: isBall ? SphereShape(radius: 0.45) : BoxShape(halfExtents: vm.Vector3(0.4, 0.4, 0.4)),
      material: PhysicsMaterial(friction: 0.6, restitution: isBall ? 0.4 : 0.15),
    ));
    _root.add(node);
    final b = _Body(node, body, isBall: isBall)
      ..spawnInfo = 'pos=${_fmt(node.position)} v=${_fmt(body.linearVelocity)} w=${_fmt(body.angularVelocity)}';
    _bodies.add(b);
    events.add(SceneEvent(
      at: elapsed,
      kind: 'spawn',
      subject: name,
      object: 'arena',
      detail: 'dropped from ${node.position.y.toStringAsFixed(1)} m, ${_bodies.length} bodies live',
    ));
    notifyListeners();
  }

  void _despawn(_Body b) {
    _bodies.remove(b);
    _root.remove(b.node);
    events.add(SceneEvent(at: elapsed, kind: 'despawn', subject: b.node.name, object: 'arena', detail: 'removed, oldest'));
  }

  void _onCollision(CollisionEvent e) {
    if (e is! CollisionBegan) return;
    // Report from the moving body's point of view; skip static-static.
    final a = _bodyFor(e.nodeA), b = _bodyFor(e.nodeB);
    if (a == null && b == null) return;
    final mover = a ?? b!;
    final other = identical(mover.node, e.nodeA) ? e.nodeB : e.nodeA;
    final speed = mover.body.linearVelocity.length;
    if (speed < 0.8) return; // resting contacts are noise, not events
    events.add(SceneEvent(
      at: elapsed,
      kind: 'collision',
      subject: mover.node.name,
      object: other.name,
      detail: '${speed.toStringAsFixed(1)} m/s'
          '${e.contacts.isNotEmpty ? ', at y=${e.contacts.first.worldPosition.y.toStringAsFixed(1)}' : ''}',
    ));
  }

  _Body? _bodyFor(Node n) {
    for (final b in _bodies) {
      if (identical(b.node, n)) return b;
    }
    return null;
  }

  /// Debug: world-space heights of the oldest bodies (should settle near 0.4).
  List<double> sampleHeights([int n = 5]) =>
      [for (final b in _bodies.take(n)) double.parse(b.node.position.y.toStringAsFixed(2))];

  void clearBodies() {
    for (final b in List.of(_bodies)) {
      _despawn(b);
    }
    notifyListeners();
  }

  @override
  void dispose() {
    _collisions?.cancel();
    super.dispose();
  }
}

class _Body {
  _Body(this.node, this.body, {required this.isBall});
  final Node node;
  final RigidBody body;
  final bool isBall;
  bool checked = false;
  String spawnInfo = '';
}

String _fmt(vm.Vector3 v) => '(${v.x.toStringAsFixed(1)},${v.y.toStringAsFixed(1)},${v.z.toStringAsFixed(1)})';
