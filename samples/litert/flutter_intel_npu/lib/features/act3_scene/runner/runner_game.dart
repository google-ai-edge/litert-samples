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

import 'package:flutter/foundation.dart';
import 'package:flutter_scene/kit.dart' show CameraShake, CameraShakeOffset;
import 'package:flutter_scene/scene.dart';
import 'package:vector_math/vector_math.dart' as vm;

import '../../../core/scene/scene_events.dart';
import '../../../core/world/chunk_mesher.dart';
import '../../../core/world/chunk_spec.dart';
import '../../../core/world/terrain_field.dart';
import '../../../core/world/time_of_day.dart';
import '../../../core/world/trail.dart';
import '../../../core/world/water_ripples.dart';
import 'dash_pilot.dart';
import 'run_situation.dart';

/// Everything the presenter can turn up or down. These are the GPU load of
/// Act 3: the scene has to sit near the frame budget on its own, or moving
/// the model onto the GPU has nothing to break.
class RunnerSettings {
  const RunnerSettings({
    this.viewChunks = 7,
    this.propDensity = 1.0,
    this.shadows = true,
    this.post = true,
    this.renderScale = 1.0,
  });

  /// Chunks kept alive ahead of Dash (64 m each) — the draw distance.
  final int viewChunks;
  final double propDensity;
  final bool shadows;
  final bool post;
  final double renderScale;

  RunnerSettings copyWith({int? viewChunks, double? propDensity, bool? shadows, bool? post, double? renderScale}) =>
      RunnerSettings(
        viewChunks: viewChunks ?? this.viewChunks,
        propDensity: propDensity ?? this.propDensity,
        shadows: shadows ?? this.shadows,
        post: post ?? this.post,
        renderScale: renderScale ?? this.renderScale,
      );
}

class _Chunk {
  _Chunk(this.spec, this.node, this.gen);
  final ChunkSpec spec;
  final Node node;

  /// The world generation (vegetation density) it was built for.
  final int gen;
}

/// Act 3's scene: Dash runs an endless trail through generated landscape
/// while the model narrates. Owns the `Scene`; the widget only displays it.
///
/// The world half (trail, heights, chunk contents) lives in `core/world` and
/// knows nothing about rendering, so chunk generation runs on a background
/// isolate and is unit-tested without a GPU.
class RunnerGame extends ChangeNotifier {
  RunnerGame({
    required this.events,
    RunnerSettings settings = const RunnerSettings(),
    int seed = 1337,
    double missRate = 0.08,
    double dayCycleSeconds = 240,
    DayPhase dayStart = DayPhase.day,
    double startZ = 4,
  })  : _settings = settings,
        world = WorldConfig(seed: seed, propDensity: settings.propDensity),
        field = TerrainField(seed: seed),
        pilot = DashPilot(seed: seed, missRate: missRate),
        clock = TimeOfDay(cycleSeconds: dayCycleSeconds, start: dayStart),
        z = startZ,
        _startZ = startZ;

  final SceneEventLog events;
  final WorldConfig world;
  final TerrainField field;
  final DashPilot pilot;
  final TimeOfDay clock;

  Trail get trail => field.trail;

  final Scene scene = Scene();
  final Node _root = Node(name: 'world');

  RunnerSettings _settings;
  RunnerSettings get settings => _settings;

  final Map<int, _Chunk> _chunks = {};

  /// Chunk index → the world generation being built for it.
  final Map<int, int> _building = {};

  /// Bumped when the vegetation density changes. Chunks of an older
  /// generation stay on screen until their replacement is installed: clearing
  /// them left Dash running on nothing for a few frames, and with no
  /// obstacle ahead the pilot did not jump the next log.
  int _worldGen = 0;

  /// Builds in flight. Each is a `compute` isolate; a slider drag used to
  /// start one per frame.
  static const _maxBuilding = 2;

  // Where chunk time actually goes: generation on the worker vs upload on the
  // render isolate. Guessing between the two costs more than measuring.
  final List<int> _genMs = [];
  final List<int> _installMs = [];

  Node? _dashPivot;
  final Map<String, AnimationClip> _clips = {};

  // Prop assets, built once and shared by every chunk's instanced meshes.
  late final Map<PropKind, (Geometry, Material)> _propAssets;
  late final Geometry _logGeometry;
  late final Geometry _boulderGeometry;
  late final Material _woodMaterial;
  late final Material _stoneMaterial;
  late final PhysicallyBasedMaterial _riverMaterial;

  // The day-night cycle mutates these in place every frame; they are built
  // once, so a settings change does not restart the sky bake.
  GradientSkySource? _sky;
  Skybox? _skybox;
  SkyEnvironment? _skyEnvironment;
  final SkyBakeGate _bakeGate = SkyBakeGate();

  // --- run state -----------------------------------------------------------

  /// Position along the trail, metres of z. The trail wanders in x, so this is
  /// not the same as distance run.
  double z;

  /// Obstacles behind the start were never in Dash's way.
  final double _startZ;

  /// Offset from the centre line, metres to the right.
  double lateral = 0;
  double y = 0;
  double _vy = 0;
  bool airborne = false;

  double speed = 8.5;
  static const baseSpeed = 8.5;

  /// The glb is 2.92 units tall; this puts Dash at about 1.2 m.
  static const dashScale = 0.42;
  static const topSpeed = 15.0;

  double elapsed = 0;
  double distance = 0;
  int obstaclesCleared = 0;
  int stumbles = 0;
  int nearMisses = 0;
  int jumps = 0;
  int jumpsCleared = 0;

  /// Trauma-decay shake: a stumble kicks it, it settles on its own. The
  /// camera applies [shakeOffset] after its smoothing, or the lag eats it.
  final CameraShake _shake = CameraShake(
    decayRate: 1.4,
    frequency: 22,
    maxTranslation: vm.Vector3(0.35, 0.25, 0.2),
    maxRotation: vm.Vector3(0.035, 0.035, 0.05),
  );
  CameraShakeOffset shakeOffset = CameraShakeOffset.zero;
  double _stumbleTimer = 0;
  bool paused = false;

  String biomeLabel = '';

  /// Obstacles cleared since the last stumble, and the run's best.
  int cleanStreak = 0;
  int bestStreak = 0;
  final List<double> _stumbleTimes = [];
  bool _presenterWasDriving = false;
  int _milestone = 0;
  final Set<int> _resolved = {};

  double? _manualLateral;
  bool _manualJump = false;

  /// The boulder Dash ran into and has to get round before running on, and
  /// the lateral offset that takes him past it.
  Obstacle? _blockedBy;
  double? _detourLateral;
  double _blockedFor = 0;

  /// Boulders Dash's body ended up inside. Should stay 0; reported so a run
  /// can prove it rather than a screenshot suggest it.
  int penetrations = 0;
  final Set<int> _penetrated = {};

  static const _dashRadius = 0.35;
  static const _logRadius = 0.36;

  bool _built = false;
  bool get isBuilt => _built;
  int get liveChunks => _chunks.length;
  int get liveProps => _chunks.values.fold(0, (a, c) => a + c.spec.props.length);

  vm.Vector3 get dashPosition {
    final p = trail.pointAt(z, lateral: lateral);
    return vm.Vector3(p.x, y, p.y);
  }

  vm.Vector2 get forward => trail.tangentAt(z);

  // --- build ---------------------------------------------------------------

  /// Call once, after `Scene.initializeStaticResources()`.
  Future<void> build() async {
    if (_built) return;
    _built = true;
    scene.add(_root);
    _buildAssets();
    _applyLook();
    await _loadDash();
    biomeLabel = field.biomeAt(z).label;
    // Build the whole visible run before the first frame. Spawning the first
    // `compute` isolate and uploading a chunk costs a few hundred ms, and a
    // 275 ms hitch two seconds into the act is exactly the kind of number
    // this act is supposed to be measuring.
    for (var i = math.max(0, _currentIndex - 1); i <= _currentIndex + _settings.viewChunks; i++) {
      await _buildChunk(i);
    }
    notifyListeners();
  }

  void _buildAssets() {
    Material leafy(double r, double g, double b) => PhysicallyBasedMaterial()
      ..baseColorFactor = vm.Vector4(r, g, b, 1)
      ..roughnessFactor = 0.95
      ..metallicFactor = 0.0;

    _propAssets = {
      // Trunk plus canopy would be two draw calls per chunk; one tapered cone
      // reads as a conifer at the distances this scene is seen from.
      PropKind.tree: (CylinderGeometry(bottomRadius: 1.15, topRadius: 0.0, height: 5.2, radialSegments: 7), leafy(0.10, 0.30, 0.12)),
      PropKind.bush: (IcosphereGeometry(radius: 0.7, subdivisions: 1), leafy(0.16, 0.34, 0.14)),
      PropKind.rock: (IcosphereGeometry(radius: 0.55, subdivisions: 1), leafy(0.36, 0.34, 0.32)),
      PropKind.grass: (CylinderGeometry(bottomRadius: 0.22, topRadius: 0.02, height: 0.75, radialSegments: 4), leafy(0.30, 0.44, 0.16)),
    };

    _logGeometry = CylinderGeometry(bottomRadius: 0.36, topRadius: 0.33, height: 1, radialSegments: 9);
    _boulderGeometry = IcosphereGeometry(radius: 1, subdivisions: 2);
    _woodMaterial = leafy(0.32, 0.20, 0.11);
    _stoneMaterial = leafy(0.42, 0.40, 0.38);
    // Smooth and slightly see-through, so it mirrors the sky (orange at
    // sunset, moonlit at night); the ripple map scrolls with the current. The
    // ribbons need no edge of their own: the banks rise over them.
    const ripples = 128;
    // Roughness is what sets how much sky the water shows here, not ior:
    // probed at sunset, ior 1.01-1.33 changed nothing, while at roughness
    // 0.05 the river mirrored the peach sky above the horizon and read as a
    // sand path. At 0.6 it is deep water at every hour, the ripples still
    // visible: blue by day, dark violet at sunset, navy at night.
    _riverMaterial = PhysicallyBasedMaterial()
      ..baseColorFactor = vm.Vector4(0.03, 0.16, 0.26, 0.95)
      ..alphaMode = AlphaMode.blend
      ..roughnessFactor = 0.6
      ..metallicFactor = 0.0
      ..normalTexture = Texture2D.fromPixels(rippleNormalMap(ripples), ripples, ripples, content: TextureContent.normal)
      ..normalScale = 0.3;
  }

  void _applyLook() {
    // Stylized: saturated and warm, to match a cartoon bird. Exposure is the
    // first knob, not the last: a physical sky lighting the whole scene plus
    // a strong sun pushes every surface into the tone curve's shoulder, where
    // channel differences compress and everything reads as washed-out white.
    scene.environmentSettings = EnvironmentSettings(
      toneMapping: ToneMappingMode.aces,
      exposure: 0.62,
      colorGradingEnabled: _settings.post,
      saturation: 1.35,
      contrast: 1.18,
      brightness: 1.0,
      temperature: 0.06,
      bloomEnabled: _settings.post,
      bloomThreshold: 1.3,
      bloomIntensity: 0.12,
      bloomScatter: 0.7,
      ambientOcclusionEnabled: _settings.post,
      ambientOcclusionHalfResolution: true,
      ambientOcclusionIntensity: 1.0,
      vignetteEnabled: _settings.post,
      vignetteIntensity: 0.25,
      // Just enough haze to separate the far hills; any more and the whole
      // frame goes milky. Its colour (and the exposure) follow the time of
      // day, see [_applyTimeOfDay].
      fogEnabled: true,
      fogMode: FogMode.exponential,
      fogDensity: 0.0011,
    );

    // Low sun: a steep one hides every shadow under the object that casts
    // it, which reads as flat. Raking light gives the hills relief and puts
    // Dash's own shadow where the audience can see it. [TimeOfDay] keeps
    // every phase, night included, inside that range.
    final light = scene.directionalLight ??= DirectionalLight(
      shadowSoftness: 0.12,
      shadowMaxDistance: 120,
    );
    light
      ..castsShadow = _settings.shadows
      ..contactShadows = _settings.shadows;

    // A gradient sky, not the physical one: at the exposure this scene needs,
    // Rayleigh scattering lands on grey-white, and a cartoon wants a stated
    // blue. The same source drives the IBL, so ambient light picks up the sky
    // colour instead of fighting it.
    if (_sky == null) {
      final sky = _sky = GradientSkySource(sunSharpness: 320);
      scene.skybox = _skybox = Skybox(sky, intensity: 1.6);
      // The sky's lighting is baked only when it has drifted (see
      // [SkyBakeGate]), at a resolution that keeps each time-sliced bake
      // cheap: a smooth gradient sky loses nothing at 32 px faces.
      final cycling = clock.cycleSeconds > 0;
      _skyEnvironment = SkyEnvironment(
        sky,
        faceResolution: cycling ? 32 : 128,
        equirectWidth: cycling ? 128 : 512,
      );
    }
    // `environmentSettings` above carries a skybox and a sky environment of
    // its own (null here) and applies them: without this, every settings
    // change after the first (the load calibration makes several) left the
    // sky black and the scene lit by the default studio map.
    scene.skybox = _skybox;
    scene.skyEnvironment = _skyEnvironment;
    scene.renderScale = _settings.renderScale;
    _applyTimeOfDay();
  }

  /// Copies the current [TimeOfDay] look onto the light, sky, fog and
  /// exposure: field writes, plus a sky bake when [SkyBakeGate] says the
  /// lighting has fallen behind what is on screen.
  void _applyTimeOfDay() {
    final look = clock.look;
    if (_bakeGate.shouldBake(look)) scene.skyEnvironment?.invalidate();
    final light = scene.directionalLight;
    if (light != null) {
      light
        ..direction = -look.towardLight
        ..color = look.lightColor
        ..intensity = look.lightIntensity;
    }
    final sky = _sky;
    if (sky != null) {
      sky
        ..zenithColor = look.zenith
        ..horizonColor = look.horizon
        ..groundColor = look.ground
        ..sunDirection = look.towardLight
        ..sunColor = look.disc;
    }
    _skybox?.intensity = look.skyIntensity;
    scene.fog.color = look.fog;
    scene.exposure = look.exposure;
  }

  /// Presenter key: fast-forward to the next part of the day. [tick] moves
  /// the clock there and reports the phase change as it crosses into it.
  void skipDayPhase() => clock.skipToNextPhase();

  void _reportDaytime(DayPhase phase) {
    events.add(SceneEvent(
      at: elapsed,
      kind: 'daytime',
      subject: 'sky',
      object: phase.name,
      detail: '${TimeOfDay.narration(phase)}, ${distance.round()} m into the run, in the $biomeLabel',
    ));
  }

  Future<void> _loadDash() async {
    final model = await Node.fromGlbAsset('assets/models/dash.glb');
    for (final name in const ['Idle', 'Run', 'JumpStart', 'JumpLand']) {
      final animation = model.findAnimationByName(name);
      if (animation == null) continue;
      final clip = model.createAnimationClip(animation)
        ..loop = name == 'Idle' || name == 'Run'
        ..weight = name == 'Idle' ? 1 : 0
        ..play();
      _clips[name] = clip;
    }
    // A pivot keeps our transform separate from the model's own hierarchy,
    // whose node names the animation channels bind to. The model is 2.92
    // units tall, which would make Dash a three-metre bird next to 5 m trees.
    final pivot = Node(name: 'dash')
      ..add(model)
      ..scale = vm.Vector3.all(dashScale);
    _root.add(pivot);
    _dashPivot = pivot;
    _placeDash();
  }

  // --- chunks --------------------------------------------------------------

  int get _currentIndex => (z / world.chunkLength).floor();

  /// Starts at most one build per call (nearest first), and none while
  /// [_maxBuilding] are in flight, so a frame never waits on two uploads.
  void _pumpChunks() {
    for (var i = _currentIndex - 1; i <= _currentIndex + _settings.viewChunks; i++) {
      if (_building.length >= _maxBuilding) break;
      if (i < 0 || _chunks[i]?.gen == _worldGen || _building[i] == _worldGen) continue;
      _buildChunk(i);
      break;
    }
    final keepFrom = _currentIndex - 1;
    final keepTo = _currentIndex + _settings.viewChunks;
    for (final index in _chunks.keys.toList()) {
      if (index < keepFrom || index > keepTo) {
        _root.remove(_chunks.remove(index)!.node);
      }
    }
  }

  Future<void> _buildChunk(int index) async {
    final gen = _worldGen;
    _building[index] = gen;
    try {
      final sw = Stopwatch()..start();
      final build = await compute(
        buildChunk,
        ChunkRequest(
          index: index,
          config: world.copyWith(propDensity: _settings.propDensity),
          biomeLength: field.biomeLength,
          trail: trail,
        ),
      );
      final genMs = sw.elapsedMilliseconds;
      // Disposed, or superseded by a build for a newer density.
      if (_building[index] != gen) return;
      _building.remove(index);
      if (gen != _worldGen) return; // stale; the pump builds it again
      sw.reset();
      _installChunk(build, gen);
      _genMs.add(genMs);
      _installMs.add(sw.elapsedMilliseconds);
    } catch (e) {
      if (_building[index] == gen) _building.remove(index);
      events.add(SceneEvent(
        at: elapsed,
        kind: 'warning',
        subject: 'chunk-$index',
        object: 'world',
        detail: 'failed to build: $e',
      ));
    }
  }

  void _installChunk(ChunkBuild build, int gen) {
    final mesh = build.mesh;
    final geometry = MeshGeometry.fromMeshData(MeshData.build(
      positions: mesh.positions,
      normals: mesh.normals,
      texCoords: mesh.texCoords,
      colors: mesh.colors,
      indices: mesh.indices,
    ));
    final ground = Node(
      name: 'chunk-${build.spec.index}',
      mesh: Mesh(geometry, PhysicallyBasedMaterial()..roughnessFactor = 1.0..metallicFactor = 0.0),
    );

    final byKind = <PropKind, List<PropInstance>>{};
    for (final p in build.spec.props) {
      byKind.putIfAbsent(p.kind, () => []).add(p);
    }
    for (final entry in byKind.entries) {
      final (geometry, material) = _propAssets[entry.key]!;
      final instanced = InstancedMesh(geometry: geometry, material: material, cullInstances: true);
      for (final p in entry.value) {
        // Cones and spheres are modelled around the origin, so lift them by
        // half their height to stand on the ground instead of sinking in.
        final lift = switch (entry.key) {
          PropKind.tree => 2.6 * p.scale,
          PropKind.bush => 0.45 * p.scale,
          PropKind.rock => 0.3 * p.scale,
          PropKind.grass => 0.35 * p.scale,
        };
        final m = _trs(
          vm.Vector3(p.x, p.y + lift, p.z),
          vm.Quaternion.axisAngle(vm.Vector3(0, 1, 0), p.yaw),
          vm.Vector3.all(p.scale),
        );
        instanced.addInstance(m, color: vm.Vector4(p.tint.$1, p.tint.$2, p.tint.$3, 1));
      }
      ground.add(Node(name: 'props-${entry.key.name}')..addComponent(InstancedMeshComponent(instanced)));
    }

    for (final o in build.spec.obstacles) {
      // A stream is where the trail meets a river; the river draws itself.
      if (o.kind != ObstacleKind.stream) ground.add(_obstacleNode(o));
    }
    for (final w in build.mesh.water) {
      final geometry = MeshGeometry.fromMeshData(MeshData.build(
        positions: w.positions,
        normals: w.normals,
        texCoords: w.texCoords,
        indices: w.indices,
      ));
      ground.add(Node(name: 'river-${w.riverId}', mesh: Mesh(geometry, _riverMaterial)));
    }

    final replaced = _chunks[build.spec.index];
    if (replaced != null) _root.remove(replaced.node);
    _root.add(ground);
    _chunks[build.spec.index] = _Chunk(build.spec, ground, gen);
    notifyListeners();
  }

  Node _obstacleNode(Obstacle o) {
    final p = trail.pointAt(o.z, lateral: o.lateral);
    final groundY = field.height(p.x, p.y, zHint: o.z);
    final heading = trail.headingAt(o.z);
    switch (o.kind) {
      case ObstacleKind.log:
        // The cylinder stands along +Y; lay it across the trail by turning it
        // onto +X first, then yawing it with the trail.
        final lie = vm.Quaternion.axisAngle(vm.Vector3(0, 1, 0), heading) *
            vm.Quaternion.axisAngle(vm.Vector3(0, 0, 1), -math.pi / 2);
        return Node(name: 'log-${o.id}', mesh: Mesh(_logGeometry, _woodMaterial))
          ..position = vm.Vector3(p.x, groundY + 0.34, p.y)
          ..rotation = lie
          ..scale = vm.Vector3(1, o.halfSpan * 2, 1);
      case ObstacleKind.boulder:
        return Node(name: 'boulder-${o.id}', mesh: Mesh(_boulderGeometry, _stoneMaterial))
          ..position = vm.Vector3(p.x, groundY + o.halfSpan * 0.55, p.y)
          ..scale = vm.Vector3(o.halfSpan, o.halfSpan * 0.85, o.halfSpan);
      case ObstacleKind.stream:
        return Node(name: 'stream-${o.id}');
    }
  }

  // --- frame ---------------------------------------------------------------

  void tick(double dt) {
    if (!_built || paused) return;
    final step = dt.clamp(0.0, 0.05); // a long frame must not teleport Dash
    elapsed += step;

    final next = nextObstacle();
    final intent = pilot.update(
      dt: step,
      z: z,
      lateral: lateral,
      halfWidth: trail.halfWidth,
      speed: speed,
      airborne: airborne,
      next: next,
    );

    if (pilot.manual != _presenterWasDriving) {
      _presenterWasDriving = pilot.manual;
      events.add(SceneEvent(
        at: elapsed,
        kind: pilot.manual ? 'takeover' : 'autopilot',
        subject: pilot.manual ? 'presenter' : 'autopilot',
        object: 'dash',
        detail: pilot.manual ? 'the presenter grabbed the controls' : 'the autopilot is back in charge',
      ));
    }

    final targetLateral = (_manualLateral ?? _detourLateral ?? intent?.lateral ?? lateral)
        .clamp(-trail.halfWidth + 0.4, trail.halfWidth - 0.4);
    lateral += (targetLateral - lateral).clamp(-6 * step, 6 * step);
    _keepOutOfBoulders();

    if ((_manualJump || (intent?.jump ?? false)) && !airborne) {
      _vy = 6.4;
      airborne = true;
      jumps++;
      _clips['JumpStart']?.seek(0);
      _clips['JumpStart']?.play();
    }
    _manualJump = false;

    // Speed ramps with distance and recovers after a stumble.
    if (_stumbleTimer > 0) {
      _stumbleTimer -= step;
    } else {
      final wanted = math.min(topSpeed, baseSpeed + distance / 260);
      speed += (wanted - speed) * math.min(1, step * 1.5);
    }
    shakeOffset = _shake.update(step);

    final turned = clock.advance(step);
    if (turned != null) _reportDaytime(turned);
    _applyTimeOfDay();
    // Ripples run down every river (u is along the course), with a little
    // drift across so the pattern never looks stamped.
    _riverMaterial.normalTextureTransform = TextureTransform(
      offset: vm.Vector2(elapsed * 0.25, elapsed * 0.02),
    );

    // Pressed against a boulder: no way forward until Dash is beside it.
    final blocked = _blockedBy;
    if (blocked != null) _blockedFor += step;
    // The time limit is a safety net: a detour that cannot finish must not
    // stop the act.
    if (blocked != null && (blocked.clearsAt(lateral) || _blockedFor > 2.5)) {
      _blockedBy = null;
      _detourLateral = null;
    }
    if (_blockedBy == null) {
      final slope = trail.slopeAt(z);
      final dz = speed * step / math.sqrt(1 + slope * slope);
      z += dz;
      distance += speed * step;
    }

    final p = trail.pointAt(z, lateral: lateral);
    final groundY = field.height(p.x, p.y, zHint: z);
    if (airborne) {
      _vy -= 17.5 * step;
      y += _vy * step;
      if (y <= groundY) {
        y = groundY;
        airborne = false;
        _vy = 0;
        _clips['JumpStart']?.weight = 0;
        _clips['JumpLand']?.seek(0);
        _clips['JumpLand']?.play();
      }
    } else {
      y = groundY;
    }

    _resolveObstacles();
    _countPenetrations();
    _updateProgress();
    _updateAnimation(step);
    _placeDash();
    _pumpChunks();
    notifyListeners();
  }

  void _updateAnimation(double dt) {
    final run = _clips['Run'];
    final idle = _clips['Idle'];
    final jump = _clips['JumpStart'];
    final land = _clips['JumpLand'];
    final moving = (speed / topSpeed).clamp(0.0, 1.0);
    final air = airborne ? 1.0 : 0.0;
    run?.weight = (1 - air) * moving;
    idle?.weight = (1 - air) * (1 - moving);
    jump?.weight = air;
    // Feet keep up with the ground instead of skating.
    run?.playbackTimeScale = 0.55 + speed / 14;
    if (land != null && land.weight > 0 && !airborne) {
      land.weight = math.max(0, land.weight - dt * 3);
    } else if (land != null && !airborne && land.playing) {
      land.weight = 0.6;
    }
  }

  void _placeDash() {
    final pivot = _dashPivot;
    if (pivot == null) return;
    final p = trail.pointAt(z, lateral: lateral);
    pivot.position = vm.Vector3(p.x, y, p.y);
    // Face along the trail, leaning into the direction Dash is stepping.
    pivot.rotation = vm.Quaternion.axisAngle(vm.Vector3(0, 1, 0), trail.headingAt(z) + math.pi);
    pivot.scale = vm.Vector3.all(dashScale);
  }

  /// The next obstacle Dash has not dealt with yet.
  Obstacle? nextObstacle({double within = 40}) {
    Obstacle? best;
    for (final chunk in _chunks.values) {
      for (final o in chunk.spec.obstacles) {
        if (o.z <= z || o.z > z + within || _resolved.contains(o.id)) continue;
        if (best == null || o.z < best.z) best = o;
      }
    }
    return best;
  }

  /// Where Dash's body meets [o] along the trail. Boulders and logs are met
  /// at their front face, not their centre line: deciding at the centre let
  /// Dash run half way into a boulder before anything happened.
  double _contactZ(Obstacle o) => switch (o.kind) {
        ObstacleKind.boulder => o.z - o.halfSpan * 0.95 - _dashRadius,
        ObstacleKind.log => o.z - _logRadius - _dashRadius,
        ObstacleKind.stream => o.z,
      };

  bool _above(Obstacle o) {
    if (!airborne) return false;
    // pointAt gives the world point (x, z); o.z is a trail z, not a world z.
    final p = trail.pointAt(o.z, lateral: lateral);
    return y > field.height(p.x, p.y, zHint: o.z) + o.height * 0.6;
  }

  void _resolveObstacles() {
    for (final chunk in _chunks.values) {
      for (final o in chunk.spec.obstacles) {
        if (_resolved.contains(o.id) || z < _contactZ(o) || o.z < _startZ) continue;
        switch (o.kind) {
          case ObstacleKind.boulder:
            // Met head-on: bump. Beside it: decided once Dash is past.
            if (!o.clearsAt(lateral) && !_above(o) && z <= o.z) {
              _resolved.add(o.id);
              _bump(o);
            } else if (z >= o.z) {
              _resolved.add(o.id);
              _cleared(o, above: _above(o));
            }
          case ObstacleKind.log:
            _resolved.add(o.id);
            if (_above(o)) {
              _cleared(o, above: true);
            } else {
              _trip(o);
            }
          case ObstacleKind.stream:
            _resolved.add(o.id);
            if (_above(o)) {
              _cleared(o, above: true);
            } else {
              _stumble(o, 'splashed through it');
            }
        }
      }
    }
  }

  void _cleared(Obstacle o, {required bool above}) {
    obstaclesCleared++;
    cleanStreak++;
    if (cleanStreak > bestStreak) bestStreak = cleanStreak;
    if (above) jumpsCleared++;
    final margin = (lateral - o.lateral).abs() - o.halfSpan;
    if (!above && margin < 0.55) {
      nearMisses++;
      _shake.addTrauma(0.2);
      events.add(SceneEvent(
        at: elapsed,
        kind: 'near_miss',
        subject: 'dash',
        object: o.label,
        detail: 'squeezed past with ${margin.toStringAsFixed(2)} m to spare',
      ));
    } else {
      events.add(SceneEvent(
        at: elapsed,
        kind: above ? 'jump_cleared' : 'dodged',
        subject: 'dash',
        object: o.label,
        detail: above ? 'jumped it cleanly' : 'ran round it',
      ));
    }
  }

  /// Ran into a boulder: knocked back off its face, then has to step round
  /// it before running on (see [tick]).
  void _bump(Obstacle o) {
    _stumble(o, 'ran smack into it');
    speed = 1.5;
    z = _contactZ(o) - 0.25;
    _blockedBy = o;
    _blockedFor = 0;
    _detourLateral = _clearOffset(o);
  }

  /// Missed the jump at a log: scrambles over it in a clumsy low hop
  /// instead of walking through the trunk.
  void _trip(Obstacle o) {
    _stumble(o, 'tripped on it and scrambled over');
    _vy = 6.0;
    airborne = true;
    _clips['JumpStart']?.seek(0);
    _clips['JumpStart']?.play();
  }

  /// The nearest lateral offset inside the corridor that passes [o].
  double _clearOffset(Obstacle o) {
    final limit = trail.halfWidth - 0.4;
    final side = o.halfSpan + 0.5; // a hair wider than [Obstacle.clearsAt]
    final options = [
      if (o.lateral + side <= limit) o.lateral + side,
      if (o.lateral - side >= -limit) o.lateral - side,
    ];
    // The world guarantees a way round every boulder, but not necessarily
    // with this margin: fall back to the far edge of the corridor.
    if (options.isEmpty) return o.lateral >= 0 ? -limit : limit;
    options.sort((a, b) => (a - lateral).abs().compareTo((b - lateral).abs()));
    return options.first;
  }

  /// Beside a boulder, Dash cannot step into its flank: whoever is steering
  /// (the pilot already heading for the next obstacle, or the presenter) is
  /// held at its edge until he is past it.
  void _keepOutOfBoulders() {
    for (final chunk in _chunks.values) {
      for (final o in chunk.spec.obstacles) {
        if (o.kind != ObstacleKind.boulder || identical(o, _blockedBy)) continue;
        if (z < _contactZ(o) || z > o.z + o.halfSpan * 0.95 + _dashRadius) continue;
        final side = o.halfSpan + 0.45;
        final d = lateral - o.lateral;
        if (d.abs() < side) lateral = o.lateral + (d >= 0 ? side : -side);
      }
    }
  }

  void _countPenetrations() {
    if (airborne) return;
    for (final chunk in _chunks.values) {
      for (final o in chunk.spec.obstacles) {
        if (o.kind != ObstacleKind.boulder || (z - o.z).abs() > 3) continue;
        final dz = z - o.z, dl = lateral - o.lateral;
        final inside = o.halfSpan * 0.8;
        if (dz * dz + dl * dl < inside * inside && _penetrated.add(o.id)) penetrations++;
      }
    }
  }

  void _stumble(Obstacle o, String what) {
    stumbles++;
    final ended = cleanStreak;
    cleanStreak = 0;
    _stumbleTimes.add(elapsed);
    speed = math.max(3.5, speed * 0.45);
    _stumbleTimer = 0.9;
    _shake.addTrauma(0.75);
    events.add(SceneEvent(
      at: elapsed,
      kind: 'stumble',
      subject: 'dash',
      object: o.label,
      detail: ended >= 3 ? '$what, ending a run of $ended clean' : what,
    ));
  }

  void _updateProgress() {
    final biome = field.biomeAt(z);
    if (biome.label != biomeLabel) {
      events.add(SceneEvent(
        at: elapsed,
        kind: 'biome',
        subject: 'dash',
        object: biome.label,
        detail: 'left the $biomeLabel and entered the ${biome.label} at ${distance.round()} m',
      ));
      biomeLabel = biome.label;
    }
    final milestone = (distance / 250).floor();
    if (milestone > _milestone) {
      _milestone = milestone;
      events.add(SceneEvent(
        at: elapsed,
        kind: 'milestone',
        subject: 'dash',
        object: '${milestone * 250} m',
        detail: '${milestone * 250} m in ${elapsed.round()} s, $obstaclesCleared cleared, $stumbles stumbles, '
            'now at ${speed.toStringAsFixed(1)} m/s in the $biomeLabel',
      ));
    }
  }

  // --- presenter controls --------------------------------------------------

  void steer(double direction) {
    pilot.takeOver();
    _manualLateral = (lateral + direction * 2.2).clamp(-trail.halfWidth + 0.4, trail.halfWidth - 0.4);
  }

  void jump() {
    pilot.takeOver();
    _manualJump = true;
  }

  void releaseControl() {
    _manualLateral = null;
  }

  void togglePause() {
    paused = !paused;
    notifyListeners();
  }

  void applySettings(RunnerSettings settings) {
    final rebuildWorld = settings.propDensity != _settings.propDensity;
    _settings = settings;
    _applyLook();
    if (rebuildWorld) _worldGen++;
    _pumpChunks();
    notifyListeners();
  }

  /// The trail half of the narrator's situation: what is coming and how Dash
  /// is doing. The screen adds the hardware half.
  RunSituation situation({
    String? backend,
    double? speedShare,
    double? frameBudget,
    double? cpuLoad,
    double? gpuLoad,
    double? npuLoad,
  }) {
    final ahead = <(double, Obstacle)>[];
    for (final chunk in _chunks.values) {
      for (final o in chunk.spec.obstacles) {
        if (o.z <= z || o.z > z + 40 || _resolved.contains(o.id)) continue;
        ahead.add((o.z - z, o));
      }
    }
    ahead.sort((a, b) => a.$1.compareTo(b.$1));

    double? riverIn;
    final crossings = field.riversBetween(z + 40, z + 80);
    if (crossings.isNotEmpty) riverIn = crossings.first.zCross - z;

    final here = trail.pointAt(z);
    final there = trail.pointAt(z + 30);
    final rise = field.height(there.x, there.y, zHint: z + 30) - field.height(here.x, here.y, zHint: z);

    String? nextBiome;
    double? nextBiomeIn;
    final boundary = ((z / field.biomeLength).floor() + 1) * field.biomeLength;
    if (boundary - z < 100) {
      nextBiome = field.biomeAt(boundary + 1).label;
      nextBiomeIn = boundary - z;
    }

    final next = ahead.isEmpty ? null : ahead.first.$2;
    final foreshadow = !pilot.manual &&
        next != null &&
        pilot.committedTo == next.id &&
        pilot.fumbling &&
        next.z - z < 25;

    _stumbleTimes.removeWhere((t) => elapsed - t > 60);
    return RunSituation(
      ahead: [for (final (d, o) in ahead.take(2)) AheadItem(o.label, d)],
      riverIn: riverIn,
      rise: rise,
      nextBiome: nextBiome,
      nextBiomeIn: nextBiomeIn,
      foreshadow: foreshadow,
      streak: cleanStreak,
      bestStreak: bestStreak,
      stumblesLastMinute: _stumbleTimes.length,
      presenterDriving: pilot.manual,
      backend: backend,
      speedShare: speedShare,
      frameBudget: frameBudget,
      cpuLoad: cpuLoad,
      gpuLoad: gpuLoad,
      npuLoad: npuLoad,
    );
  }

  int _avg(List<int> xs) => xs.isEmpty ? 0 : xs.reduce((a, b) => a + b) ~/ xs.length;

  Map<String, Object?> toJson() => {
        'distance_m': distance.round(),
        'biome': biomeLabel,
        'speed': double.parse(speed.toStringAsFixed(2)),
        'obstacles_cleared': obstaclesCleared,
        'jumps': jumps,
        'jumps_cleared': jumpsCleared,
        'near_misses': nearMisses,
        'stumbles': stumbles,
        'penetrations': penetrations,
        'chunks': liveChunks,
        'props': liveProps,
        'chunk_gen_ms': _avg(_genMs),
        'chunk_install_ms': _avg(_installMs),
        'chunk_gen_worst_ms': _genMs.isEmpty ? 0 : _genMs.reduce(math.max),
        'chunk_install_worst_ms': _installMs.isEmpty ? 0 : _installMs.reduce(math.max),
        'view_chunks': _settings.viewChunks,
        'prop_density': _settings.propDensity,
        'shadows': _settings.shadows,
        'post': _settings.post,
        'day_phase': clock.phase.name,
        'day_fraction': double.parse(clock.fraction.toStringAsFixed(3)),
        'day_cycle_s': clock.cycleSeconds,
      };

  @override
  void dispose() {
    _building.clear();
    super.dispose();
  }
}

/// One transform per instance, and a chunk places a few thousand of them, so
/// this stays a plain function on the hot path rather than a chain of matrix
/// multiplications.
vm.Matrix4 _trs(vm.Vector3 position, vm.Quaternion rotation, vm.Vector3 scale) =>
    vm.Matrix4.identity()..setFromTranslationRotationScale(position, rotation, scale);
