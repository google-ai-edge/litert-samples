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

// ignore_for_file: experimental_member_use

import 'dart:convert';
import 'dart:io';
import 'dart:ui' as ui;

import 'package:flutter/rendering.dart';
import 'package:flutter/services.dart';
import 'package:flutter/widgets.dart';
import 'package:flutter_edge_ai/flutter_edge_ai.dart';

import 'dash_pilot.dart';
import 'runner_game.dart';

/// Drives Dash from the screen alone (`ACT3_EYES_MODEL`): fixed parts of the
/// scene in front of him go to EmbeddingGemma 2, and a decider per question
/// names what each part shows. The centre of the band ahead says trail,
/// grass, rock or something to jump; the near part just over Dash says
/// whether to jump now; the whole band says which way to go round a rock. An
/// [EyesReflex] turns that into the three buttons. Nothing here reads the
/// obstacles: no positions, no distances. Besides the pixels the eyes know
/// only Dash's own place across the trail, the trail's width and whether he
/// is in the air.
class VisionPilot extends ChangeNotifier {
  VisionPilot._(this._model, this._zones, this._near, this._steer, this.sceneKey);

  /// The parts, as fractions of the frame (left, top, right, bottom). The
  /// examples in `assets/eyes/` are these crops of 1028x682 frames.
  static const parts = {
    'C': (207 / 514, 150 / 341, 307 / 514, 240 / 341),
    'N': (150 / 514, 190 / 341, 364 / 514, 262 / 341),
    'S': (107 / 514, 150 / 341, 407 / 514, 240 / 341),
  };

  /// The questions, the part each one looks at, and its answers. Each
  /// answer's examples are `assets/eyes/<question>/<answer>/*.png`.
  static const questions = {
    'zones': ('C', ['trail', 'grass', 'rock', 'jump']),
    'near': ('N', ['jump', 'run']),
    'steer': ('S', ['left', 'right', 'straight']),
  };

  /// Looks that fail in a row before the eyes give Dash back to the
  /// autopilot.
  static const maxFailures = 5;

  /// A [stop] still closing; the next [load] waits for it, because the
  /// plugin hands a second screen the same embedder.
  static Future<void>? _closing;

  final EmbeddingModel _model;
  final Decider _zones;
  final Decider _near;
  final Decider _steer;
  final GlobalKey sceneKey;

  /// Where the eyes' model runs.
  PreferredBackend? get backend => _model.activeBackend;

  /// Loads [modelPath] on [backend] and a decider per question: a bundled
  /// `assets/eyes/deciders/<model>/<question>.json` fitted on this model file
  /// (its name and size) and backend, else one fitted on the examples in
  /// `assets/eyes/`. A fitted decider is kept in the temp folder (what
  /// `toArtifact()` writes) and loaded on the next start; one fitted on
  /// another model, backend or set of examples is fitted again.
  static Future<VisionPilot> load({
    required String modelPath,
    required PreferredBackend backend,
    required GlobalKey sceneKey,
    void Function(String line)? log,
  }) async {
    await _closing;
    final w = Stopwatch()..start();
    await FlutterEdgeAi.installEmbedder().modelFromFile(modelPath).install();
    final installed = w.elapsedMilliseconds;
    final model = await FlutterEdgeAi.getActiveEmbedder(preferredBackend: backend, supportImage: true);
    log?.call('ACT3_EYES_MODEL ${model.activeBackend?.name}: install $installed ms, load ${w.elapsedMilliseconds - installed} ms');
    final built = <Decider>[];
    try {
      final manifest = await AssetManifest.loadFromAssetBundle(rootBundle);
      final file = File(modelPath);
      final cacheDir = Directory('${Directory.systemTemp.path}/intel_npu_demo_eyes')..createSync(recursive: true);

      // A decider fitted on another model or backend is passed over, and the
      // log says why.
      Future<Decider?> tryLoad(String json, String from) async {
        try {
          return await model.loadDecider(jsonDecode(json) as Map<String, Object?>);
        } on DeciderSpaceMismatchException catch (e) {
          log?.call('ACT3_EYES_SKIPPED $from: ${e.message}');
          return null;
        }
      }

      Future<Decider> decider(String question) async {
        // A bundled file that does not parse is a broken build: it fails.
        for (final bundled in manifest.listAssets().where((a) => a.startsWith('assets/eyes/deciders/') && a.endsWith('/$question.json'))) {
          final d = await tryLoad(await rootBundle.loadString(bundled), bundled);
          if (d != null) {
            log?.call('ACT3_EYES_LOADED $question from $bundled');
            return d;
          }
        }
        final (_, answers) = questions[question]!;
        final options = <DeciderOption>[];
        // FNV-1a over each example's path and bytes: a relabelled or replaced
        // example names another cache file.
        var hash = 0x811c9dc5;
        void mix(List<int> data) {
          for (final b in data) {
            hash = ((hash ^ b) * 0x01000193) & 0xffffffff;
          }
        }

        for (final answer in answers) {
          final prefix = 'assets/eyes/$question/$answer/';
          final files = manifest.listAssets().where((a) => a.startsWith(prefix)).toList()..sort();
          if (files.isEmpty) throw StateError('no examples under $prefix');
          final images = <Uint8List>[];
          for (final f in files) {
            final bytes = (await rootBundle.load(f)).buffer.asUint8List();
            mix(utf8.encode(f));
            mix(bytes);
            images.add(bytes);
          }
          options.add(DeciderOption(answer, examples: [for (final i in images) EmbeddingInput.image(i)]));
        }
        final name = file.uri.pathSegments.last;
        final examples = hash.toRadixString(16).padLeft(8, '0');
        final cache = File('${cacheDir.path}/$question-${backend.name}-$name-${file.lengthSync()}-$examples.json');
        if (cache.existsSync()) {
          // A damaged cache file is fitted again.
          try {
            final d = await tryLoad(cache.readAsStringSync(), cache.path);
            if (d != null) {
              log?.call('ACT3_EYES_LOADED $question from ${cache.path}');
              return d;
            }
          } on FormatException catch (e) {
            log?.call('ACT3_EYES_SKIPPED ${cache.path}: $e');
          } on TypeError catch (e) {
            log?.call('ACT3_EYES_SKIPPED ${cache.path}: $e');
          }
        }
        final w = Stopwatch()..start();
        final d = await model.createDecider(options: options);
        log?.call('ACT3_EYES_FITTED $question from ${options.fold(0, (n, o) => n + o.examples.length)} examples in ${w.elapsedMilliseconds} ms');
        // The cache only saves the next start a fit: failing to write it
        // keeps the decider.
        try {
          final part = File('${cache.path}.part')..writeAsStringSync(jsonEncode(d.toArtifact()));
          part.renameSync(cache.path);
        } on FileSystemException catch (e) {
          log?.call('ACT3_EYES_CACHE_NOT_WRITTEN ${cache.path}: $e');
        }
        return d;
      }

      for (final q in questions.keys) {
        built.add(await decider(q));
      }
      return VisionPilot._(model, built[0], built[1], built[2], sceneKey);
    } catch (_) {
      for (final d in built) {
        await d.close();
      }
      await model.close();
      rethrow;
    }
  }

  /// What the last look saw, per part, for the overlay and the log.
  Map<String, String> seen = const {};
  int looks = 0;

  /// Why the eyes stopped driving: [maxFailures] looks failed in a row. Null
  /// while they drive.
  Object? error;

  final EyesReflex _reflex = EyesReflex();
  bool _stopped = false;
  Future<void>? _loop;

  /// Looks again and again until [stop], or until looks keep failing; then
  /// the eyes set [error] and hand Dash back to the autopilot. Every look
  /// reads the near part and the centre; the whole band only while a rock is
  /// in front. While the presenter drives or the game is paused the eyes
  /// rest, and they start afresh when they are back.
  Future<void> run(RunnerGame game, void Function(String line) log) => _loop ??= _run(game, log);

  Future<void> _run(RunnerGame game, void Function(String line) log) async {
    var failures = 0;
    while (!_stopped) {
      if (game.pilot.manual || game.paused) {
        _reflex.reset();
        await Future<void>.delayed(const Duration(milliseconds: 100));
        continue;
      }
      final w = Stopwatch()..start();
      try {
        final n = await _see('N', _near);
        final c = await _see('C', _zones);
        if (_stopped) break;
        if (n == null || c == null) {
          await Future<void>.delayed(const Duration(milliseconds: 50));
          continue;
        }
        looks++;
        final now = {'C': c.id, 'N': n.id};
        var act = _reflex.look(near: n.id, centre: c.id, airborne: game.airborne);
        // A rock in front: the band says left, right, or hold the line.
        String? s;
        if (_reflex.wantsSteer(act)) {
          final steer = await _see('S', _steer);
          if (_stopped) break;
          if (steer != null) {
            now['S'] = steer.id;
            s = steer.mark;
            act = _reflex.steer(steer.id, act: act, lateral: game.lateral);
          }
        }
        failures = 0;
        seen = now;
        notifyListeners();
        log('ACT3_EYES ${game.elapsed.toStringAsFixed(2)} ${c.mark}|${n.mark}|${s ?? '-'} ${w.elapsedMilliseconds}ms ${act.isEmpty ? '.' : act}');
      } catch (e) {
        if (_stopped) break;
        log('ACT3_EYES_FAILED $e');
        _reflex.reset();
        if (++failures >= maxFailures) {
          // The eyes stop driving, and say so: the autopilot takes over.
          error = e;
          game.eyes = null;
          seen = const {};
          notifyListeners();
          log('ACT3_EYES_STOPPED after $failures failed looks');
          return;
        }
        await Future<void>.delayed(const Duration(milliseconds: 200));
      }
    }
  }

  /// Called by the game every frame while the presenter is not driving: its
  /// intent replaces the scripted pilot's, which still runs to time out the
  /// presenter's hold.
  PilotIntent intent(RunnerGame game) => _reflex.intent(lateral: game.lateral, halfWidth: game.trail.halfWidth);

  /// One part of the scene, read back alone (not the whole frame), and what
  /// the decider calls it. When the decider abstains (not sure enough) the
  /// eyes still take its best guess, marked not sure in the log: the reflex
  /// needs an answer every look, and two looks in a row must agree before it
  /// acts.
  Future<_Look?> _see(String part, Decider decider) async {
    final boundary = sceneKey.currentContext?.findRenderObject() as RenderRepaintBoundary?;
    if (boundary == null || !boundary.hasSize) return null;
    final (l, t, r, b) = parts[part]!;
    final size = boundary.size;
    final rect = Rect.fromLTRB(
      (l * size.width).floorToDouble(),
      (t * size.height).floorToDouble(),
      (r * size.width).floorToDouble(),
      (b * size.height).floorToDouble(),
    );
    // ignore: invalid_use_of_protected_member
    final layer = boundary.layer! as OffsetLayer;
    final image = await layer.toImage(rect);
    final png = await image.toByteData(format: ui.ImageByteFormat.png);
    image.dispose();
    final answer = await decider.choose(DeciderInput(EmbeddingInput.image(png!.buffer.asUint8List())), topK: 1);
    final choice = answer.choice;
    return choice != null ? (id: choice, sure: true) : (id: answer.ranked.first.id, sure: false);
  }

  /// Stops looking, waits for the look in flight, and closes the deciders
  /// and the model.
  Future<void> stop() {
    _stopped = true;
    final done = () async {
      await _loop;
      await _zones.close();
      await _near.close();
      await _steer.close();
      await _model.close();
    }();
    _closing = done.then((_) {}, onError: (_) {});
    return done;
  }
}

typedef _Look = ({String id, bool sure});

extension on _Look {
  /// The answer for the log: a guess the decider was not sure of ends in `?`.
  String get mark => sure ? id : '$id?';
}

/// The eyes' reflex: what the answers of the last looks mean for the three
/// buttons. An answer counts once two looks in a row agree; a rock in front
/// sends Dash to the side the band names, and three looks without one let
/// him go back.
class EyesReflex {
  final List<String> _centre = [];
  final List<String> _nearby = [];
  bool _jumped = false;
  bool _jumpNow = false;
  double? _side; // -1 left, 1 right, while passing a rock
  double? _hold; // the line to hold while the band says straight past a rock
  int _clearLooks = 0;

  /// Forgets everything: the eyes start afresh after the presenter drove,
  /// after a pause and after a failed look.
  void reset() {
    _centre.clear();
    _nearby.clear();
    _jumped = false;
    _jumpNow = false;
    _side = null;
    _hold = null;
    _clearLooks = 0;
  }

  /// Takes one look's near and centre answers. Returns what the reflex does:
  /// `jump`, `pick` (a rock: ask the band which way), the way it keeps
  /// passing a rock, or nothing.
  String look({required String near, required String centre, required bool airborne}) {
    for (final (list, v) in [(_nearby, near), (_centre, centre)]) {
      list.add(v);
      if (list.length > 2) list.removeAt(0);
    }
    bool twice(List<String> seen, String l) => seen.length == 2 && seen.every((c) => c == l);
    var act = '';
    // The near part says jump: jump, once, until it says run again.
    if (twice(_nearby, 'jump') && !_jumped && !airborne) {
      _jumped = true;
      _jumpNow = true;
      act = 'jump';
    } else if (twice(_nearby, 'run')) {
      _jumped = false;
    }
    // A rock in front: the band picks the way round.
    if (twice(_centre, 'rock')) {
      _clearLooks = 0;
      if (_side == null && _hold == null) return 'pick';
      act = _side == null ? 'straight' : (_side! < 0 ? 'left' : 'right');
    } else if ((_side != null || _hold != null) && _centre.last != 'rock' && ++_clearLooks >= 3) {
      _side = null;
      _hold = null;
    }
    return act;
  }

  /// Whether this look also reads the band: a rock to pick a way round, or
  /// one being passed.
  bool wantsSteer(String act) => act == 'pick' || _side != null || _hold != null;

  /// Takes the band's answer, `left`, `right` or `straight`. Returns [act],
  /// or the answer when the look was a new rock (`pick`).
  String steer(String answer, {required String act, required double lateral}) {
    if (answer == 'straight') {
      _hold ??= lateral;
      _side = null;
    } else {
      _side = answer == 'left' ? -1.0 : 1.0;
      _hold = null;
    }
    return act == 'pick' ? answer : act;
  }

  /// The game's intent for this frame. Passing a rock: all the way to that
  /// edge of the trail; holding a line: that line; otherwise drifting back
  /// towards the centre.
  PilotIntent intent({required double lateral, required double halfWidth}) {
    final jump = _jumpNow;
    _jumpNow = false;
    final side = _side;
    final target = side != null ? side * (halfWidth - 0.4) : _hold ?? lateral * 0.98;
    return PilotIntent(lateral: target, jump: jump);
  }
}
