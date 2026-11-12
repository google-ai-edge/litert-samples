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
/// whether to jump now; the whole band says which way to go round a rock. A
/// reflex turns that into the three buttons. Nothing here reads the world:
/// no obstacle, no distance, only pixels and Dash's own place on the trail.
class VisionPilot extends ChangeNotifier {
  VisionPilot._(this._model, this._zones, this._near, this._steer, this.sceneKey);

  /// The parts, as fractions of the frame (left, top, right, bottom). The
  /// examples in `assets/eyes/` are these crops of 1028x681 frames.
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

  final EmbeddingModel _model;
  final Decider _zones;
  final Decider _near;
  final Decider _steer;
  final GlobalKey sceneKey;

  /// Loads [modelPath] on [backend] and a decider per question: a bundled
  /// `assets/eyes/deciders/<model>/<question>.json` fitted on this model and
  /// backend (Google's and the fine-tuned NPU files), else one fitted on the
  /// examples in `assets/eyes/`. A fitted decider is kept in the temp folder
  /// (what `toArtifact()` writes) and loaded on the next start; one fitted on
  /// another model, backend or set of examples is fitted again.
  static Future<VisionPilot> load({
    required String modelPath,
    required PreferredBackend backend,
    required GlobalKey sceneKey,
    void Function(String line)? log,
  }) async {
    final w = Stopwatch()..start();
    await FlutterEdgeAi.installEmbedder().modelFromFile(modelPath).install();
    final installed = w.elapsedMilliseconds;
    final model = await FlutterEdgeAi.getActiveEmbedder(preferredBackend: backend, supportImage: true);
    log?.call('ACT3_EYES_MODEL ${model.activeBackend?.name}: install $installed ms, load ${w.elapsedMilliseconds - installed} ms');
    final manifest = await AssetManifest.loadFromAssetBundle(rootBundle);
    final file = File(modelPath);
    final cacheDir = Directory('${Directory.systemTemp.path}/intel_npu_demo_eyes')..createSync(recursive: true);
    // A decider fitted on another model or backend, or a damaged file, is
    // passed over.
    Future<Decider?> load(String json) async {
      try {
        return await model.loadDecider(jsonDecode(json) as Map<String, Object?>);
      } on DeciderSpaceMismatchException {
        return null;
      } on FormatException {
        return null;
      }
    }

    Future<Decider> decider(String question) async {
      for (final bundled in manifest.listAssets().where((a) => a.startsWith('assets/eyes/deciders/') && a.endsWith('/$question.json'))) {
        final d = await load(await rootBundle.loadString(bundled));
        if (d != null) {
          log?.call('ACT3_EYES_LOADED $question from $bundled');
          return d;
        }
      }
      final (_, answers) = questions[question]!;
      final options = <DeciderOption>[];
      var bytes = 0;
      for (final answer in answers) {
        final prefix = 'assets/eyes/$question/$answer/';
        final files = manifest.listAssets().where((a) => a.startsWith(prefix)).toList()..sort();
        if (files.isEmpty) throw StateError('no examples under $prefix');
        final images = [for (final f in files) (await rootBundle.load(f)).buffer.asUint8List()];
        bytes += images.fold(0, (n, i) => n + i.length);
        options.add(DeciderOption(answer, examples: [for (final i in images) EmbeddingInput.image(i)]));
      }
      final name = file.uri.pathSegments.last;
      final cache = File('${cacheDir.path}/$question-${backend.name}-$name-${file.lengthSync()}-$bytes.json');
      if (cache.existsSync()) {
        final d = await load(cache.readAsStringSync());
        if (d != null) {
          log?.call('ACT3_EYES_LOADED $question from ${cache.path}');
          return d;
        }
      }
      final w = Stopwatch()..start();
      final d = await model.createDecider(options: options);
      cache.writeAsStringSync(jsonEncode(d.toArtifact()));
      log?.call('ACT3_EYES_FITTED $question from ${options.fold(0, (n, o) => n + o.examples.length)} examples in ${w.elapsedMilliseconds} ms');
      return d;
    }

    return VisionPilot._(model, await decider('zones'), await decider('near'), await decider('steer'), sceneKey);
  }

  /// What the last look saw, per part, for the overlay and the log.
  Map<String, String> seen = const {};
  int looks = 0;

  bool _stopped = false;
  final List<String> _centre = [];
  final List<String> _nearby = [];
  bool _jumped = false;
  bool _jumpNow = false;
  double? _side; // -1 left, 1 right, while passing a rock
  double? _hold; // the line to hold while the band says straight past a rock
  int _clearLooks = 0;

  /// Looks again and again until [stop]. Every look reads the near part and
  /// the centre; the whole band only while a rock is in front.
  Future<void> run(RunnerGame game, void Function(String line) log) async {
    while (!_stopped) {
      final w = Stopwatch()..start();
      try {
        final n = await _see('N', _near);
        final c = await _see('C', _zones);
        if (n == null || c == null) {
          await Future<void>.delayed(const Duration(milliseconds: 50));
          continue;
        }
        for (final (list, v) in [(_nearby, n), (_centre, c)]) {
          list.add(v);
          if (list.length > 2) list.removeAt(0);
        }
        looks++;
        final now = {'C': c, 'N': n};
        var act = _react(game);
        // A rock in front: the band says left, right, or hold the line.
        if (act == 'pick' || _side != null || _hold != null) {
          final s = await _see('S', _steer) ?? 'straight';
          now['S'] = s;
          if (s == 'straight') {
            _hold ??= game.lateral;
            _side = null;
          } else {
            _side = s == 'left' ? -1.0 : 1.0;
            _hold = null;
          }
          if (act == 'pick') act = s;
        }
        seen = now;
        notifyListeners();
        log('ACT3_EYES ${game.elapsed.toStringAsFixed(2)} $c|$n|${now['S'] ?? '-'} ${w.elapsedMilliseconds}ms ${act.isEmpty ? '.' : act}');
      } catch (e) {
        log('ACT3_EYES_FAILED $e');
        await Future<void>.delayed(const Duration(milliseconds: 200));
      }
    }
  }

  /// The reflex. An answer counts once two looks in a row agree.
  String _react(RunnerGame game) {
    bool twice(List<String> seen, String l) => seen.length == 2 && seen.every((c) => c == l);
    var act = '';
    // The near part says jump: jump, once, until it says run again.
    if (twice(_nearby, 'jump') && !_jumped && !game.airborne) {
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

  /// Called by the game every frame instead of the scripted pilot.
  PilotIntent intent(RunnerGame game) {
    final jump = _jumpNow;
    _jumpNow = false;
    final half = game.trail.halfWidth - 0.4;
    final side = _side;
    // Passing a rock: all the way to that edge of the trail.
    final lateral = side != null ? side * half : _hold ?? game.lateral * 0.98;
    return PilotIntent(lateral: lateral, jump: jump);
  }

  /// One part of the scene, read back alone (not the whole frame), and what
  /// the decider calls it.
  Future<String?> _see(String part, Decider decider) async {
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
    return answer.choice ?? answer.ranked.first.id;
  }

  Future<void> stop() async {
    _stopped = true;
    await _zones.close();
    await _near.close();
    await _steer.close();
    await _model.close();
  }
}
