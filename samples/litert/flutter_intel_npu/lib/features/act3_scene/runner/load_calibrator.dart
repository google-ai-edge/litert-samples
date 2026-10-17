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

import 'package:flutter/foundation.dart';

import '../../../core/telemetry/frame_time_probe.dart';
import 'runner_game.dart';

/// One rung of the load ladder. Ordered by cost: draw distance first (cheap
/// to raise, adds depth to the shot), then vegetation, then pixels.
class LoadStep {
  const LoadStep(this.viewChunks, this.propDensity, this.renderScale);
  final int viewChunks;
  final double propDensity;
  final double renderScale;

  String get label => '${viewChunks * 64} m · veg ×${propDensity.toStringAsFixed(1)} · scale ${renderScale.toStringAsFixed(2)}';
}

/// The frame rate Act 3 budgets for, and whether the display caps the scene
/// at that rate.
///
/// The calibrator raises the load while the scene runs well above the budget
/// ([LoadCalibrator.climbAbove]). That works on a 120 Hz Mac aiming at 60,
/// and never on a 60 Hz screen aiming at 60: vsync caps the scene at the
/// refresh rate, so it can never look like it has headroom, and the scene
/// stays at its lightest — the one setting where the act shows nothing. On
/// such a display the budget is the refresh rate itself, and the calibrator
/// keeps the heaviest load that still holds it.
class FrameBudget {
  const FrameBudget(this.fps, {required this.vsyncCapped, required this.refreshRate});

  /// [requested] is `ACT3_TARGET_FPS` (0 = the display's refresh rate);
  /// [refreshRate] is the display's, 0 when unknown (taken as 60).
  factory FrameBudget.resolve({required int requested, required double refreshRate}) {
    final refresh = refreshRate > 0 ? refreshRate.round() : 60;
    if (requested <= 0 || refresh < requested * LoadCalibrator.climbAbove) {
      return FrameBudget(refresh, vsyncCapped: true, refreshRate: refresh);
    }
    return FrameBudget(requested, vsyncCapped: false, refreshRate: refresh);
  }

  final int fps;
  final bool vsyncCapped;
  final int refreshRate;

  Map<String, Object?> toJson() => {'target_fps': fps, 'refresh_rate': refreshRate, 'vsync_capped': vsyncCapped};
}

/// Puts the scene on the edge of the frame budget, whatever machine it is
/// running on.
///
/// This is not polish: it is what makes Act 3 work at all. A scene with 50 %
/// headroom shows nothing when the model takes the GPU — on an M4 Pro at the
/// default dials the narrator cost exactly zero frames, because there was
/// nothing to take. The act needs the scene to have no slack, so that any GPU
/// the model steals is visible immediately.
class LoadCalibrator extends ChangeNotifier {
  LoadCalibrator({
    required this.probe,
    required this.apply,
    required this.budget,
    this.startLevel = 1,
    this.settle = const Duration(milliseconds: 900),
    this.sample = const Duration(milliseconds: 1400),
  });

  final FrameTimeProbe probe;
  final void Function(LoadStep step) apply;
  final FrameBudget budget;

  /// Uncapped display: climb while the scene runs above 1.30 × budget, keep a
  /// rung that runs above 1.08 × budget. Aim a little ABOVE the budget, not
  /// at it: the scene's own cost swings with the biome, and a scene
  /// calibrated to sit exactly on 60 fps in a meadow is already missing
  /// frames by the next treeline.
  static const climbAbove = 1.30;
  static const keepAbove = 1.08;

  /// Display capped at the budget: "holds the refresh rate".
  static const holds = 0.97;
  final int startLevel;
  final Duration settle;
  final Duration sample;

  static const ladder = <LoadStep>[
    LoadStep(5, 0.6, 1.0),
    LoadStep(7, 1.0, 1.0),
    LoadStep(8, 1.5, 1.0),
    LoadStep(9, 2.0, 1.0),
    LoadStep(10, 2.5, 1.15),
    LoadStep(11, 3.0, 1.3),
    LoadStep(12, 3.5, 1.5),
    LoadStep(12, 4.0, 1.75),
    LoadStep(13, 4.5, 2.0),
    LoadStep(14, 5.0, 2.25),
    LoadStep(14, 5.5, 2.5),
    LoadStep(14, 6.0, 3.0),
    LoadStep(14, 6.0, 3.5),
    LoadStep(14, 6.5, 4.0),
  ];

  int level = 1;
  bool running = false;
  bool done = false;
  String status = '';
  final List<String> log = [];

  LoadStep get step => ladder[level];

  /// Walks the ladder until the scene sits just under the frame budget, then
  /// stops. Returns the level it settled on.
  Future<int> run() async {
    if (running) return level;
    running = true;
    done = false;
    level = startLevel.clamp(0, ladder.length - 1);
    var lastDirection = 0;
    _set('measuring ${ladder[level].label}');
    try {
      for (var attempt = 0; attempt < ladder.length; attempt++) {
        apply(ladder[level]);
        await Future<void>.delayed(settle);
        probe.reset();
        await Future<void>.delayed(sample);
        final fps = probe.sustainedFps;
        log.add('level $level (${ladder[level].label}): ${fps.toStringAsFixed(1)} fps');

        final climb = budget.fps * (budget.vsyncCapped ? holds : climbAbove);
        final keep = budget.fps * (budget.vsyncCapped ? holds : keepAbove);
        if (fps >= climb && level < ladder.length - 1) {
          if (lastDirection < 0) break; // already came down from here
          level++;
          lastDirection = 1;
          _set('more load: ${ladder[level].label}');
          continue;
        }
        // Below budget — back off one rung and keep it.
        if (fps < keep && level > 0) {
          if (lastDirection > 0) {
            level--;
            apply(ladder[level]);
            break;
          }
          level--;
          lastDirection = -1;
          _set('less load: ${ladder[level].label}');
          continue;
        }
        break;
      }
    } finally {
      running = false;
      done = true;
      _set('locked at ${ladder[level].label}');
      probe.reset();
    }
    return level;
  }

  void _set(String s) {
    status = s;
    notifyListeners();
  }

  Map<String, Object?> toJson() => {
        'level': level,
        'view_chunks': step.viewChunks,
        'prop_density': step.propDensity,
        'render_scale': step.renderScale,
        ...budget.toJson(),
        'log': log,
      };
}

/// Applies a ladder step to the game, keeping the dials the calibrator does
/// not touch.
void applyStepTo(RunnerGame game, LoadStep step) {
  game.applySettings(game.settings.copyWith(
    viewChunks: step.viewChunks,
    propDensity: step.propDensity,
    renderScale: step.renderScale,
  ));
}
