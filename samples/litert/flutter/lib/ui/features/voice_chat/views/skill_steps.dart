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

import 'package:flutter/material.dart';

import '../../../../domain/models/skill_step.dart';

/// Keys for tests.
abstract final class SkillStepKeys {
  static const panel = ValueKey('skill-steps');

  /// The [index]th step line (0-based).
  static ValueKey<String> step(int index) => ValueKey('skill-step-$index');
}

/// Under a reply: what the agent did, one line per step —
/// `loadSkill(current-time)`, `runIntent(current_time, {})`, the result
/// — so the user sees the arguments the model wrote (the fp16 GPU path may
/// copy digits wrongly) and what the app answered.
class SkillStepsPanel extends StatelessWidget {
  const SkillStepsPanel({super.key, required this.steps, required this.color});

  final List<SkillStep> steps;
  final Color color;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final style = theme.textTheme.labelSmall?.copyWith(
      color: color,
      fontFamily: 'Menlo',
      fontFamilyFallback: const ['monospace', 'Courier'],
    );
    return Column(
      key: SkillStepKeys.panel,
      crossAxisAlignment: CrossAxisAlignment.start,
      spacing: 2,
      children: [
        for (var i = 0; i < steps.length; i++)
          Row(
            key: SkillStepKeys.step(i),
            crossAxisAlignment: CrossAxisAlignment.start,
            spacing: 4,
            children: [
              Icon(
                _icon(steps[i]),
                size: 14,
                color: steps[i] is IntentFailed
                    ? theme.colorScheme.error
                    : color,
              ),
              Expanded(child: Text(_line(steps[i]), style: style)),
            ],
          ),
      ],
    );
  }

  static IconData _icon(SkillStep step) => switch (step) {
    SkillLoaded(found: true) => Icons.menu_book_outlined,
    SkillLoaded() => Icons.help_outline,
    IntentCalled() => Icons.play_arrow,
    IntentSucceeded() => Icons.check,
    IntentFailed() => Icons.error_outline,
  };

  static String _line(SkillStep step) {
    final at = '${(step.at.inMilliseconds / 1000).toStringAsFixed(1)} s';
    return switch (step) {
      SkillLoaded(:final name, :final found) =>
        'loadSkill($name)${found ? '' : ' — not found'} · $at',
      IntentCalled(:final intent, :final parameters) =>
        'runIntent($intent${parameters.isEmpty ? '' : ', $parameters'}) · $at',
      IntentSucceeded(:final result, :final elapsed) =>
        '$result (${elapsed.inMilliseconds} ms)',
      IntentFailed(:final intent, :final message) =>
        '${intent ?? 'tool'} failed: $message',
    };
  }
}
