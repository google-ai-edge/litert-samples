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

import 'package:flutter_edge_ai_agent/flutter_edge_ai_agent.dart'
    show parseSkillMd;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/domain/models/assistant_event.dart';
import 'package:litert_edge_demos/domain/models/diagnostics_snapshot.dart';
import 'package:litert_edge_demos/domain/models/skill_catalog.dart';
import 'package:litert_edge_demos/domain/models/skill_step.dart';
import 'package:litert_edge_demos/ui/core/debug_overlay.dart';

/// The overlay's skills count and the last agent turn's tool
/// timings.
void main() {
  List<String> skillLines(DiagnosticsSnapshot s) =>
      debugOverlayLines(s)
          .where((l) => l.startsWith('skills') || l.startsWith('tools'))
          .toList();

  final skill = parseSkillMd(
    '---\nname: timer\ndescription: Timers.\n---\nCall the `run_intent` tool '
    'with intent `start_timer`.',
  );

  test('the scan: ok and error counts, or why there is no folder', () {
    expect(
      skillLines(
        DiagnosticsSnapshot(
          skills: SkillCatalog(
            directory: '/d',
            skills: [LoadedSkill(skill: skill, path: 'timer/SKILL.md')],
            errors: const [SkillLoadError(path: 'x.md', message: 'bad')],
            fingerprint: 'f',
          ),
        ),
      ),
      ['skills 1 ok / 1 error'],
    );
    expect(
      skillLines(
        const DiagnosticsSnapshot(
          skills: SkillCatalog(
            directory: null,
            fingerprint: 'unavailable',
            storeError: 'External storage is not available',
          ),
        ),
      ),
      ['skills unavailable: External storage is not available'],
    );
    expect(skillLines(const DiagnosticsSnapshot()), isEmpty);
  });

  test('a reset after an interrupted skill call is not called "budget"', () {
    final lines = debugOverlayLines(
      const DiagnosticsSnapshot(
        lastGeneration: GenerationMetrics(
          timeToFirstToken: null,
          chunks: 0,
          tokensPerSecond: null,
          tokensPerSecondSource: TokenRateSource.chunks,
          total: Duration.zero,
          stopped: false,
          contextReset: true,
          contextResetReason: ContextResetReason.interruptedSkill,
        ),
      ),
    );
    final ctx = lines.singleWhere((l) => l.startsWith('ctx'));
    expect(ctx, endsWith('reset (interrupted skill)'));
  });

  test('tool timings: each step at its time since the turn started, the '
      "intent's own work, then the first text", () {
    final lines = skillLines(
      const DiagnosticsSnapshot(
        lastGeneration: GenerationMetrics(
          timeToFirstToken: Duration(milliseconds: 2600),
          chunks: 6,
          tokensPerSecond: 30,
          tokensPerSecondSource: TokenRateSource.native,
          total: Duration(milliseconds: 2900),
          stopped: false,
          toolRounds: 2,
          skillSteps: [
            SkillLoaded('timer', found: true, at: Duration(milliseconds: 840)),
            IntentCalled(
              'start_timer',
              '{"seconds": 10}',
              at: Duration(milliseconds: 1710),
            ),
            IntentSucceeded(
              'start_timer',
              'Started a timer for ten seconds.',
              elapsed: Duration(milliseconds: 2),
              at: Duration(milliseconds: 1712),
            ),
          ],
        ),
      ),
    );

    expect(lines, [
      'tools 2 · loadSkill(timer) 0.84 s → start_timer 1.71 s (+2 ms) → '
          'text 2.60 s',
    ]);
  });
}
