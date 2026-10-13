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

import 'package:flutter_edge_ai_agent/flutter_edge_ai_agent.dart' show Skill;

/// A SKILL.md that parsed and passed the checks.
final class const LoadedSkill({
  required final Skill skill,

  /// Relative to the skills directory, e.g. `current-time/SKILL.md`.
  required final String path,
});

/// A file in the skills directory that is not a usable skill, and why:
/// a parse error, bad UTF-8, too large, a duplicate name, an unknown
/// intent.
final class const SkillLoadError({
  /// Relative to the skills directory.
  required final String path,
  required final String message,
});

/// One scan of the runtime skills directory: the skills the agent gets
/// and the files that failed, for the Skills sheet and the overlay.
final class const SkillCatalog({
  /// Where users drop skills; null when the directory is unavailable.
  required final String? directory,
  final List<LoadedSkill> skills = const [],
  final List<SkillLoadError> errors = const [],

  /// A hash of every scanned file's path and bytes: a reload applies only
  /// when it changed (applying drops the conversation's history).
  required final String fingerprint,

  /// The directory could not be used at all (no skills then).
  final String? storeError,
}) {
  List<Skill> get agentSkills => [for (final s in skills) s.skill];
}
