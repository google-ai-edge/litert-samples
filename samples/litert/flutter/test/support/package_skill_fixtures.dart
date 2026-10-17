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

// SKILL.md files that ship with the agent package (flutter_edge_ai_agent,
// MIT; its bundled skills are Apache-2.0, derived from google-ai-edge/gallery),
// copied verbatim: skills written for the package, not for this
// app, which the tests read as they are. package_skill_fixtures_test.dart
// compares each one with the resolved package's file.

/// The package's own skill directory, relative to its root.
const kPackageSkillsDir = 'assets/skills';

/// `assets/skills/get-current-time/SKILL.md` of flutter_edge_ai_agent 0.3.0
/// (unchanged since 0.2.10): the `- intent: name` list form, with a built-in
/// intent of the package's executor ([packageGetCurrentTimeIntent]).
const packageGetCurrentTimeSkillMd = '''
---
name: get-current-time
description: Get the user's current local date, time, and day of the week.
---

# Get current time

## Instructions

Call the `run_intent` tool with the following exact parameters:

- intent: get_current_date_and_time
- parameters: the JSON string `{}`

Then tell the user the date and time returned.
''';

/// The intent [packageGetCurrentTimeSkillMd] names: a built-in of the
/// package's `NativeIntentExecutor` (`getCurrentDateAndTime`), which the app
/// runs next to its own intents, so the skill loads and runs.
const packageGetCurrentTimeIntent = 'get_current_date_and_time';
