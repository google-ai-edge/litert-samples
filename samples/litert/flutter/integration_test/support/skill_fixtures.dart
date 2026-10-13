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

// SKILL.md files the skills tests drop into the skills folder at run time:
// never bundled. Plain Dart so the unit tests import the same text the
// integration test writes.

/// The runtime-only example skill: composes `current_time` under a new
/// trigger and a new way of saying it, without code.
const kidClockSkillMd = '''
---
name: kid-clock
description: Tell the time the way you would to a small child, when the user asks you to tell their kid or a child what time it is.
---
# Kid clock

## Instructions
Call the `run_intent` tool with intent `current_time` and parameters {}.
Then say the time in one short sentence a five-year-old understands, for example "It is quarter past six in the evening."
''';

/// No front matter at all: listed with its parse error.
const brokenSkillMd = '''
# Broken skill

This file has no front matter, so it has no name or description.
''';
