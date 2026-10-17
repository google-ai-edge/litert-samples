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
    show Skill, SkillType;

/// The intents the app adds to the agent package's `NativeIntentExecutor`
/// (`buildAppIntents`), next to the package's built-in ones
/// (`get_current_date_and_time`, `send_email`, …). A SKILL.md names an intent
/// as intent `current_time` (or `intent: current_time`, the agent package's
/// form) and calls `run_intent` with it; Markdown combines intents, it never
/// adds Dart.
///
/// There is no camera-watch or timer intent: simpler, no camera in the chat,
/// less memory.
abstract final class AppIntent {
  static const deviceInfo = 'device_info';
  static const currentTime = 'current_time';

  static const all = {deviceInfo, currentTime};
}

/// The intents a SKILL.md body names, as written, in the forms skills are
/// written in: intent `name` (in backticks, the form every bundled skill
/// uses), or `intent: name`, also as a list item (`- intent: name`, the agent
/// package's own skills). No case or `-`/`_` folding: the executor runs an
/// intent only by its exact name (`NativeIntentExecutor` trims, nothing
/// more), so the scan checks the names as the model will copy them, and a
/// typo is listed as an error, not discovered by the model at run time.
///
/// Prose never names an intent: "this intent tells the time" has neither a
/// colon nor backticks, and after a colon a bare word counts only when it
/// looks like an identifier (it has `_`, as every intent name does), so
/// `## Intent: tell the user the time`, "the user's intent: they may want
/// the time" and `## Intent: real-time clock` name nothing. In backticks or
/// quotes the value always counts.
Set<String> intentsNamedIn(String body) => {
  for (final match in _namedIntent.allMatches(body)) ?_namedBy(match),
};

/// The app intents [skills] offer: the ones in [AppIntent.all] that their
/// intent-type skills name ([intentsNamedIn]). An instruction-only skill
/// that merely quotes an intent offers nothing. Over the installed skills,
/// these are the requests the app may answer itself (`ChatTurnResponder`),
/// with tools on or off.
Set<String> intentsOfferedBy(Iterable<Skill> skills) => {
  for (final skill in skills)
    if (skill.type == SkillType.intent)
      for (final intent in intentsNamedIn(skill.instructions))
        if (AppIntent.all.contains(intent)) intent,
};

/// The name [match] gives, or null when it is a prose word after a colon.
String? _namedBy(RegExpMatch match) {
  if ((match[2] ?? match[4])?.trim() case final quoted?
      when quoted.isNotEmpty) {
    return quoted;
  }
  final bare = match[3];
  // `_` marks an identifier; `-` does not (prose: "real-time",
  // "kid-friendly").
  return bare != null && bare.contains('_') ? bare : null;
}

/// After `intent` and a colon (only spaces around it, so a colon at a
/// line's end does not take the next line's first word): a value in
/// backticks or quotes (group 2) or a bare word (group 3). Or intent `name`
/// (group 4). `\b` keeps `run_intent` out: `_` is a word character.
final _namedIntent = RegExp(
  r'''\bintent(?:[ \t]*:[ \t]*(?:([`"'])([A-Za-z0-9_-]+)\1|([A-Za-z0-9_-]+))|\s+`([^`\n]+)`)''',
  caseSensitive: false,
);
