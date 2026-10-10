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

/// The app's intents: the only actions a Markdown skill can run (Markdown
/// combines intents, it never adds Dart). A SKILL.md names them as intent
/// `current_time` and calls `run_intent` with one.
///
/// There is no camera-watch or timer intent: simpler, no camera in the chat,
/// less memory.
abstract final class AppIntent {
  static const deviceInfo = 'device_info';
  static const currentTime = 'current_time';

  static const all = {deviceInfo, currentTime};

  /// `device_info, current_time`, for error results the model reads.
  static String get listed => all.join(', ');

  /// [raw] as an intent name: trimmed, lowercased, `-` → `_` (small models
  /// echo `current-time`). Null when that is not one of [all].
  static String? normalize(String raw) {
    final name = raw.trim().toLowerCase().replaceAll('-', '_');
    return all.contains(name) ? name : null;
  }
}

/// The intents a SKILL.md body names, written as intent `name` (in
/// backticks, the form every bundled skill uses). The scan checks them
/// against [AppIntent.all] so a typo is listed as an error, not discovered
/// by the model at run time.
Set<String> intentsNamedIn(String body) => {
  for (final match in _namedIntent.allMatches(body)) match[1]!.trim(),
};

final _namedIntent = RegExp(r'\bintent\s+`([^`\n]+)`');
