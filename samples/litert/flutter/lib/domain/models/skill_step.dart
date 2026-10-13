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

/// One step of an agent turn, for the steps panel under the reply and the
/// overlay's tool timings. [at] counts from the start of the turn (`ask`),
/// so consecutive steps show how long each generation took.
sealed class SkillStep {
  const SkillStep({required this.at});

  final Duration at;
}

/// `loadSkill(name)`: the model read a skill's instructions. [found] false:
/// no skill has that name (the loop told the model so).
final class SkillLoaded extends SkillStep {
  const SkillLoaded(this.name, {required this.found, required super.at});

  final String name;
  final bool found;

  @override
  String toString() => 'loadSkill($name${found ? '' : ', not found'})';
}

/// `runIntent(intent, parameters)`: the model called an intent; it runs
/// now.
final class IntentCalled extends SkillStep {
  const IntentCalled(this.intent, this.parameters, {required super.at});

  final String intent;

  /// As the model wrote it (a JSON string, or whatever it sent instead).
  final String parameters;

  @override
  String toString() => 'runIntent($intent, $parameters)';
}

/// The intent ran; [result] is what the model was told.
final class IntentSucceeded extends SkillStep {
  const IntentSucceeded(
    this.intent,
    this.result, {
    required this.elapsed,
    required super.at,
  });

  final String intent;
  final String result;

  /// From [IntentCalled] to the result: the app's own work.
  final Duration elapsed;

  @override
  String toString() => '$intent → $result';
}

/// A tool call failed (an error result, an unknown skill or tool); the model
/// was told [message] so it can recover.
final class IntentFailed extends SkillStep {
  const IntentFailed(
    this.intent,
    this.message, {
    this.elapsed,
    required super.at,
  });

  /// The intent or tool that failed; null when unknown.
  final String? intent;
  final String message;
  final Duration? elapsed;

  @override
  String toString() => '${intent ?? 'tool'} failed: $message';
}
