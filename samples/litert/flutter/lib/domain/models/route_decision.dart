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

/// What a fast answer is about.
enum FastIntent {
  /// "What do you see?"
  inventory,

  /// "Is there a dog?"
  presence,

  /// "How many cats?"
  count,
}

/// Where a camera question goes. [rule] names the rule
/// that decided, for the chip and the overlay.
sealed class RouteDecision {
  const RouteDecision(this.rule);

  final String rule;
}

/// Answered from the detection summary with a template: no LLM.
final class FastRoute extends RouteDecision {
  const FastRoute(this.intent, super.rule, {this.cls});

  final FastIntent intent;

  /// The class asked about; null for [FastIntent.inventory].
  final int? cls;

  @override
  String toString() => 'FastRoute(${intent.name}, cls=$cls, rule=$rule)';
}

/// Needs the frame and Gemma.
final class DetailedRoute extends RouteDecision {
  const DetailedRoute(super.rule);

  @override
  String toString() => 'DetailedRoute(rule=$rule)';
}
