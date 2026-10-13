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

import '../models/detection_summary.dart';
import '../models/route_decision.dart';
import '../vision/coco_vocabulary.dart';

/// Turns a fast route and the detection summary into one spoken sentence:
/// no LLM, deterministic. Negative answers are hedged
/// ("right now"): the detector misses things, and "look closer" goes to
/// Gemma.
final class FastAnswerComposer {
  const FastAnswerComposer([this._vocabulary = kCocoVocabulary]);

  final CocoVocabulary _vocabulary;

  /// At most this many classes in an inventory answer.
  static const maxListed = 5;

  String compose(FastRoute route, DetectionSummary summary) =>
      switch ((route.intent, route.cls)) {
        (FastIntent.count, final int cls) => _count(cls, summary.countOf(cls)),
        (FastIntent.presence, final int cls) => _presence(
          cls,
          summary.countOf(cls),
        ),
        (FastIntent.inventory, _) ||
        (FastIntent.count || FastIntent.presence, null) => _inventory(summary),
      };

  /// What an answer was based on, for the route chip: "cat ×2 · remote".
  String basis(DetectionSummary summary) =>
      summary.label((cls) => _vocabulary.spokenName(cls, 1));

  String _count(int cls, int n) => switch (n) {
    0 => "I don't see any ${_vocabulary.spokenName(cls, 2)} right now.",
    _ => 'I count ${_vocabulary.counted(cls, n)}.',
  };

  String _presence(int cls, int n) => switch (n) {
    0 => "I don't see ${_vocabulary.withArticle(cls)} right now.",
    1 => 'Yes, I see ${_vocabulary.withArticle(cls)}.',
    _ => 'Yes, I see ${_vocabulary.counted(cls, n)}.',
  };

  String _inventory(DetectionSummary summary) {
    if (summary.isEmpty) return "I don't see anything I recognize right now.";
    final items = [
      for (final MapEntry(key: cls, value: n) in summary.counts.entries.take(
        maxListed,
      ))
        n == 1 ? _vocabulary.withArticle(cls) : _vocabulary.counted(cls, n),
    ];
    final list = items.length == 1
        ? items.single
        : '${items.sublist(0, items.length - 1).join(', ')} and ${items.last}';
    return 'I see $list.';
  }
}
