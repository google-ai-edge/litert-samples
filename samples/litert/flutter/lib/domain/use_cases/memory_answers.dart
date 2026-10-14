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

import '../models/visual_memory.dart';
import '../vision/coco_vocabulary.dart';
import 'camera_prompt.dart' show kCameraPromptMaxDetections;

/// The visual memory's spoken answers that need no LLM, and the prompt for
/// the one that does.
final class MemoryAnswers {
  const MemoryAnswers([this._vocabulary = kCocoVocabulary]);

  final CocoVocabulary _vocabulary;

  /// Nothing is remembered yet.
  static const empty =
      "I don't remember anything yet: keep the camera running for a few "
      'seconds.';

  /// No frame was accepted; [span] is how far back the memory reaches.
  String notSeen(Duration span) =>
      "I haven't seen that in the last ${_spanWords(span)}.";

  /// The memory is off: [reason] (already a sentence part).
  String off(String reason) => "My visual memory is off: ${_clause(reason)}.";

  /// The "Visual memory" switch in the camera settings is off.
  static const disabled = 'Visual memory is off (camera settings).';

  /// A hit while the chat model has images off (or the camera chat is not
  /// open): when, and what the detector saw in that frame. An ambiguous hit
  /// says first that it may have been in more than one place.
  String seen(MemoryHit hit) {
    final when = hit.latest
        ? 'The last frame I remember is from ${agoWords(hit.age)}.'
        : hit.ambiguous
        ? ambiguousLead(hit)
        : 'I saw it ${agoWords(hit.age)}.';
    final counts = hit.entry.counts;
    if (counts.isEmpty) {
      return "$when The detector didn't recognize anything in that frame.";
    }
    return '$when In that frame the detector saw ${_list(counts)}.';
  }

  /// Said before the chat model's answer about an ambiguous hit's frame (and
  /// before the detector's list with images off): the object may have been
  /// seen in more than one place, and this is the latest.
  String ambiguousLead(MemoryHit hit) =>
      'I may have seen that in more than one place; the latest was '
      '${agoWords(hit.age)}.';

  /// The question to the chat model about a remembered frame (sent with
  /// its PNG), in the detailed path's style: what the frame is, the
  /// detector's list for it as a hint labelled possibly incomplete, then the
  /// question.
  String prompt(String question, MemoryHit hit) {
    final counts = hit.entry.counts;
    final hint = counts.isEmpty
        ? ''
        : ' Objects a detector found in it (may be incomplete): '
              '${_list(counts, max: kCameraPromptMaxDetections)}.';
    final match = hit.latest
        ? 'the most recent frame in its visual memory'
        : hit.ambiguous
        ? 'the most recent of several frames from different places that '
              'matched the question about equally well'
        : 'the best match in its visual memory for the question';
    return 'This frame is from the camera ${agoWords(hit.age)}: $match.$hint'
        '\n\nQuestion: ${question.trim()}';
  }

  /// "two cats and a remote".
  String _list(Map<int, int> counts, {int max = 5}) {
    final items = [
      for (final MapEntry(key: cls, value: n) in counts.entries.take(max))
        n == 1 ? _vocabulary.withArticle(cls) : _vocabulary.counted(cls, n),
    ];
    return items.length == 1
        ? items.single
        : '${items.sublist(0, items.length - 1).join(', ')} and ${items.last}';
  }

  static String _clause(String reason) {
    final trimmed = reason.trim();
    return trimmed.endsWith('.')
        ? trimmed.substring(0, trimmed.length - 1)
        : trimmed;
  }
}

/// "a moment ago", "12 seconds ago", "about 2 minutes ago": how long ago a
/// remembered frame was seen, spoken.
String agoWords(Duration age) {
  final s = age.inSeconds;
  if (s < 3) return 'a moment ago';
  if (s < 60) return '$s seconds ago';
  final m = (s / 60).round();
  return m == 1 ? 'about a minute ago' : 'about $m minutes ago';
}

/// "12 s ago", "2 min ago": the result card's label.
String agoLabel(Duration age) {
  final s = age.inSeconds;
  if (s < 60) return '$s s ago';
  return '${(s / 60).round()} min ago';
}

/// "20 seconds", "minute", "3 minutes": how far back the memory reaches, in
/// "I haven't seen that in the last …".
String _spanWords(Duration span) {
  final s = span.inSeconds;
  if (s < 60) {
    final n = s < 1 ? 1 : s;
    return '$n second${n == 1 ? '' : 's'}';
  }
  final m = (s / 60).round();
  return m == 1 ? 'minute' : '$m minutes';
}

/// "37 frames · 2 min": the memory chip's figures.
String memorySpanLabel(int frames, Duration span) {
  final s = span.inSeconds;
  final reach = s < 60 ? '$s s' : '${(s / 60).round()} min';
  return '$frames frame${frames == 1 ? '' : 's'} · $reach';
}
