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

/// One citation marker in a reply: `[2]`, `[1, 3]`, `[1-3]` or `[1–3]`.
final class const CitationMarker({
  /// Where the marker starts and ends in the text (`]` included).
  required final int start,
  required final int end,

  /// The excerpt numbers it cites, ranges expanded.
  required final Set<int> numbers,
});

/// The citation markers in [text] for a prompt that carried [count]
/// excerpts. A bracket group of numbers is a citation only when
/// every number in it is in 1..[count] and it is not inside backticks, so a
/// tensor shape such as `[1, 3, 640, 640]` or `[1, 80, 3000]`, and code such
/// as `` `[1]` ``, is never mistaken for one. One place decides this for the
/// citation chips and the speech sanitizer.
Iterable<CitationMarker> findCitations(
  String text, {
  required int count,
}) sync* {
  if (count < 1) return;
  final code = _codeSpans(text);
  for (final match in _group.allMatches(text)) {
    if (code.any((span) => match.start >= span.$1 && match.start < span.$2)) {
      continue;
    }
    final numbers = _numbers(match[1]!);
    if (numbers == null || numbers.isEmpty) continue;
    if (numbers.any((n) => n < 1 || n > count)) continue;
    yield CitationMarker(start: match.start, end: match.end, numbers: numbers);
  }
}

/// The numbers of one bracket group, ranges expanded (at most 16 per range);
/// null when a range runs backwards or is too long to be a citation.
Set<int>? _numbers(String inner) {
  final numbers = <int>{};
  for (final part in inner.split(',')) {
    final bounds = [for (final s in part.split(_dash)) ?int.tryParse(s.trim())];
    if (bounds.isEmpty) continue;
    final from = bounds.first;
    final to = bounds.last;
    if (to < from || to - from >= 16) return null;
    for (var n = from; n <= to; n++) {
      numbers.add(n);
    }
  }
  return numbers;
}

/// `[start, end)` of every inline code span or fence: text between a run of
/// backticks and the next run. An unpaired last run opens nothing.
List<(int, int)> _codeSpans(String text) {
  final runs = _backticks.allMatches(text).toList();
  return [
    for (var i = 0; i + 1 < runs.length; i += 2)
      (runs[i].start, runs[i + 1].end),
  ];
}

final _group = RegExp(r'\[(\d+(?:\s*[,–-]\s*\d+)*)\]');
final _dash = RegExp('[–-]');
final _backticks = RegExp('`+');
