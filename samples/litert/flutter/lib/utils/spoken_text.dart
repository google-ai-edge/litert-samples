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

import '../config/knowledge_config.dart';
import 'citations.dart';

/// [clause] as TTS should read it: what the chat shows minus what must not be
/// spoken — citation markers (`[1]`, `[1, 2]`, `[1-3]`), markdown syntax,
/// links and URLs; an echoed excerpt label's `›` is read as a comma. The UI
/// keeps showing the raw text. Returns '' when nothing speakable is left.
///
/// A bracket group is a citation only when every number in it is in
/// 1..[maxCitation] (the excerpts a prompt can carry) and it is not inside
/// backticks: a tensor shape such as `[1, 3, 640, 640]` is read out.
String toSpokenText(String clause, {int maxCitation = kKbTopK}) {
  var text = clause;
  // [text](url) → text, before bare URLs and brackets are touched.
  text = text.replaceAllMapped(_mdLink, (m) => m[1] ?? '');
  text = text.replaceAll(_url, '');
  text = _withoutCitations(text, maxCitation);
  // "Title › Section" from a knowledge-base excerpt.
  text = text.replaceAll(_breadcrumb, ', ');
  text = text.replaceAll(_codeFence, '');
  // Line-leading markers: headings, bullets, quotes, numbered lists keep
  // their number (it is read as "one", which is fine).
  text = text.replaceAll(_heading, '');
  text = text.replaceAll(_bullet, '');
  text = text.replaceAll(_quote, '');
  // Emphasis and inline code markers; the words stay.
  text = text.replaceAll(_emphasis, '');
  text = text.replaceAll(_whitespace, ' ');
  return text.trim();
}

/// [text] with its citation markers removed, each with the whitespace before
/// it, and a comma left dangling between removed markers (`[1], [2].`).
String _withoutCitations(String text, int maxCitation) {
  final markers = findCitations(text, count: maxCitation).toList();
  if (markers.isEmpty) return text;
  final out = StringBuffer();
  var from = 0;
  for (final marker in markers) {
    var start = marker.start;
    while (start > from && _space.hasMatch(text[start - 1])) {
      start--;
    }
    out.write(text.substring(from, start));
    from = marker.end;
  }
  out.write(text.substring(from));
  return out.toString().replaceAll(_danglingComma, '');
}

final _space = RegExp(r'\s');
final _danglingComma = RegExp(r'\s*,(?=\s*(?:[.!?]|$))');
final _mdLink = RegExp(r'\[([^\]]*)\]\([^)]*\)');
final _url = RegExp(r'https?://\S+|www\.\S+');
final _breadcrumb = RegExp(r'\s*›\s*');
final _codeFence = RegExp(r'```[a-zA-Z]*');
final _heading = RegExp(r'^\s{0,3}#{1,6}\s+', multiLine: true);
final _bullet = RegExp(r'^\s*[-*+•]\s+', multiLine: true);
final _quote = RegExp(r'^\s*>\s?', multiLine: true);
final _emphasis = RegExp(r'\*{1,3}|_{2,3}|`|~~');
final _whitespace = RegExp(r'\s+');
