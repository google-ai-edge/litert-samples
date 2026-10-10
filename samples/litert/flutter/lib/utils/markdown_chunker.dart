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

import 'dart:convert';

/// One knowledge-base chunk: what is embedded and stored, plus the metadata
/// a citation shows.
final class const KbChunk({
  /// The document's asset file name, e.g. `litert-overview.md`.
  required final String doc,

  /// 0-based position within [doc]; stable while the document and the
  /// chunker version stay the same.
  required final int index,
  required final String title,

  /// `H2`, `H2 › H3`, or `Overview` for text before the first H2.
  required final String section,
  required final String body,

  /// The first URL of the front matter's `source`, when there is one.
  final String? source,
}) {
  /// `doc#n`.
  String get id => '$doc#$index';

  /// What is embedded and stored: `Title › Section`, a blank line, the body.
  String get content => '$title › $section\n\n$body';

  /// The JSON stored next to the vector; citations are built from it. Every
  /// key is always present (`source` may be null).
  String get metadataJson => jsonEncode({
    'doc': doc,
    'title': title,
    'section': section,
    'chunk': index,
    'source': source,
  });
}

/// Splits a Markdown document into retrieval chunks:
///
/// 1. CRLF → LF, BOM stripped.
/// 2. Front matter: only a leading `---` … `---` block of flat `key: value`
///    lines (`title`, the first URL of `source`).
/// 3. Fenced code (``` or ~~~, closed by the same character with at least
///    the same length) is dropped, and headings inside it are ignored. An
///    unclosed fence throws [FormatException].
/// 4. H1 is the fallback title; H2 and H3 start sections (H3 path
///    `H2 › H3`); H4+ stay in the body; text before the first H2 is
///    `Overview`.
/// 5. Blocks are split on blank lines; table separator rows are dropped.
/// 6. Cost: 1 per prose character, 2 per table character. A section within
///    [maxCost] is one chunk, never merged with another. A larger one is
///    split into about `ceil(cost / maxCost)` balanced parts at block
///    boundaries; an oversized table splits into row groups that repeat its
///    header, oversized prose into sentences.
/// 7. A part after the first starts with the previous part's last sentence
///    when that sentence is at most [overlapMaxChars] long, still fits, and
///    neither side of the cut is a table.
final class MarkdownChunker {
  const MarkdownChunker({this.maxCost = 1400, this.overlapMaxChars = 300});

  /// Part of the index hash: bump it with any change to the output,
  /// so the next launch re-indexes instead of serving stale chunks.
  static const version = 'md-chunker-1';

  final int maxCost;
  final int overlapMaxChars;

  /// Chunks [markdown]; [docId] becomes [KbChunk.doc] and the id prefix.
  List<KbChunk> chunk(String markdown, {required String docId}) {
    var text = markdown.replaceAll('\r\n', '\n').replaceAll('\r', '\n');
    if (text.startsWith('\uFEFF')) text = text.substring(1);
    final lines = text.split('\n');

    final (:title, :source, :bodyStart) = _frontMatter(lines, docId);
    final (:h1, :sections) = _sections(lines, bodyStart, docId);
    final docTitle = title ?? h1 ?? docId;

    final chunks = <KbChunk>[];
    for (final section in sections) {
      final blocks = _blocks(section.lines);
      if (blocks.isEmpty) continue;
      for (final body in _split(blocks)) {
        chunks.add(
          KbChunk(
            doc: docId,
            index: chunks.length,
            title: docTitle,
            section: section.path,
            body: body,
            source: source,
          ),
        );
      }
    }
    return chunks;
  }

  // --- 2. Front matter -----------------------------------------------------

  ({String? title, String? source, int bodyStart}) _frontMatter(
    List<String> lines,
    String docId,
  ) {
    if (lines.isEmpty || lines.first.trimRight() != '---') {
      return (title: null, source: null, bodyStart: 0);
    }
    final end = lines.indexWhere((l) => l.trimRight() == '---', 1);
    if (end < 0) {
      throw FormatException('$docId: front matter is not closed by "---"');
    }
    String? title;
    String? source;
    for (final line in lines.sublist(1, end)) {
      final match = _keyValue.firstMatch(line);
      if (match == null) continue;
      final value = _unquote(match[2]!.trim());
      switch (match[1]!.toLowerCase()) {
        case 'title' when value.isNotEmpty:
          title = value;
        case 'source':
          source = _firstUrl.firstMatch(value)?[0];
      }
    }
    return (title: title, source: source, bodyStart: end + 1);
  }

  static String _unquote(String value) {
    if (value.length >= 2 &&
        ((value.startsWith('"') && value.endsWith('"')) ||
            (value.startsWith("'") && value.endsWith("'")))) {
      return value.substring(1, value.length - 1);
    }
    return value;
  }

  // --- 3–4. Fences and headings --------------------------------------------

  ({String? h1, List<_Section> sections}) _sections(
    List<String> lines,
    int start,
    String docId,
  ) {
    String? h1;
    String? h2;
    final sections = <_Section>[];
    var current = _Section('Overview');
    String? fenceChar;
    var fenceLength = 0;
    var fenceLine = 0;

    for (var i = start; i < lines.length; i++) {
      final line = lines[i];
      if (fenceChar != null) {
        final close = _fenceClose.firstMatch(line);
        if (close != null &&
            close[1]![0] == fenceChar &&
            close[1]!.length >= fenceLength) {
          fenceChar = null;
        }
        continue; // code is dropped
      }
      final open = _fenceOpen.firstMatch(line);
      if (open != null) {
        fenceChar = open[1]![0];
        fenceLength = open[1]!.length;
        fenceLine = i + 1;
        // The code is gone; keep the text on either side apart.
        current.lines.add('');
        continue;
      }
      final heading = _heading.firstMatch(line);
      if (heading != null) {
        final level = heading[1]!.length;
        final text = heading[2]!.trim();
        if (level == 1) {
          h1 ??= text;
          continue;
        }
        if (level == 2) {
          sections.add(current);
          h2 = text;
          current = _Section(text);
          continue;
        }
        if (level == 3) {
          sections.add(current);
          current = _Section(h2 == null ? text : '$h2 › $text');
          continue;
        }
        // H4 and deeper stay in the body.
      }
      current.lines.add(line);
    }
    if (fenceChar != null) {
      throw FormatException(
        '$docId: the code fence opened on line $fenceLine is never closed',
      );
    }
    sections.add(current);
    return (h1: h1, sections: sections);
  }

  // --- 5. Blocks -------------------------------------------------------------

  List<_Block> _blocks(List<String> lines) {
    final blocks = <_Block>[];
    var pending = <String>[];
    void flush() {
      if (pending.isNotEmpty) {
        final table = pending.every((l) => l.trimLeft().startsWith('|'));
        blocks.add(_Block(pending.join('\n'), table: table));
        pending = <String>[];
      }
    }

    for (final raw in lines) {
      final line = raw.trimRight();
      if (line.trim().isEmpty) {
        flush();
        continue;
      }
      if (_isTableSeparator(line)) continue;
      pending.add(line);
    }
    flush();
    return blocks;
  }

  static bool _isTableSeparator(String line) =>
      line.contains('|') && line.contains('-') && _separatorRow.hasMatch(line);

  // --- 6–7. Budget, balanced split, overlap ------------------------------

  List<String> _split(List<_Block> blocks) {
    final whole = [
      for (var b = 0; b < blocks.length; b++)
        _Unit(
          blocks[b].text,
          block: b,
          table: blocks[b].table,
          joinWithPrevious: false,
        ),
    ];
    final total = _cost(whole);
    if (total <= maxCost) return [_join(whole)];

    final units = [
      for (var b = 0; b < blocks.length; b++)
        ...switch (blocks[b]) {
          final block
              when _unitCost(block.text, table: block.table) <= maxCost =>
            [whole[b]],
          _Block(:final text, table: true) => _rowGroups(text, b),
          _Block(:final text) => _sentenceUnits(text, b),
        },
    ];

    final parts = _pack(units, _ceilDiv(total, maxCost));
    final bodies = <String>[];
    for (var p = 0; p < parts.length; p++) {
      var part = parts[p];
      if (p > 0) part = _withOverlap(parts[p - 1], part);
      bodies.add(_join(part));
    }
    return bodies;
  }

  /// Packs [units] into parts of at most [maxCost], each as close as it can
  /// get to an even share of what is left, aiming for [wanted] parts.
  List<List<_Unit>> _pack(List<_Unit> units, int wanted) {
    final parts = <List<_Unit>>[];
    var current = <_Unit>[];
    var currentCost = 0;
    var remaining = _cost(units);
    for (final unit in units) {
      if (current.isEmpty) {
        current = [unit];
        currentCost = unit.cost;
        continue;
      }
      final partsLeft = [
        wanted - parts.length,
        _ceilDiv(remaining, maxCost),
        1,
      ].reduce((a, b) => a > b ? a : b);
      final target = remaining / partsLeft;
      final withUnit = currentCost + _joinCost(current.last, unit) + unit.cost;
      final closer =
          (withUnit - target).abs() <= (currentCost - target).abs() ||
          withUnit <= target;
      if (withUnit > maxCost || !closer) {
        parts.add(current);
        remaining -= currentCost;
        current = [unit];
        currentCost = unit.cost;
      } else {
        current.add(unit);
        currentCost = withUnit;
      }
    }
    if (current.isNotEmpty) parts.add(current);
    return parts;
  }

  List<_Unit> _withOverlap(List<_Unit> previous, List<_Unit> part) {
    final last = previous.last;
    final first = part.first;
    if (last.table || first.table) return part;
    final sentences = _sentences(last.text);
    final sentence = sentences.isEmpty ? '' : sentences.last.trim();
    if (sentence.isEmpty || sentence.length > overlapMaxChars) return part;
    final overlap = _Unit(
      sentence,
      block: last.block,
      table: false,
      joinWithPrevious: false,
    );
    final continued = first.withJoin(join: first.block == last.block);
    final candidate = [overlap, continued, ...part.skip(1)];
    return _cost(candidate) <= maxCost ? candidate : part;
  }

  List<_Unit> _rowGroups(String table, int block) {
    final rows = table.split('\n');
    final header = rows.first;
    final body = rows.skip(1).toList();
    if (body.isEmpty) {
      return [_Unit(table, block: block, table: true, joinWithPrevious: false)];
    }
    final headerCost = _unitCost(header, table: true) + 2; // + newline
    final rowsCost = body.fold(0, (sum, r) => sum + _unitCost(r, table: true));
    final room = (maxCost - headerCost).clamp(1, maxCost);
    final groups = _ceilDiv(rowsCost, room);
    final target = rowsCost / groups;
    final units = <_Unit>[];
    var current = <String>[];
    var currentCost = 0;
    void flush() {
      if (current.isEmpty) return;
      units.add(
        _Unit(
          [header, ...current].join('\n'),
          block: block,
          table: true,
          joinWithPrevious: false,
        ),
      );
      current = <String>[];
      currentCost = 0;
    }

    for (final row in body) {
      final cost = _unitCost(row, table: true) + 2;
      if (current.isNotEmpty &&
          (currentCost + cost > room || currentCost >= target)) {
        flush();
      }
      current.add(row);
      currentCost += cost;
    }
    flush();
    return units;
  }

  List<_Unit> _sentenceUnits(String prose, int block) {
    final units = <_Unit>[];
    for (final sentence in _sentences(prose)) {
      for (final piece in _hardWrap(sentence)) {
        units.add(
          _Unit(
            piece,
            block: block,
            table: false,
            joinWithPrevious: units.isNotEmpty,
          ),
        );
      }
    }
    return units;
  }

  /// A "sentence" longer than [maxCost] (a run-on list, a URL dump) split at
  /// word boundaries so no unit can break the budget.
  List<String> _hardWrap(String sentence) {
    if (sentence.length <= maxCost) return [sentence];
    final pieces = <String>[];
    final buffer = StringBuffer();
    for (final word in sentence.split(' ')) {
      if (buffer.isNotEmpty && buffer.length + 1 + word.length > maxCost) {
        pieces.add(buffer.toString());
        buffer.clear();
      }
      if (buffer.isNotEmpty) buffer.write(' ');
      buffer.write(word);
    }
    if (buffer.isNotEmpty) pieces.add(buffer.toString());
    return pieces;
  }

  static List<String> _sentences(String prose) => [
    for (final s in prose.split(_sentenceBreak))
      if (s.trim().isNotEmpty) s.trim(),
  ];

  static String _join(List<_Unit> units) {
    final out = StringBuffer();
    for (var i = 0; i < units.length; i++) {
      if (i > 0) out.write(units[i].joinWithPrevious ? ' ' : '\n\n');
      out.write(units[i].text);
    }
    return out.toString();
  }

  static int _cost(List<_Unit> units) {
    var cost = 0;
    for (var i = 0; i < units.length; i++) {
      if (i > 0) cost += _joinCost(units[i - 1], units[i]);
      cost += units[i].cost;
    }
    return cost;
  }

  static int _joinCost(_Unit previous, _Unit next) =>
      next.joinWithPrevious ? 1 : 2;

  static int _unitCost(String text, {required bool table}) =>
      table ? text.length * 2 : text.length;

  static int _ceilDiv(int a, int b) => (a + b - 1) ~/ b;
}

final class _Section {
  _Section(this.path);

  final String path;
  final List<String> lines = [];
}

final class const _Block(final String text, {required final bool table});

final class const _Unit(
  final String text, {
  required final int block,
  required final bool table,

  /// A later sentence of the same paragraph: joined with a space, not a
  /// blank line.
  required final bool joinWithPrevious,
}) {
  int get cost => MarkdownChunker._unitCost(text, table: table);

  _Unit withJoin({required bool join}) =>
      _Unit(text, block: block, table: table, joinWithPrevious: join);
}

final _keyValue = RegExp(r'^([A-Za-z_][\w-]*)\s*:\s*(.*)$');
final _firstUrl = RegExp(r'https?://[^\s;,]+');
final _fenceOpen = RegExp(r'^\s{0,3}(`{3,}|~{3,})');
final _fenceClose = RegExp(r'^\s{0,3}(`{3,}|~{3,})\s*$');
final _heading = RegExp(r'^\s{0,3}(#{1,6})[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$');
final _separatorRow = RegExp(r'^[\s|:\-]+$');
final _sentenceBreak = RegExp(
  r'(?<!\b(?:e\.g|i\.e|vs|etc))(?<=[.!?])\s+(?=[A-Z0-9"(\[`])',
);
