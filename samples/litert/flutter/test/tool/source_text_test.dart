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

import 'dart:io';

import 'package:flutter_test/flutter_test.dart';

/// Invisible and look-alike characters (a byte-order mark, the no-break,
/// zero-width and typographic spaces, a soft hyphen, the bidirectional
/// controls) look like nothing, or like a plain space, in an editor and in
/// a diff, and the bidi ones can reorder what a reader sees. In Dart they are
/// written as escapes (`'\u00A0'`) so a reader sees what the code compares
/// with; in scripts, docs and YAML they have no business at all.
void main() {
  final invisible = <int, String>{
    0x00A0: 'U+00A0 (no-break space)',
    0x00AD: 'U+00AD (soft hyphen)',
    for (var c = 0x2000; c <= 0x200A; c++)
      c: 'U+${_hex(c)} (typographic space)',
    0x200B: 'U+200B (zero-width space)',
    0x200C: 'U+200C (zero-width non-joiner)',
    0x200D: 'U+200D (zero-width joiner)',
    0x200E: 'U+200E (left-to-right mark)',
    0x200F: 'U+200F (right-to-left mark)',
    for (var c = 0x202A; c <= 0x202E; c++)
      c: 'U+${_hex(c)} (bidirectional embedding or override)',
    0x202F: 'U+202F (narrow no-break space)',
    0x2060: 'U+2060 (word joiner)',
    for (var c = 0x2066; c <= 0x2069; c++)
      c: 'U+${_hex(c)} (bidirectional isolate)',
    0xFEFF: 'U+FEFF (byte-order mark)',
  };

  /// Files that carry such characters as data, with why.
  const exempt = {
    // Its vocabulary holds these characters as tokens.
    'assets/models/moonshine_tiny_tokenizer.json',
  };

  test('no source, script, doc or YAML file carries an invisible character '
      'literally', () {
    final files = _repoFiles()
        .where((p) => _checked.any(p.endsWith) && !exempt.contains(p))
        .toList();
    expect(files, isNotEmpty, reason: 'the file listing found nothing');
    expect(files, contains('lib/main.dart'));
    expect(files, contains('README.md'));
    expect(files, contains('tool/fetch_models.sh'));
    expect(files, contains('pubspec.yaml'));

    final found = <String>[];
    for (final path in files) {
      final file = File(path);
      if (!file.existsSync()) continue; // deleted, not yet staged
      final lines = file.readAsLinesSync();
      for (var i = 0; i < lines.length; i++) {
        for (final rune in lines[i].runes) {
          if (invisible[rune] case final name?) {
            found.add('$path:${i + 1}: $name');
          }
        }
      }
    }
    expect(found, isEmpty, reason: 'write them as \\u escapes in Dart');
  });

  test('the check knows every character it was asked to catch', () {
    for (final c in [
      0x00A0,
      0x00AD,
      for (var c = 0x2000; c <= 0x200F; c++) c,
      for (var c = 0x202A; c <= 0x202F; c++) c,
      for (var c = 0x2066; c <= 0x2069; c++) c,
      0x2060,
      0xFEFF,
    ]) {
      expect(invisible, contains(c), reason: 'U+${_hex(c)}');
    }
  });
}

/// The extensions checked.
const _checked = ['.dart', '.sh', '.md', '.yaml'];

/// The repository's files (tracked and untracked, not ignored), relative to
/// the package root; without git, the folders that hold its sources.
List<String> _repoFiles() {
  try {
    final git = Process.runSync('git', [
      'ls-files',
      '--cached',
      '--others',
      '--exclude-standard',
    ]);
    if (git.exitCode == 0) {
      return (git.stdout as String)
          .split('\n')
          .where((l) => l.isNotEmpty)
          .toList();
    }
  } on ProcessException {
    // No git here: walk the folders below.
  }
  return [
    for (final entry in Directory('.').listSync())
      if (entry is File) entry.path.substring(2),
    for (final root in const [
      'lib',
      'test',
      'integration_test',
      'tool',
      'docs',
      'assets',
    ])
      if (Directory(root).existsSync())
        for (final entry in Directory(root).listSync(recursive: true))
          if (entry is File) entry.path,
  ];
}

String _hex(int c) => c.toRadixString(16).toUpperCase().padLeft(4, '0');
