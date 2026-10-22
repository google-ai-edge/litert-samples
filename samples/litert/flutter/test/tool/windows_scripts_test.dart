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

// tool/windows/: run.cmd, run.ps1 and package.ps1, the Windows counterparts
// of tool/linux/run.sh and package.sh. They run in Windows PowerShell 5.1,
// which no test host here has: what is checked is what breaks them there
// silently (a non-ASCII byte, PowerShell 7 syntax, a licence package.sh ships
// and package.ps1 forgets), plus PowerShell's own parser when pwsh is
// installed (the counterpart of shellcheck).

import 'dart:io';

import 'package:flutter_test/flutter_test.dart';

const _scripts = [
  'tool/windows/run.cmd',
  'tool/windows/run.ps1',
  'tool/windows/package.ps1',
];

/// The script's lines without its `#` comments (good enough for these
/// scripts: no `#` inside their strings starts a comment).
List<String> _code(String path) => [
  for (final line in File(path).readAsLinesSync())
    if (!line.trimLeft().startsWith('#')) line,
];

void main() {
  test('ASCII only and no byte-order mark: Windows PowerShell 5.1 reads a '
      'script without one in the ANSI code page', () {
    for (final path in _scripts) {
      final bytes = File(path).readAsBytesSync();
      final offending = [
        for (final (i, b) in bytes.indexed)
          if (b > 0x7e || (b < 0x20 && b != 0x0a && b != 0x0d && b != 0x09)) i,
      ];
      expect(offending, isEmpty, reason: '$path: bytes at $offending');
    }
  });

  test('run.cmd starts run.ps1 beside it with every argument, past the '
      "execution policy, and returns the app's exit code", () {
    final lines = File('tool/windows/run.cmd').readAsLinesSync();
    expect(
      lines,
      contains(
        r'"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" '
        '-NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*',
      ),
    );
    expect(lines.last, 'exit /b %ERRORLEVEL%');
  });

  test('a Windows checkout keeps LF in the knowledge-base articles (the '
      'prebuilt index is keyed by their bytes) and CRLF in the launchers', () {
    final git = Process.runSync('git', [
      'check-attr',
      'eol',
      '--',
      'assets/kb/embeddinggemma.md',
      'tool/windows/run.cmd',
      'tool/linux/run.sh',
    ]);
    if (git.exitCode != 0) {
      markTestSkipped('not a git checkout: ${git.stderr}');
      return;
    }
    expect((git.stdout as String).trim().split('\n'), [
      'assets/kb/embeddinggemma.md: eol: lf',
      'tool/windows/run.cmd: eol: crlf',
      'tool/linux/run.sh: eol: lf',
    ]);
  });

  test('no PowerShell 7 syntax (5.1 is the default on Windows 11)', () {
    final seven = <String, RegExp>{
      'pipeline chain && or ||': RegExp(r'&&|\|\|'),
      'null-coalescing ?? or ??=': RegExp(r'\?\?'),
      'null-conditional ?. or ?[': RegExp(r'\$\w+\?(\.|\[)'),
      'ternary ? :': RegExp(r'\s\?\s[^:]+\s:\s'),
      'ConvertFrom-Json -AsHashtable': RegExp('-AsHashtable'),
      'ForEach-Object -Parallel': RegExp('-Parallel'),
      'Join-Path -AdditionalChildPath': RegExp('-AdditionalChildPath'),
      r'$PSStyle': RegExp(r'\$PSStyle'),
    };
    for (final path in _scripts.where((p) => p.endsWith('.ps1'))) {
      final code = _code(path);
      for (final MapEntry(key: what, value: pattern) in seven.entries) {
        final hits = [
          for (final (i, line) in code.indexed)
            if (pattern.hasMatch(line)) '${i + 1}: $line',
        ];
        expect(hits, isEmpty, reason: '$path uses $what');
      }
    }
  });

  test('package.ps1 ships every licence file package.sh ships (libjpeg-turbo '
      'only goes with the Linux bundle, which ships the library) under the '
      'same names', () {
    final sh = File('tool/linux/package.sh').readAsStringSync();
    final ps1 = File('tool/windows/package.ps1').readAsStringSync();
    final copies = RegExp(r'cp "\$ROOT/([^"]+)" "\$DEST/licenses/([^"]+)"')
        .allMatches(sh)
        .map((m) => (source: m.group(1)!, name: m.group(2)!));
    final shared = copies
        .where((c) => !c.source.contains('libjpeg-turbo'))
        .toList();
    expect(shared, hasLength(4), reason: 'the copies package.sh makes');
    for (final c in shared) {
      expect(
        ps1,
        contains("'${c.source.replaceAll('/', r'\')}'"),
        reason: c.source,
      );
      expect(ps1, contains("'${c.name}'"), reason: c.name);
    }
    expect(sh, contains('LICENSE-app-Apache-2.0.txt'));
    expect(ps1, contains("'LICENSE-app-Apache-2.0.txt'"));
    // What Windows adds: the notices of the LiteRT-LM native bundle beside
    // the app (DirectX Shader Compiler, OpenVINO, oneTBB, ...) and
    // flutter_litert's licence for its TFLite C library.
    expect(ps1, contains("'flutter_edge_ai_litertlm-NOTICES.txt'"));
    expect(ps1, contains("'flutter_litert-LICENSE.txt'"));
  });

  test('pwsh parses both scripts without errors (when installed)', () {
    final which = Process.runSync('/bin/sh', ['-c', 'command -v pwsh']);
    final pwsh = (which.stdout as String).trim();
    if (pwsh.isEmpty) {
      markTestSkipped('pwsh not installed; the text checks above still run');
      return;
    }
    for (final path in _scripts.where((p) => p.endsWith('.ps1'))) {
      final result = Process.runSync(pwsh, [
        '-NoProfile',
        '-Command',
        r'$e = $null; '
            '[System.Management.Automation.Language.Parser]::ParseFile('
            "'${File(path).absolute.path}', [ref]\$null, [ref]\$e) | "
            r'Out-Null; $e | ForEach-Object { $_.ToString() }; '
            r'exit $e.Count',
      ]);
      expect(result.exitCode, 0, reason: '$path: ${result.stdout}');
    }
  });
}
