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

// tool/models.lock, the list tool/fetch_models.sh fetches the built-in
// models from (they are not in git), agrees with what the app expects of
// each file (bundled_model_files.dart, detector_spec.dart), with the notice
// (assets/models/NOTICE.md), with the asset list in pubspec.yaml and with
// .gitignore. None of this needs the model files themselves.
import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/services/model_store/bundled_model_files.dart';
import 'package:litert_edge_demos/domain/models/detector_spec.dart';

const _lock = 'tool/models.lock';

/// One `hf` or `yolo26n-rawhead` line of the lock.
typedef _Entry = ({
  String kind,
  String dest,
  int size,
  String sha256,
  String access,
  String repo,
  String revision,
  String path,
  List<String> extra,
});

List<List<String>> _lines() => [
  for (final line in File(_lock).readAsLinesSync())
    if (line.trim().isNotEmpty && !line.startsWith('#'))
      line.trim().split(RegExp(r'\s+')),
];

List<_Entry> _entries() => [
  for (final f in _lines())
    if (f.first != 'pip')
      (
        kind: f[0],
        dest: f[1],
        size: int.parse(f[2]),
        sha256: f[3],
        access: f[4],
        repo: f[5],
        revision: f[6],
        path: f[7],
        extra: f.sublist(8),
      ),
];

void main() {
  test('every line is a known kind with the right number of fields', () {
    for (final fields in _lines()) {
      switch (fields.first) {
        case 'pip':
          expect(fields.length, greaterThan(1), reason: fields.join(' '));
          for (final requirement in fields.skip(1)) {
            expect(
              requirement,
              matches(RegExp(r'^[A-Za-z0-9_.-]+==[0-9][A-Za-z0-9.]*$')),
              reason: 'pinned exactly',
            );
          }
        case 'hf':
          expect(fields, hasLength(8), reason: fields.join(' '));
        case 'yolo26n-rawhead':
          expect(fields, hasLength(10), reason: fields.join(' '));
        default:
          fail('unknown kind: ${fields.join(' ')}');
      }
    }
  });

  test('each file is pinned: a full commit, a repo, a SHA-256, an access', () {
    final entries = _entries();
    expect(entries, isNotEmpty);
    for (final e in entries) {
      expect(e.revision, matches(RegExp(r'^[0-9a-f]{40}$')), reason: e.dest);
      expect(e.repo, matches(RegExp(r'^[\w.-]+/[\w.-]+$')), reason: e.dest);
      expect(e.sha256, matches(RegExp(r'^[0-9a-f]{64}$')), reason: e.dest);
      expect(e.access, anyOf('public', 'gated'), reason: e.dest);
      expect(e.size, greaterThan(0), reason: e.dest);
    }
    expect(
      {for (final e in entries) e.dest},
      hasLength(entries.length),
      reason: 'one line per file',
    );
    expect(
      {
        for (final e in entries)
          if (e.access == 'gated') e.repo,
      },
      {'litert-community/embeddinggemma-300m'},
      reason: 'fetch_models.sh asks for HF_TOKEN for this repo only',
    );
  });

  test('the files are exactly the ones the app checks, with the same size '
      'and SHA-256', () {
    final app = {
      for (final f in kBundledModelFiles) f.asset: (f.sizeBytes, f.sha256),
      kDetModelAsset: (kDetModelBytes, kDetModelSha256),
    };
    final lock = {
      for (final e in _entries()) 'assets/models/${e.dest}': (e.size, e.sha256),
    };
    expect(lock, app);
  });

  test('the YOLO26n line derives from Arm\'s original and pins the Python '
      'packages', () {
    final yolo = _entries().singleWhere((e) => e.kind == 'yolo26n-rawhead');
    expect('assets/models/${yolo.dest}', kDetModelAsset);
    expect(yolo.repo, 'Arm/yolo26n-fp16-litert');
    expect(yolo.extra, hasLength(2));
    expect(int.parse(yolo.extra[0]), isNot(yolo.size), reason: 'the original');
    expect(yolo.extra[1], matches(RegExp(r'^[0-9a-f]{64}$')));
    final pip = _lines().singleWhere((f) => f.first == 'pip');
    expect(pip, contains(startsWith('ai-edge-litert==')));
    expect(File('tool/prune_yolo26n_head.py').existsSync(), isTrue);
  });

  test('NOTICE.md documents every file\'s SHA-256 and repo', () {
    final notice = File('assets/models/NOTICE.md').readAsStringSync();
    for (final e in _entries()) {
      expect(notice, contains(e.sha256), reason: e.dest);
      expect(notice, contains(e.repo), reason: e.dest);
    }
  });

  test('pubspec.yaml declares every file as an asset (itself or its '
      'folder)', () {
    final declared = {
      for (final line in File('pubspec.yaml').readAsLinesSync())
        if (RegExp(r'^\s+- (assets/\S+)$').firstMatch(line) case final m?)
          m.group(1)!,
    };
    for (final e in _entries()) {
      final asset = 'assets/models/${e.dest}';
      final folder = '${asset.substring(0, asset.lastIndexOf('/'))}/';
      expect(
        declared.contains(asset) || declared.contains(folder),
        isTrue,
        reason: '$asset is not in pubspec.yaml',
      );
    }
  });

  test('git tracks only NOTICE.md under assets/models and ignores every '
      'file of the lock', () {
    final ProcessResult tracked;
    try {
      tracked = Process.runSync('git', ['ls-files', 'assets/models']);
    } on ProcessException {
      markTestSkipped('git is not installed');
      return;
    }
    if (tracked.exitCode != 0) {
      markTestSkipped('not a git checkout: ${tracked.stderr}');
      return;
    }
    expect(LineSplitter.split(tracked.stdout as String), [
      'assets/models/NOTICE.md',
    ]);
    final dests = [for (final e in _entries()) 'assets/models/${e.dest}'];
    final ignored = Process.runSync('git', [
      'check-ignore',
      '--no-index',
      ...dests,
    ]);
    expect(LineSplitter.split(ignored.stdout as String), dests);
  });
}
