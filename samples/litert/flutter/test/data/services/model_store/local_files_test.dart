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
import 'package:litert_edge_demos/data/services/model_store/local_files.dart';
import 'package:path/path.dart' as p;

void main() {
  Future<Directory> Function() docs(String path) =>
      () async => Directory(path);

  group('Windows paths', () {
    Future<String> resolve(String path) => resolveLocalPath(
      path,
      style: p.windows,
      documents: docs(r'C:\Users\me\Documents'),
    );

    test('a drive path, either slash, a UNC share or a rooted path is '
        'absolute and kept as given', () async {
      for (final path in [
        r'C:\Users\me\litert-demos\models\gemma-4-E2B-it.litertlm',
        'D:/models/gemma.litertlm',
        r'\\nas\share\gemma.litertlm',
        r'\models\gemma.litertlm',
      ]) {
        expect(await resolve(path), path, reason: path);
      }
    });

    test('a bare name or a relative path is put under the documents '
        'directory with backslashes', () async {
      expect(
        await resolve('gemma.litertlm'),
        r'C:\Users\me\Documents\gemma.litertlm',
      );
      expect(
        await resolve(r'models\gemma.litertlm'),
        r'C:\Users\me\Documents\models\gemma.litertlm',
      );
    });
  });

  group('POSIX paths', () {
    Future<String> resolve(String path) => resolveLocalPath(
      path,
      style: p.posix,
      documents: docs('/home/me/Documents'),
    );

    test('/… is absolute and kept as given', () async {
      expect(
        await resolve('/home/me/Downloads/gemma.litertlm'),
        '/home/me/Downloads/gemma.litertlm',
      );
    });

    test('a relative path goes under the documents directory; a Windows '
        'drive path is relative there', () async {
      expect(
        await resolve('models/gemma.litertlm'),
        '/home/me/Documents/models/gemma.litertlm',
      );
      expect(
        await resolve(r'C:\gemma.litertlm'),
        r'/home/me/Documents/C:\gemma.litertlm',
      );
    });
  });

  test('without a style, the running platform decides', () async {
    final absolute = Platform.isWindows ? r'C:\m.litertlm' : '/m.litertlm';
    expect(
      await resolveLocalPath(absolute, documents: docs('unused')),
      absolute,
    );
  });
}
