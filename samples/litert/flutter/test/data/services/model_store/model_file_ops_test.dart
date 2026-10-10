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
import 'package:litert_edge_demos/data/services/model_store/model_file_ops.dart';

import '../../../support/failing_file_workers.dart';

void main() {
  late Directory dir;
  late File file;

  setUp(() {
    dir = Directory.systemTemp.createTempSync('model_file_ops_test');
    file = File('${dir.path}/model.bin')..writeAsBytesSync(List.filled(64, 7));
  });

  tearDown(() => dir.deleteSync(recursive: true));

  // A worker failure must be an Exception: every handler above the file
  // operations (the model store's) catches `on Exception`, and an Error
  // escaped them as an uncaught zone error, the row stuck on Verifying.
  for (final (what, worker, message) in [
    ('crashes', crashingFileWorker, 'file worker crashed: '),
    ('reports an error', erroringFileWorker, 'the worker reported a failure'),
    ('exits without a result', silentFileWorker, 'without a result'),
  ]) {
    test('a worker that $what fails the hash and the copy with '
        'FileOpFailedException, an Exception', () async {
      final ops = ModelFileOps(worker: worker);

      await expectLater(
        ops.sha256OfFile(file.path, onProgress: (_, _) {}),
        throwsA(
          isA<Exception>().having((e) => '$e', 'message', contains(message)),
        ),
      );
      await expectLater(
        ops.copyAndHash(file.path, '${dir.path}/copy', onProgress: (_, _) {}),
        throwsA(isA<FileOpFailedException>()),
      );
    });
  }

  test('the real worker still hashes', () async {
    final hex = await const ModelFileOps().sha256OfFile(
      file.path,
      onProgress: (_, _) {},
    );
    expect(hex, hasLength(64));
  });
}
