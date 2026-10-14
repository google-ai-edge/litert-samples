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

import 'dart:async';
import 'dart:typed_data';

import 'package:fake_async/fake_async.dart';
import 'package:flutter_test/flutter_test.dart';

import '../../../integration_test/support/fixtures.dart';

void main() {
  test('the fixture mic plays its recording in 64 ms chunks from each start '
      'and says when every byte was sent', () {
    fakeAsync((async) {
      final pcm = Uint8List.fromList(List.generate(5000, (i) => i % 251));
      final mic = FixtureMicService(pcm);
      final got = <int>[];
      unawaited(
        mic
            .startPcm16(sampleRate: 16000, onFormatChanged: (_) {})
            .then((stream) => stream.listen(got.addAll)),
      );
      async.flushMicrotasks();
      expect(mic.starts, 1);
      expect(mic.played, isFalse);

      async.elapse(const Duration(milliseconds: 128));
      expect(got, hasLength(4096), reason: 'two 2048-byte chunks');
      expect(mic.played, isFalse);

      async.elapse(const Duration(milliseconds: 64));
      expect(got, pcm);
      expect(mic.played, isTrue);

      // The next start plays from the beginning again.
      unawaited(mic.startPcm16(sampleRate: 16000, onFormatChanged: (_) {}));
      async.flushMicrotasks();
      expect(mic.played, isFalse);
      async.elapse(const Duration(milliseconds: 192));
      expect(mic.played, isTrue);
      unawaited(mic.dispose());
      async.flushMicrotasks();
    });
  });
}
