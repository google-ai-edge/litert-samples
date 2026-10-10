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

import 'package:flutter/foundation.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/domain/use_cases/turn_responder_factory.dart';

void main() {
  group('whenPreviousReplyEnded', () {
    late ValueNotifier<bool> generating;
    late List<String?> logs;
    late DebugPrintCallback previous;

    setUp(() {
      generating = ValueNotifier(false);
      logs = [];
      previous = debugPrint;
      debugPrint = (message, {wrapWidth}) => logs.add(message);
    });

    tearDown(() {
      debugPrint = previous;
      generating.dispose();
    });

    test('idle: returns at once, says nothing', () async {
      await whenPreviousReplyEnded(
        generating,
        wait: const Duration(seconds: 5),
        tag: 'T',
      );
      expect(logs, isEmpty);
    });

    test('a reply that ends in time: returns when it ends', () async {
      generating.value = true;
      var done = false;
      final waiting = whenPreviousReplyEnded(
        generating,
        wait: const Duration(seconds: 5),
        tag: 'T',
      ).then((_) => done = true);
      await pumpEventQueue();
      expect(done, isFalse);

      generating.value = false;
      await waiting;
      expect(logs, isEmpty);
    });

    test('a reply that outlives the wait: logged with the tag, then it '
        'returns anyway', () async {
      generating.value = true;
      await whenPreviousReplyEnded(
        generating,
        wait: const Duration(milliseconds: 10),
        tag: 'CameraTurn',
      );
      expect(logs, [
        '[CameraTurn] the previous reply still runs after 0s; asking anyway',
      ]);
    });
  });
}
