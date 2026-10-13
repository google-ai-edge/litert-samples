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

import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/repositories/conversation_repository.dart';
import 'package:litert_edge_demos/domain/use_cases/chat_model_switcher.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../fakes/fake_conversation_repository.dart';

/// Review (exclusivity): one owner of the chat model's engine; reloads,
/// unloads, the self-test and the model setup never overlap, and [busy]
/// covers them from the call on.
void main() {
  late FakeConversationRepository conversation;
  late List<String> log;
  late ChatModelSwitcher switcher;
  Completer<void>? reloadGate;

  setUp(() {
    conversation = FakeConversationRepository();
    log = [];
    reloadGate = null;
    switcher = ChatModelSwitcher(
      conversation: conversation,
      reloadChatModel: () async {
        log.add('reload start');
        await reloadGate?.future;
        log.add('reload end');
        return const Result.ok(null);
      },
      unloadChatModel: () async => log.add('unload'),
      refuseChatModelLoads: (reason) => log.add('refuse: $reason'),
    );
  });
  tearDown(() => conversation.close());

  test('busy turns true synchronously, before any await, and false when '
      'the work ends', () async {
    expect(switcher.busy.value, isFalse);
    reloadGate = Completer<void>();

    final reloading = switcher.reload();

    expect(switcher.busy.value, isTrue, reason: 'no window after the call');
    await pumpEventQueue();
    expect(switcher.busy.value, isTrue);
    reloadGate!.complete();
    expect(await reloading, isA<Ok<void>>());
    expect(switcher.busy.value, isFalse);
    expect(conversation.releaseCalls, 1, reason: 'the chat goes first');
  });

  test('operations run one after the other; busy stays true until the last '
      'ends', () async {
    final gate = Completer<void>();
    final first = switcher.exclusive((ops) async {
      log.add('test start');
      await ops.unload();
      await gate.future;
      log.add('test end');
    });
    final second = switcher.reload();
    await pumpEventQueue();

    expect(log, ['test start', 'unload'], reason: 'the reload waits');
    gate.complete();
    await first;
    expect(switcher.busy.value, isTrue, reason: 'the reload still runs');
    await second;

    expect(log, [
      'test start',
      'unload',
      'test end',
      'reload start',
      'reload end',
    ]);
    expect(switcher.busy.value, isFalse);
  });

  group('the chat cannot be released (its stopped reply still generates '
      'past the stop timeout)', () {
    setUp(
      () => conversation.releaseError = const ConversationNotReadyException(
        'The previous reply did not finish within 5s of being stopped',
      ),
    );

    test('a reload is refused before the engine is touched, with a message '
        'the user can act on', () async {
      final result = await switcher.reload();

      expect(
        result,
        isA<Error<void>>().having(
          (e) => e.error,
          'error',
          isA<ReplyStillStoppingException>().having(
            (e) => '$e',
            'message',
            'The previous reply is still stopping; try again in a moment',
          ),
        ),
      );
      expect(log, isEmpty, reason: 'neither closed nor loaded');
      expect(switcher.busy.value, isFalse);
    });

    test('an unload is refused the same way; once the reply has ended, both '
        'run', () async {
      final result = await switcher.unload();

      expect(
        result,
        isA<Error<void>>().having(
          (e) => e.error,
          'error',
          isA<ReplyStillStoppingException>(),
        ),
      );
      expect(log, isEmpty);

      conversation.releaseError = null;
      expect(await switcher.unload(), isA<Ok<void>>());
      expect(await switcher.reload(), isA<Ok<void>>());
      expect(log, ['unload', 'reload start', 'reload end']);
    });
  });

  test('a body that throws releases the lock and busy', () async {
    await expectLater(
      switcher.exclusive<void>((_) async => throw StateError('crash')),
      throwsStateError,
    );
    expect(switcher.busy.value, isFalse);
    expect(await switcher.reload(), isA<Ok<void>>(), reason: 'not wedged');
  });
}
