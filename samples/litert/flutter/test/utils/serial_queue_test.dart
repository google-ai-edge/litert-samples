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
import 'package:litert_edge_demos/utils/serial_queue.dart';

void main() {
  test('runs one operation at a time, in the order queued', () async {
    final queue = SerialQueue();
    final log = <String>[];
    final first = Completer<void>();
    final a = queue.run(() async {
      log.add('a start');
      await first.future;
      log.add('a end');
      return 'a';
    });
    final b = queue.run(() async {
      log.add('b');
      return 'b';
    });
    await pumpEventQueue();
    expect(log, ['a start'], reason: 'b waits for a');

    first.complete();
    expect(await a, 'a');
    expect(await b, 'b');
    expect(log, ['a start', 'a end', 'b']);
  });

  test('a failed operation fails its own caller only; onError sees it and '
      'the next one runs', () async {
    final seen = <Object>[];
    final queue = SerialQueue(onError: (e, _) => seen.add(e));
    final failed = queue.run<int>(() async => throw StateError('broken'));
    final thrownSync = queue.run<int>(() => throw const FormatException('x'));
    final next = queue.run(() async => 3);

    await expectLater(failed, throwsStateError);
    await expectLater(thrownSync, throwsFormatException);
    expect(await next, 3);
    expect(seen, [isStateError, isFormatException]);
  });

  test('with onError, a failed operation whose future was dropped is no '
      'uncaught error; without it, it is', () async {
    Future<List<Object>> uncaughtWith(SerialQueue queue) async {
      final uncaught = <Object>[];
      await runZonedGuarded(() async {
        unawaited(queue.run<void>(() async => throw StateError('dropped')));
        await queue.idle;
        await pumpEventQueue();
      }, (e, _) => uncaught.add(e));
      return uncaught;
    }

    expect(await uncaughtWith(SerialQueue(onError: (_, _) {})), isEmpty);
    expect(await uncaughtWith(SerialQueue()), [isStateError]);
  });

  test('default: every operation starts after the code that queued it, the '
      'first one and one queued when idle too', () async {
    final queue = SerialQueue();
    final log = <String>[];
    final first = queue.run(() async => log.add('first'));
    log.add('caller');
    await first;
    final later = queue.run(() async => log.add('later'));
    log.add('caller again');
    await later;
    expect(log, ['caller', 'first', 'caller again', 'later']);
  });

  test('default: a request made right after one is queued can supersede it '
      '(the owner checks inside the operation)', () async {
    final queue = SerialQueue();
    String? requested;
    final ran = <String>[];
    Future<void> activate(String id) {
      requested = id;
      return queue.run(() async {
        if (requested != id) return; // superseded
        ran.add(id);
      });
    }

    await Future.wait([activate('a'), activate('b')]);
    expect(ran, ['b']);
  });

  test('firstAtOnce: the first operation starts inside run, later ones after '
      'the code that queued them', () async {
    final queue = SerialQueue.firstAtOnce();
    final log = <String>[];
    final first = queue.run(() async => log.add('first'));
    log.add('caller');
    await first;
    final later = queue.run(() async => log.add('later'));
    log.add('caller again');
    await later;
    expect(log, ['first', 'caller', 'caller again', 'later']);
  });

  test('atOnceWhenIdle: with none pending it starts inside run; queued '
      'behind another it waits; idle again it starts at once', () async {
    final queue = SerialQueue.atOnceWhenIdle();
    final log = <String>[];
    final hold = Completer<void>();
    final a = queue.run(() async {
      log.add('a');
      await hold.future;
    });
    final b = queue.run(() async => log.add('b'));
    log.add('caller');
    expect(log, ['a', 'caller']);

    hold.complete();
    await a;
    await b;
    expect(log, ['a', 'caller', 'b']);

    final c = queue.run(() async => log.add('c'));
    expect(log.last, 'c', reason: 'idle again: synchronous start');
    await c;
  });

  for (final (name, make) in <(String, SerialQueue Function())>[
    ('default', SerialQueue.new),
    ('firstAtOnce', SerialQueue.firstAtOnce),
    ('atOnceWhenIdle', SerialQueue.atOnceWhenIdle),
  ]) {
    test('$name: idle completes once everything queued so far has ended, '
        'never with an error', () async {
      final queue = make();
      await queue.idle; // nothing queued: at once

      final hold = Completer<void>();
      var ended = false;
      final failing = queue.run<void>(() async {
        await hold.future;
        throw StateError('broken');
      });
      final last = queue.run(() async => ended = true);
      var idle = false;
      unawaited(queue.idle.then((_) => idle = true));
      await pumpEventQueue();
      expect(idle, isFalse);

      hold.complete();
      await expectLater(failing, throwsStateError);
      await last;
      await queue.idle;
      expect((ended, idle), (true, true));
    });
  }
}
