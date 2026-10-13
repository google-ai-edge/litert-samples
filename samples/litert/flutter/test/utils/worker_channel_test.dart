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
import 'dart:isolate';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/utils/worker_channel.dart';

/// How long a test waits for something that must happen. Generous: a
/// loaded machine is slower, never different.
const _patience = Duration(seconds: 15);

/// The test worker. Boot: (reply port, mode, monitor port).
///
/// - `serve` answers `(id, op, payload)` requests with `(id, value, error)`;
///   `null` makes it exit without answering, `close` answers, then exits.
/// - `refuse` says why it cannot start instead of sending its port.
/// - `silent` never answers the handshake and pings the monitor until it is
///   killed.
/// - `crash` and `exit` end before the handshake.
void _worker((SendPort, String, SendPort?) boot) {
  final (replies, mode, monitor) = boot;
  switch (mode) {
    case 'refuse':
      replies.send('no library here');
      return;
    case 'silent':
      Timer.periodic(const Duration(milliseconds: 10), (_) {
        monitor?.send(true);
      });
      return;
    case 'crash':
      throw StateError('no handshake today');
    case 'exit':
      return; // no open ports: the isolate exits
  }
  final requests = ReceivePort('test worker requests');
  replies.send(requests.sendPort);
  requests.listen((message) {
    switch (message) {
      case null:
        requests.close(); // no open ports left: the isolate exits
      case (final int id, 'reverse', final TransferableTypedData bytes):
        final reversed = Uint8List.fromList(
          receiveBytes(bytes).reversed.toList(),
        );
        replies.send((id, transferBytes(reversed), null));
      case (final int id, 'fail', _):
        replies.send((id, null, 'bad input'));
      case (_, 'crash', _):
        throw StateError('the test worker blew up');
      case (_, 'exit', _):
        Isolate.exit();
      case (_, 'hang', _):
        return; // never answers
      case (_, 'busy', _):
        // Stuck, like a native call that never returns (but killable),
        // pinging the monitor every 10 ms while it runs.
        final watch = Stopwatch()..start();
        var pinged = 0;
        while (watch.elapsed < const Duration(seconds: 20)) {
          if (watch.elapsedMilliseconds - pinged >= 10) {
            pinged = watch.elapsedMilliseconds;
            monitor?.send(true);
          }
        }
      case (final int id, 'stray', _):
        replies.send(42); // not a reply this protocol knows
        replies.send((id, 'ok', null));
      case (final int id, 'close', _):
        replies.send((id, 'bye', null));
        requests.close();
    }
  });
}

/// The protocol's exception: which failure, and the channel's reason.
final class _TestFailure implements Exception {
  const _TestFailure(this.kind, this.reason);

  final WorkerFailure kind;
  final String reason;

  @override
  String toString() => '${kind.name}: $reason';
}

Matcher _fails(WorkerFailure kind, Object? reason) => throwsA(
  isA<_TestFailure>()
      .having((e) => e.kind, 'kind', kind)
      .having((e) => e.reason, 'reason', reason),
);

/// A protocol that records its log, closing as [closeMessage] says.
final class _Harness {
  _Harness({Object? Function(int id)? closeMessage})
    : _closeMessage = closeMessage ?? ((id) => (id, 'close', null));

  final Object? Function(int id) _closeMessage;
  final List<(WorkerEvent, String)> events = [];
  final List<WorkerChannel> _channels = [];

  late final WorkerProtocol protocol = WorkerProtocol(
    name: 'test',
    noun: 'the test worker',
    parseReply: (message) => switch (message) {
      (final int id, final Object? value, final String? error) => (
        id: id,
        value: value,
        error: error,
      ),
      _ => null,
    },
    closeMessage: _closeMessage,
    failure: _TestFailure.new,
    log: (event, detail) => events.add((event, detail)),
  );

  List<String> details(WorkerEvent event) => [
    for (final (e, detail) in events)
      if (e == event) detail,
  ];

  Future<WorkerChannel> spawn({
    String mode = 'serve',
    SendPort? monitor,
    Duration handshakeTimeout = _patience,
  }) async {
    final channel = await WorkerChannel.spawn(
      protocol: protocol,
      entryPoint: _worker,
      boot: (replyTo) => (replyTo, mode, monitor),
      handshakeTimeout: handshakeTimeout,
    );
    _channels.add(channel);
    return channel;
  }

  void killAll() {
    for (final channel in _channels) {
      channel.kill();
    }
  }
}

Future<Object?> _op(WorkerChannel channel, String op, [Object? payload]) =>
    channel.request((id) => (id, op, payload));

/// Counts the worker's pings: it is alive while they arrive.
final class _Monitor {
  _Monitor() {
    _port.listen((_) => pings++);
  }

  final ReceivePort _port = ReceivePort('test monitor');
  int pings = 0;

  SendPort get sendPort => _port.sendPort;

  /// Waits for the pings in flight to drain, then expects no more (a
  /// window in which something must NOT happen; a slow machine only
  /// widens it).
  Future<void> expectSilent() async {
    await Future<void>.delayed(const Duration(milliseconds: 200));
    final drained = pings;
    await Future<void>.delayed(const Duration(milliseconds: 300));
    expect(pings, drained, reason: 'the worker was killed');
  }

  void close() => _port.close();
}

void main() {
  late _Harness harness;

  setUp(() => harness = _Harness());
  tearDown(() => harness.killAll());

  test('requests are answered by id, bytes move both ways as '
      'TransferableTypedData, and a reply error is the protocol\'s '
      'exception', () async {
    final channel = await harness.spawn();

    final first = _op(
      channel,
      'reverse',
      transferBytes(Uint8List.fromList([1, 2, 3])),
    );
    final second = _op(
      channel,
      'reverse',
      transferBytes(Uint8List.fromList([7, 8])),
    );

    expect(receiveBytes((await second)! as TransferableTypedData), [8, 7]);
    expect(receiveBytes((await first)! as TransferableTypedData), [3, 2, 1]);
    await expectLater(
      _op(channel, 'fail'),
      _fails(WorkerFailure.replyError, 'bad input'),
    );
    expect(await _op(channel, 'stray'), 'ok', reason: 'still serving');
    expect(harness.details(WorkerEvent.unexpectedMessage), ['42']);
    expect(channel.failure, isNull);

    await channel.close(timeout: _patience).timeout(_patience);
    expect(harness.details(WorkerEvent.closeTimedOut), isEmpty);
    expect(harness.details(WorkerEvent.died), isEmpty);
  });

  group('start', () {
    test('a handshake that never comes times out: spawn fails and the '
        'worker is killed', () async {
      final monitor = _Monitor();
      addTearDown(monitor.close);
      final watch = Stopwatch()..start();

      await expectLater(
        harness.spawn(
          mode: 'silent',
          monitor: monitor.sendPort,
          handshakeTimeout: const Duration(milliseconds: 300),
        ),
        _fails(WorkerFailure.startFailed, contains('TimeoutException')),
      );

      expect(watch.elapsed, lessThan(const Duration(seconds: 5)));
      expect(monitor.pings, greaterThan(0), reason: 'the worker ran');
      await monitor.expectSilent();
      expect(harness.details(WorkerEvent.died), isEmpty);
    });

    test('a worker that cannot start says why', () async {
      await expectLater(
        harness.spawn(mode: 'refuse'),
        _fails(WorkerFailure.startFailed, 'no library here'),
      );
    });

    test('a crash before the handshake fails the start with its cause, '
        'logged with the stack', () async {
      const reason = 'the test worker crashed: Bad state: no handshake today';

      await expectLater(
        harness.spawn(mode: 'crash'),
        _fails(WorkerFailure.startFailed, reason),
      );

      expect(
        harness.details(WorkerEvent.crashed).single,
        startsWith('Bad state: no handshake today\n'),
      );
      expect(harness.details(WorkerEvent.died), [reason]);
    });

    test('an exit before the handshake fails the start', () async {
      await expectLater(
        harness.spawn(mode: 'exit'),
        _fails(WorkerFailure.startFailed, 'the test worker exited'),
      );
      expect(harness.details(WorkerEvent.died), ['the test worker exited']);
    });
  });

  group('a worker that dies', () {
    test('a crash mid-request fails every pending request with the cause, '
        'is logged with its error and stack, and later requests fail at '
        'once', () async {
      final channel = await harness.spawn();
      const reason =
          'the test worker crashed: Bad state: the test worker '
          'blew up';

      final hanging = expectLater(
        _op(channel, 'hang'),
        _fails(WorkerFailure.lost, reason),
      );
      final crashing = expectLater(
        _op(channel, 'crash'),
        _fails(WorkerFailure.lost, reason),
      );

      await crashing;
      await hanging;
      expect(channel.failure, reason);
      final crash = harness.details(WorkerEvent.crashed).single;
      expect(crash, startsWith('Bad state: the test worker blew up\n'));
      expect(crash, contains('_worker'), reason: 'the stack');
      expect(harness.details(WorkerEvent.died), [reason]);
      await expectLater(
        _op(channel, 'reverse'),
        _fails(WorkerFailure.notRunning, reason),
      );
      // Closing a dead worker does not wait for it.
      await channel.close(timeout: _patience).timeout(_patience);
      expect(harness.details(WorkerEvent.closeTimedOut), isEmpty);
    });

    test('an exit mid-request fails it with "exited"', () async {
      final channel = await harness.spawn();

      await expectLater(
        _op(channel, 'exit'),
        _fails(WorkerFailure.lost, 'the test worker exited'),
      );
      expect(channel.failure, 'the test worker exited');
    });
  });

  group('close', () {
    test('with a request pending: the worker exits (no answer to the close '
        'message), the request fails and close returns', () async {
      harness = _Harness(closeMessage: (_) => null);
      final channel = await harness.spawn();
      final hanging = _op(channel, 'hang');
      final failed = expectLater(
        hanging,
        _fails(WorkerFailure.lost, 'the test worker exited'),
      );

      await channel.close(timeout: _patience).timeout(_patience);

      await failed;
      expect(harness.details(WorkerEvent.closeTimedOut), isEmpty);
      expect(harness.details(WorkerEvent.closeFailed), isEmpty);
      expect(harness.details(WorkerEvent.died), isEmpty, reason: 'closing');
      expect(channel.failure, isNull, reason: 'closed, not failed');
    });

    test('a worker stuck in a request is killed after the close timeout; the '
        'request fails', () async {
      final monitor = _Monitor();
      addTearDown(monitor.close);
      final channel = await harness.spawn(monitor: monitor.sendPort);
      final busy = _op(channel, 'busy');
      final failed = expectLater(busy, _fails(WorkerFailure.lost, 'closed'));
      final watch = Stopwatch()..start();

      await channel.close(timeout: const Duration(milliseconds: 300));

      expect(watch.elapsed, lessThan(const Duration(seconds: 5)));
      await failed;
      expect(harness.details(WorkerEvent.closeTimedOut), ['300 ms']);
      expect(monitor.pings, greaterThan(0), reason: 'it was busy');
      await monitor.expectSilent();
    });

    test('a worker that crashes on the close message: close reports it and '
        'returns at once', () async {
      harness = _Harness(closeMessage: (id) => (id, 'crash', null));
      final channel = await harness.spawn();
      const reason =
          'the test worker crashed: Bad state: the test worker '
          'blew up';

      await channel.close(timeout: _patience).timeout(_patience);

      expect(harness.details(WorkerEvent.closeFailed), ['lost: $reason']);
      expect(
        harness.details(WorkerEvent.crashed).single,
        startsWith('Bad state: the test worker blew up\n'),
      );
      expect(harness.details(WorkerEvent.died), isEmpty, reason: 'closing');
      expect(harness.details(WorkerEvent.closeTimedOut), isEmpty);
    });

    test('a request during or after close fails at once; a second close '
        'waits for the first', () async {
      final channel = await harness.spawn();

      final closing = channel.close(timeout: _patience);
      await expectLater(
        _op(channel, 'reverse'),
        _fails(WorkerFailure.notRunning, 'closed'),
      );
      expect(channel.close(timeout: _patience), same(closing));
      await closing.timeout(_patience);

      await expectLater(
        _op(channel, 'reverse'),
        _fails(WorkerFailure.notRunning, 'closed'),
      );
      expect(channel.failure, isNull);
    });

    test(
      'kill fails what is pending at once, without a close message',
      () async {
        final channel = await harness.spawn();
        final hanging = expectLater(
          _op(channel, 'hang'),
          _fails(WorkerFailure.lost, 'closed'),
        );

        channel.kill();

        await hanging;
        expect(harness.details(WorkerEvent.died), isEmpty);
        await expectLater(
          _op(channel, 'reverse'),
          _fails(WorkerFailure.notRunning, 'closed'),
        );
      },
    );
  });
}
