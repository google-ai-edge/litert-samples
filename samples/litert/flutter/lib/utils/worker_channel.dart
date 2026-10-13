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

/// Why a [WorkerChannel] call failed; the [WorkerProtocol] turns it into the
/// service's own exception.
enum WorkerFailure {
  /// The isolate did not spawn, or its handshake failed or timed out.
  startFailed,

  /// A request after the worker died or [WorkerChannel.close] began.
  notRunning,

  /// The worker answered the request with an error.
  replyError,

  /// The worker died, or was closed or killed, with the request pending.
  lost,
}

/// What a [WorkerChannel] reports for the service's log.
enum WorkerEvent {
  /// A message the protocol does not know (detail: the message).
  unexpectedMessage,

  /// An uncaught error ended the worker (detail: the error and its stack).
  crashed,

  /// The worker exited or crashed on its own (detail: the reason). Not
  /// reported once a close or kill began.
  died,

  /// The close request failed (detail: the exception).
  closeFailed,

  /// The worker neither answered the close request nor exited within the
  /// close timeout (detail: the timeout); it is killed.
  closeTimedOut,
}

/// A worker's answer to request [id]: [value], or [error] when not null.
typedef WorkerReply = ({int id, Object? value, String? error});

/// How one kind of worker talks, and how its failures read: a service's
/// half of a [WorkerChannel].
final class const WorkerProtocol({
  /// The isolate's debug name; it also names the ports (`detector replies`).
  required final String name,

  /// How a reason names the worker: `the worker` gives `the worker exited`.
  required final String noun,

  /// The reply [message] carries; null for a message the protocol does not
  /// know (logged as [WorkerEvent.unexpectedMessage]).
  required final WorkerReply? Function(Object? message) parseReply,

  /// The message, sent as request `id`, that asks the worker to free what it
  /// holds and exit. The worker may answer it, or just exit.
  required final Object? Function(int id) closeMessage,

  /// The service's exception for a failure of [WorkerFailure] `kind`.
  required final Exception Function(WorkerFailure kind, String reason) failure,

  /// The service's log line for an event, if it has one.
  required final void Function(WorkerEvent event, String detail) log,
});

/// [bytes] packed to move to another isolate: one copy here, none when the
/// receiver calls [receiveBytes].
TransferableTypedData transferBytes(Uint8List bytes) =>
    TransferableTypedData.fromList([bytes]);

/// The bytes [data] carries. A [TransferableTypedData] gives them up once.
Uint8List receiveBytes(TransferableTypedData data) =>
    data.materialize().asUint8List();

/// A long-lived worker isolate spoken to by request and reply.
///
/// - [spawn] starts the isolate with `onError` and `onExit` ports and waits,
///   at most the handshake timeout, for its first message: the [SendPort]
///   requests go to, or a [String] saying why it could not start.
/// - [request] sends the message its builder makes for a fresh id; the reply
///   with that id (as the protocol parses it) completes it. Messages may carry
///   [TransferableTypedData] both ways ([transferBytes], [receiveBytes]).
/// - A crash (logged with its error and stack) or an exit fails every pending
///   request, and every later one at once.
/// - [close] sends the protocol's close message, waits for its answer or the
///   exit (at most the timeout), then kills the isolate; [kill] does not ask.
///   Both close the ports and fail whatever is pending.
///
/// Every failure is the protocol's exception ([WorkerProtocol.failure]).
final class WorkerChannel {
  WorkerChannel._(this._protocol)
    : _replies = ReceivePort('${_protocol.name} replies'),
      _errors = ReceivePort('${_protocol.name} errors'),
      _exits = ReceivePort('${_protocol.name} exit') {
    // A spawn that throws leaves the handshake to fail with nobody waiting.
    _ready.future.ignore();
  }

  final WorkerProtocol _protocol;
  final ReceivePort _replies;
  final ReceivePort _errors;
  final ReceivePort _exits;
  final Completer<SendPort> _ready = Completer();
  final Completer<void> _exited = Completer();
  final Map<int, Completer<Object?>> _pending = {};

  /// Set by the handshake, before [spawn] returns the channel.
  late final SendPort _requests;
  Isolate? _isolate;
  int _nextId = 0;
  String? _deadReason;
  bool _closing = false;
  Future<void>? _closed;

  /// Spawns `entryPoint(boot(replyTo))` and waits up to [handshakeTimeout]
  /// for the worker's [SendPort]. A failure kills the isolate and throws
  /// [WorkerFailure.startFailed] as the protocol's exception.
  static Future<WorkerChannel> spawn<B>({
    required WorkerProtocol protocol,
    required void Function(B boot) entryPoint,
    required B Function(SendPort replyTo) boot,
    required Duration handshakeTimeout,
  }) async {
    final channel = WorkerChannel._(protocol);
    channel._replies.listen(channel._onReply);
    channel._errors.listen(channel._onError);
    channel._exits.listen((_) => channel._onExit());
    try {
      channel._isolate = await Isolate.spawn<B>(
        entryPoint,
        boot(channel._replies.sendPort),
        onError: channel._errors.sendPort,
        onExit: channel._exits.sendPort,
        debugName: protocol.name,
      );
      channel._requests = await channel._ready.future.timeout(handshakeTimeout);
      return channel;
    } catch (e) {
      channel.kill();
      throw protocol.failure(
        WorkerFailure.startFailed,
        e is _StartFailure ? e.reason : '$e',
      );
    }
  }

  /// Why the worker stopped on its own (it exited or crashed); null while
  /// it runs, and once [close] or [kill] began.
  String? get failure => _closing ? null : _deadReason;

  /// Sends the message [build] makes for a fresh request id and completes
  /// with the worker's value, or fails with the protocol's exception.
  Future<Object?> request(Object? Function(int id) build) {
    final reason = _deadReason ?? (_closing ? 'closed' : null);
    if (reason != null) {
      return Future.error(_protocol.failure(WorkerFailure.notRunning, reason));
    }
    return _send(build);
  }

  /// Asks the worker to exit (the protocol's close message) and waits for
  /// its answer or its exit, at most [timeout], then kills it. Pending
  /// requests fail. Safe to call more than once: later calls wait for the
  /// first.
  Future<void> close({required Duration timeout}) =>
      _closed ??= _close(timeout);

  Future<void> _close(Duration timeout) async {
    _closing = true;
    if (_deadReason == null) {
      try {
        await Future.any<Object?>([
          _send(_protocol.closeMessage),
          _exited.future,
        ]).timeout(timeout);
      } on TimeoutException {
        _protocol.log(
          WorkerEvent.closeTimedOut,
          '${timeout.inMilliseconds} ms',
        );
      } on Exception catch (e) {
        _protocol.log(WorkerEvent.closeFailed, '$e');
      }
    }
    kill();
  }

  /// Ends the isolate without asking (it may be stuck in a native call),
  /// fails whatever is pending and closes the ports.
  void kill() {
    _closing = true;
    _isolate?.kill(priority: Isolate.immediate);
    _die('closed');
  }

  Future<Object?> _send(Object? Function(int id) build) {
    final id = _nextId++;
    // An unsendable message throws here, before anything waits for it.
    _requests.send(build(id));
    return (_pending[id] = Completer<Object?>()).future;
  }

  void _onReply(Object? message) {
    if (!_ready.isCompleted) {
      switch (message) {
        case final SendPort requests:
          _ready.complete(requests);
          return;
        case final String reason:
          _ready.completeError(_StartFailure(reason));
          return;
      }
    }
    final reply = _protocol.parseReply(message);
    if (reply == null) {
      _protocol.log(WorkerEvent.unexpectedMessage, '$message');
      return;
    }
    final pending = _pending.remove(reply.id);
    if (pending == null) return; // no request with this id is waiting
    if (reply.error case final error?) {
      pending.completeError(_protocol.failure(WorkerFailure.replyError, error));
    } else {
      pending.complete(reply.value);
    }
  }

  void _onError(Object? message) {
    final description = switch (message) {
      [final error, final stack] => '$error\n$stack',
      _ => '$message',
    };
    _protocol.log(WorkerEvent.crashed, description);
    _die('${_protocol.noun} crashed: ${description.split('\n').first}');
  }

  void _onExit() {
    // Before the requests fail: a close waiting for the exit sees it first.
    if (!_exited.isCompleted) _exited.complete();
    _die('${_protocol.noun} exited');
  }

  /// Fails the handshake and everything pending; later requests fail at
  /// once. The first reason is the one kept.
  void _die(String reason) {
    final first = _deadReason == null;
    final dead = _deadReason ??= reason;
    if (first && !_closing) _protocol.log(WorkerEvent.died, dead);
    if (!_ready.isCompleted) _ready.completeError(_StartFailure(dead));
    final pending = [..._pending.values];
    _pending.clear();
    for (final request in pending) {
      request.completeError(_protocol.failure(WorkerFailure.lost, dead));
    }
    _replies.close();
    _errors.close();
    _exits.close();
  }
}

/// Why the handshake failed, as the worker or [WorkerChannel._die] said it.
final class const _StartFailure(final String reason) implements Exception {
  @override
  String toString() => reason;
}
