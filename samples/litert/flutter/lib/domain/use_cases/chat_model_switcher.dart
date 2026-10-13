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

import 'package:flutter/foundation.dart';

import '../../data/repositories/conversation_repository.dart';
import '../../utils/result.dart';
import '../../utils/serial_queue.dart';

/// The one owner of the chat model's engine: loads it again after its choice or
/// settings changed, frees it for the self-test, and runs the model setup.
/// These never overlap: each runs inside [exclusive], one after the other, and
/// [busy] is true from the moment one is asked for until the last ends, so the
/// screens disable what would touch the engine (Apply, Use this model, import,
/// download, Run self-test) without a window between a tap and the work
/// starting.
///
/// The open chat goes first on every reload or unload: its native session
/// dies with the model, and a chat built on the old model must never outlive
/// it.
final class ChatModelSwitcher {
  ChatModelSwitcher({
    required this._conversation,
    required this._reloadChatModel,
    required this._unloadChatModel,
    required this._refuseChatModelLoads,
  });

  final ConversationRepository _conversation;

  /// `ModelRepository.reloadChatModel`.
  final Future<Result<void>> Function() _reloadChatModel;

  /// `ModelRepository.unloadChatModel`.
  final Future<void> Function() _unloadChatModel;

  /// `ModelRepository.refuseChatModelLoads`: the gate every chat model load
  /// (a reload, the model setup) goes through.
  final void Function(String reason) _refuseChatModelLoads;

  /// App-scoped like the switcher (never disposed: screens may still be
  /// detaching their listeners when the app closes).
  final ValueNotifier<bool> _busy = ValueNotifier(false);
  int _holders = 0;
  final SerialQueue _queue = SerialQueue();

  /// An exclusive operation runs or waits: the chat model is in use.
  ValueListenable<bool> get busy => _busy;

  /// Runs [body] alone, after any earlier one. [busy] turns true
  /// synchronously, before this returns; [body] reloads and unloads through
  /// its [ChatModelOps] (calling [reload] or [unload] inside it would wait
  /// for itself).
  Future<T> exclusive<T>(Future<T> Function(ChatModelOps ops) body) {
    _holders++;
    _busy.value = true;
    return _queue.run(() async {
      try {
        return await body(ChatModelOps._(this));
      } finally {
        if (--_holders == 0) _busy.value = false;
      }
    });
  }

  /// Releases the chat, then closes the chat model and loads it from the
  /// current plan. [ReplyStillStoppingException] when the chat cannot be
  /// released: then the engine is left as it is.
  Future<Result<void>> reload() => exclusive((ops) => ops.reload());

  /// Releases the chat and closes the chat model (flutter_edge_ai holds one
  /// model per process; the self-test loads its own).
  /// [ReplyStillStoppingException] when the chat cannot be released: then
  /// the engine is left as it is.
  Future<Result<void>> unload() => exclusive((ops) => ops.unload());

  Future<Result<void>> _reloadNow() async {
    if (await _release() case final refused?) return refused;
    return _reloadChatModel();
  }

  Future<Result<void>> _unloadNow() async {
    if (await _release() case final refused?) return refused;
    await _unloadChatModel();
    return const Result.ok(null);
  }

  /// Releases the chat; null once it is gone. A stopped reply that still
  /// generates past the conversation's stop timeout keeps it: closing the
  /// chat or the model under a running native generation is never safe, so
  /// the reload or unload is refused with a reason the user can act on.
  Future<Result<void>?> _release() async {
    switch (await _conversation.release()) {
      case Ok():
        return null;
      case Error(:final error):
        debugPrint(
          '[ChatModel] switch refused: the chat could not be released: '
          '$error',
        );
        return const Result.error(ReplyStillStoppingException());
    }
  }
}

/// A reload or unload refused because the chat's last reply is still
/// stopping (its native turn ignored the stop past the stop timeout): the
/// chat and the engine are left as they are.
final class ReplyStillStoppingException implements Exception {
  const ReplyStillStoppingException();

  @override
  String toString() =>
      'The previous reply is still stopping; try again in a moment';
}

/// What a body run by [ChatModelSwitcher.exclusive] may do with the chat
/// model, without waiting for itself.
final class ChatModelOps {
  ChatModelOps._(this._switcher);

  final ChatModelSwitcher _switcher;

  Future<Result<void>> reload() => _switcher._reloadNow();

  Future<Result<void>> unload() => _switcher._unloadNow();

  /// The engine is held by something the app can no longer close (a
  /// self-test that did not end): no chat model loads again until the app
  /// restarts, and the chat slot says [reason].
  void refuseLoads(String reason) => _switcher._refuseChatModelLoads(reason);
}
