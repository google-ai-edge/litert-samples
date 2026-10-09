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
import 'package:flutter_edge_ai_agent/flutter_edge_ai_agent.dart' show Skill;

import '../../../config/model_catalog.dart';
import '../../../utils/result.dart';
import '../conversation_repository.dart';

/// Builds the chat for a profile with a skill set (the chat slot's
/// rebuild).
typedef ChatRebuild = Future<Result<void>> Function(
  ConversationProfile profile,
  List<Skill> skills,
);

/// The conversation's opens and resets, served one rebuild at a time, the
/// newest request winning: a queued open that a later one replaces before it
/// runs is never built.
final class OpenQueue {
  /// [rebuild] builds one chat; it is never called twice at once.
  OpenQueue(this._rebuild);

  final ChatRebuild _rebuild;

  /// The profile of the newest [open] call, built or still queued.
  ConversationProfile? _requestedProfile;

  /// The skills of the newest [open]; a reset rebuilds with them.
  List<Skill> _requestedSkills = const [];
  final List<_OpenRequest> _queue = [];
  bool _serving = false;

  /// The queue's server while it runs.
  Future<void>? _server;

  /// Why each running [refusing] body refuses, in the order they began;
  /// a refused request is told the newest reason. A body that ends takes
  /// its own reason out, so the ones still running keep theirs, whether the
  /// bodies nest or overlap.
  final List<String> _refusals = [];

  /// Queues a rebuild with [profile] and [skills]; they become the requested
  /// ones at once. Inside [refusing] it fails at once instead, and nothing
  /// changes.
  Future<Result<void>> open(ConversationProfile profile, List<Skill> skills) {
    if (_refusals.isNotEmpty) {
      final reason = _refusals.last;
      debugPrint('[Conversation] open(${profile.name}) refused: $reason');
      return Future.value(Result.error(ConversationNotReadyException(reason)));
    }
    _requestedProfile = profile;
    _requestedSkills = List.unmodifiable(skills);
    return _enqueue(
      _OpenRequest(profile, onlyIfCurrent: false, skills: _requestedSkills),
    );
  }

  /// Queues a rebuild with [ifCurrent] and the newest [open]'s skills (a reset
  /// keeps the skill set), which only runs while [ifCurrent] is still the
  /// requested profile when its turn comes. Inside [refusing] it is moot (Ok,
  /// nothing built): the history goes with the chat anyway.
  Future<Result<void>> reset(ConversationProfile ifCurrent) =>
      _refusals.isNotEmpty
      ? Future.value(const Result.ok(null))
      : _enqueue(_OpenRequest(ifCurrent, onlyIfCurrent: true));

  /// Runs [body] while opens and resets are refused: an [open] requested
  /// meanwhile fails at once with [ConversationNotReadyException]([reason])
  /// and a [reset] is moot, so nothing requested during [body] builds a chat
  /// after it. Requests queued before still run ([drain] waits for them).
  Future<T> refusing<T>(String reason, Future<T> Function() body) async {
    _refusals.add(reason);
    try {
      return await body();
    } finally {
      // Equal reasons are interchangeable: which entry goes does not matter.
      _refusals.remove(reason);
    }
  }

  /// Completes once nothing is queued or being built, opens queued meanwhile
  /// included.
  Future<void> drain() async {
    while (_serving) {
      await _server;
    }
  }

  /// No profile is requested any more (the chat was released): a reset is
  /// moot until the next [open].
  void forgetRequested() => _requestedProfile = null;

  Future<Result<void>> _enqueue(_OpenRequest request) {
    _queue.add(request);
    if (!_serving) {
      _serving = true;
      unawaited(_server = _serve());
    }
    return request.done.future;
  }

  /// Serves queued opens and resets, one rebuild at a time. Each pass takes
  /// everything queued so far, then:
  /// - drops a reset whose profile is no longer the one most recently passed
  ///   to [open] (it is moot: Ok, nothing built);
  /// - builds only the newest remaining request. Earlier ones are superseded:
  ///   they share its result when they asked for the same profile, otherwise
  ///   they fail with [ConversationSupersededException].
  ///
  /// The flag is cleared in the same synchronous step as the last empty-queue
  /// check, so a request can never be left queued with no server.
  Future<void> _serve() async {
    try {
      while (_queue.isNotEmpty) {
        final batch = List.of(_queue);
        _queue.clear();
        final requested = _requestedProfile;
        final live = <_OpenRequest>[];
        for (final request in batch) {
          if (request.onlyIfCurrent && request.profile != requested) {
            request.done.complete(const Result.ok(null));
          } else {
            live.add(request);
          }
        }
        if (live.isEmpty) continue;
        final winner = live.last.profile;
        final skills = live.last.skills ?? _requestedSkills;
        Result<void> result;
        try {
          result = await _rebuild(winner, skills);
        } catch (e, st) {
          debugPrint('[Conversation] rebuild failed: $e\n$st');
          result = Result.error(asException(e));
        }
        for (final request in live) {
          request.done.complete(
            request.profile == winner
                ? result
                : Result.error(
                    ConversationSupersededException(
                      'open(${request.profile.name}) was replaced by '
                      'open(${winner.name}) before it ran',
                    ),
                  ),
          );
        }
      }
    } finally {
      _serving = false;
    }
  }
}

/// One queued [OpenQueue.open] or [OpenQueue.reset].
final class _OpenRequest {
  _OpenRequest(this.profile, {required this.onlyIfCurrent, this.skills});

  final ConversationProfile profile;

  /// A reset: only meaningful while [profile] is still the requested one.
  final bool onlyIfCurrent;

  /// The open's skills; null for a reset (the newest open's apply).
  final List<Skill>? skills;
  final Completer<Result<void>> done = Completer();
}
