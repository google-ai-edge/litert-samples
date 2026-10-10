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
import 'package:flutter_edge_ai/flutter_edge_ai.dart';
import 'package:flutter_edge_ai_agent/flutter_edge_ai_agent.dart' show Skill;

import '../../../config/model_catalog.dart';
import '../../../domain/models/assistant_event.dart';
import '../../../domain/models/chat_capabilities.dart';
import '../../../utils/result.dart';
import '../conversation_repository.dart';
import 'context_budget.dart';
import 'image_context_tracker.dart';
import 'native_chat.dart';

/// The open chat and its profile, as a turn found them.
typedef OpenChat = ({InferenceChat chat, ConversationProfile profile});

/// What a turn sends and measures against ([LiveChat.prepareSend]): the
/// image to send (null when the live chat already sees it, or none is
/// attached), why it is a re-send, whether LiteRT-LM rebuilds the native
/// conversation as this turn starts, and LiteRT-LM's metrics before it.
typedef TurnSend = ({
  Uint8List? image,
  ImageLoss? resent,
  bool rebuildPending,
  SessionMetrics? before,
});

/// The model's single chat slot (the `createChat` lane), shared by the
/// repository facade and the two turn paths: the live chat with its agent
/// and profile, what its native conversation holds, and the one turn that
/// may run on it.
final class LiveChat {
  LiveChat(this._chats, {required this._stopTimeout});

  /// Builds every chat on the loaded chat model.
  final ChatFactory _chats;
  final Duration _stopTimeout;

  final ValueNotifier<bool> _isGenerating = ValueNotifier(false);
  InferenceChat? _chat;
  ConversationProfile? _profile;

  /// The open chat's agent (null for a plain chat).
  AgentChat? _agent;

  /// The open chat was asked for skills, but the chat model has tools off, so
  /// it was built plain.
  bool _skillsNeedTools = false;

  /// The image the live chat can see, and why it lost the last one.
  final ImageContextTracker _images = ImageContextTracker();

  /// The last sent turn was stopped: LiteRT-LM rebuilds the native
  /// conversation when the next turn starts, so that turn's prefill cannot be
  /// read from the metrics delta.
  bool _nativeRebuildPending = false;

  /// The last agent turn left staged input behind (LiteRT-LM buffers
  /// `addQueryChunk` until the next generation and would prepend it, as text,
  /// to the next user message, so the next turn rebuilds the chat first).
  bool _staleToolTail = false;

  Completer<void>? _turnDone;
  bool _stopRequested = false;
  Stopwatch? _sinceStop;
  bool _closed = false;

  /// Set when [close] begins, before it waits for the running turn: turns
  /// and rebuilds refuse from then on, so no turn starts and no chat is
  /// built on what close is about to close.
  bool _closing = false;
  Future<void>? _closeRun;

  /// The chat being built now and its hand-over ([_create]); [close] waits
  /// for it.
  Future<void>? _building;

  /// True from the start of a turn to its last event.
  ValueListenable<bool> get isGenerating => _isGenerating;

  /// What the loaded chat model allows; every chat is built from it.
  ChatCapabilities get capabilities => _chats.capabilities;

  /// A chat exists.
  bool get isOpen => _chat != null;

  /// The open chat's profile; null when no chat is open.
  ConversationProfile? get profile => _chat == null ? null : _profile;

  /// The open chat's agent; null for a plain chat (or none). Kept while an
  /// agent turn rebuilds its chat.
  AgentChat? get agent => _agent;

  /// The open chat was asked for skills but is plain (tools off).
  bool get skillsNeedTools => _chat != null && _skillsNeedTools;

  /// The image the live chat can see.
  Uint8List? get imageInContext => _images.inContext;

  /// [close] has closed the chat.
  bool get closed => _closed;

  /// A stop was requested for the running (or last) turn.
  bool get stopRequested => _stopRequested;

  /// From the stop request until now; null without one.
  Duration? get stopLatency => _sinceStop?.elapsed;

  /// The last agent turn left input staged.
  bool get staleToolTail => _staleToolTail;

  // ---- The slot's lifecycle ----

  /// Replaces the chat with a fresh one for [profile] and [skills] (the open
  /// queue's rebuild): a running turn is stopped first.
  Future<Result<void>> rebuild(
    ConversationProfile profile,
    List<Skill> skills,
  ) async {
    if (_closing) {
      return const Result.error(
        ConversationNotReadyException('The conversation is closed'),
      );
    }
    // Leaving a demo mid-reply stops without waiting; the entering demo's
    // open lands here while that turn drains. stop() (re)sends the stop and
    // waits for the turn's end, bounded by _stopTimeout.
    if (_isGenerating.value) await stop();
    // Still generating after the stop timeout: give up and leave the previous
    // chat (and its profile) in place; the caller shows the error and can
    // open again once the turn has ended.
    if (_isGenerating.value) return Result.error(stillGenerating());
    try {
      final previous = _chat;
      _chat = null;
      _agent = null;
      _profile = null;
      _images.lose(ImageLoss.reset);
      _nativeRebuildPending = false;
      _staleToolTail = false;
      _skillsNeedTools = false;
      await previous?.close();
      final (chat, agent) = await _create(() => _chats.build(profile, skills));
      _chat = chat;
      _agent = agent;
      _profile = profile;
      _skillsNeedTools =
          agent == null &&
          profile.skillsTemplate != null &&
          skills.isNotEmpty &&
          !capabilities.tools;
      debugPrint(
        '[Conversation] opened profile=${profile.name} '
        'model=${capabilities.modelName} images=${capabilities.images} '
        'tools=${capabilities.tools}'
        '${agent == null ? '' : ' skills=${agent.skills.map((s) => s.name).join(',')}'}'
        '${_skillsNeedTools ? ' (skills off: the chat model has tools off)' : ''}',
      );
      return const Result.ok(null);
    } on ConversationNotReadyException catch (e) {
      return Result.error(e); // a close began
    } catch (e, st) {
      debugPrint('[Conversation] open(${profile.name}) failed: $e\n$st');
      return Result.error(asException(e));
    }
  }

  /// Why a rebuild or a release gives up: the stopped turn still generates
  /// after [stop] waited the stop timeout for it.
  ConversationNotReadyException stillGenerating() =>
      ConversationNotReadyException(
        'The previous reply did not finish within '
        '${_stopTimeout.inSeconds}s of being stopped',
      );

  /// Stops the running turn and completes once it has ended (bounded by the
  /// stop timeout). No-op when idle.
  Future<void> stop() async {
    final done = _turnDone;
    if (done == null || done.isCompleted) return;
    if (!_stopRequested) {
      _stopRequested = true;
      _sinceStop = Stopwatch()..start();
    }
    // Null while the budget guard replaces the chat; the turn checks the
    // stop flag before it sends anything.
    final chat = _chat;
    if (chat != null) {
      try {
        await chat.stopGeneration();
      } catch (e, st) {
        // The chunk loop still drains on _stopRequested, so the turn ends.
        debugPrint('[Conversation] stopGeneration failed: $e\n$st');
      }
    }
    await done.future.timeout(
      _stopTimeout,
      onTimeout: () => debugPrint(
        '[Conversation] the stopped turn did not end within '
        '${_stopTimeout.inSeconds}s',
      ),
    );
  }

  /// Closes the chat and forgets it, leaving the slot usable. The caller
  /// has waited for queued opens and stopped the running turn.
  Future<void> release() async {
    final chat = _chat;
    _chat = null;
    _agent = null;
    _profile = null;
    _skillsNeedTools = false;
    _images.lose(ImageLoss.reset);
    _nativeRebuildPending = false;
    _staleToolTail = false;
    try {
      await chat?.close();
    } catch (e, st) {
      debugPrint('[Conversation] release failed: $e\n$st');
    }
    debugPrint('[Conversation] released the chat (the chat model changes)');
  }

  /// Refuses new turns and rebuilds from now on ([_closing]); [close] does
  /// the rest. A rebuild already building its chat closes what it builds.
  /// The repository calls it before it waits for that rebuild, so nothing
  /// starts on the chat meanwhile.
  void beginClose() => _closing = true;

  /// Refuses new turns and rebuilds at once ([_closing]), stops the running
  /// turn, waits for a chat still being built, then closes the chat. Every
  /// call gets the same close.
  Future<void> close() => _closeRun ??= _close();

  Future<void> _close() async {
    _closing = true;
    await stop();
    // A turn's recreate may outlive the stop timeout: the chat model must
    // not be closed under its createChat.
    await _building;
    // Still generating after the stop timeout: like [rebuild] and the
    // repository's release, nothing is closed under the native generation;
    // the chat is left to it.
    final stillGenerating = _isGenerating.value;
    _closed = true;
    final chat = _chat;
    _chat = null;
    if (stillGenerating) {
      debugPrint(
        '[Conversation] close: the stopped turn is still generating after '
        'the stop timeout; its chat is left open, not closed under it',
      );
    } else {
      try {
        await chat?.close();
      } catch (e, st) {
        debugPrint('[Conversation] close failed: $e\n$st');
      }
    }
    _isGenerating.dispose();
  }

  // ---- One turn at a time ----

  /// The checks before either kind of turn starts: the open chat and its
  /// profile, or why no turn can start (then nothing reaches the engine).
  Result<OpenChat> precheck(Uint8List? image) {
    final chat = _chat;
    final profile = _profile;
    const notOpen = Result<OpenChat>.error(
      ConversationNotReadyException('The chat is not open'),
    );
    if (_closing) return notOpen;
    // Before the open check: a running turn that rebuilds its chat (budget
    // guard, interrupted skill) leaves the slot empty meanwhile.
    if (_isGenerating.value) {
      return const Result.error(
        ConversationNotReadyException('A reply is already being generated'),
      );
    }
    if (chat == null || profile == null) return notOpen;
    if (image != null && !chat.supportsImages) {
      // The engine would drop the image without a word.
      return const Result.error(
        ConversationImageUnsupportedException(
          'This chat was built without image support, so the image would be '
          'ignored',
        ),
      );
    }
    return Result.ok((chat: chat, profile: profile));
  }

  /// Starts a turn: [isGenerating] until the returned completer is passed
  /// to [endTurn]; [stop] waits for it.
  Completer<void> beginTurn() {
    _isGenerating.value = true;
    _stopRequested = false;
    _sinceStop = null;
    return _turnDone = Completer<void>();
  }

  /// Ends the turn [beginTurn] started.
  void endTurn(Completer<void> done) {
    if (!_closed) _isGenerating.value = false;
    done.complete();
  }

  /// The budget guard for a turn with [prompt] and [image] on [chat] (an
  /// agent chat when [agent] is given): measures what the chat holds and
  /// what the turn needs ([ContextBudget]), logs it, and says whether the
  /// chat must be recreated first. Throws [ConversationTooLongException] when
  /// even a fresh chat could not hold the turn. [turnTime]: the turn's clock,
  /// for the plain guard's log line.
  Future<({int promptTokens, bool reset})> checkBudget(
    InferenceChat chat,
    String prompt,
    ConversationProfile profile,
    Uint8List? image,
    Duration Function() turnTime, {
    AgentChat? agent,
  }) async {
    final native = sessionMetricsOf(chat)?.totalTokens ?? 0;
    final promptTokens = await chat.session.sizeInTokens(prompt);
    final guardTime = turnTime();
    final budget = ContextBudget(
      nativeTokens: native,
      chatTokens: chat.currentTokens,
      promptTokens: promptTokens,
      replyTokens: profile.maxOutputTokens,
      maxTokens: chat.maxTokens,
      agentReserve: agent == null ? null : await agent.reserve(chat),
    );
    final check = budget.check(
      image: image != null,
      sendsImage: _images.toSend(image) != null,
    );
    switch (check) {
      case BudgetTooLong(:final needFresh):
        throw budget.tooLong(needFresh);
      case BudgetFits(:final needNow) || BudgetReset(:final needNow):
        debugPrint(
          '[Conversation] ${budget.summary(needNow)}'
          '${agent == null ? ' (${guardTime.inMicroseconds} µs)' : ''}',
        );
        if (check is BudgetReset) {
          debugPrint(
            '[Conversation] context reset (budget): used=${budget.used} '
            'need=$needNow limit=${budget.limit}',
          );
        }
    }
    return (promptTokens: promptTokens, reset: check is BudgetReset);
  }

  /// Replaces the chat with a fresh plain one on the same profile, inside
  /// the running turn (still "generating", so no open can interleave). The
  /// chat is null meanwhile, so a stop never reaches a closing chat.
  Future<InferenceChat> recreatePlain(ConversationProfile profile) async {
    final previous = _chat;
    _chat = null;
    _images.lose(ImageLoss.budget);
    _nativeRebuildPending = false;
    await previous?.close();
    final (fresh, _) = await _create(
      () async => (await _chats.plain(profile), null),
    );
    _chat = fresh;
    return fresh;
  }

  /// Replaces the agent chat with a fresh one on the same profile and
  /// skills, inside the running turn (like [recreatePlain]).
  Future<(InferenceChat, AgentChat)> recreateAgent(
    ConversationProfile profile,
    AgentChat current,
    ImageLoss reason,
  ) async {
    final previous = _chat;
    _chat = null;
    _images.lose(reason);
    _nativeRebuildPending = false;
    _staleToolTail = false;
    await previous?.close();
    final (fresh, agent) = await _create(
      () => _chats.build(profile, current.skills),
    );
    if (agent == null) {
      await fresh.close();
      throw const ConversationNotReadyException('The conversation is closed');
    }
    _chat = fresh;
    _agent = agent;
    return (fresh, agent);
  }

  /// Builds a chat with [build] — refused once a close has begun — and
  /// closes it again when a close began meanwhile (it would have no owner).
  /// [close] waits for all of it, so the chat model is never closed under
  /// an in-flight `createChat`. One build at a time: [rebuild] never runs
  /// during a turn, and a turn recreates its chat at most once at a time.
  Future<(InferenceChat, AgentChat?)> _create(
    Future<(InferenceChat, AgentChat?)> Function() build,
  ) {
    final created = () async {
      const closed = ConversationNotReadyException(
        'The conversation is closed',
      );
      if (_closing) throw closed;
      final (chat, agent) = await build();
      if (_closing) {
        await chat.close();
        throw closed;
      }
      return (chat, agent);
    }();
    // Only a marker for close to wait on: the caller, awaiting `created`,
    // gets and reports the failure.
    _building = created.then<void>((_) {}, onError: (Object _) {});
    return created;
  }

  /// What a turn with [image] attached sends on [chat]: the image only when
  /// the live chat cannot see it (a re-send is logged with why it was
  /// lost), plus what the turn's metrics are measured against.
  TurnSend prepareSend(InferenceChat chat, Uint8List? image) {
    final toSend = _images.toSend(image);
    final resent = toSend == null ? null : _images.resendReason(toSend);
    if (resent != null) {
      debugPrint('[Conversation] image resent (lost: ${resent.name})');
    }
    return (
      image: toSend,
      resent: resent,
      rebuildPending: _nativeRebuildPending,
      before: sessionMetricsOf(chat),
    );
  }

  /// A turn that reached the model ended, normally or [stopped]: a stop
  /// loses the image and makes LiteRT-LM rebuild the conversation as the
  /// next turn starts; a normal end with a [sentImage] puts it in context.
  void settleTurn({required bool stopped, required Uint8List? sentImage}) {
    _images.settle(stopped: stopped, sentImage: sentImage);
    _nativeRebuildPending = stopped;
  }

  /// A turn that reached the model failed. No rebuild follows a failure: what
  /// the model saw counts as gone, so the next turn sends the image again.
  void failTurn({required Uint8List? sentImage}) {
    _images.lose(ImageLoss.failure, sentImage: sentImage);
    _nativeRebuildPending = _stopRequested;
  }

  /// A turn that reached the model was cancelled mid-turn: LiteRT-LM
  /// cancels native generation and rebuilds the conversation from text, like
  /// a stop.
  void cancelTurn({required Uint8List? sentImage}) {
    _images.lose(ImageLoss.stop, sentImage: sentImage);
    _nativeRebuildPending = true;
  }

  /// The agent turn ended with input staged, a stale tool tail: the next agent
  /// turn rebuilds the chat first.
  void markStaleToolTail() => _staleToolTail = true;
}
