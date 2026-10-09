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
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/repositories/conversation_repository.dart';
import 'package:litert_edge_demos/domain/models/assistant_event.dart';
import 'package:litert_edge_demos/domain/models/chat_capabilities.dart';
import 'package:litert_edge_demos/domain/models/skill_step.dart';
import 'package:litert_edge_demos/utils/result.dart';

/// Hand-driven [ConversationRepository]: a test calls [emit], [finish] or
/// [fail] to play the model's part of a turn.
///
/// Holds to the real contract where a caller can get it wrong: [ask] is
/// lazy (nothing happens until the stream is listened to, like `async*`)
/// and answers [AssistantFailed] when the chat is not open, a reply is
/// already running, or the chat cannot take the image; a listener that
/// cancels ends the turn; [close] is idempotent.
class FakeConversationRepository implements ConversationRepository {
  final ValueNotifier<bool> _isGenerating = ValueNotifier(false);
  StreamController<AssistantEvent>? _turn;
  ConversationProfile? _profile;
  bool _closed = false;

  /// Whether the open chat takes images: fixed when it opens, like the real
  /// chat's `supportsImages`.
  bool _chatTakesImages = false;

  Result<void> openResult = const Result.ok(null);
  final List<String> prompts = [];

  /// The image passed with each [ask], in order (null for none). The fake
  /// has no re-send rule: that lives in the real repository and is tested
  /// there.
  final List<Uint8List?> images = [];
  final List<ConversationProfile> openedProfiles = [];
  int openCalls = 0;
  int stopCalls = 0;
  int resetCalls = 0;

  /// When set, [open] waits for it, like the real one waiting for a stopped
  /// turn to drain; the previous chat (and its profile) stays meanwhile.
  Completer<void>? openGate;

  /// When set, [stop] waits for it before ending the turn, like a turn that
  /// takes a while to drain after a stop.
  Completer<void>? stopGate;

  /// Plays "another demo left the chat open with [profile]".
  void leaveOpen(ConversationProfile profile) {
    _profile = profile;
    _chatTakesImages = capabilities.images;
  }

  StreamController<AssistantEvent> get _activeTurn =>
      _turn ?? (throw StateError('No turn is running'));

  @override
  ValueListenable<bool> get isGenerating => _isGenerating;

  @override
  bool get isOpen => _profile != null;

  @override
  ConversationProfile? get profile => _profile;

  /// Like the real one: stops a running turn first; a failed open leaves no
  /// chat.
  @override
  Future<Result<void>> open(
    ConversationProfile profile, {
    List<Skill> skills = const [],
  }) async {
    openCalls++;
    openedProfiles.add(profile);
    openedSkills.add(skills);
    await openGate?.future;
    if (_turn != null) await stop();
    final ok = openResult is Ok<void>;
    _profile = ok ? profile : null;
    _chatTakesImages = ok && capabilities.images;
    final wantsSkills = profile.skillsTemplate != null && skills.isNotEmpty;
    _hasSkills = ok && wantsSkills && capabilities.tools;
    _skillsNeedTools = ok && wantsSkills && !capabilities.tools;
    return openResult;
  }

  /// The skills passed with each [open], in order.
  final List<List<Skill>> openedSkills = [];
  bool _hasSkills = false;
  bool _skillsNeedTools = false;

  @override
  bool get hasSkills => _hasSkills;

  /// What the loaded chat model allows; Gemma 4 E2B's by default. A test
  /// sets images or tools off to play a custom model without them.
  @override
  ChatCapabilities capabilities = const ChatCapabilities(
    modelName: 'Gemma 4 E2B',
    images: true,
    tools: true,
  );

  @override
  bool get skillsNeedTools => _skillsNeedTools;

  int releaseCalls = 0;

  /// When set, [release] refuses with it and changes nothing, like the real
  /// one while a stopped turn still generates past the stop timeout.
  Exception? releaseError;

  /// Like the real one: the chat is gone, the repository stays usable.
  @override
  Future<Result<void>> release() async {
    releaseCalls++;
    if (_turn != null) await stop();
    if (releaseError case final error?) return Result.error(error);
    _profile = null;
    _hasSkills = false;
    _skillsNeedTools = false;
    return const Result.ok(null);
  }

  /// The running turn's step listener.
  void Function(SkillStep step)? _onStep;

  @override
  Uint8List? imageInContext;

  /// Lazy like the real `async*`: the turn starts when the stream is
  /// listened to, and is refused like the real one (`ask` in
  /// conversation_repository_edge_ai.dart). [prompts] and [images] record
  /// every ask listened to, refused ones included.
  @override
  Stream<AssistantEvent> ask(
    String prompt, {
    Uint8List? image,
    void Function(SkillStep step)? onStep,
  }) async* {
    prompts.add(prompt);
    images.add(image);
    if (_closed || _profile == null) {
      yield const AssistantFailed(
        ConversationNotReadyException('The chat is not open'),
      );
      return;
    }
    if (_isGenerating.value) {
      yield const AssistantFailed(
        ConversationNotReadyException('A reply is already being generated'),
      );
      return;
    }
    if (image != null && !_chatTakesImages) {
      yield const AssistantFailed(
        ConversationImageUnsupportedException(
          'This chat was built without image support, so the image would be '
          'ignored',
        ),
      );
      return;
    }
    _onStep = onStep;
    _isGenerating.value = true;
    final turn = _turn = StreamController<AssistantEvent>();
    try {
      yield* turn.stream;
    } finally {
      // The listener cancelled: the turn ends, like the real one settling
      // as stopped.
      if (identical(_turn, turn)) {
        _turn = null;
        if (!_closed) _isGenerating.value = false;
        unawaited(turn.close());
      }
    }
  }

  void emit(String text) => _activeTurn.add(AssistantTextDelta(text));

  /// Plays a skill step of the running turn.
  void emitStep(SkillStep step) => _onStep?.call(step);

  /// Plays the budget guard recreating the chat before the turn.
  void emitContextReset({
    ContextResetReason reason = ContextResetReason.budget,
  }) => _activeTurn.add(AssistantContextReset(reason: reason));

  /// Ends the turn; [metricsOverride] plays what the real repository
  /// reports (an image re-send, a context reset).
  Future<void> finish({
    bool stopped = false,
    GenerationMetrics? metricsOverride,
  }) async {
    final turn = _activeTurn;
    _turn = null;
    turn.add(AssistantDone(metricsOverride ?? metrics(stopped: stopped)));
    _isGenerating.value = false;
    await turn.close();
  }

  Future<void> fail(Exception error) async {
    final turn = _activeTurn;
    _turn = null;
    turn.add(AssistantFailed(error));
    _isGenerating.value = false;
    await turn.close();
  }

  @override
  Future<void> stop() async {
    stopCalls++;
    await stopGate?.future;
    if (_turn != null) await finish(stopped: true);
  }

  @override
  Future<Result<void>> reset({required ConversationProfile ifCurrent}) async {
    resetCalls++;
    // Like the real one: a reset for a profile that is no longer current is
    // moot.
    if (_profile != ifCurrent) return const Result.ok(null);
    return open(ifCurrent);
  }

  /// Idempotent, like the real one.
  @override
  Future<void> close() async {
    if (_closed) return;
    // First: the turn's own cleanup then leaves the notifier alone.
    _closed = true;
    await _turn?.close();
    _turn = null;
    _isGenerating.dispose();
  }

  static GenerationMetrics metrics({
    bool stopped = false,
    bool imageAttached = false,
    bool imageSent = false,
    ImageLoss? imageResent,
    bool contextReset = false,
  }) => GenerationMetrics(
    timeToFirstToken: const Duration(milliseconds: 120),
    chunks: 2,
    tokensPerSecond: 20,
    tokensPerSecondSource: TokenRateSource.chunks,
    total: const Duration(milliseconds: 300),
    stopped: stopped,
    stopLatency: stopped ? const Duration(milliseconds: 40) : null,
    imageAttached: imageAttached,
    imageSent: imageSent,
    imageResent: imageResent,
    contextReset: contextReset,
  );
}
