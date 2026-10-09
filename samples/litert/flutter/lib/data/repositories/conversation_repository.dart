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
import 'package:flutter_edge_ai_agent/flutter_edge_ai_agent.dart' show Skill;

import '../../config/model_catalog.dart';
import '../../domain/models/assistant_event.dart';
import '../../domain/models/chat_capabilities.dart';
import '../../domain/models/skill_step.dart';
import '../../utils/result.dart';

/// The single chat on the loaded model (the model has one chat slot, the
/// `createChat` lane; `openSession` is never used). All turns go through
/// here, one at a time. Both demos share it: the entering demo opens it with
/// its [ConversationProfile], which drops the previous history.
///
/// Images: the caller passes the attached image on every turn and this
/// repository decides whether the live chat needs it sent, so a stop, a
/// failure, a reset or the budget guard never leaves a follow-up question
/// without its picture.
abstract interface class ConversationRepository {
  /// True from the start of a turn to its last event.
  ValueListenable<bool> get isGenerating;

  /// Whether a chat exists (the last [open] succeeded).
  bool get isOpen;

  /// The open chat's profile; null when no chat is open.
  ConversationProfile? get profile;

  /// Builds a fresh chat with [profile]. A running turn is stopped first, and
  /// a stopped turn that is still draining is waited for (bounded), so this
  /// never fails just because a demo was left mid-reply. Rebuilds run one at
  /// a time; a queued open that a later one replaces before it runs is not
  /// built (it fails with [ConversationSupersededException], or shares the
  /// later result when both asked for the same profile).
  ///
  /// If the draining turn does not end within the stop timeout, this fails
  /// and the previous chat stays open with its own profile; open again once
  /// the turn has ended.
  ///
  /// [skills]: with a profile that has a `skillsTemplate`, a non-empty list
  /// makes the chat an agent chat — tools `loadSkill` and `runIntent`, every
  /// skill listed in the system prompt — whose turns run through
  /// `AgentSession`. Otherwise it is ignored (Demo 3 stays plain). A new skill
  /// set needs a new chat: open again.
  Future<Result<void>> open(
    ConversationProfile profile, {
    List<Skill> skills = const [],
  });

  /// The open chat is an agent chat (opened with skills).
  bool get hasSkills;

  /// The open chat was asked for skills but is plain, because the chat model
  /// has tools off: skills that need tool calls are off; the app's direct
  /// intents still run.
  bool get skillsNeedTools;

  /// What the loaded chat model allows: images, tools, and its name for the
  /// labels that say why something is off.
  ChatCapabilities get capabilities;

  /// Stops a running turn and closes the chat, but stays usable (unlike
  /// [close]): the chat model is about to be unloaded or replaced. The next
  /// [open] builds on the model loaded then.
  ///
  /// A [ConversationNotReadyException] when the stopped turn still generates
  /// after the stop timeout (a native turn that ignores the stop): then
  /// nothing is closed, and the caller must not close the chat model either.
  Future<Result<void>> release();

  /// Sends [prompt] and streams the reply. The stream ends with exactly one
  /// [AssistantDone] or [AssistantFailed]; failures, a busy chat included,
  /// arrive as [AssistantFailed], never as stream errors. Cancelling the
  /// subscription stops generation too.
  ///
  /// On an agent chat [onStep] receives each skill step as it happens
  /// (loadSkill, runIntent and its result); nothing is spoken before a tool
  /// runs, because a Gemma 4 tool-call turn is pure JSON. A step is not an
  /// [AssistantEvent], so code that switches over those stays unchanged. A
  /// cancelled subscription does not stop native generation on the agent path:
  /// the repository stops it and drains the turn, keeping [isGenerating] true
  /// until it has ended. Too many tool rounds end the turn as [AssistantFailed]
  /// with a [SkillLoopException].
  ///
  /// [image]: a PNG or JPEG (normally from `normalizeForLlm` or a camera
  /// snapshot). Pass the SAME object every turn while it stays attached: it
  /// is sent only when the live chat cannot see it — the first time, and
  /// again after a stop, a failure, an [open]/[reset] or a budget reset
  /// (compared with `identical`, see [imageInContext]). A new object is sent
  /// as a new image.
  ///
  /// Before each turn a context budget guard compares the tokens in use plus
  /// what this turn needs with the model's window; when it would not fit, the
  /// chat is recreated with the same profile first (the history is dropped;
  /// [GenerationMetrics.contextReset] says so).
  Stream<AssistantEvent> ask(
    String prompt, {
    Uint8List? image,
    void Function(SkillStep step)? onStep,
  });

  /// The image the live chat can see: the last one a turn sent and that has
  /// not been lost since. Null after a stop, a failure, [open], [reset] or a
  /// budget reset.
  Uint8List? get imageInContext;

  /// Stops the running turn and completes once it has ended. No-op when idle.
  Future<void> stop();

  /// Drops the history by rebuilding the chat with [ifCurrent] — only while
  /// [ifCurrent] is still the profile most recently passed to [open], checked
  /// when the reset's turn in the open queue comes. Otherwise it does nothing
  /// and returns Ok: a late, unawaited reset from a demo the user has left
  /// can never reopen that demo's profile over the next one's.
  Future<Result<void>> reset({required ConversationProfile ifCurrent});

  /// Closes the chat, after any chat still being built: the chat model may
  /// be closed once it returns. Safe to call more than once.
  Future<void> close();
}

/// [ConversationRepository.ask] while a turn is running or before [open].
final class ConversationNotReadyException implements Exception {
  const ConversationNotReadyException(this.message);

  final String message;

  @override
  String toString() => message;
}

/// [ConversationRepository.ask] with an image on a chat built without image
/// support (the engine would drop the image silently).
final class ConversationImageUnsupportedException implements Exception {
  const ConversationImageUnsupportedException(this.message);

  final String message;

  @override
  String toString() => message;
}

/// The prompt (with its image and the reply allowance) does not fit the
/// model's context window even in a fresh chat.
final class ConversationTooLongException implements Exception {
  const ConversationTooLongException(this.message);

  final String message;

  @override
  String toString() => message;
}

/// An agent turn used up its generations without a plain answer (the model
/// kept calling tools).
final class SkillLoopException implements Exception {
  const SkillLoopException(this.iterations);

  final int iterations;

  @override
  String toString() =>
      'The model kept calling skills for $iterations rounds without '
      'answering';
}

/// A queued [ConversationRepository.open] that a later open for another
/// profile replaced before it ran.
final class ConversationSupersededException implements Exception {
  const ConversationSupersededException(this.message);

  final String message;

  @override
  String toString() => message;
}
