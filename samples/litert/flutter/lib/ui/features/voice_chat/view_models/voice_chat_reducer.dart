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

import '../../../../domain/models/assistant_event.dart'
    show ContextResetReason, GenerationMetrics;
import '../../../../domain/models/chat_entry.dart';
import '../../../../domain/models/chat_side_event.dart';
import '../../../../domain/models/knowledge.dart';
import '../../../../domain/models/skill_step.dart';
import '../../../../domain/use_cases/prompt_builder.dart';
import '../../../../domain/use_cases/voice_assistant.dart';
import '../../../core/voice_notices.dart';

/// What Demo 1's turns have put on screen, and the running turn's facts its
/// reply takes when it commits.
final class const VoiceChatState({
  /// The committed conversation, oldest first.
  final List<ChatEntry> entries = const [],

  /// The last failure, shown to the user; cleared by the next turn.
  final String? error,

  /// The running turn's retrieval, attached to its reply when that commits.
  final Retrieval? turnRetrieval,

  /// The running turn's skill steps, attached to its reply when it commits;
  /// the streaming bubble shows them meanwhile. `const []` when empty.
  final List<SkillStep> turnSteps = const [],
}) {
  /// This state with the given fields replaced; a null keeps the field,
  /// [clearError] and [clearTurnRetrieval] drop theirs.
  VoiceChatState copyWith({
    List<ChatEntry>? entries,
    String? error,
    bool clearError = false,
    Retrieval? turnRetrieval,
    bool clearTurnRetrieval = false,
    List<SkillStep>? turnSteps,
  }) => VoiceChatState(
    entries: entries ?? this.entries,
    error: clearError ? null : error ?? this.error,
    turnRetrieval: clearTurnRetrieval
        ? null
        : turnRetrieval ?? this.turnRetrieval,
    turnSteps: turnSteps ?? this.turnSteps,
  );

  /// This state with [entry] committed after the others.
  VoiceChatState adding(ChatEntry entry) =>
      copyWith(entries: List.unmodifiable([...entries, entry]));

  /// This state with [message] shown: as the error, and as an entry (with
  /// [steps], how far a failed skill call got).
  VoiceChatState showingError(
    String message, {
    List<SkillStep> steps = const [],
  }) =>
      copyWith(error: message)
          .adding(ChatEntry(role: ChatRole.error, text: message, steps: steps));

  /// A new conversation: nothing on screen, no turn facts.
  static const VoiceChatState empty = VoiceChatState();
}

/// One step of [VoiceChatReducer]: the next state, and what the view model
/// does about it, in this order — take the state (the streaming bubble's
/// steps follow its [VoiceChatState.turnSteps]), record the figures,
/// rebuild.
final class const VoiceChatUpdate({
  required final VoiceChatState state,

  /// Gemma's figures for the overlay (also from a turn a barge-in replaced:
  /// its stop latency is the point).
  final GenerationMetrics? generation,

  /// The turn's retrieval for the overlay (also a replaced turn's).
  final Retrieval? retrieval,

  /// The screen changed: rebuild. Tokens, steps and figures do not.
  final bool notify = true,
});

/// Demo 1's projection of its voice events onto the chat, as a pure
/// function: [reduce] takes the state and one event, and returns the next
/// state plus the effects for the view model to apply.
///
/// The rules it owns:
/// - the user's words start the turn: the last error goes, and the steps
///   start empty;
/// - a committed reply takes the turn's retrieval (with the excerpts it
///   cites) and steps, so the next reply starts without them; a reply with
///   no text, not interrupted and without steps adds no entry;
/// - a capture that made no LLM call is a notice and drops the turn's
///   retrieval; a failed turn is an error entry with the steps it got to;
/// - a turn a barge-in replaced (detached) is already committed: its
///   retrieval and its steps belong to no entry, except an intent that ran,
///   which is a notice; its figures and its context reset still count;
/// - a context reset is a notice wherever the turn ends, because the next
///   question goes to the chat that forgot.
final class VoiceChatReducer {
  const VoiceChatReducer();

  VoiceChatUpdate reduce(
    VoiceChatState state,
    VoiceAssistantEvent<ChatSideEvent> event,
  ) {
    switch (event) {
      case UserSaid(:final text, :final image):
        return VoiceChatUpdate(
          state: state
              .copyWith(clearError: true, turnSteps: const [])
              .adding(ChatEntry(role: ChatRole.user, text: text, image: image)),
        );
      case AssistantSaid(:final text, :final interrupted):
        final steps = state.turnSteps;
        final next = state.copyWith(
          clearTurnRetrieval: true,
          turnSteps: const [],
        );
        if (text.isEmpty && !interrupted && steps.isEmpty) {
          return VoiceChatUpdate(state: next, notify: false);
        }
        return VoiceChatUpdate(
          state: next.adding(
            ChatEntry(
              role: ChatRole.assistant,
              text: text,
              interrupted: interrupted,
              knowledge: _knowledge(state.turnRetrieval, text),
              steps: steps,
            ),
          ),
        );
      case NotHeard(:final reason):
        return VoiceChatUpdate(
          state: state
              .copyWith(clearTurnRetrieval: true)
              .adding(
                ChatEntry(role: ChatRole.notice, text: notHeardText(reason)),
              ),
        );
      case MicUnavailable(:final message):
        return VoiceChatUpdate(state: state.showingError(message));
      case TurnFailed(:final error):
        // The steps show how far a failed skill call got.
        return VoiceChatUpdate(
          state: state
              .copyWith(clearTurnRetrieval: true, turnSteps: const [])
              .showingError('The reply failed: $error', steps: state.turnSteps),
        );
      case SideEvent(:final event, :final detached):
        return _side(state, event, detached: detached);
    }
  }

  VoiceChatUpdate _side(
    VoiceChatState state,
    ChatSideEvent event, {
    required bool detached,
  }) {
    switch (event) {
      // Recorded for detached (barged-in) turns too: their stop latency is
      // the point.
      case ChatGenerationDone(:final metrics):
        return VoiceChatUpdate(
          state: state,
          generation: metrics,
          notify: false,
        );
      case ChatContextReset(:final reason):
        // The history was dropped before this turn (budget guard, or an
        // interrupted skill call): say so where the user reads, not only in
        // the overlay — also for a turn a barge-in replaced or that failed
        // afterwards, because the next question goes to the chat that
        // forgot.
        return VoiceChatUpdate(
          state: state.adding(
            ChatEntry(role: ChatRole.notice, text: contextResetTextFor(reason)),
          ),
        );
      case ChatRetrieval(:final retrieval):
        return VoiceChatUpdate(
          // A barged-in turn's partial reply is already committed; its
          // excerpts belong to no entry.
          state: detached ? state : state.copyWith(turnRetrieval: retrieval),
          retrieval: retrieval,
          notify: false,
        );
      case ChatSkillStep(:final step) when detached:
        // A barged-in turn's reply is already committed, but an intent that
        // ran is a real outcome: say so.
        if (step case IntentSucceeded(:final result)) {
          return VoiceChatUpdate(
            state: state.adding(
              ChatEntry(
                role: ChatRole.notice,
                text: '$skillAfterInterruptionText $result',
                steps: [step],
              ),
            ),
          );
        }
        return VoiceChatUpdate(state: state, notify: false);
      case ChatSkillStep(:final step):
        // One update per step, to the bubble only: two or three silent
        // generations must not look like a hang.
        return VoiceChatUpdate(
          state: state.copyWith(
            turnSteps: List.unmodifiable([...state.turnSteps, step]),
          ),
          notify: false,
        );
    }
  }

  /// [retrieval] with the excerpts [reply] cites; null without one.
  ReplyKnowledge? _knowledge(Retrieval? retrieval, String reply) {
    if (retrieval == null) return null;
    return ReplyKnowledge(
      retrieval: retrieval,
      cited: PromptBuilder.citedNumbers(reply, retrieval.passages.length),
    );
  }
}

/// Before the result of an intent that ran after a barge-in.
const skillAfterInterruptionText = 'Done after you interrupted:';

/// The notice when the context budget guard started the chat over.
const contextResetText =
    'The conversation got too long for the model, so it started over; '
    'earlier turns are forgotten.';

/// The notice when an interrupted, failed or runaway skill call left the
/// chat unusable and it was started over.
const interruptedSkillResetText =
    'The last skill call did not finish, so the conversation started over; '
    'earlier turns are forgotten.';

/// The notice for a [ContextResetReason].
String contextResetTextFor(ContextResetReason reason) => switch (reason) {
  ContextResetReason.budget => contextResetText,
  ContextResetReason.interruptedSkill => interruptedSkillResetText,
};
