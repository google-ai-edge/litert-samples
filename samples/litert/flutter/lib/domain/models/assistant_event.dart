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

import 'skill_step.dart';

/// Where a tokens-per-second figure came from.
enum TokenRateSource {
  /// LiteRT-LM's benchmark info for the last decode turn.
  native,

  /// Dart-side count of streamed text chunks over the decode time.
  chunks,
}

/// Why the chat no longer saw an image it had been sent, so the next turn
/// sent it again.
enum ImageLoss {
  /// A stopped turn: LiteRT-LM rebuilds the conversation from text only.
  stop,

  /// A failed turn: whatever reached the model is treated as gone.
  failure,

  /// `open` or `reset` built a new chat.
  reset,

  /// The context budget guard recreated the chat.
  budget,
}

/// Timing of one assistant turn.
final class const GenerationMetrics({
  /// From the start of the turn to the first visible text; null when no text
  /// arrived (stopped during prefill, or failed).
  required final Duration? timeToFirstToken,

  /// Streamed text chunks (about one per token on LiteRT-LM).
  required final int chunks,
  required final double? tokensPerSecond,
  required final TokenRateSource tokensPerSecondSource,
  required final Duration total,

  /// True when the turn ended because of a stop request.
  required final bool stopped,

  /// From the stop request to the end of the turn.
  final Duration? stopLatency,

  /// The turn had an image attached: sent now, or already in the context.
  final bool imageAttached = false,

  /// The image went to the model with this turn (first time or again).
  final bool imageSent = false,

  /// Set when [imageSent] re-sent an image the chat had lost: why it was lost.
  final ImageLoss? imageResent,

  /// The chat was recreated before this turn (the history and any image in
  /// it were dropped), for [contextResetReason].
  final bool contextReset = false,
  final ContextResetReason contextResetReason = ContextResetReason.budget,

  /// Tokens in the live conversation after the turn: the larger of
  /// LiteRT-LM's prefill + decode count and flutter_edge_ai's own count.
  final int? contextTokens,

  /// This turn's prefill tokens from LiteRT-LM (prompt, image, template);
  /// null when not measurable (the native conversation was rebuilt during
  /// the turn after a stop, or its metrics were unavailable).
  final int? prefillTokens,

  /// The text in this turn's prefill, counted by the model's tokenizer: the
  /// prompt, plus the system instruction on a chat's first turn with an
  /// image (LiteRT-LM prefills it with the first message).
  final int? promptTokens,

  /// Tool calls in this agent turn (loadSkill and runIntent each
  /// count; every one cost a generation before the answer).
  final int toolRounds = 0,

  /// The turn's skill steps with their times, for the overlay.
  final List<SkillStep> skillSteps = const [],
}) {
  /// What the image cost, estimated as this turn's prefill minus its text
  /// (so it includes a few chat-template tokens); null unless the image was
  /// sent and both counts are known.
  int? get imageTokens => switch ((imageSent, prefillTokens, promptTokens)) {
    (true, final int prefill, final int prompt) => prefill - prompt,
    _ => null,
  };
}

/// What one `ConversationRepository.ask` stream emits.
sealed class const AssistantEvent();

/// A piece of the reply, in order.
final class const AssistantTextDelta(final String text) extends AssistantEvent;

/// Why the chat was recreated before a turn, dropping its history.
enum ContextResetReason {
  /// The context budget guard: the next turn would not have fit.
  budget,

  /// The last agent turn was stopped, ran out of tool rounds or
  /// failed, leaving input staged that LiteRT-LM would glue onto the next
  /// message.
  interruptedSkill,
}

/// The chat was recreated before this turn ([reason]): the earlier history
/// (and any image in it) is gone. Arrives before any text, whatever the
/// turn's outcome afterwards (done, stopped, failed or cancelled), so the UI
/// can say so even for a turn a barge-in replaced.
final class const AssistantContextReset({
  final ContextResetReason reason = ContextResetReason.budget,
}) extends AssistantEvent;

/// The turn ended, normally or because of a stop. Always the last event of a
/// turn that did not fail.
final class const AssistantDone(final GenerationMetrics metrics)
    extends AssistantEvent;

/// The turn failed. Always the last event of a failed turn.
final class const AssistantFailed(final Exception error) extends AssistantEvent;
