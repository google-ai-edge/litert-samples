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

import 'dart:math' as math;

import '../../../config/model_catalog.dart';
import '../conversation_repository.dart';

/// What an agent chat must hold on top of a plain turn. [system]:
/// the system prompt (template + skill list) and the two tool declarations,
/// which the first prefill carries. [rounds]: the tool rounds a skill call
/// takes before the answer — each round's call and result, plus the largest
/// skill body `loadSkill` returns.
typedef AgentReserve = ({int system, int rounds});

/// The context budget guard's figures for one turn and its verdict: the tokens
/// the live chat holds plus what this turn needs, against the model's window
/// less [kContextHeadroomTokens]. When the turn would not fit, the chat is
/// recreated first ([BudgetReset]); when it would not fit even a fresh chat,
/// the turn fails ([BudgetTooLong]).
///
/// Used tokens are the larger of LiteRT-LM's count and flutter_edge_ai's
/// (which leaves out user text, so it under-reads); the max also keeps
/// InferenceChat's own trim from firing (it replays the history onto one
/// user message).
final class const ContextBudget({
  /// LiteRT-LM's count for the live conversation (0 when unreadable).
  required final int nativeTokens,

  /// flutter_edge_ai's count (`InferenceChat.currentTokens`).
  required final int chatTokens,

  /// The prompt, counted by the model's tokenizer.
  required final int promptTokens,

  /// The reply allowance (the profile's `maxOutputTokens`).
  required final int replyTokens,

  /// The model's context window (`InferenceChat.maxTokens`).
  required final int maxTokens,

  /// Set on an agent chat: what it holds on top of a plain turn.
  final AgentReserve? agentReserve,
}) {
  /// Tokens already in the live conversation.
  int get used => math.max(nativeTokens, chatTokens);

  /// `maxTokens` less the headroom.
  int get limit => maxTokens - kContextHeadroomTokens;

  /// The system prompt this turn still prefills: an agent chat's, before its
  /// conversation's first prefill.
  int get systemNow => switch (agentReserve) {
    (:final system, rounds: _) when nativeTokens == 0 => system,
    _ => 0,
  };

  /// The prompt, the reply allowance, the template and the tool rounds,
  /// without the system prompt or an image.
  int get _turn =>
      promptTokens +
      replyTokens +
      kTurnOverheadTokens +
      (agentReserve?.rounds ?? 0);

  /// What a fresh chat must hold for this turn: the system prompt and, when
  /// one is attached, the image included.
  int needFresh({required bool image}) =>
      _turn + (agentReserve?.system ?? 0) + (image ? kImageTokenAllowance : 0);

  /// What this turn adds to the live chat: the image only when it is sent.
  int needNow({required bool sendsImage}) =>
      _turn + systemNow + (sendsImage ? kImageTokenAllowance : 0);

  /// The verdict for a turn with an [image] attached that [sendsImage] (the
  /// live chat cannot see it).
  BudgetCheck check({required bool image, required bool sendsImage}) {
    final fresh = needFresh(image: image);
    if (fresh > limit) return BudgetTooLong(fresh);
    final now = needNow(sendsImage: sendsImage);
    return used + now > limit ? BudgetReset(now) : BudgetFits(now);
  }

  /// The failure for a turn that needs [needFresh] tokens.
  ConversationTooLongException tooLong(int needFresh) =>
      ConversationTooLongException(
        'The question needs about $needFresh tokens with its '
        '${agentReserve == null ? '' : 'skills and '}reply, more than the '
        '$limit the model can hold',
      );

  /// The guard's log line for a turn that needs [needNow] tokens.
  String summary(int needNow) => switch (agentReserve) {
    null => 'budget used=$used need=$needNow limit=$limit',
    (:final rounds, system: _) =>
      'agent budget used=$used need=$needNow (rounds $rounds, system '
          '$systemNow) limit=$limit',
  };
}

/// The budget guard's verdict for one turn.
sealed class const BudgetCheck();

/// Even a fresh chat could not hold the turn's [needFresh] tokens.
final class const BudgetTooLong(final int needFresh) extends BudgetCheck;

/// The turn's [needNow] tokens fit the live chat.
final class const BudgetFits(final int needNow) extends BudgetCheck;

/// The turn's [needNow] tokens do not fit the live chat: it is recreated
/// first (the history is dropped).
final class const BudgetReset(final int needNow) extends BudgetCheck;
