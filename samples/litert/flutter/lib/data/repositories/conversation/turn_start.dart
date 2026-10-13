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
import 'package:flutter_edge_ai/flutter_edge_ai.dart';

import '../../../config/model_catalog.dart';
import '../../../domain/models/assistant_event.dart';
import 'generation_metrics_builder.dart';
import 'live_chat.dart';
import 'native_chat.dart';

/// The chat a turn sends on once the budget guard has run ([guardBudget]).
typedef GuardedChat = ({
  /// The chat to send on: a fresh one after a reset.
  InferenceChat chat,

  /// Its agent; null on a plain chat.
  AgentChat? agent,

  /// The prompt's tokens.
  int promptTokens,

  /// The chat was recreated: the caller announces it at once
  /// ([AssistantContextReset]), whatever happens to the turn next.
  bool reset,
});

/// What a turn sends ([LiveChat.prepareSend]) and the text in its prefill
/// ([startSend]).
typedef SendStart = ({TurnSend send, int textTokens});

/// The budget guard both turn paths run first: measures the turn on [chat]
/// (with [agent]'s reserve on an agent chat) and, when it would not fit,
/// recreates the chat in place — not via the open queue — recording the reset
/// in [metrics]. Throws [ConversationTooLongException] when even a fresh chat
/// could not hold the turn.
Future<GuardedChat> guardBudget(
  LiveChat live,
  InferenceChat chat,
  AgentChat? agent, {
  required String prompt,
  required ConversationProfile profile,
  required Uint8List? image,
  required GenerationMetricsBuilder metrics,
}) async {
  final budget = await live.checkBudget(
    chat,
    prompt,
    profile,
    image,
    () => metrics.elapsed,
    agent: agent,
  );
  final (InferenceChat, AgentChat?) current = switch ((budget.reset, agent)) {
    (false, _) => (chat, agent),
    (true, null) => (await live.recreatePlain(profile), null),
    (true, final agent?) => await live.recreateAgent(
      profile,
      agent,
      ImageLoss.budget,
    ),
  };
  if (budget.reset) metrics.contextWasReset(ContextResetReason.budget);
  return (
    chat: current.$1,
    agent: current.$2,
    promptTokens: budget.promptTokens,
    reset: budget.reset,
  );
}

/// The turn's end when a stop arrived before anything reached the model:
/// nothing generated and nothing lost (a stop on an idle conversation is
/// harmless). Null when the turn goes on.
AssistantDone? stoppedBeforeSending(
  LiveChat live,
  GenerationMetricsBuilder metrics,
  int promptTokens,
) {
  if (!live.stopRequested) return null;
  metrics.stop();
  return AssistantDone(
    metrics.stoppedBeforeSending(
      stopLatency: live.stopLatency,
      promptTokens: promptTokens,
    ),
  );
}

/// What the turn sends on [chat] — the image only when the live chat
/// cannot see it — and the text in its prefill, for the image-token
/// estimate: the [promptTokens], plus [systemTokens] when an image goes into
/// a conversation still empty (LiteRT-LM prefills the system prompt with
/// the first message).
Future<SendStart> startSend(
  LiveChat live,
  InferenceChat chat,
  Uint8List? image, {
  required int promptTokens,
  required Future<int> Function() systemTokens,
}) async {
  final send = live.prepareSend(chat, image);
  final firstPrefill =
      send.image != null &&
      !send.rebuildPending &&
      send.before?.inputTokens == 0;
  return (
    send: send,
    textTokens: promptTokens + (firstPrefill ? await systemTokens() : 0),
  );
}
