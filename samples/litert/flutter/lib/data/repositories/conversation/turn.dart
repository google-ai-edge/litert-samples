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

import '../../../domain/models/assistant_event.dart';
import '../../../domain/models/skill_step.dart';
import '../../../utils/result.dart';
import 'agent_turn.dart';
import 'generation_metrics_builder.dart';
import 'live_chat.dart';
import 'native_chat.dart';
import 'turn_start.dart';

/// One turn on [live]'s chat, as `ConversationRepository.ask` describes it.
/// An agent chat runs it through [askAgent]; a plain chat runs it here, in
/// this generator's own body rather than behind a second `yield*`, so its
/// checks run in the same microtask after `listen` as the agent path's.
Stream<AssistantEvent> askTurn(
  LiveChat live,
  String prompt, {
  Uint8List? image,
  void Function(SkillStep step)? onStep,
}) async* {
  // An agent chat runs its turn through AgentSession.
  if (live.agent case final agent?) {
    yield* askAgent(live, agent, prompt, image, onStep);
    return;
  }
  final OpenChat ready;
  switch (live.precheck(image)) {
    case Ok(:final value):
      ready = value;
    case Error(:final error):
      yield AssistantFailed(error);
      return;
  }
  final profile = ready.profile;
  final done = live.beginTurn();
  final metrics = GenerationMetricsBuilder(imageAttached: image != null);
  Uint8List? imageToSend;
  var sent = false;

  /// The image bookkeeping ran for a turn that reached the model. A caller
  /// that cancels the subscription skips everything but `finally` (async*
  /// semantics), which then settles the turn as stopped.
  var settled = false;
  try {
    // 1. Budget guard, then the stop check.
    final guarded = await guardBudget(
      live,
      ready.chat,
      null,
      prompt: prompt,
      profile: profile,
      image: image,
      metrics: metrics,
    );
    final chat = guarded.chat;
    // Announced now, whatever happens to the turn next.
    if (guarded.reset) yield const AssistantContextReset();
    if (stoppedBeforeSending(live, metrics, guarded.promptTokens)
        case final done?) {
      yield done;
      return;
    }

    // 2. Send the image only when the live chat cannot see it; its prefill
    //    includes the system instruction on an empty conversation.
    final (:send, :textTokens) = await startSend(
      live,
      chat,
      image,
      promptTokens: guarded.promptTokens,
      systemTokens: () => chat.session.sizeInTokens(profile.systemInstruction),
    );
    imageToSend = send.image;
    final (image: _, :resent, :rebuildPending, :before) = send;
    sent = true;
    await chat.addQueryChunk(
      Message(text: prompt, isUser: true, imageBytes: imageToSend),
    );
    // The turn always starts, even if a stop already arrived: the FFI
    // session buffers the prompt until generation, so skipping it would glue
    // this prompt onto the next one.
    var stopResent = false;
    await for (final response in chat.generateChatResponseAsync()) {
      if (live.stopRequested) {
        // Drain, never break: breaking would cancel the stream and skip
        // InferenceChat's end-of-turn bookkeeping (reply into its history,
        // token count), while LiteRT-LM's conversation still records the
        // partial reply, so the two histories would drift. Same approach as
        // VoiceSession's barge-in. A stop that landed before native
        // generation began had nothing to cancel; re-send it once now.
        if (!stopResent) {
          stopResent = true;
          requestNativeStop(chat);
        }
        continue;
      }
      switch (response) {
        case TextResponse(:final token):
          if (token.isNotEmpty) {
            metrics.chunk();
            yield AssistantTextDelta(token);
          }
        case ThinkingResponse() ||
            FunctionCallResponse() ||
            ParallelFunctionCallResponse():
          // Thinking is off and a plain chat declares no tools (an agent
          // chat runs through askAgent).
          debugPrint('[Conversation] unexpected ${response.runtimeType}');
      }
    }
    metrics.stop();
    final stopped = live.stopRequested;
    // 3. Before the terminal event: the caller may ask again at once.
    live.settleTurn(stopped: stopped, sentImage: imageToSend);
    settled = true;
    final after = sessionMetricsOf(chat);
    yield AssistantDone(
      metrics.finished(
        stopped: stopped,
        stopLatency: live.stopLatency,
        imageSent: imageToSend != null,
        imageResent: resent,
        before: before,
        after: after,
        rebuildPending: rebuildPending,
        chatTokens: chat.currentTokens,
        promptTokens: textTokens,
      ),
    );
  } catch (e, st) {
    debugPrint('[Conversation] turn failed: $e\n$st');
    if (sent) {
      // No rebuild follows a failure: treat what the model saw as gone, so the
      // next turn sends the image again.
      live.failTurn(sentImage: imageToSend);
      settled = true;
    }
    yield AssistantFailed(asException(e));
  } finally {
    if (sent && !settled) {
      // Cancelled mid-turn: LiteRT-LM cancels native generation and
      // rebuilds the conversation from text, like a stop.
      live.cancelTurn(sentImage: imageToSend);
    }
    live.endTurn(done);
  }
}
