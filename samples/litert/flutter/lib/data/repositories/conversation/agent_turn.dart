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

import '../../../domain/models/assistant_event.dart';
import '../../../domain/models/skill_step.dart';
import '../../../utils/result.dart';
import '../conversation_repository.dart';
import 'agent_turn_fold.dart';
import 'generation_metrics_builder.dart';
import 'live_chat.dart';
import 'native_chat.dart';
import 'turn_start.dart';

/// One turn on [live]'s agent chat [agent], as a stream. The agent's own stream
/// is listened to internally and always drained to its end: cancelling it would
/// only set a flag while native generation goes on. An outer cancel (a
/// VoiceSession giving up after its drain) therefore stops the turn and lets it
/// drain; [LiveChat.isGenerating] stays true until it has ended, so nothing can
/// start on the chat meanwhile.
Stream<AssistantEvent> askAgent(
  LiveChat live,
  AgentChat agent,
  String prompt,
  Uint8List? image,
  void Function(SkillStep step)? onStep,
) {
  late final StreamController<AssistantEvent> out;
  var listening = true;
  var ended = false;
  out = StreamController<AssistantEvent>(
    onListen: () => unawaited(
      _runAgentTurn(
        live,
        agent,
        prompt,
        image,
        onStep: onStep,
        emit: (event) {
          if (listening && !out.isClosed) out.add(event);
        },
      ).whenComplete(() {
        ended = true;
        return out.close();
      }),
    ),
    onCancel: () {
      listening = false;
      // While the turn runs, the current turn is this one.
      if (!ended) unawaited(live.stop());
    },
  );
  return out.stream;
}

/// Runs one agent turn to its end and reports it through [emit]: the same
/// grammar as the plain path (text deltas, at most one context reset
/// first, exactly one [AssistantDone] or [AssistantFailed] last), with the
/// skill steps going to [onStep]. Never throws.
Future<void> _runAgentTurn(
  LiveChat live,
  AgentChat agentChat,
  String prompt,
  Uint8List? image, {
  required void Function(AssistantEvent event) emit,
  required void Function(SkillStep step)? onStep,
}) async {
  final OpenChat ready;
  // No check that `agentChat` is still the open agent: `askTurn` read it
  // from the slot in the same synchronous run (its `yield*` listens to
  // `askAgent`'s controller at once, and `onListen` runs this at once).
  switch (live.precheck(image)) {
    case Ok(:final value):
      ready = value;
    case Error(:final error):
      emit(AssistantFailed(error));
      return;
  }
  final profile = ready.profile;
  final done = live.beginTurn();
  final metrics = GenerationMetricsBuilder(imageAttached: image != null);
  var chat = ready.chat;
  var agent = agentChat;
  Uint8List? imageToSend;
  var sent = false;
  final fold = AgentTurnFold(clock: () => metrics.elapsed, onStep: onStep);
  try {
    // 0. The last agent turn ended with input staged (LiteRT-LM would prepend
    //    it to the next user message). The model forgets the history while the
    //    screen still shows it: announce it like a budget reset, with its own
    //    reason.
    if (live.staleToolTail) {
      debugPrint(
        '[Conversation] the last agent turn ended without a final '
        'generation, leaving input staged: rebuilding the chat first '
        '(history lost)',
      );
      (chat, agent) = await live.recreateAgent(profile, agent, ImageLoss.stop);
      metrics.contextWasReset(ContextResetReason.interruptedSkill);
      emit(
        const AssistantContextReset(
          reason: ContextResetReason.interruptedSkill,
        ),
      );
    }

    // 1. Budget guard, then the stop check: the plain guard's prompt +
    //    reply + turn overhead, plus the tool rounds a skill call takes and,
    //    before the first prefill, the system prompt with the skill list and
    //    the tool declarations.
    final guarded = await guardBudget(
      live,
      chat,
      agent,
      prompt: prompt,
      profile: profile,
      image: image,
      metrics: metrics,
    );
    chat = guarded.chat;
    agent = guarded.agent ?? agent;
    // Announced now, whatever happens to the turn next.
    if (guarded.reset) emit(const AssistantContextReset());
    if (stoppedBeforeSending(live, metrics, guarded.promptTokens)
        case final done?) {
      emit(done);
      return;
    }

    // 2. Send the image only when the live chat cannot see it; its prefill
    //    includes, on an empty conversation, the system prompt (template,
    //    skill list) and the tool declarations: the budget reserve's figure,
    //    with the declarations estimated (kToolDeclarationTokens).
    final (:send, :textTokens) = await startSend(
      live,
      chat,
      image,
      promptTokens: guarded.promptTokens,
      systemTokens: () async => (await agent.reserve(chat)).system,
    );
    imageToSend = send.image;
    final (image: _, :resent, :rebuildPending, :before) = send;
    sent = true;

    // 3. Drain the agent's stream to its end (never cancel it).
    var stopResent = false;
    await for (final event in agent.session.ask(
      prompt,
      imageBytes: imageToSend,
      isCancelled: () => live.stopRequested,
    )) {
      if (live.stopRequested && !stopResent) {
        // A stop that landed before native generation began, or between
        // generations, had nothing to cancel then; send it once more.
        stopResent = true;
        requestNativeStop(chat);
      }
      if (fold.on(event, stopRequested: live.stopRequested) case final text?) {
        metrics.chunk();
        emit(AssistantTextDelta(text));
      }
    }
    metrics.stop();
    // Without DoneEvent or MaxIterationsEvent the loop saw the cancel.
    final stopped = fold.stopped(stopRequested: live.stopRequested);
    // 4. Before the terminal event: the caller may ask again at once.
    live.settleTurn(stopped: stopped, sentImage: imageToSend);
    // Core ends a run without a final call-free generation only at an
    // isCancelled poll, and every poll follows staged input (the prompt, a tool
    // response, or the `cancelled` answers it writes itself without an event),
    // or by running out of rounds after staging the last responses. A stop that
    // cuts a generation ends with DoneEvent and stages nothing, so the chat is
    // kept then.
    if (!fold.finished) live.markStaleToolTail();
    if (fold.maxIterations > 0) {
      emit(AssistantFailed(SkillLoopException(fold.maxIterations)));
      return;
    }
    final after = sessionMetricsOf(chat);
    emit(
      AssistantDone(
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
          toolRounds: fold.toolRounds,
          skillSteps: fold.steps,
        ),
      ),
    );
  } catch (e, st) {
    debugPrint('[Conversation] agent turn failed: $e\n$st');
    if (sent) {
      live.failTurn(sentImage: imageToSend);
      // Core balances a failed generation by staging failure answers.
      live.markStaleToolTail();
    }
    emit(AssistantFailed(asException(e)));
  } finally {
    live.endTurn(done);
  }
}
