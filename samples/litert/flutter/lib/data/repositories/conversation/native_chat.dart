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
import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:flutter_edge_ai/flutter_edge_ai.dart';
import 'package:flutter_edge_ai_agent/flutter_edge_ai_agent.dart'
    show AgentSession, Skill, SkillExecutor, SkillRegistry;

import '../../../config/model_catalog.dart';
import '../../../domain/models/chat_capabilities.dart';
import '../../services/llm/llm_service.dart';
import 'context_budget.dart';

/// Builds the conversation's native chats (`InferenceChat`, the model's
/// single chat slot) on the loaded chat model, with the shared sampler and a
/// [ConversationProfile].
final class ChatFactory {
  ChatFactory({
    required this._llm,
    required this._sampler,
    required this._executors,
    required this._maxIterations,
    required this._agentTools,
  });

  final LlmService _llm;
  final SamplerConfig _sampler;

  /// What runs the skills' tool calls on an agent chat (the app
  /// passes `NativeIntentExecutor` and `TextSkillExecutor`). Explicit, so the
  /// global executor registry is never consulted (`_resolveExecutors` in
  /// flutter_edge_ai_agent's `agent_session.dart`).
  final List<SkillExecutor> _executors;
  final int _maxIterations;

  /// The tool declarations of an agent chat ([kAgentTools]).
  final List<Tool> _agentTools;

  /// What the loaded chat model allows; every chat is built from it.
  ChatCapabilities get capabilities => switch (_llm.loaded) {
    final loaded? => ChatCapabilities(
      modelName: loaded.name,
      images: loaded.llm.supportImage,
      tools: loaded.tools,
      audio: loaded.llm.supportAudio,
    ),
    null => const ChatCapabilities(
      modelName: 'no chat model',
      images: false,
      tools: false,
    ),
  };

  /// Whether flutter_edge_ai_litertlm (1.15.0 on) rewinds a stopped turn in
  /// place: as the next turn starts, the native conversation is put back to
  /// before the stopped turn and given that turn again — the message as it was
  /// sent, image and recording included, and the partial reply — so it keeps
  /// every earlier image and recording too (`RecoveringConversationHandle`,
  /// `canRewind` in `ffi_inference_model.dart`). True on the GPU and the CPU.
  /// On the NPU, whose executor rolls back one token at most, it rebuilds the
  /// conversation from the turns' text instead. A backend the engine does not
  /// report counts as the NPU here (the plugin rewinds then, but the app
  /// would rather send an image twice than lose it). On the GPU and the CPU
  /// the plugin also falls back to that rebuild when a rewind fails or the
  /// stopped turn does not wind down within 5 s, and the app cannot tell
  /// (only a log line says so).
  bool get stopRewinds =>
      _llm.isLoaded &&
      switch (_llm.model.activeBackend) {
        PreferredBackend.gpu || PreferredBackend.cpu => true,
        PreferredBackend.npu || null => false,
      };

  /// A plain chat: no tool declarations at all (`tools` stays empty, so the
  /// native session gets no `tools_json`). Built with the chat model's image
  /// and audio support (a chat without `supportAudio` would drop a recording
  /// without a word).
  Future<InferenceChat> plain(ConversationProfile profile) =>
      _llm.model.createChat(
        temperature: _sampler.temperature,
        topK: _sampler.topK,
        maxOutputTokens: profile.maxOutputTokens,
        enableThinking: kThinking,
        supportImage: capabilities.images,
        supportAudio: capabilities.audio,
        systemInstruction: profile.systemInstruction,
      );

  /// The chat for [profile]: an agent chat when the profile has a skills
  /// template and [skills] is not empty, otherwise the plain one.
  ///
  /// Every skill is added selected: `addAll` defaults to unselected, which
  /// would leave the prompt's skill list empty. Built over our own `createChat`
  /// with the shared [kSampler], only the two tools Demo 1 uses
  /// ([kAgentTools]) and the chat model's image and audio support (audio is
  /// what `AgentSession.ask(audioBytes:)` requires). `AgentSession.fromModel`
  /// (flutter_edge_ai_agent 0.3.0) takes the sampler too, but it always
  /// declares all four `agentTools` (`runSkill` and `runMcp` as well, which no
  /// executor here runs) and always passes a `modelType` to the chat
  /// (`gemma4` unless told). No `modelType` here: the native session takes its
  /// tool format from the type the model was installed with
  /// (flutter_edge_ai_litertlm 1.14.0 `FfiInferenceModel._nativeToolsJson`
  /// reads the model's type, not the chat's), and a chat-only type would
  /// disagree with it. A chat model with tools off always gets the plain chat.
  Future<(InferenceChat, AgentChat?)> build(
    ConversationProfile profile,
    List<Skill> skills,
  ) async {
    final template = profile.skillsTemplate;
    if (template == null || skills.isEmpty || !capabilities.tools) {
      return (await plain(profile), null);
    }
    final registry = SkillRegistry()..addAll(skills, selected: true);
    final systemPrompt = AgentSession.buildSystemPrompt(
      registry,
      systemPromptTemplate: template,
    );
    final chat = await _llm.model.createChat(
      temperature: _sampler.temperature,
      topK: _sampler.topK,
      maxOutputTokens: profile.maxOutputTokens,
      enableThinking: kThinking,
      supportImage: capabilities.images,
      supportAudio: capabilities.audio,
      tools: _agentTools,
      supportsFunctionCalls: true,
      toolChoice: ToolChoice.auto,
      systemInstruction: systemPrompt,
    );
    // The package's own answer to an empty generation right after
    // `loadSkill`: one retry with its nudge (`emptyAfterSkillRetries` 1, the
    // default), then an `EmptyAnswerEvent`. No retry of the app's own.
    final session = AgentSession(
      chat: chat,
      registry: registry,
      executors: _executors,
      maxIterations: _maxIterations,
    );
    return (chat, AgentChat(session, systemPrompt, List.unmodifiable(skills)));
  }
}

/// An agent chat: its session, the system prompt it was built with, its
/// skills, and the budget reserve, counted once per chat.
final class AgentChat {
  AgentChat(this.session, this.systemPrompt, this.skills);

  final AgentSession session;
  final String systemPrompt;
  final List<Skill> skills;
  AgentReserve? _reserve;

  /// What this chat holds on top of a plain turn, counted on the
  /// first call.
  Future<AgentReserve> reserve(InferenceChat chat) async {
    if (_reserve case final known?) return known;
    var largestSkill = 0;
    for (final skill in skills) {
      final tokens = await chat.session.sizeInTokens(
        '${skill.name}\n${skill.description}\n${skill.instructions}',
      );
      largestSkill = math.max(largestSkill, tokens);
    }
    final system =
        await chat.session.sizeInTokens(systemPrompt) + kToolDeclarationTokens;
    final rounds =
        kAgentToolRounds * (kToolCallTokens + kToolResultTokens) + largestSkill;
    return _reserve = (system: system, rounds: rounds);
  }
}

/// LiteRT-LM's metrics for [chat]'s live conversation; null when they
/// cannot be read.
SessionMetrics? sessionMetricsOf(InferenceChat chat) {
  try {
    return chat.session.getSessionMetrics();
  } catch (e) {
    debugPrint('[Conversation] session metrics unavailable: $e');
    return null;
  }
}

/// Asks native to stop without awaiting it: called from inside the chunk
/// loop, where awaiting could hold up the stream it is stopping.
void requestNativeStop(InferenceChat chat) {
  unawaited(
    chat.stopGeneration().catchError(
      (Object e, StackTrace st) =>
          debugPrint('[Conversation] stopGeneration failed: $e\n$st'),
    ),
  );
}
