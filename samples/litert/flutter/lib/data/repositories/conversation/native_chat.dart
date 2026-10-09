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
  /// passes `AppIntentExecutor` and `TextSkillExecutor`). Explicit, so the
  /// global executor registry is never consulted (`agent_session.dart:27`).
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
    ),
    null => const ChatCapabilities(
      modelName: 'no chat model',
      images: false,
      tools: false,
    ),
  };

  /// A plain chat: no tool declarations at all (`tools` stays empty, so the
  /// native session gets no `tools_json`).
  Future<InferenceChat> plain(ConversationProfile profile) =>
      _llm.model.createChat(
        temperature: _sampler.temperature,
        topK: _sampler.topK,
        maxOutputTokens: profile.maxOutputTokens,
        enableThinking: kThinking,
        supportImage: capabilities.images,
        systemInstruction: profile.systemInstruction,
      );

  /// The chat for [profile]: an agent chat when the profile has a skills
  /// template and [skills] is not empty, otherwise the plain one.
  ///
  /// Every skill is added selected: `addAll` defaults to unselected, which
  /// would leave the prompt's skill list empty. Built over our own `createChat`
  /// (not `AgentSession.fromModel`, whose sampler defaults are
  /// `topK: 1, temperature: .8`) with only the two tools Demo 1 uses and the
  /// chat model's image support. No `modelType` here: the native session takes
  /// its tool format from the type the model was installed with
  /// (flutter_edge_ai_litertlm 1.9.0 `FfiInferenceModel._nativeToolsJson` reads
  /// the model's type, not the chat's), and a chat-only type would disagree
  /// with it. A chat model with tools off always gets the plain chat.
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
      tools: _agentTools,
      supportsFunctionCalls: true,
      toolChoice: ToolChoice.auto,
      systemInstruction: systemPrompt,
    );
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
