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
import 'package:flutter_edge_ai_agent/flutter_edge_ai_agent.dart'
    show Skill, SkillExecutor;

import '../../config/model_catalog.dart';
import '../../domain/models/assistant_event.dart';
import '../../domain/models/chat_capabilities.dart';
import '../../domain/models/skill_step.dart';
import '../../utils/result.dart';
import '../services/llm/llm_service.dart';
import 'conversation/live_chat.dart';
import 'conversation/native_chat.dart';
import 'conversation/open_queue.dart';
import 'conversation/turn.dart';
import 'conversation_repository.dart';

/// [ConversationRepository] over one flutter_edge_ai `InferenceChat`
/// (`createChat`, the model's single chat slot) with the shared [kSampler]
/// and the open [ConversationProfile].
///
/// Images: LiteRT-LM keeps an image in the native conversation until the
/// conversation is rebuilt — after a stopped turn it is rebuilt from text
/// only — so this repository tracks which image the live chat can see
/// ([imageInContext]) and re-sends the caller's image when it cannot.
///
/// A facade over the pieces in `conversation/`: [OpenQueue] serves [open]
/// and [reset], [LiveChat] holds the chat slot and what its native
/// conversation holds, [askTurn] runs a turn (an agent chat's through
/// `askAgent`), and [ChatFactory] builds the native chats.
final class EdgeAiConversationRepository implements ConversationRepository {
  /// [executors] run the skills' tool calls on an agent chat (the app
  /// passes `AppIntentExecutor` and `TextSkillExecutor`); [agentTools] are
  /// an agent chat's tool declarations.
  EdgeAiConversationRepository({
    required LlmService llm,
    SamplerConfig sampler = kSampler,
    Duration stopTimeout = const Duration(seconds: 5),
    List<SkillExecutor> executors = const [],
    int maxIterations = kAgentMaxIterations,
    List<Tool> agentTools = kAgentTools,
  }) : _live = LiveChat(
         ChatFactory(
           llm: llm,
           sampler: sampler,
           executors: executors,
           maxIterations: maxIterations,
           agentTools: agentTools,
         ),
         stopTimeout: stopTimeout,
       );

  /// The chat slot: the live chat, its native state and the running turn.
  final LiveChat _live;

  /// Opens and resets, one rebuild at a time, the newest winning.
  late final OpenQueue _opens = OpenQueue(_live.rebuild);
  Future<void>? _closeRun;

  @override
  ValueListenable<bool> get isGenerating => _live.isGenerating;

  @override
  bool get hasSkills => _live.agent != null;

  @override
  bool get skillsNeedTools => _live.skillsNeedTools;

  /// What the loaded chat model allows; every chat is built from it.
  @override
  ChatCapabilities get capabilities => _live.capabilities;

  @override
  Uint8List? get imageInContext => _live.imageInContext;

  @override
  bool get isOpen => _live.isOpen;

  @override
  ConversationProfile? get profile => _live.profile;

  @override
  Future<Result<void>> open(
    ConversationProfile profile, {
    List<Skill> skills = const [],
  }) => _opens.open(profile, skills);

  /// Rebuilds with the skills of the newest [open]: a reset keeps the skill
  /// set.
  @override
  Future<Result<void>> reset({required ConversationProfile ifCurrent}) =>
      _opens.reset(ifCurrent);

  @override
  Stream<AssistantEvent> ask(
    String prompt, {
    Uint8List? image,
    void Function(SkillStep step)? onStep,
  }) => askTurn(_live, prompt, image: image, onStep: onStep);

  @override
  Future<void> stop() => _live.stop();

  /// Stops a running turn and closes the chat, leaving the repository
  /// usable: the next [open] builds a chat on whatever model is loaded then.
  /// Before the chat model is unloaded or replaced (its sessions die with
  /// it). Waits for opens already queued, and refuses those that arrive
  /// while it runs (a reset is moot then), so none builds on the old model
  /// after this returns.
  ///
  /// A turn still generating after the stop timeout is refused like
  /// [LiveChat.rebuild] refuses it: the chat, its profile and the native
  /// generation are left as they are, and the caller keeps the model.
  @override
  Future<Result<void>> release() async {
    if (_live.closed) return const Result.ok(null);
    return _opens.refusing(
      'The chat model is changing: open the chat again once it has loaded',
      () async {
        await _opens.drain();
        await _live.stop();
        if (_live.isGenerating.value) {
          debugPrint(
            '[Conversation] release refused: the stopped turn is still '
            'generating',
          );
          return Result.error(_live.stillGenerating());
        }
        _opens.forgetRequested();
        await _live.release();
        return const Result.ok(null);
      },
    );
  }

  /// Refuses new turns and opens at once, waits for a rebuild already
  /// building its chat (so the chat model is never closed under an
  /// in-flight `createChat`), stops the running turn, then closes the chat.
  /// Every call gets the same close.
  @override
  Future<void> close() =>
      _closeRun ??= _opens.refusing('The conversation is closed', () async {
        _live.beginClose();
        await _opens.drain();
        await _live.close();
      });
}
