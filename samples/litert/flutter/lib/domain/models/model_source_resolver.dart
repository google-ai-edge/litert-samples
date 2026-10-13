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

import '../../config/env.dart';
import '../../config/model_catalog.dart';
import 'chat_model.dart';
import 'chat_model_config.dart';

/// The `--dart-define` that names a chat model file while none is chosen
/// ([kGemmaModelPath]).
const kGemmaModelPathDefine = 'GEMMA_MODEL_PATH';

/// Where the chat model slot's file comes from.
sealed class const ChatModelSource();

/// The user's own `.litertlm`, chosen in the Chat model card: it wins over
/// `GEMMA_MODEL_PATH`.
final class const CustomChatSource(final CustomChatPlan plan)
    extends ChatModelSource;

/// `GEMMA_MODEL_PATH` while no model is chosen: [path] as given (absolute,
/// or relative to the documents directory), run with [config] (Gemma 4
/// E2B's settings).
final class const DefineChatSource({
  required final String path,
  required final ChatModelConfig config,
}) extends ChatModelSource {
  /// `GEMMA_MODEL_PATH=<path>`, where the slot's file came from.
  String get label => '$kGemmaModelPathDefine=$path';
}

/// The chosen model cannot load ([reason]); never replaced by another one.
final class const BlockedChatSource(final String reason)
    extends ChatModelSource;

/// No model chosen and no define: the slot stays empty ([note]: why, after a
/// retired saved choice).
final class const NoChatSource({final String? note}) extends ChatModelSource;

/// The precedence rule for the chat model's source. `ModelRepository` loads
/// by it, the Models screen shows it and the self-test resolves by it, so
/// the self-test loads exactly what the app would. Pure: paths are not
/// resolved or checked here.
///
/// The model chosen in the Chat model card (or its blocked choice); with
/// none chosen, `GEMMA_MODEL_PATH` when set, else nothing. Every other model
/// is built into the app.
final class const ModelSourceResolver({
  /// `GEMMA_MODEL_PATH`; the build's own by default.
  final String gemmaModelPath = kGemmaModelPath,

  /// What `GEMMA_MODEL_PATH` loads with.
  final ChatModelConfig defineChatModel = kDefineChatModel,
}) {
  ChatModelSource chat(ChatModelPlan plan) => switch (plan) {
    CustomChatPlan() => CustomChatSource(plan),
    ChatPlanBlocked(:final reason) => BlockedChatSource(reason),
    NoChatModelPlan(:final note) =>
      gemmaModelPath.isEmpty
          ? NoChatSource(note: note)
          : DefineChatSource(path: gemmaModelPath, config: defineChatModel),
  };
}
