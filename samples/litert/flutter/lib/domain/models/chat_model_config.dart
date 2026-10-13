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

import 'package:flutter_edge_ai/flutter_edge_ai.dart'
    show ModelType, PreferredBackend;

/// The model-level arguments of `FlutterEdgeAi.getActiveModel`.
///
/// The model is a process-wide singleton: any argument that differs from the
/// previous call rebuilds it and closes the old handle
/// (`ActiveModelParams.firstDifference` in flutter_edge_ai 2.1.0).
/// The app's own set (`kLlmConfig` in `config/model_catalog.dart`) is
/// therefore the only argument set ever passed, and it
/// enables images from the start, so the first image turn does not rebuild
/// the model.
final class const LlmConfig({
  required final int maxTokens,
  required final PreferredBackend backend,
  required final bool supportImage,
  required final int maxNumImages,
});

/// The chat model as the loaders use it: the install type, the [LlmConfig] of
/// `getActiveModel`, and whether chats may declare tools. A user's own
/// `.litertlm` gets one built from its saved settings; `GEMMA_MODEL_PATH` (a
/// `--dart-define`) runs with `kDefineChatModel`.
final class const ChatModelConfig({
  /// Shown everywhere the chat model is named (`Gemma 4 E2B`).
  required final String name,

  /// Given to `installModel` only: the native `.litertlm` session takes its
  /// tool format from the installed type, so chats never pass another one
  /// (flutter_edge_ai_litertlm 1.9.0 `FfiInferenceModel._nativeToolsJson`,
  /// `lib/src/ffi/ffi_inference_model.dart`).
  required final ModelType modelType,
  required final LlmConfig llm,

  /// Off: every chat is plain (`tools: const []`); skills that need tool
  /// calls are disabled.
  required final bool tools,
});
