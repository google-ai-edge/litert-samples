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
/// (`ActiveModelParams.firstDifference` in flutter_edge_ai 2.4.0,
/// `lib/src/registry/runtime_config.dart`). Each chat model therefore has
/// one argument set, passed from its load on: `kLlmConfig`
/// (`config/model_catalog.dart`) for `GEMMA_MODEL_PATH`, a user's own model
/// its saved settings with its file's towers. So an image turn never
/// rebuilds the model.
///
/// No tower backend is part of it: `preferredVisionBackend` and
/// `preferredAudioBackend` stay null, and the engine places each encoder from
/// the file (flutter_edge_ai_litertlm 1.14.0); where it put them is
/// `InferenceModel.activeVisionBackend` / `activeAudioBackend` after the load.
final class const LlmConfig({
  required final int maxTokens,
  required final PreferredBackend backend,
  required final int maxNumImages,

  /// `supportImage`: the vision encoder is loaded, so chats can send photos
  /// and camera frames. For a user's own model: the file has one
  /// (`LiteRtLmBundleInfo.visionEncoder`).
  final bool supportImage = false,

  /// `supportAudio`: the audio encoder is loaded, so chats can take a spoken
  /// question as audio. For a user's own model: the file has one
  /// (`LiteRtLmBundleInfo.audioEncoder`).
  final bool supportAudio = false,
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
  /// (flutter_edge_ai_litertlm 1.14.0 `FfiInferenceModel._nativeToolsJson`,
  /// `lib/src/ffi/ffi_inference_model.dart`).
  required final ModelType modelType,
  required final LlmConfig llm,

  /// Off: every chat is plain (`tools: const []`); skills that need tool
  /// calls are disabled.
  required final bool tools,
});
