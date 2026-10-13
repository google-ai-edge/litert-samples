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

/// Every model the app loads at setup, in load order.
enum ModelId {
  chat,
  whisperBase,
  inflectNano,
  yolo26n,
  moonshineTiny,
  embeddingGemma,
}

/// What the setup screen shows about a model. Every model but the chat model
/// is built into the app: its files, sizes and hashes are in
/// `data/services/model_store/bundled_model_files.dart`.
final class const ModelSpec({
  required final ModelId id,
  required final String displayName,

  /// Setup fails without a required model. An optional model's failure is
  /// shown on its row and on the demos that need it, and setup continues.
  final bool required = true,
});

/// The chat model: not shipped with the app, a `.litertlm` the user
/// chooses (the Chat model card). Required: the demos need it.
const kChatSpec = ModelSpec(id: ModelId.chat, displayName: 'Chat model');

/// Whisper base int8, Demo 1's speech recognizer: multilingual, a
/// 30 s window, ~1.8 s per question on an M4 Pro (padded to 30 s).
const kWhisperSpec = ModelSpec(
  id: ModelId.whisperBase,
  displayName: 'Whisper base (STT)',
);

/// moonshine-tiny f32, Demo 3's speech recognizer: English only, a 5 s
/// window, ~65 ms per question (Whisper base: 1.8 s); router 96/98 and
/// must-detailed 43/43 on its transcripts of the router's golden questions.
/// Optional: without it Demo 3 is unavailable and says why; Demo 1 is not
/// affected.
const kMoonshineSpec = ModelSpec(
  id: ModelId.moonshineTiny,
  displayName: 'moonshine-tiny (STT, Live camera)',
  required: false,
);

/// Inflect-nano-v2, the voice of both demos, built into the app: its
/// two models and the four Matcha G2P files it reuses.
const kInflectSpec = ModelSpec(
  id: ModelId.inflectNano,
  displayName: 'Inflect-nano-v2 (TTS)',
);

/// YOLO26n raw-head, derived locally from
/// `Arm/yolo26n-fp16-litert` (`tool/prune_yolo26n_head.py`); built into the
/// app as an asset (AGPL-3.0, `assets/models/NOTICE.md`).
const kYolo26nSpec = ModelSpec(
  id: ModelId.yolo26n,
  displayName: 'YOLO26n detector',
  required: false,
);

/// EmbeddingGemma-300M, the knowledge base's embedder. Optional:
/// without it Demo 1 chats without the knowledge base and says so.
const kEmbeddingGemmaSpec = ModelSpec(
  id: ModelId.embeddingGemma,
  displayName: 'EmbeddingGemma 300M (knowledge base)',
  required: false,
);

extension ModelIdSpec on ModelId {
  ModelSpec get spec => switch (this) {
    ModelId.chat => kChatSpec,
    ModelId.whisperBase => kWhisperSpec,
    ModelId.inflectNano => kInflectSpec,
    ModelId.yolo26n => kYolo26nSpec,
    ModelId.moonshineTiny => kMoonshineSpec,
    ModelId.embeddingGemma => kEmbeddingGemmaSpec,
  };
}
