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

import '../domain/models/model_id.dart';

/// The demos the launcher offers and the models each one needs before its
/// tile is enabled. Both speak: their recognizer ([stt])
/// and the TTS are required.
enum Demo {
  voiceChat(
    title: 'Voice chat',
    subtitle: 'Talk or type to the chat model',
    stt: ModelId.whisperBase,
    models: {ModelId.chat, ModelId.whisperBase, ModelId.inflectNano},
    knowledgeBase: true,
  ),
  liveCamera(
    title: 'Live camera',
    subtitle: 'Live detector boxes, questions about the scene',
    stt: ModelId.moonshineTiny,
    models: {
      ModelId.chat,
      ModelId.moonshineTiny,
      ModelId.inflectNano,
      ModelId.yolo26n,
    },
  );

  const Demo({
    required this.title,
    required this.subtitle,
    required this.stt,
    required this.models,
    this.knowledgeBase = false,
  });

  final String title;
  final String subtitle;

  /// The speech recognizer this demo makes active on entry: Whisper base for
  /// Demo 1 (multilingual, 30 s window), moonshine-tiny for Demo 3 (~65 ms,
  /// so a fast answer reaches the ear in well under a second).
  final ModelId stt;
  final Set<ModelId> models;

  /// Answers from the knowledge base. It is not required: without
  /// it the tile stays enabled and says why the knowledge base is missing.
  final bool knowledgeBase;
}
