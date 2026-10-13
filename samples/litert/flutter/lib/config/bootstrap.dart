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

import 'package:flutter_edge_ai/flutter_edge_ai.dart';
import 'package:flutter_edge_ai_embeddings/flutter_edge_ai_embeddings.dart'
    show GemmaEmbeddingTokenizers;
import 'package:flutter_edge_ai_litertlm/flutter_edge_ai_litertlm.dart';
import 'package:flutter_edge_ai_speech/flutter_edge_ai_speech.dart';

/// One-time flutter_edge_ai setup: the LiteRT-LM engine, the LiteRT speech
/// backends, the LiteRT embedding backend and Gemma's embedding
/// tokenizer. The sqlite-vec store is no longer
/// registered here: since flutter_edge_ai 2.0 RAG is `flutter_edge_ai_rag`,
/// opened by `VectorStoreService`. Registering loads nothing; models load
/// in `getActive*`. Keep this the only `initialize` call: a second one
/// returns early and silently ignores its arguments.
Future<void> initEdgeAi() => FlutterEdgeAi.initialize(
  inferenceEngines: const [LiteRtLmEngine()],
  sttBackends: const [LiteRtSttBackend()],
  ttsBackends: const [LiteRtTtsBackend()],
  embeddingBackends: const [LiteRtEmbeddingBackend()],
  embeddingTokenizers: const [GemmaEmbeddingTokenizers()],
);
