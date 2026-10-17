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
import 'package:flutter_edge_ai_litertlm/flutter_edge_ai_litertlm.dart';
import 'package:flutter_edge_ai_speech/flutter_edge_ai_speech.dart';

import 'model_catalog.dart' show kEmbedderVisionTokensPerImage;

/// `FlutterEdgeAi.initialize` with the registrations the app passes: the
/// seam [initEdgeAi] and the self-test's `initSelfTestEdgeAi` call it
/// through, so a test can see what they register.
typedef EdgeAiInitialize = Future<void> Function({
  List<InferenceEngineProvider> inferenceEngines,
  List<EmbeddingBackendProvider> embeddingBackends,
  List<SttBackendProvider> sttBackends,
  List<TtsBackendProvider> ttsBackends,
});

/// One-time flutter_edge_ai setup: the LiteRT-LM engine, the LiteRT speech
/// backends and the LiteRT embedding backend, with EmbeddingGemma 2's images
/// at [kEmbedderVisionTokensPerImage] tokens. No embedding tokenizer is
/// registered: the built-in EmbeddingGemma 2 `.litertlm` carries its own. The
/// sqlite-vec store is no longer registered here: since flutter_edge_ai 2.0
/// RAG is `flutter_edge_ai_rag`, opened by `VectorStoreService`. Registering
/// loads nothing; models load in `getActive*`. Keep this the only
/// `initialize` call: a second one keeps the first one's service settings
/// (flutter_edge_ai 2.4.0 `ServiceRegistry.initialize` returns early) and
/// appends the engines and backends not yet registered, after the first
/// ones, which win a tie of equal priority.
///
/// Both are strict ([kStrictLiteRtLmEngine], [kStrictEmbeddingBackend]): a
/// chat model loads on the backend asked for or fails with the plugin's
/// reason, and an embedder load that writes no weight cache at all fails
/// instead of running slower with more memory.
///
/// [initialize] is `FlutterEdgeAi.initialize`; a test passes one that keeps
/// what is registered.
Future<void> initEdgeAi({
  EdgeAiInitialize initialize = FlutterEdgeAi.initialize,
}) => initialize(
  inferenceEngines: const [kStrictLiteRtLmEngine],
  sttBackends: const [LiteRtSttBackend()],
  ttsBackends: const [LiteRtTtsBackend()],
  embeddingBackends: const [kStrictEmbeddingBackend],
);

/// The LiteRT-LM engine without its backend fallback
/// (flutter_edge_ai_litertlm 1.14.0): one attempt, on the backend the load
/// asks for; a backend that cannot run the model is a
/// `BackendInitException` or a `LiteRtLmBundleException`, never a model on
/// another backend. The app and the self-test register this one.
const kStrictLiteRtLmEngine = LiteRtLmEngine(backendFallback: false);

/// The LiteRT embedding backend with EmbeddingGemma 2's images at
/// [kEmbedderVisionTokensPerImage] tokens and `requireWeightCache`
/// (flutter_edge_ai_litertlm 1.14.0): a load that writes no weight cache at
/// all fails with an `EmbeddingLoadException` (the cache directory, the path
/// length, the likely causes), shown as the embedder's load error. One
/// tower's cache missing beside the others stays a printed warning, and a
/// disk that fills while a cache is written aborts natively. The
/// app and the self-test register this one.
const kStrictEmbeddingBackend = LiteRtEmbeddingBackend(
  visionTokensPerImage: kEmbedderVisionTokensPerImage,
  requireWeightCache: true,
);
