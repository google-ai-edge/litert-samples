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
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/bootstrap.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/selftest/self_test_mode.dart';

/// An `EdgeAiInitialize` that keeps what one call registered.
final class _Registrations {
  List<InferenceEngineProvider>? engines;
  List<EmbeddingBackendProvider>? embedding;
  List<SttBackendProvider>? stt;
  List<TtsBackendProvider>? tts;
  var calls = 0;

  Future<void> call({
    List<InferenceEngineProvider> inferenceEngines = const [],
    List<EmbeddingBackendProvider> embeddingBackends = const [],
    List<SttBackendProvider> sttBackends = const [],
    List<TtsBackendProvider> ttsBackends = const [],
  }) async {
    calls++;
    engines = inferenceEngines;
    embedding = embeddingBackends;
    stt = sttBackends;
    tts = ttsBackends;
  }
}

/// The LiteRT-LM engine without its backend fallback, and nothing else.
void expectStrictEngine(List<InferenceEngineProvider>? engines) {
  expect(engines, hasLength(1));
  expect(
    engines!.single,
    isA<LiteRtLmEngine>().having(
      (e) => e.backendFallback,
      'backendFallback',
      isFalse,
    ),
  );
  expect(identical(engines.single, kStrictLiteRtLmEngine), isTrue);
}

/// The app's embedding backend, and nothing else. Its `requireWeightCache`
/// is private in flutter_edge_ai_litertlm 1.14.0 (`_requireWeightCache`),
/// so it is pinned through const identity: two const constructions with the
/// same arguments are one object, with other arguments another.
void expectStrictEmbedding(List<EmbeddingBackendProvider>? backends) {
  expect(backends, hasLength(1));
  expect(backends!.single, isA<LiteRtEmbeddingBackend>());
  expect(identical(backends.single, kStrictEmbeddingBackend), isTrue);
}

void main() {
  test('the LiteRT-LM engine the app and the self-test register has no '
      'backend fallback: a load runs on the backend asked for or fails (the '
      'app has no check of its own any more)', () {
    expect(kStrictLiteRtLmEngine.backendFallback, isFalse);
  });

  test('the embedding backend requires its weight cache, with '
      "EmbeddingGemma 2's image token count (pinned by const identity: the "
      'field is private)', () {
    expect(
      identical(
        kStrictEmbeddingBackend,
        const LiteRtEmbeddingBackend(
          visionTokensPerImage: kEmbedderVisionTokensPerImage,
          requireWeightCache: true,
        ),
      ),
      isTrue,
    );
    expect(
      identical(
        kStrictEmbeddingBackend,
        const LiteRtEmbeddingBackend(
          visionTokensPerImage: kEmbedderVisionTokensPerImage,
        ),
      ),
      isFalse,
      reason: 'without requireWeightCache it is another backend',
    );
  });

  test('initEdgeAi registers the strict engine, the weight-cache embedding '
      'backend and the LiteRT speech backends, in one call', () async {
    final registered = _Registrations();

    await initEdgeAi(initialize: registered.call);

    expect(registered.calls, 1);
    expectStrictEngine(registered.engines);
    expectStrictEmbedding(registered.embedding);
    expect(registered.stt?.single, isA<LiteRtSttBackend>());
    expect(registered.tts?.single, isA<LiteRtTtsBackend>());
  });

  test("the self-test registers the app's strict engine and embedding "
      'backend, and no speech', () async {
    final registered = _Registrations();

    await initSelfTestEdgeAi(initialize: registered.call);

    expect(registered.calls, 1);
    expectStrictEngine(registered.engines);
    expectStrictEmbedding(registered.embedding);
    expect(registered.stt, isEmpty);
    expect(registered.tts, isEmpty);
  });
}
