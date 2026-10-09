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
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/domain/models/chat_model.dart';

void main() {
  final sha = 'a1' * 32;
  final model = CustomChatModel(
    displayName: 'Gemma 3 1B NPU',
    source: UrlModelSource(
      Uri.parse('https://example.com/g3_ekv1280.litertlm'),
      sha256: sha,
      sizeBytes: 1234,
    ),
    file: CustomModelFile(
      name: 'g3_ekv1280.litertlm',
      sizeBytes: 1234,
      sha256: sha,
      checksumMatched: true,
    ),
    modelType: ModelType.gemmaIt,
    backend: PreferredBackend.npu,
    maxTokens: 1280,
    tools: true,
  );

  test('context limits per backend', () {
    expect(CustomChatModel.contextProblem(896, PreferredBackend.npu), isNull);
    expect(
      CustomChatModel.contextProblem(896, PreferredBackend.gpu),
      contains('at least 1024'),
    );
    expect(CustomChatModel.contextProblem(64, PreferredBackend.npu), isNotNull);
    expect(
      CustomChatModel.contextProblem(1 << 20, PreferredBackend.cpu),
      contains('at most'),
    );
  });

  test('an NPU build name announces its context', () {
    expect(
      contextHintFromFileName('Gemma3-1B-IT_q4_ekv1280_sm8750.litertlm'),
      1280,
    );
    expect(contextHintFromFileName('gemma-4-E2B-it.litertlm'), isNull);
  });

  test('the settings line says everything the card needs', () {
    expect(
      model.settingsLine,
      'npu · ctx 1280 · images off · tools on · gemmaIt',
    );
  });

  test('the model type a file name announces', () {
    expect(modelTypeFromFileName('gemma-4-E2B-it.litertlm'), ModelType.gemma4);
    expect(
      modelTypeFromFileName('gemma4_2b_SM8850.litertlm'),
      ModelType.gemma4,
    );
    expect(
      modelTypeFromFileName('Gemma3-1B-IT_q4_ekv1280.litertlm'),
      ModelType.gemmaIt,
    );
    expect(modelTypeFromFileName('gemma-3n-E2B.litertlm'), ModelType.gemmaIt);
    expect(modelTypeFromFileName('Qwen3-0.6B.litertlm'), ModelType.qwen3);
    expect(modelTypeFromFileName('qwen3.5-1.7b.litertlm'), ModelType.qwen35);
    expect(modelTypeFromFileName('Qwen2.5-1.5B.litertlm'), ModelType.qwen);
    expect(modelTypeFromFileName('Llama-3.2-1B.litertlm'), ModelType.llama);
    expect(modelTypeFromFileName('Phi-4-mini.litertlm'), ModelType.phi);
    expect(
      modelTypeFromFileName('DeepSeek-R1-Distill.litertlm'),
      ModelType.deepSeek,
    );
    expect(modelTypeFromFileName('model.litertlm'), ModelType.gemma4);
  });
}
