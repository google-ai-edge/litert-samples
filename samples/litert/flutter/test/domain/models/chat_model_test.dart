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
import 'package:litert_edge_demos/domain/models/chat_model_config.dart';

import '../../support/fake_bundles.dart';

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

  test('the settings line names the towers the file has, never where they '
      'will run', () {
    expect(model.settingsLine(), 'npu · ctx 1280 · tools on · gemmaIt');
    expect(
      model.settingsLine(BundleRead(bundleInfo(npu: kTestNpuTarget))),
      'npu · ctx 1280 · towers none · tools on · gemmaIt',
      reason: 'a text-only file',
    );
    expect(
      model.settingsLine(
        BundleRead(bundleInfo(vision: true, audio: true, npu: kTestNpuTarget)),
      ),
      'npu · ctx 1280 · towers vision+audio · tools on · gemmaIt',
    );
    expect(towersLine(BundleRead(bundleInfo(vision: true))), 'towers vision');
    expect(towersLine(BundleRead(bundleInfo(audio: true))), 'towers audio');
  });

  test('the plan loads every tower the file has and asks for no tower '
      'backend, on any backend', () {
    LlmConfig llm(CustomChatModel m, LiteRtLmBundleInfo bundle) =>
        CustomChatPlan(
          path: '/models/g3.litertlm',
          model: m,
          bundle: BundleRead(bundle),
        ).config.llm;

    final both = llm(
      model,
      bundleInfo(vision: true, audio: true, npu: kTestNpuTarget),
    );
    expect(both.supportImage && both.supportAudio, isTrue);
    expect(both.backend, PreferredBackend.npu);
    expect(both.maxTokens, 1280);

    final textOnly = llm(model, bundleInfo(npu: kTestNpuTarget));
    expect(textOnly.supportImage || textOnly.supportAudio, isFalse);

    final visionOnly = llm(
      model.copyWith(backend: PreferredBackend.gpu, maxTokens: 4096),
      bundleInfo(vision: true),
    );
    expect(visionOnly.supportImage, isTrue);
    expect(visionOnly.supportAudio, isFalse);
    expect(visionOnly.backend, PreferredBackend.gpu);
  });

  test('a file whose header the plugin could not read is planned text-only '
      '(no tower is known) and says so in its settings line', () {
    const reason = LiteRtLmBundleException(
      LiteRtLmBundleProblem.unsupportedVersion,
      '/models/g3.litertlm is a version 2 .litertlm container; this package '
      'reads version 1',
    );
    final plan = CustomChatPlan(
      path: '/models/g3.litertlm',
      model: model,
      bundle: const BundleNotRead(reason),
    );

    expect(plan.config.llm.supportImage, isFalse);
    expect(plan.config.llm.supportAudio, isFalse);
    expect(plan.bundle.info, isNull);
    expect(
      model.settingsLine(plan.bundle),
      'npu · ctx 1280 · towers not read · tools on · gemmaIt',
    );
  });

  test('npuOnlyReason: in the plugin\'s terms for a build compiled for an '
      'NPU, null for any other file', () {
    expect(
      npuOnlyReason(bundleInfo(npu: kTestNpuTarget)),
      'compiled for the Qualcomm SM8750 NPU: runs only on the NPU',
    );
    expect(npuOnlyReason(bundleInfo(vision: true, audio: true)), isNull);
    expect(npuOnlyReason(null), isNull);
  });

  test('the model type a file name announces', () {
    expect(modelTypeFromFileName('gemma-4-E2B-it.litertlm'), ModelType.gemma4);
    expect(
      modelTypeFromFileName('gemma4_e2b_npu_sm8850.litertlm'),
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
