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

import '../../support/litertlm_header.dart';

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

  test('the settings line names each tower the load takes, with where it '
      'runs', () {
    expect(
      model.settingsLine(ModelTowers.none),
      'npu · ctx 1280 · tools on · gemmaIt',
      reason: 'a text-only file',
    );
    expect(
      model.settingsLine(
        model.towersOf(litertlmSections(kQualcommSm8750Sections)),
      ),
      'npu · ctx 1280 · vision npu · audio npu · tools on · gemmaIt',
    );
    expect(
      model
          .withTowerChoices(vision: PreferredBackend.cpu, audio: null)
          .settingsLine(
            model
                .withTowerChoices(vision: PreferredBackend.cpu, audio: null)
                .towersOf(litertlmSections(kQualcommSm8750Sections)),
          ),
      'npu · ctx 1280 · vision cpu · audio npu · tools on · gemmaIt',
    );
    expect(
      model.settingsLine(model.towersOf(litertlmSections(kQwen3Sections))),
      'npu · ctx 1280 · tools on · gemmaIt',
      reason: 'a text-only file: no tower',
    );
  });

  test('the plan derives the towers from the header and the backend', () {
    CustomChatPlan plan(CustomChatModel m, List<HeaderSection> header) =>
        CustomChatPlan(
          path: '/models/g3.litertlm',
          model: m,
          header: litertlmSections(header),
        );

    final allNpu = plan(model, kQualcommSm8750Sections).config.llm;
    expect(allNpu.supportImage && allNpu.supportAudio, isTrue);
    expect(allNpu.visionBackend, PreferredBackend.npu);
    expect(allNpu.audioBackend, PreferredBackend.npu);
    expect(allNpu.undeclaredTowersOnNpu, isTrue);

    final chosenCpu = plan(
      model.withTowerChoices(
        vision: PreferredBackend.cpu,
        audio: PreferredBackend.cpu,
      ),
      kQualcommSm8750Sections,
    ).config.llm;
    expect(chosenCpu.visionBackend, PreferredBackend.cpu);
    expect(chosenCpu.audioBackend, PreferredBackend.cpu);
    expect(chosenCpu.undeclaredTowersOnNpu, isFalse);

    final declared = plan(model, kQualcommQcs8275Sections).config.llm;
    expect(declared.visionBackend, PreferredBackend.cpu);
    expect(declared.audioBackend, PreferredBackend.cpu);
    expect(
      declared.undeclaredTowersOnNpu,
      isFalse,
      reason: 'the file sets both: nothing on the card moves them',
    );

    final textOnly = plan(model, kTextOnlyNpuSections).config.llm;
    expect(textOnly.supportImage || textOnly.supportAudio, isFalse);
    expect(textOnly.visionBackend, isNull);
    expect(textOnly.audioBackend, isNull);

    final gpu = plan(
      model.copyWith(backend: PreferredBackend.gpu, maxTokens: 4096),
      kGemma4E2bSections,
    ).config.llm;
    expect(gpu.visionBackend, PreferredBackend.cpu);
    expect(gpu.audioBackend, PreferredBackend.cpu);
  });

  test('copyWith keeps the towers\' choices; withTowerChoices sets them', () {
    final chosen = model.withTowerChoices(
      vision: PreferredBackend.cpu,
      audio: PreferredBackend.npu,
    );
    final moved = chosen.copyWith(backend: PreferredBackend.gpu);
    expect(moved.visionChoice, PreferredBackend.cpu);
    expect(moved.audioChoice, PreferredBackend.npu);
    final reset = moved.withTowerChoices(vision: null, audio: null);
    expect(reset.visionChoice, isNull);
    expect(reset.audioChoice, isNull);
    expect(reset.backend, PreferredBackend.gpu);
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
