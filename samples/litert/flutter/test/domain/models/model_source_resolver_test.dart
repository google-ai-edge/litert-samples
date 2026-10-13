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
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/domain/models/chat_model.dart';
import 'package:litert_edge_demos/domain/models/chat_model_config.dart';
import 'package:litert_edge_demos/domain/models/model_source_resolver.dart';

/// One precedence for the chat model, shared by the app and the self-test.
void main() {
  const withDefine = ModelSourceResolver(gemmaModelPath: '/dev/gemma.litertlm');

  const custom = CustomChatPlan(
    path: '/store/custom/g3.litertlm',
    model: CustomChatModel(
      displayName: 'Gemma 3 1B NPU',
      source: ImportedModelSource('/picked/g3.litertlm'),
      backend: PreferredBackend.npu,
      maxTokens: 1280,
    ),
  );

  test('the chosen model wins over GEMMA_MODEL_PATH', () {
    final source = withDefine.chat(custom);

    expect(source, isA<CustomChatSource>());
    expect((source as CustomChatSource).plan, same(custom));
  });

  test('a blocked choice stays blocked, even with GEMMA_MODEL_PATH set', () {
    final source = withDefine.chat(
      const ChatPlanBlocked('g3.litertlm is gone'),
    );

    expect(
      source,
      isA<BlockedChatSource>().having(
        (s) => s.reason,
        'reason',
        'g3.litertlm is gone',
      ),
    );
  });

  test("none chosen: GEMMA_MODEL_PATH with Gemma 4 E2B's settings", () {
    final source = withDefine.chat(const NoChatModelPlan(note: 'retired'));

    final define = source as DefineChatSource;
    expect(define.path, '/dev/gemma.litertlm');
    expect(define.label, 'GEMMA_MODEL_PATH=/dev/gemma.litertlm');
    expect(define.config, same(kDefineChatModel));
  });

  test("the define's settings can be replaced", () {
    const config = ChatModelConfig(
      name: 'Gemma 4 E2B',
      modelType: ModelType.gemma4,
      llm: LlmConfig(
        maxTokens: 4096,
        backend: PreferredBackend.cpu,
        supportImage: false,
        maxNumImages: 1,
      ),
      tools: true,
    );

    final source = const ModelSourceResolver(
      gemmaModelPath: '/dev/gemma.litertlm',
      defineChatModel: config,
    ).chat(const NoChatModelPlan());

    expect((source as DefineChatSource).config, same(config));
  });

  test('none chosen and no define: nothing, with the note', () {
    final source = const ModelSourceResolver(gemmaModelPath: '')
        .chat(const NoChatModelPlan(note: 'Gemma 4 E2B is retired.'));

    expect(
      source,
      isA<NoChatSource>().having(
        (s) => s.note,
        'note',
        'Gemma 4 E2B is retired.',
      ),
    );
  });

  test("by default, the build's own GEMMA_MODEL_PATH (none in tests)", () {
    const sources = ModelSourceResolver();

    expect(sources.gemmaModelPath, isEmpty);
    expect(sources.chat(const NoChatModelPlan()), isA<NoChatSource>());
  });
}
