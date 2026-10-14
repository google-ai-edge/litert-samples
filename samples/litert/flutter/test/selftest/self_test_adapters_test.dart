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

import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter_edge_ai/flutter_edge_ai.dart'
    show ModelType, PreferredBackend;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/knowledge_config.dart'
    show kKbAudioWarmUpAsset;
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/services/knowledge/embedder_service.dart';
import 'package:litert_edge_demos/data/services/model_store/bundled_model_files.dart';
import 'package:litert_edge_demos/domain/models/chat_model.dart';
import 'package:litert_edge_demos/domain/models/chat_model_config.dart';
import 'package:litert_edge_demos/domain/models/knowledge_query.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/selftest/self_test_adapters.dart';
import 'package:litert_edge_demos/selftest/self_test_runner.dart'
    show
        GemmaLoad,
        SelfTestChatModel,
        SelfTestEmbedderLoad,
        SelfTestImagesLoad,
        gemmaLoadConfigLine;
import 'package:litert_edge_demos/utils/result.dart';

import '../fakes/fake_bundled_files.dart';
import '../fakes/fake_knowledge.dart';
import '../fakes/fake_llm_service.dart';
import '../support/litertlm_header.dart';

/// The self-test loads exactly what the app would (GEMMA_MODEL_PATH must
/// not override the user's own model), except for an explicit
/// `--gemma`.
void main() {
  // The warm-up clip comes from the asset bundle.
  TestWidgetsFlutterBinding.ensureInitialized();
  late Directory tmp;
  late String defineFile;
  late String customFile;
  late String argumentFile;

  setUp(() {
    tmp = Directory.systemTemp.createTempSync('self_test_adapters');
    String file(String name) =>
        (File('${tmp.path}/$name')..writeAsStringSync('x')).path;
    defineFile = file('define.litertlm');
    customFile = file('custom.litertlm');
    argumentFile = file('argument.litertlm');
  });
  tearDown(() => tmp.deleteSync(recursive: true));

  CustomChatPlan custom({List<HeaderSection> header = const []}) =>
      CustomChatPlan(
        path: customFile,
        header: litertlmSections(header),
        model: CustomChatModel(
          displayName: 'My Gemma',
          source: ImportedModelSource(customFile),
          file: const CustomModelFile(
            name: 'custom.litertlm',
            sizeBytes: 1,
            sha256: 'abc',
            checksumMatched: false,
          ),
          backend: PreferredBackend.gpu,
          maxTokens: 4096,
        ),
      );

  Future<SelfTestChatModel> resolve(
    ChatModelPlan plan, {
    String? argument,
    String define = '',
  }) =>
      resolveSelfTestChatModel(argument: argument, define: define, plan: plan);

  test('your own model wins over GEMMA_MODEL_PATH, like the app', () async {
    final chosen = await resolve(custom(), define: defineFile);

    expect(chosen.file.path, customFile);
    expect(chosen.settingsSource, 'your own .litertlm, saved settings');
    expect(chosen.config.name, 'My Gemma');
    expect(chosen.sha256, 'abc');
  });

  test('a file used in place says so, not "model store"', () async {
    final local = custom();
    final chosen = await resolve(
      CustomChatPlan(
        path: customFile,
        header: local.header,
        model: local.model.copyWith(source: LocalModelSource(customFile)),
      ),
    );

    expect(chosen.file.path, customFile);
    expect(chosen.file.source, 'in place');
    expect((await resolve(local)).file.source, 'model store, custom/');
  });

  test('no chat model chosen: GEMMA_MODEL_PATH when set, else a problem '
      '(the app ships none)', () async {
    final fromDefine = await resolve(
      const NoChatModelPlan(),
      define: defineFile,
    );
    expect(fromDefine.file.path, defineFile);
    expect(fromDefine.file.source, 'GEMMA_MODEL_PATH');
    expect(fromDefine.sha256, isNull);

    final none = await resolve(const NoChatModelPlan(note: 'retired'));
    expect(none.file.path, isNull);
    expect(none.file.problem, startsWith('No chat model yet'));
    expect(none.file.problem, endsWith('retired'));
  });

  test('an explicit --gemma wins over everything (the CLI override)', () async {
    final chosen = await resolve(
      custom(),
      argument: argumentFile,
      define: defineFile,
    );

    expect(chosen.file.path, argumentFile);
    expect(chosen.file.source, '--gemma');
  });

  group('--gemma-backend places the towers of your own model as the app '
      'does', () {
    test('towers declared by the file (the official qcs8275 build): they '
        'stay on the CPU (the file sets them)', () async {
      final chosen = await resolve(custom(header: kQualcommQcs8275Sections));
      final onNpu = chosen.withBackend(PreferredBackend.npu);

      expect(onNpu.config.llm.backend, PreferredBackend.npu);
      expect(onNpu.config.llm.visionBackend, PreferredBackend.cpu);
      expect(onNpu.config.llm.audioBackend, PreferredBackend.cpu);
      expect(onNpu.custom, same(chosen.custom), reason: 'carried over');
      expect(onNpu.header, same(chosen.header));
    });

    test(
      'undeclared towers (the official sm8750 build): both towers go to the '
      'NPU with it, as the app places them; the saved choices still apply',
      () async {
        final chosen = await resolve(custom(header: kQualcommSm8750Sections));
        expect(
          chosen.config.llm.visionBackend,
          PreferredBackend.cpu,
          reason: 'saved on the GPU: off the NPU',
        );
        final onNpu = chosen.withBackend(PreferredBackend.npu).config.llm;
        expect(onNpu.visionBackend, PreferredBackend.npu);
        expect(onNpu.audioBackend, PreferredBackend.npu);

        final plan = custom(header: kQualcommSm8750Sections);
        final cpuChoices = CustomChatPlan(
          path: plan.path,
          header: plan.header,
          model: plan.model.withTowerChoices(
            vision: PreferredBackend.cpu,
            audio: PreferredBackend.cpu,
          ),
        );
        final sm8750 = (await resolve(cpuChoices))
            .withBackend(PreferredBackend.npu)
            .config
            .llm;
        expect(sm8750.visionBackend, PreferredBackend.cpu);
        expect(sm8750.audioBackend, PreferredBackend.cpu);
      },
    );

    test("--gemma and GEMMA_MODEL_PATH keep the app's fixed Gemma 4 E2B "
        'towers (vision on the CPU, no audio) on any backend', () async {
      for (final chosen in [
        await resolve(custom(), argument: argumentFile),
        await resolve(const NoChatModelPlan(), define: defineFile),
      ]) {
        expect(chosen.custom, isNull);
        final onNpu = chosen.withBackend(PreferredBackend.npu).config.llm;
        expect(onNpu.backend, PreferredBackend.npu);
        expect(onNpu.visionBackend, PreferredBackend.cpu);
        expect(onNpu.audioBackend, isNull);
      }
    });
  });

  test('the chat model load warms its towers up as the app does: the image '
      'and, with kLlmAudioWarmUp, the audio tower', () async {
    final llm = FakeLlmService()..followRequested = true;
    final gemma = AppSelfTestGemma(llm: llm);
    final plan = custom(header: kGemma4E2bSections);

    final loaded = await gemma.load(plan.path, plan.config);

    expect(loaded, isA<Ok<GemmaLoad>>(), reason: '$loaded');
    expect(llm.warmUpsWithImage.single, isTrue);
    expect(llm.warmUpsWithAudio.single, kLlmAudioWarmUp);

    final textOnly = custom(header: kTextOnlyNpuSections);
    await gemma.load(textOnly.path, textOnly.config);
    expect(llm.warmUpsWithImage.last, isFalse);
    expect(llm.warmUpsWithAudio.last, isFalse);
  });

  test('an explicit --detector wins over the built-in one; without it, the '
      'built-in one', () async {
    final chosen = await resolveSelfTestDetector(argument: argumentFile);

    expect(chosen.path, argumentFile);
    expect(chosen.source, '--detector');
    expect(chosen.bundled, isFalse);

    final missing = await resolveSelfTestDetector(
      argument: '${tmp.path}/nowhere.tflite',
    );
    expect(missing.path, isNull);
    expect(missing.problem, startsWith('not readable'));

    final builtIn = await resolveSelfTestDetector(argument: null);
    expect(builtIn.bundled, isTrue);
    expect(builtIn.problem, isNull);
  });

  test('the load line says what was asked for, and the engine\'s context '
      'when it differs', () {
    expect(
      gemmaLoadConfigLine(kDefineChatModel, engineContext: 8192),
      'maxTokens 8192 · vision cpu · tools on · type gemma4',
    );
    expect(
      gemmaLoadConfigLine(kDefineChatModel, engineContext: 4096),
      'maxTokens 8192 (engine 4096) · vision cpu · tools on · type gemma4',
    );
    // Each loaded tower is named with where it runs.
    const npu = ChatModelConfig(
      name: 'Gemma 4 E2B NPU',
      modelType: ModelType.gemma4,
      llm: LlmConfig(
        maxTokens: 4096,
        backend: PreferredBackend.npu,
        maxNumImages: 1,
        towers: ModelTowers(
          vision: TowerPlacement(
            PreferredBackend.npu,
            reason: TowerReason.choice,
          ),
          audio: TowerPlacement(
            PreferredBackend.cpu,
            reason: TowerReason.declared,
          ),
        ),
      ),
      tools: false,
    );
    expect(
      gemmaLoadConfigLine(npu, engineContext: 4096),
      'maxTokens 4096 · vision npu · audio cpu · tools off · type gemma4',
    );
  });

  test('a blocked choice stays blocked, never Gemma 4 E2B', () async {
    final chosen = await resolve(
      const ChatPlanBlocked('the file changed'),
      define: defineFile,
    );

    expect(chosen.file.path, isNull);
    expect(chosen.file.problem, 'the file changed');
  });

  group('the embedder of steps 7a/7b', () {
    test('in the app, the app\'s own: refused with its state until setup has '
        'it ready; then used as it is and never closed', () async {
      final model = FakeEmbeddingModel();
      final service = await loadedEmbedder(model);
      final states = ValueNotifier<Map<ModelId, ModelState>>({
        ModelId.embeddingGemma: const ModelLoading(),
      });
      addTearDown(states.dispose);
      var imageLoads = 0;
      final borrowed = BorrowedSelfTestEmbedder(
        service,
        states,
        enableImages: () async {
          imageLoads++;
          return const Result.ok(
            EmbedderImagesLoad(loadTime: Duration(seconds: 2)),
          );
        },
      );

      final loading = await borrowed.prepare();
      expect('${(loading as Error).error}', contains('is loading'));

      states.value = {
        ModelId.embeddingGemma: const ModelReady(
          LoadedModelInfo(
            modelId: 'embeddinggemma-2-740m',
            backend: 'cpu',
            loadTime: Duration(seconds: 2),
            warmUpTime: Duration.zero,
          ),
        ),
      };
      final ready =
          (await borrowed.prepare() as Ok<SelfTestEmbedderLoad>).value;
      expect(ready.borrowed, isTrue);
      expect(ready.media, 'audio+text', reason: 'setup loads no images');
      expect(ready.loadTime, isNull);
      expect(
        await borrowed.embed(const KnowledgeQuery.text('What time is it?')),
        isA<Ok<List<double>>>(),
      );
      // Step 7c loads images the app's way (the row follows).
      final images =
          (await borrowed.enableImages() as Ok<SelfTestImagesLoad>).value;
      expect(images.loadTime, const Duration(seconds: 2));
      expect(imageLoads, 1);
      await borrowed.close();
      expect(model.closeCalls, 0, reason: 'the knowledge base still uses it');
      expect(service.isLoaded, isTrue);
    });

    test('headless, one loaded for the run from the built-in file, warmed '
        'up as the app warms it up (the text question and the spoken clip, '
        'so step 7a runs a warm audio encoder), and closed after it; a build '
        'without the file fails', () async {
      final model = FakeEmbeddingModel();
      final file = File('${tmp.path}/${kBundledEmbedderModel.name}')
        ..writeAsStringSync('x');
      final own = AppSelfTestEmbedder(
        bundled: FakeBundledFiles.at({kBundledEmbedderModel.asset: file.path}),
        service: appEmbedderService(
          install: (config, source, onProgress) async =>
              'embeddinggemma-2-740m',
          load: (config) async => model.loadedWith(config),
        ),
      );

      final prepared = await own.prepare();
      expect(prepared, isA<Ok<SelfTestEmbedderLoad>>(), reason: '$prepared');
      final load = (prepared as Ok<SelfTestEmbedderLoad>).value;
      expect(load.borrowed, isFalse);
      expect(load.media, 'audio+text', reason: 'images come with step 7c');
      expect(load.loadTime, isNotNull);
      expect(model.queries, hasLength(1), reason: 'the warm-up question');
      final clip = await loadAudioWarmUpClip();
      expect(
        model.inputs.where((i) => i.$1.audio != null).single.$1.audio,
        clip,
        reason: 'the warm-up clip, $kKbAudioWarmUpAsset',
      );
      await own.close();
      expect(model.closeCalls, 1);

      final missing = AppSelfTestEmbedder(bundled: FakeBundledFiles());
      expect(
        '${(await missing.prepare() as Error).error}',
        contains('No app bundle'),
      );
    });
  });
}
