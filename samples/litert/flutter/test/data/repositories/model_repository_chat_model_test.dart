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

import 'package:flutter_edge_ai/flutter_edge_ai.dart'
    show ModelType, PreferredBackend;
import 'package:flutter_edge_ai_litertlm/flutter_edge_ai_litertlm.dart'
    show NpuDispatchUnavailableException;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/repositories/model_repository.dart';
import 'package:litert_edge_demos/data/services/knowledge/embedder_service.dart';
import 'package:litert_edge_demos/data/services/llm/llm_service.dart';
import 'package:litert_edge_demos/domain/models/chat_model.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/domain/ports/chat_model_planner.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../fakes/fake_bundled_files.dart';
import '../../fakes/fake_detector_runtime.dart';
import '../../fakes/fake_llm_service.dart';
import '../../fakes/fake_speech.dart';
import '../../support/fake_bundles.dart';

/// A planner the test switches by hand.
final class _Planner implements ChatModelPlanner {
  _Planner(this.plan);

  @override
  ChatModelPlan plan;
}

final _sha = 'cd' * 32;

CustomChatModel _custom({
  PreferredBackend backend = PreferredBackend.npu,
  bool tools = false,
}) => CustomChatModel(
  displayName: 'Gemma 3 1B NPU',
  source: const ImportedModelSource('/picked/g3.litertlm'),
  file: CustomModelFile(
    name: 'g3.litertlm',
    sizeBytes: 1000,
    sha256: _sha,
    checksumMatched: false,
  ),
  modelType: ModelType.gemmaIt,
  backend: backend,
  maxTokens: 1280,
  tools: tools,
);

/// The plan for [model] whose file is [bundle] (by default a text-only
/// per-SoC build: no towers).
CustomChatPlan _plan(
  CustomChatModel model, {
  FakeBundle bundle = FakeBundle.textOnlyNpu,
}) => CustomChatPlan(
  path: '/store/custom/g3.litertlm',
  model: model,
  bundle: BundleRead(bundle.info(4096)),
);

/// The chat slot loads exactly what the plan says, with its own type,
/// backend and settings, and never Gemma 4 E2B in place of a custom model
/// that cannot load.
void main() {
  late FakeLlmService llm;
  late _Planner planner;
  late ModelRepository models;

  setUp(() {
    llm = FakeLlmService();
    planner = _Planner(_plan(_custom()));
    models = ModelRepository(
      bundled: FakeBundledFiles(),
      detector: fakeDetectorService(),
      bundledDetector: fakeBundledDetector,
      gemmaModelPath: '',
      llm: llm,
      stt: fakeSttService(),
      tts: fakeTtsService(),
      chatModels: planner,
      embedder: EmbedderService(),
    );
    addTearDown(models.close);
  });

  test('a custom plan installs its file with its type and loads its own '
      'backend, context, images and tools', () async {
    expect(await models.prepareAll(), isA<Ok<void>>());

    expect(llm.installs.single, '/store/custom/g3.litertlm');
    expect(llm.installTypes.single, ModelType.gemmaIt);
    final loaded = llm.models.single;
    expect(loaded.name, 'Gemma 3 1B NPU');
    expect(loaded.llm.backend, PreferredBackend.npu);
    expect(loaded.llm.maxTokens, 1280);
    expect(loaded.llm.supportImage, isFalse);
    expect(loaded.tools, isFalse);
    expect(llm.warmUpsWithImage.single, isFalse, reason: 'no vision encoder');
    expect(llm.warmUpsWithAudio.single, isFalse, reason: 'no audio encoder');

    final info = (models.states.value[ModelId.chat] as ModelReady).info;
    expect(info.backend, 'npu');
    final chat = info.chat!;
    expect(chat.name, 'Gemma 3 1B NPU');
    expect(chat.custom, isTrue);
    expect(chat.source, 'imported from /picked/g3.litertlm');
    expect(chat.sha256, _sha);
    expect(chat.requestedBackend, 'npu');
    expect(chat.contextTokens, 1280);
    expect(chat.images, isFalse);
    expect(chat.tools, isFalse);
    expect(chat.modelType, 'gemmaIt');
  });

  test('no chat model chosen: the slot is unavailable with how to choose '
      'one, and everything else still loads', () async {
    planner.plan = const NoChatModelPlan(note: 'Gemma 4 E2B is retired.');

    expect(await models.prepareAll(), isA<Error<void>>());

    expect(llm.installs, isEmpty);
    expect(
      models.states.value[ModelId.chat],
      isA<ModelUnavailable>().having(
        (s) => s.reason,
        'reason',
        allOf(startsWith('No chat model yet'), endsWith('retired.')),
      ),
    );
    for (final id in [ModelId.whisperBase, ModelId.inflectNano]) {
      expect(models.states.value[id], isA<ModelReady>(), reason: '$id');
    }
    expect(models.requiredReady, isFalse);

    // Chosen afterwards: the reload loads only the chat model.
    planner.plan = _plan(_custom());
    expect(await models.reloadChatModel(), isA<Ok<void>>());
    expect(models.requiredReady, isTrue);
  });

  test('a chosen model wins over GEMMA_MODEL_PATH (the define only fills an '
      'empty slot)', () async {
    final both = ModelRepository(
      bundled: FakeBundledFiles(),
      detector: fakeDetectorService(),
      bundledDetector: fakeBundledDetector,
      gemmaModelPath: '/dev/gemma.litertlm',
      llm: llm,
      stt: fakeSttService(),
      tts: fakeTtsService(),
      chatModels: planner,
      embedder: EmbedderService(),
    );
    addTearDown(both.close);

    expect(await both.prepareAll(), isA<Ok<void>>());

    expect(llm.installs.single, '/store/custom/g3.litertlm');
    final chat = (both.states.value[ModelId.chat] as ModelReady).info.chat!;
    expect(chat.custom, isTrue);
    expect(chat.name, 'Gemma 3 1B NPU');
  });

  test('GEMMA_MODEL_PATH without a chosen model loads with Gemma 4 E2B\'s '
      'settings (a developer define)', () async {
    final define = ModelRepository(
      bundled: FakeBundledFiles(),
      detector: fakeDetectorService(),
      bundledDetector: fakeBundledDetector,
      gemmaModelPath: '/dev/gemma.litertlm',
      llm: llm,
      stt: fakeSttService(),
      tts: fakeTtsService(),
      chatModels: _Planner(const NoChatModelPlan()),
      embedder: EmbedderService(),
    );
    addTearDown(define.close);

    expect(await define.prepareAll(), isA<Ok<void>>());

    final chat = (define.states.value[ModelId.chat] as ModelReady).info.chat!;
    expect(chat.name, 'Gemma 4 E2B');
    expect(chat.custom, isFalse);
    expect(chat.source, 'GEMMA_MODEL_PATH=/dev/gemma.litertlm');
    expect(chat.requestedBackend, kLlmConfig.backend.name);
    expect(llm.installTypes.single, ModelType.gemma4);
    // Fixed towers, not read from the file: vision on, no audio; the engine
    // put the vision encoder on the CPU.
    expect(chat.visionBackend, 'cpu');
    expect(chat.audioBackend, isNull);
  });

  group('GEMMA_MODEL_PATH follows its file', () {
    late Directory tmp;
    setUp(() => tmp = Directory.systemTemp.createTempSync('define_file'));
    tearDown(() => tmp.deleteSync(recursive: true));

    /// The slot over a define naming a file of [kind], on a Qualcomm phone.
    Future<ModelRepository> launch(FakeBundle kind, String name) async {
      final path = writeFakeLitertlm('${tmp.path}/$name', kind).path;
      final define = ModelRepository(
        bundled: FakeBundledFiles(),
        detector: fakeDetectorService(),
        bundledDetector: fakeBundledDetector,
        gemmaModelPath: path,
        llm: llm,
        stt: fakeSttService(),
        tts: fakeTtsService(),
        chatModels: _Planner(const NoChatModelPlan()),
        embedder: EmbedderService(),
        readBundle: fakeReadBundle,
        abi: 'android_arm64',
      );
      addTearDown(define.close);
      await define.prepareAll();
      return define;
    }

    test(
      "a file not compiled for an NPU: exactly Gemma 4 E2B's settings "
      '(images on, no audio tower: the integration tests rely on them)',
      () async {
        await launch(FakeBundle.gemma4E2b, 'gemma-4-E2B-it.litertlm');

        expect(llm.models.single, same(kDefineChatModel));
      },
    );

    test("a build compiled for an NPU: the app's defaults for it (the NPU, "
        'the context its name gives, tools off, its towers), not the GPU the '
        'plugin would refuse', () async {
      final define = await launch(
        FakeBundle.officialNpu,
        'gemma-4-E2B-it_qualcomm_sm8750_ekv2048.litertlm',
      );

      final loaded = llm.models.single;
      expect(loaded.llm.backend, PreferredBackend.npu);
      expect(loaded.llm.maxTokens, 2048);
      expect(loaded.tools, isFalse);
      expect(loaded.llm.supportImage && loaded.llm.supportAudio, isTrue);
      final chat = (define.states.value[ModelId.chat] as ModelReady).info.chat!;
      expect(chat.requestedBackend, 'npu');
      expect(chat.custom, isFalse);
    });

    test('a build for an NPU this build does not run here: refused before '
        'any install, not retryable', () async {
      final define = await launch(
        FakeBundle.intelNpu,
        'e2b_intel_lnl.litertlm',
      );

      final failed = define.states.value[ModelId.chat]! as ModelFailed;
      expect(failed.retryable, isFalse);
      expect(
        failed.message,
        contains(
          'cannot run here: it is compiled for the IntelOpenVINO LNL NPU, '
          'which this build runs on windows_x64 only, and this is '
          'android_arm64.',
        ),
      );
      expect(llm.installs, isEmpty);
    });
  });

  test('a blocked custom model fails the slot with its reason; Gemma 4 E2B '
      'is not loaded instead', () async {
    planner.plan = const ChatPlanBlocked('g3.litertlm is gone');

    expect(await models.prepareAll(), isA<Error<void>>());

    expect(llm.installs, isEmpty);
    expect(
      models.states.value[ModelId.chat],
      isA<ModelFailed>().having(
        (s) => s.message,
        'message',
        'g3.litertlm is gone',
      ),
    );
  });

  test('an NPU the strict engine cannot reach fails the slot with the '
      "plugin's reason, naming the model", () async {
    llm.loadError = const ChatModelLoadException(
      modelName: 'Gemma 3 1B NPU',
      requested: PreferredBackend.npu,
      cause: NpuDispatchUnavailableException(
        'no NPU dispatch stack ships for macos',
      ),
    );

    expect(await models.prepareAll(), isA<Error<void>>());

    final failed = models.states.value[ModelId.chat] as ModelFailed;
    expect(
      failed.message,
      'Gemma 3 1B NPU did not load on npu: '
      'NpuDispatchUnavailableException: no NPU dispatch stack ships for macos',
    );
    expect(llm.models, hasLength(1), reason: 'one attempt, nothing else');
  });

  test(
    'reloadChatModel closes the loaded model and loads the new plan',
    () async {
      await models.prepareAll();
      planner.plan = _plan(_custom(backend: PreferredBackend.gpu));

      expect(await models.reloadChatModel(), isA<Ok<void>>());

      expect(llm.unloadCalls, 1);
      expect(llm.models.last.llm.backend, PreferredBackend.gpu);
      expect(
        (models.states.value[ModelId.chat] as ModelReady).info.backend,
        'gpu',
      );
    },
  );

  test(
    'after a failed load the reload with another backend continues setup',
    () async {
      llm.loadError = ChatModelLoadException(
        modelName: 'Gemma 3 1B NPU',
        requested: PreferredBackend.npu,
        cause: Exception('wrong SoC'),
      );
      expect(await models.prepareAll(), isA<Error<void>>());
      expect(
        models.states.value[ModelId.whisperBase],
        isNot(isA<ModelReady>()),
      );

      planner.plan = _plan(_custom(backend: PreferredBackend.cpu));
      expect(await models.reloadChatModel(), isA<Ok<void>>());

      expect(models.requiredReady, isTrue);
    },
  );

  test("a failure at any step is the slot's error as the plugin or the "
      'engine said it: nothing added, nothing retried', () async {
    planner.plan = _plan(_custom(), bundle: FakeBundle.npuWithTowers);
    final cause = Exception('Failed to create conversation');
    llm.warmUpError = cause;

    final error = (await models.prepareAll() as Error<void>).error;

    expect(error, same(cause));
    expect(
      (models.states.value[ModelId.chat] as ModelFailed).message,
      'Exception: Failed to create conversation',
    );
    expect(llm.models, hasLength(1), reason: 'no retry');
    expect(llm.unloadCalls, 1, reason: 'the failed model is released');
  });

  test('unloadChatModel frees the model and marks the slot pending', () async {
    await models.prepareAll();

    await models.unloadChatModel();

    expect(llm.isLoaded, isFalse);
    expect(models.states.value[ModelId.chat], isA<ModelPending>());
  });

  test('a chat model loaded with its audio tower warms it up too '
      '(kLlmAudioWarmUp), and the row says where the engine put each '
      'tower', () async {
    planner.plan = _plan(
      _custom(backend: PreferredBackend.gpu),
      bundle: FakeBundle.gemma4E2b,
    );
    expect(await models.prepareAll(), isA<Ok<void>>());
    expect(llm.models.single.llm.supportAudio, isTrue);
    expect(llm.warmUpsWithAudio.single, kLlmAudioWarmUp);
    expect(kLlmAudioWarmUp, isTrue);
    final chat = (models.states.value[ModelId.chat] as ModelReady).info.chat!;
    expect(chat.audio, isTrue);
    expect(chat.visionBackend, 'cpu');
    expect(chat.audioBackend, 'cpu');
    expect(chat.capabilityLine, contains('vision cpu · audio cpu'));
  });

  test('the row shows where the engine put each tower '
      '(activeVisionBackend / activeAudioBackend), nothing the app '
      'predicted; a file without towers has none', () async {
    // Encoders that declare their backends: the plugin runs vision on the
    // GPU and audio on the CPU.
    llm
      ..visionBackend = PreferredBackend.gpu
      ..audioBackend = PreferredBackend.cpu;
    planner.plan = _plan(
      _custom(backend: PreferredBackend.gpu),
      bundle: FakeBundle.constrainedTowers,
    );

    expect(await models.prepareAll(), isA<Ok<void>>());

    final loaded = llm.models.single.llm;
    expect(loaded.supportImage && loaded.supportAudio, isTrue);
    final chat = (models.states.value[ModelId.chat] as ModelReady).info.chat!;
    expect(chat.visionBackend, 'gpu');
    expect(chat.audioBackend, 'cpu');

    planner.plan = _plan(_custom(), bundle: FakeBundle.textOnlyNpu);
    expect(await models.reloadChatModel(), isA<Ok<void>>());
    final textOnly =
        (models.states.value[ModelId.chat] as ModelReady).info.chat!;
    expect(textOnly.visionBackend, isNull);
    expect(textOnly.audioBackend, isNull);
  });
}
