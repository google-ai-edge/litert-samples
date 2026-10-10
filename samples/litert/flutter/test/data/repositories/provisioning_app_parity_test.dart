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
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/repositories/model_repository.dart';
import 'package:litert_edge_demos/data/repositories/provisioning_repository.dart';
import 'package:litert_edge_demos/data/services/detector/detector_service.dart';
import 'package:litert_edge_demos/data/services/knowledge/embedder_service.dart';
import 'package:litert_edge_demos/data/services/model_store/bundled_model_files.dart';
import 'package:litert_edge_demos/data/services/model_store/model_store.dart';
import 'package:litert_edge_demos/domain/models/chat_model.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/domain/models/provisioning.dart';
import 'package:litert_edge_demos/domain/ports/chat_model_planner.dart';

import '../../fakes/fake_bundled_files.dart';
import '../../fakes/fake_detector_runtime.dart';
import '../../fakes/fake_knowledge.dart';
import '../../fakes/fake_llm_service.dart';
import '../../fakes/fake_speech.dart';

/// The Models screen says present exactly what setup loads: for the same
/// choice and define, `ProvisioningRepository.presenceOf` (which gates
/// `requiredPresent`) names the source `ModelRepository` loads (both ask
/// `ModelSourceResolver`).
void main() {
  late Directory tmp;
  late String defineFile;
  late String customFile;
  late Map<String, String> bundledEmbedder;

  setUp(() {
    tmp = Directory.systemTemp.createTempSync('provisioning_parity');
    String file(String name) =>
        (File('${tmp.path}/$name')..writeAsStringSync('x')).path;
    defineFile = file('define.litertlm');
    customFile = file('custom.litertlm');
    bundledEmbedder = {
      for (final f in [kBundledEmbedderModel, kBundledEmbedderTokenizer])
        f.asset: file('bundled_${f.name}'),
    };
  });
  tearDown(() => tmp.deleteSync(recursive: true));

  ProvisioningRepository provisioning(
    String gemmaModelPath,
    ChatModelPlanner? planner,
  ) {
    final store = ModelStore(root: () async => tmp);
    addTearDown(store.close);
    return ProvisioningRepository(
      store: store,
      gemmaModelPath: gemmaModelPath,
      chatModels: planner,
    );
  }

  CustomChatPlan custom({bool inPlace = false}) => CustomChatPlan(
    path: customFile,
    model: CustomChatModel(
      displayName: 'My Gemma 3',
      source: inPlace
          ? LocalModelSource(customFile)
          : ImportedModelSource(customFile),
      modelType: ModelType.gemmaIt,
      backend: PreferredBackend.cpu,
      maxTokens: 2048,
    ),
  );

  group('the chat model', () {
    // name: (the plan, null = no planner; GEMMA_MODEL_PATH set; the
    // presence the Models screen must show).
    final cases = <String, (ChatModelPlan? Function(), bool, Type)>{
      'your own model': (custom, false, PresentAsCustomChatModel),
      'your own model, used in place': (
        () => custom(inPlace: true),
        false,
        PresentAsCustomChatModel,
      ),
      'your own model and GEMMA_MODEL_PATH': (
        custom,
        true,
        PresentAsCustomChatModel,
      ),
      'GEMMA_MODEL_PATH, none chosen': (
        () => const NoChatModelPlan(note: 'retired'),
        true,
        PresentByDefine,
      ),
      'GEMMA_MODEL_PATH, no planner': (() => null, true, PresentByDefine),
      'none chosen, no define': (
        () => const NoChatModelPlan(note: 'retired'),
        false,
        ChatModelNotChosen,
      ),
      'no planner, no define': (() => null, false, ChatModelNotChosen),
      'a blocked choice': (
        () => const ChatPlanBlocked('the file changed'),
        false,
        CustomChatModelBlocked,
      ),
      'a blocked choice and GEMMA_MODEL_PATH': (
        () => const ChatPlanBlocked('the file changed'),
        true,
        CustomChatModelBlocked,
      ),
    };

    for (final MapEntry(key: name, value: (plan, withDefine, shown))
        in cases.entries) {
      test(name, () async {
        final define = withDefine ? defineFile : '';
        final llm = FakeLlmService()..followRequested = true;
        final planned = plan();
        final planner = planned == null ? null : _Planner(planned);
        final models = ModelRepository(
          bundled: FakeBundledFiles(),
          detector: fakeDetectorService(),
          bundledDetector: fakeBundledDetector,
          gemmaModelPath: define,
          llm: llm,
          stt: fakeSttService(),
          tts: fakeTtsService(),
          chatModels: planner,
          embedder: EmbedderService(),
        );
        addTearDown(models.close);
        await models.prepareAll();
        final appPath = llm.installs.singleOrNull;
        final slot = models.states.value[ModelId.chat];

        final repo = provisioning(define, planner);
        final presence = repo.presenceOf(ModelId.chat);

        expect(presence.runtimeType, shown);
        switch (presence) {
          case PresentByDefine(:final define, :final value):
            expect(define, 'GEMMA_MODEL_PATH');
            expect(appPath, value);
            expect(llm.models.single.name, kDefineChatModel.name);
          case PresentAsCustomChatModel(:final name):
            expect(appPath, (planned! as CustomChatPlan).path);
            expect(llm.models.single.name, name);
          case CustomChatModelBlocked(:final reason):
            expect(appPath, isNull);
            expect(
              slot,
              isA<ModelFailed>().having((s) => s.message, 'message', reason),
            );
          case ChatModelNotChosen(:final note):
            expect(appPath, isNull);
            expect(slot, isA<ModelUnavailable>());
            expect(note, (planned as NoChatModelPlan?)?.note);
          case PresentBundled():
            fail('the chat model is never built in');
        }
        // Setup may start exactly when the app would try (or show why not).
        expect(
          ProvisioningRepository.isPresent(presence),
          slot is! ModelUnavailable,
        );
        expect(
          repo.requiredPresent,
          ProvisioningRepository.isPresent(presence),
          reason: 'every other required model is built in',
        );
      });
    }
  });

  test('the built-in models: shown built in and loaded from the app, '
      'whatever GEMMA_MODEL_PATH says', () async {
    final detector = InProcessDetectorService();
    final embedderSources = <EmbedderSource>[];
    final models = ModelRepository(
      bundled: FakeBundledFiles.at(bundledEmbedder),
      detector: detector,
      bundledDetector: fakeBundledDetector,
      gemmaModelPath: defineFile,
      embedder: EmbedderService(
        install: (config, source, onProgress) async {
          embedderSources.add(source);
          return 'eg';
        },
        load: (config) async => FakeEmbeddingModel(),
      ),
      llm: FakeLlmService()..followRequested = true,
      stt: fakeSttService(),
      tts: fakeTtsService(),
    );
    addTearDown(models.close);
    await models.prepareAll();
    final repo = provisioning(defineFile, null);

    expect(repo.presenceOf(ModelId.yolo26n), isA<PresentBundled>());
    expect(detector.sources.single, isA<DetectorBytes>());
    expect(repo.presenceOf(ModelId.embeddingGemma), isA<PresentBundled>());
    expect(embedderSources.single, isA<EmbedderFromFiles>());
    for (final id in [
      ModelId.whisperBase,
      ModelId.moonshineTiny,
      ModelId.inflectNano,
    ]) {
      expect(repo.presenceOf(id), isA<PresentBundled>(), reason: '$id');
    }
    expect(repo.requiredPresent, isTrue);
  });
}

/// A planner fixed at [plan].
final class _Planner implements ChatModelPlanner {
  _Planner(this.plan);

  @override
  final ChatModelPlan plan;
}
