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
import 'package:litert_edge_demos/data/repositories/model_repository.dart';
import 'package:litert_edge_demos/data/services/detector/detector_service.dart';
import 'package:litert_edge_demos/data/services/knowledge/embedder_service.dart';
import 'package:litert_edge_demos/domain/models/chat_model.dart';
import 'package:litert_edge_demos/domain/models/chat_model_config.dart';
import 'package:litert_edge_demos/domain/ports/chat_model_planner.dart';
import 'package:litert_edge_demos/selftest/self_test_adapters.dart';

import '../fakes/fake_bundled_files.dart';
import '../fakes/fake_detector_runtime.dart';
import '../fakes/fake_llm_service.dart';
import '../fakes/fake_speech.dart';

/// The self-test loads exactly what the app would: for the same choice and
/// defines, the file and settings `ModelRepository` loads are the ones the
/// self-test resolves (both ask `ModelSourceResolver`).
void main() {
  late Directory tmp;
  late String defineFile;
  late String customFile;

  setUp(() {
    tmp = Directory.systemTemp.createTempSync('self_test_parity');
    String file(String name) =>
        (File('${tmp.path}/$name')..writeAsStringSync('x')).path;
    defineFile = file('define.litertlm');
    customFile = file('custom.litertlm');
  });
  tearDown(() => tmp.deleteSync(recursive: true));

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
      tools: true,
    ),
  );

  /// The file and settings the app's chat slot loads; nulls when none.
  Future<(String?, ChatModelConfig?)> appLoads(
    ChatModelPlan plan,
    String define,
  ) async {
    final llm = FakeLlmService()..followRequested = true;
    final models = ModelRepository(
      bundled: FakeBundledFiles(),
      detector: fakeDetectorService(),
      bundledDetector: fakeBundledDetector,
      gemmaModelPath: define,
      llm: llm,
      stt: fakeSttService(),
      tts: fakeTtsService(),
      chatModels: _Planner(plan),
      embedder: EmbedderService(),
    );
    addTearDown(models.close);
    await models.prepareAll();
    return (llm.installs.singleOrNull, llm.models.singleOrNull);
  }

  void expectSameSettings(ChatModelConfig selfTest, ChatModelConfig app) {
    expect(selfTest.name, app.name);
    expect(selfTest.modelType, app.modelType);
    expect(selfTest.tools, app.tools);
    expect(selfTest.llm.maxTokens, app.llm.maxTokens);
    expect(selfTest.llm.backend, app.llm.backend);
    expect(selfTest.llm.supportImage, app.llm.supportImage);
    expect(selfTest.llm.maxNumImages, app.llm.maxNumImages);
  }

  final cases = <String, (ChatModelPlan Function(), bool)>{
    'your own model': (custom, false),
    'your own model, used in place': (() => custom(inPlace: true), false),
    'your own model and GEMMA_MODEL_PATH': (custom, true),
    'GEMMA_MODEL_PATH, none chosen': (NoChatModelPlan.new, true),
  };
  for (final MapEntry(key: name, value: (plan, withDefine)) in cases.entries) {
    test('the chat model: $name', () async {
      final define = withDefine ? defineFile : '';
      final (appPath, appConfig) = await appLoads(plan(), define);

      final selfTest = await resolveSelfTestChatModel(
        argument: null,
        define: define,
        plan: plan(),
      );

      expect(appPath, isNotNull);
      expect(selfTest.file.path, appPath);
      expect(selfTest.file.problem, isNull);
      expectSameSettings(selfTest.config, appConfig!);
    });
  }

  for (final (name, plan) in [
    ('a blocked choice', const ChatPlanBlocked('the file changed')),
    ('none chosen, no define', const NoChatModelPlan()),
  ]) {
    test('the chat model: $name loads nothing in either', () async {
      final define = plan is ChatPlanBlocked ? defineFile : '';
      final (appPath, _) = await appLoads(plan, define);

      final selfTest = await resolveSelfTestChatModel(
        argument: null,
        define: define,
        plan: plan,
      );

      expect(appPath, isNull);
      expect(selfTest.file.path, isNull);
      expect(selfTest.file.problem, isNotNull);
    });
  }

  test('the detector: built in, in both', () async {
    final detector = InProcessDetectorService();
    final models = ModelRepository(
      bundled: FakeBundledFiles(),
      detector: detector,
      bundledDetector: fakeBundledDetector,
      gemmaModelPath: defineFile,
      llm: FakeLlmService(),
      stt: fakeSttService(),
      tts: fakeTtsService(),
      embedder: EmbedderService(),
    );
    addTearDown(models.close);
    await models.prepareAll();

    final selfTest = await resolveSelfTestDetector(argument: null);

    expect(detector.sources.single, isA<DetectorBytes>());
    expect(selfTest.bundled, isTrue);
  });
}

/// A planner fixed at [plan].
final class _Planner implements ChatModelPlanner {
  _Planner(this.plan);

  @override
  final ChatModelPlan plan;
}
