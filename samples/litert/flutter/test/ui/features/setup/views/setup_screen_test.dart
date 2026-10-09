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

import 'package:flutter/material.dart';
import 'package:flutter_edge_ai/flutter_edge_ai.dart' show PreferredBackend;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/repositories/chat_model_repository.dart';
import 'package:litert_edge_demos/data/repositories/model_repository.dart';
import 'package:litert_edge_demos/data/repositories/provisioning_repository.dart';
import 'package:litert_edge_demos/data/services/detector/detector_service.dart';
import 'package:litert_edge_demos/data/services/knowledge/embedder_service.dart';
import 'package:litert_edge_demos/data/services/model_store/bundled_model_files.dart';
import 'package:litert_edge_demos/data/services/model_store/model_store.dart';
import 'package:litert_edge_demos/data/services/settings/typed_settings.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/npu_availability.dart';
import 'package:litert_edge_demos/domain/use_cases/chat_model_switcher.dart';
import 'package:litert_edge_demos/ui/core/warning_color.dart';
import 'package:litert_edge_demos/ui/features/setup/view_models/chat_model_view_model.dart';
import 'package:litert_edge_demos/ui/features/setup/view_models/self_test_view_model.dart';
import 'package:litert_edge_demos/ui/features/setup/view_models/setup_view_model.dart';
import 'package:litert_edge_demos/ui/features/setup/views/setup_screen.dart';
import 'package:provider/provider.dart';

import '../../../../fakes/fake_bundled_files.dart';
import '../../../../fakes/fake_conversation_repository.dart';
import '../../../../fakes/fake_detector_runtime.dart';
import '../../../../fakes/fake_llm_service.dart';
import '../../../../fakes/fake_model_file_picker.dart';
import '../../../../fakes/fake_model_files.dart';
import '../../../../fakes/fake_provisioning_repository.dart';
import '../../../../fakes/fake_self_test_launcher.dart';
import '../../../../fakes/fake_settings_store.dart';
import '../../../../fakes/fake_speech.dart';

void main() {
  Future<void> pumpScreen(
    WidgetTester tester, {
    required ModelRepository models,
    required ProvisioningRepository provisioning,
    SetupMode mode = SetupMode.firstRun,
    VoidCallback? onReady,
  }) async {
    // Every row on screen (the list builds lazily).
    tester.view.physicalSize = const Size(1200, 4000);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.reset);
    await tester.pumpWidget(const SizedBox.shrink()); // a fresh view model
    final filePicker = FakeModelFilePicker();
    // The Chat model card over fakes: nothing saved, no models folder, no
    // NPU; its store is never written.
    final chatStore = ModelStore(root: () async => Directory.systemTemp);
    final chatModels = ChatModelRepository(
      settings: TypedSettings(store: InMemorySettingsStore()),
      store: chatStore,
      npu: () => const NpuUnavailable('no NPU in a widget test'),
      folders: const [],
    );
    final conversation = FakeConversationRepository();
    addTearDown(() async {
      chatModels.dispose();
      await chatStore.close();
      await conversation.close();
    });
    final switcher = ChatModelSwitcher(
      conversation: conversation,
      reloadChatModel: models.reloadChatModel,
      unloadChatModel: models.unloadChatModel,
      refuseChatModelLoads: models.refuseChatModelLoads,
    );
    await tester.pumpWidget(
      MaterialApp(
        home: MultiProvider(
          providers: [
            ChangeNotifierProvider(
              create: (_) => SetupViewModel(
                models: models,
                prepareModels: models.prepareAll,
                provisioning: provisioning,
                mode: mode,
              ),
            ),
            ChangeNotifierProvider(
              create: (_) => ChatModelViewModel(
                chatModels: chatModels,
                models: models,
                switcher: switcher,
                picker: filePicker,
              ),
            ),
            ChangeNotifierProvider(
              create: (_) =>
                  SelfTestViewModel(launcher: FakeSelfTestLauncher()),
            ),
          ],
          child: SetupScreen(onReady: onReady),
        ),
      ),
    );
    await tester.pump(); // start (after the first build)
    await tester.pump(); // setup
  }

  ModelRepository fakeModels({FakeLlmService? llm}) => ModelRepository(
    gemmaModelPath: kTestChatModelPath,
    bundled: FakeBundledFiles(),
    detector: fakeDetectorService(),
    bundledDetector: fakeBundledDetector,
    llm: llm ?? FakeLlmService(),
    stt: fakeSttService(),
    tts: fakeTtsService(),
    embedder: EmbedderService(),
  );

  testWidgets('a non-GPU backend blocks setup with the error and a Retry', (
    tester,
  ) async {
    final llm = FakeLlmService()..activeBackend = PreferredBackend.cpu;
    final models = fakeModels(llm: llm);
    var readyCalls = 0;

    await pumpScreen(
      tester,
      models: models,
      provisioning: FakeProvisioningRepository(),
      onReady: () => readyCalls++,
    );

    expect(find.text('Failed'), findsOneWidget);
    // On the chat model's row (the Chat model card above says it too).
    expect(
      find.descendant(
        of: find.byKey(SetupKeys.row(ModelId.chat)),
        matching: find.textContaining('Fallback is disabled'),
      ),
      findsOneWidget,
    );
    expect(find.byKey(SetupKeys.retry), findsOneWidget);
    expect(readyCalls, 0, reason: 'never moves on to the chat on a CPU load');

    llm.activeBackend = PreferredBackend.gpu;
    await tester.tap(find.byKey(SetupKeys.retry));
    await tester.pump(); // prepare runs
    await tester.pump(); // post-frame hand-over

    expect(find.textContaining('Ready · gpu'), findsOneWidget);
    expect(readyCalls, 1);

    await tester.pumpWidget(const SizedBox.shrink());
    await models.close();
  });

  testWidgets('a detector on the explicit CPU is Ready in amber with its '
      'label; GEMMA_MODEL_PATH shows as set by the build', (tester) async {
    final models = ModelRepository(
      gemmaModelPath: kTestChatModelPath,
      bundled: FakeBundledFiles(),
      bundledDetector: fakeBundledDetector,
      llm: FakeLlmService(),
      stt: fakeSttService(),
      tts: fakeTtsService(),
      detector: DetectorService(
        runtime: const FakeDetectorRuntime(fullyAccelerated: false),
        expectedModelBytes: kFakeBundledDetectorBytes,
      ),
      detectorBackend: 'cpu',
      embedder: EmbedderService(),
    );
    await tester.runAsync(
      models.prepareAll,
    ); // the worker isolate needs real time

    await pumpScreen(
      tester,
      models: models,
      provisioning: FakeProvisioningRepository(),
      onReady: () {},
    );
    await tester.pump();

    final label = find.textContaining('Ready · CPU (chosen)');
    expect(label, findsOneWidget);
    expect(tester.widget<Text>(label).style?.color, kWarningColor);
    expect(
      tester.widget<Text>(find.textContaining('Ready · gpu')).style?.color,
      isNot(kWarningColor),
    );
    expect(
      find.text('Set by the build: GEMMA_MODEL_PATH=/store/chat/model'),
      findsOneWidget,
    );

    await tester.pumpWidget(const SizedBox.shrink());
    await tester.runAsync(models.close);
  });

  testWidgets('every model built in: each row with its size and where it '
      'comes from; no manifest, download or import', (tester) async {
    final models = fakeModels();
    await pumpScreen(
      tester,
      models: models,
      provisioning: FakeProvisioningRepository(),
      mode: SetupMode.manage,
    );

    expect(find.text('Models'), findsOneWidget);
    final whisperRow = find.byKey(SetupKeys.row(ModelId.whisperBase));
    expect(
      find.descendant(
        of: whisperRow,
        matching: find.text(
          SetupViewModel.sizeLabel(bundledModelBytes(ModelId.whisperBase)),
        ),
      ),
      findsOneWidget,
    );
    expect(
      find.descendant(
        of: whisperRow,
        matching: find.text('Part of the app: nothing to download'),
      ),
      findsOneWidget,
    );
    expect(find.text('Download all'), findsNothing);
    expect(find.textContaining('Manifest'), findsNothing);
    expect(find.text('Import from folder…'), findsNothing);
    expect(find.text('Download'), findsNothing);

    await tester.pumpWidget(const SizedBox.shrink());
    await models.close();
  });
}
