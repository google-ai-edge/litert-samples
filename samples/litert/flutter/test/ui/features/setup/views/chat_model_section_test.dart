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
import 'package:litert_edge_demos/data/services/knowledge/embedder_service.dart';
import 'package:litert_edge_demos/data/services/llm/llm_service.dart';
import 'package:litert_edge_demos/data/services/model_store/model_store.dart';
import 'package:litert_edge_demos/data/services/model_store/models_folder.dart';
import 'package:litert_edge_demos/data/services/settings/typed_settings.dart';
import 'package:litert_edge_demos/domain/models/chat_model.dart';
import 'package:litert_edge_demos/domain/models/npu_availability.dart';
import 'package:litert_edge_demos/domain/ports/model_file_picker.dart';
import 'package:litert_edge_demos/domain/use_cases/chat_model_switcher.dart';
import 'package:litert_edge_demos/ui/features/setup/view_models/chat_model_view_model.dart';
import 'package:litert_edge_demos/ui/features/setup/views/chat_model_section.dart';

import '../../../../fakes/fake_bundled_files.dart';
import '../../../../fakes/fake_conversation_repository.dart';
import '../../../../fakes/fake_detector_runtime.dart';
import '../../../../fakes/fake_llm_service.dart';
import '../../../../fakes/fake_model_file_picker.dart';
import '../../../../fakes/fake_settings_store.dart';
import '../../../../fakes/fake_speech.dart';
import '../../../../support/test_bytes.dart';

/// Alternates real time (file I/O) and frames (continuations in the test's
/// fake zone) while [running] holds: bounded by real time, generously, not
/// by a count of 5 ms steps (a loaded machine is slower, never different).
/// The frames advance no fake time.
Future<void> whileRunning(WidgetTester tester, bool Function() running) async {
  final watch = Stopwatch()..start();
  while (running() && watch.elapsed < const Duration(seconds: 15)) {
    await tester.runAsync(
      () => Future<void>.delayed(const Duration(milliseconds: 5)),
    );
    await tester.pump();
  }
}

void main() {
  late Directory root;
  late ModelStore store;
  late FakeLlmService llm;
  late FakeConversationRepository conversation;
  late ModelRepository models;
  late ChatModelRepository chatModels;
  late ChatModelViewModel vm;

  const fileName = 'gemma3_ekv1280_sm8750.litertlm';

  Future<void> pumpSection(
    WidgetTester tester, {
    NpuAvailability npu = const NpuUnavailable(
      'no NPU dispatch stack ships for macos',
    ),
    ImportSupport support = const ImportFromFolder(),
    bool withFile = false,
  }) async {
    root = Directory.systemTemp.createTempSync('chat_section');
    llm = FakeLlmService()..followRequested = true;
    conversation = FakeConversationRepository();
    store = ModelStore(root: () async => root);
    chatModels = ChatModelRepository(
      settings: TypedSettings(store: InMemorySettingsStore()),
      store: store,
      npu: () => npu,
      folders: [
        ModelsFolder(
          directory: () async => Directory('${root.path}/models_folder'),
        ),
      ],
    );
    models = ModelRepository(
      bundled: FakeBundledFiles(),
      detector: fakeDetectorService(),
      bundledDetector: fakeBundledDetector,
      gemmaModelPath: '',
      llm: llm,
      stt: fakeSttService(),
      tts: fakeTtsService(),
      chatModels: chatModels,
      embedder: EmbedderService(),
    );
    await tester.runAsync(() async {
      await chatModels.load();
      if (withFile) {
        final source = File('${root.path}/$fileName')
          ..writeAsBytesSync(testBytes(4096));
        await chatModels.importFile(source.path);
      }
      await models.prepareAll();
    });
    vm = ChatModelViewModel(
      chatModels: chatModels,
      models: models,
      switcher: ChatModelSwitcher(
        conversation: conversation,
        reloadChatModel: models.reloadChatModel,
        unloadChatModel: models.unloadChatModel,
        refuseChatModelLoads: models.refuseChatModelLoads,
      ),
      picker: FakeModelFilePicker(support: support),
    );
    // The models folder is listed with real I/O.
    await whileRunning(tester, () => vm.rescan.running);
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: ListView(children: [ChatModelSection(viewModel: vm)]),
        ),
      ),
    );
  }

  Future<void> tearDownSection(WidgetTester tester) async {
    await tester.pumpWidget(const SizedBox.shrink());
    vm.dispose();
    await tester.runAsync(() async {
      await models.close();
      chatModels.dispose();
      await store.close();
      await conversation.close();
    });
    root.deleteSync(recursive: true);
  }

  testWidgets('no chat model yet (the app ships none): the card leads with '
      'the models folder, Path…, Import and Download from URL', (tester) async {
    await pumpSection(tester);

    expect(
      tester.widget<Text>(find.byKey(ChatModelKeys.activeLine)).data,
      startsWith('No chat model yet'),
    );
    expect(find.textContaining('The app ships no chat model'), findsOneWidget);
    expect(find.byKey(ChatModelKeys.folderPath), findsOneWidget);
    expect(find.byKey(ChatModelKeys.pathButton), findsOneWidget);
    expect(find.byKey(ChatModelKeys.importFile), findsOneWidget);
    expect(find.byKey(ChatModelKeys.downloadUrl), findsOneWidget);
    expect(find.text('No file yet.'), findsOneWidget);
    await tearDownSection(tester);
  });

  testWidgets(
    'with a file: the editor; NPU is disabled with flutter_edge_ai\'s '
    'reason on this host',
    (tester) async {
      await pumpSection(tester, withFile: true);

      expect(
        tester.widget<Text>(find.byKey(ChatModelKeys.fileLine)).data,
        contains(fileName),
      );
      final backend = tester.widget<SegmentedButton<PreferredBackend>>(
        find.byKey(ChatModelKeys.backend),
      );
      final npu = backend.segments.firstWhere(
        (s) => s.value == PreferredBackend.npu,
      );
      expect(npu.enabled, isFalse);
      expect(backend.selected, {PreferredBackend.gpu});
      expect(
        tester.widget<Text>(find.byKey(ChatModelKeys.npuReason)).data,
        contains('no NPU dispatch stack ships for macos'),
      );
      expect(find.byKey(ChatModelKeys.apply), findsOneWidget);
      expect(find.text('Use this model'), findsOneWidget);
      await tearDownSection(tester);
    },
  );

  testWidgets('the URL dialog refuses a bad link and says why', (tester) async {
    await pumpSection(tester);

    await tester.tap(find.byKey(ChatModelKeys.downloadUrl));
    await tester.pumpAndSettle();
    await tester.enterText(find.byKey(ChatModelKeys.urlField), 'not a link');
    await tester.tap(find.byKey(ChatModelKeys.urlSubmit));
    await tester.pump();

    expect(
      tester.widget<Text>(find.byKey(ChatModelKeys.urlError)).data,
      contains('https://'),
    );
    expect(find.byKey(ChatModelKeys.urlField), findsOneWidget, reason: 'open');
    await tearDownSection(tester);
  });

  testWidgets('a failed load: the reason and Run on GPU / Run on CPU; Run on '
      'CPU reloads on the CPU', (tester) async {
    await pumpSection(
      tester,
      npu: const NpuAvailable(soc: 'QTI SM8750'),
      withFile: true,
    );
    llm.loadError = ChatModelLoadException(
      modelName: 'gemma3_ekv1280_sm8750',
      requested: PreferredBackend.npu,
      cause: Exception('Failed to create engine'),
    );

    await tester.runAsync(() => vm.apply.execute());
    await tester.pump();

    expect(
      tester.widget<Text>(find.byKey(ChatModelKeys.loadError)).data,
      contains('did not load on npu'),
    );
    for (final action in ChatModelAction.values) {
      expect(find.byKey(ChatModelKeys.action(action)), findsOneWidget);
    }
    expect(find.textContaining('Gemma 4 E2B'), findsNothing);

    await tester.tap(
      find.byKey(ChatModelKeys.action(ChatModelAction.runOnCpu)),
    );
    // The tapped command runs in the test's fake zone (its continuations
    // need pumps) and its store work in real time: alternate both.
    await whileRunning(tester, () => vm.runOn.running || vm.busy);
    expect(vm.busy, isFalse, reason: 'the reload finished');

    expect(llm.models.last.llm.backend, PreferredBackend.cpu);
    expect(find.byKey(ChatModelKeys.loadError), findsNothing);
    expect(chatModels.state.value.kind, ChatModelKind.custom);
    await tearDownSection(tester);
  });

  testWidgets('Android: no Import button, the reason and the URL way instead', (
    tester,
  ) async {
    await pumpSection(
      tester,
      support: const ImportUnsupported('not on Android'),
    );

    expect(find.byKey(ChatModelKeys.importFile), findsNothing);
    expect(
      tester.widget<Text>(find.byKey(ChatModelKeys.importUnavailable)).data,
      contains('Download from URL'),
    );
    expect(find.byKey(ChatModelKeys.downloadUrl), findsOneWidget);
    await tearDownSection(tester);
  });
}
