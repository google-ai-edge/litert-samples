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
import 'package:litert_edge_demos/selftest/self_test_adapters.dart'
    show resolveSelfTestChatModel;
import 'package:litert_edge_demos/ui/features/setup/view_models/chat_model_view_model.dart';
import 'package:litert_edge_demos/ui/features/setup/views/chat_model_section.dart';

import '../../../../fakes/fake_bundled_files.dart';
import '../../../../fakes/fake_conversation_repository.dart';
import '../../../../fakes/fake_detector_runtime.dart';
import '../../../../fakes/fake_llm_service.dart';
import '../../../../fakes/fake_model_file_picker.dart';
import '../../../../fakes/fake_settings_store.dart';
import '../../../../fakes/fake_speech.dart';
import '../../../../support/fake_bundles.dart';

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
  late InMemorySettingsStore prefs;
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
    FakeBundle? inPlace,
  }) async {
    root = Directory.systemTemp.createTempSync('chat_section');
    llm = FakeLlmService();
    conversation = FakeConversationRepository();
    store = ModelStore(root: () async => root);
    prefs = InMemorySettingsStore();
    chatModels = ChatModelRepository(
      settings: TypedSettings(store: prefs),
      store: store,
      npu: () => npu,
      readBundle: fakeReadBundle,
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
        // A text-only model for the GPU and the CPU, imported.
        final source = writeFakeLitertlm(
          '${root.path}/$fileName',
          FakeBundle.textOnly,
          filler: 4096,
        );
        await chatModels.importFile(source.path);
      }
      if (inPlace != null) {
        // A file of this kind, used in place.
        final file = writeFakeLitertlm(
          '${root.path}/in_place.litertlm',
          inPlace,
          filler: 4096,
        );
        await chatModels.useLocalFile(file.path);
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
    "with a file: the editor; NPU is disabled with the app's NPU check's "
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

  /// Every text in the row [key], in order: `Vision | in the file · runs on
  /// the CPU`.
  String row(WidgetTester tester, Key key) => tester
      .widgetList<Text>(
        find.descendant(of: find.byKey(key), matching: find.byType(Text)),
      )
      .map((t) => t.data)
      .join(' | ');

  SegmentedButton<PreferredBackend> control(WidgetTester tester, Key key) =>
      tester.widget<SegmentedButton<PreferredBackend>>(find.byKey(key));

  const sm8750 = NpuAvailable(soc: 'QTI SM8750');
  const notLoaded = 'in the file (where it runs shows once it is loaded)';

  testWidgets('no Images, Audio or Encoders controls: a read-only Towers '
      'block under Context, before Tools, with nothing to choose; an imported '
      'text-only model: neither tower in the file', (tester) async {
    await pumpSection(tester, withFile: true);

    expect(find.byKey(ChatModelKeys.towers), findsOneWidget);
    expect(find.byType(SwitchListTile), findsOneWidget, reason: 'Tools only');
    expect(find.byKey(ChatModelKeys.tools), findsOneWidget);
    expect(find.text('Images'), findsNothing);
    expect(find.text('Encoders'), findsNothing);
    expect(
      find.descendant(
        of: find.byKey(ChatModelKeys.towers),
        matching: find.byType(SegmentedButton<PreferredBackend>),
      ),
      findsNothing,
      reason: 'no CPU | NPU tower control',
    );
    final contextY = tester.getTopLeft(find.byKey(ChatModelKeys.context)).dy;
    final towersY = tester.getTopLeft(find.byKey(ChatModelKeys.towers)).dy;
    final toolsY = tester.getTopLeft(find.byKey(ChatModelKeys.tools)).dy;
    expect(contextY < towersY && towersY < toolsY, isTrue);

    expect(find.byKey(ChatModelKeys.towerText), findsNothing);
    expect(
      row(tester, ChatModelKeys.towerVision),
      'Vision | not in this file: photos in Voice chat and detailed answers '
      'in Live camera are off',
    );
    expect(
      row(tester, ChatModelKeys.towerAudio),
      'Audio | not in this file: voice questions are transcribed first',
    );
    await tearDownSection(tester);
  });

  testWidgets('an all-NPU build, with towers: the NPU, the text model line; '
      'each tower in the file, then where the engine put it once it is '
      'loaded (the NPU: its encoders are compiled for it)', (tester) async {
    await pumpSection(tester, npu: sm8750, inPlace: FakeBundle.npuWithTowers);

    expect(vm.draftBackend, PreferredBackend.npu);
    expect(
      tester.widget<Text>(find.byKey(ChatModelKeys.towerText)).data,
      'Text model: compiled for the Qualcomm SM8750 NPU',
    );
    expect(row(tester, ChatModelKeys.towerVision), 'Vision | $notLoaded');
    expect(row(tester, ChatModelKeys.towerAudio), 'Audio | $notLoaded');

    llm
      ..visionBackend = PreferredBackend.npu
      ..audioBackend = PreferredBackend.npu;
    await tester.runAsync(() => vm.apply.execute());
    await tester.pump();

    expect(llm.models.last.llm.supportImage, isTrue);
    expect(
      row(tester, ChatModelKeys.towerVision),
      'Vision | in the file · runs on the NPU',
    );
    expect(
      row(tester, ChatModelKeys.towerAudio),
      'Audio | in the file · runs on the NPU',
    );
    await tearDownSection(tester);
  });

  testWidgets('the standard Gemma 4 E2B on the GPU: both towers in the file, '
      'on the CPU once loaded', (tester) async {
    await pumpSection(tester, inPlace: FakeBundle.gemma4E2b);

    expect(vm.draftBackend, PreferredBackend.gpu);
    expect(find.byKey(ChatModelKeys.towerText), findsNothing);
    expect(row(tester, ChatModelKeys.towerVision), 'Vision | $notLoaded');

    await tester.runAsync(() => vm.apply.execute());
    await tester.pump();

    expect(
      row(tester, ChatModelKeys.towerVision),
      'Vision | in the file · runs on the CPU',
    );
    expect(
      row(tester, ChatModelKeys.towerAudio),
      'Audio | in the file · runs on the CPU',
    );
    await tearDownSection(tester);
  });

  testWidgets('a build compiled for the NPU on a host without one: the NPU '
      'is selected but off, and the card says why it cannot be applied', (
    tester,
  ) async {
    await pumpSection(tester, inPlace: FakeBundle.textOnlyNpu);

    final backend = control(tester, ChatModelKeys.backend);
    expect(backend.selected, {PreferredBackend.npu});
    expect(
      backend.segments
          .firstWhere((s) => s.value == PreferredBackend.npu)
          .enabled,
      isFalse,
    );
    // It runs only on the NPU: GPU and CPU are off too, in the app's own
    // words for the plugin's refusal.
    for (final segment in backend.segments.where(
      (s) => s.value != PreferredBackend.npu,
    )) {
      expect(segment.enabled, isFalse, reason: '${segment.value}');
      expect(
        segment.tooltip,
        'GPU and CPU: not for this file (compiled for the Qualcomm SM8750 '
        'NPU: runs only on the NPU).',
      );
    }
    expect(
      tester.widget<Text>(find.byKey(ChatModelKeys.gpuCpuReason)).data,
      'GPU and CPU: not for this file (compiled for the Qualcomm SM8750 NPU: '
      'runs only on the NPU).',
    );
    expect(
      tester.widget<Text>(find.byKey(ChatModelKeys.draftProblem)).data,
      'The NPU is not available here: no NPU dispatch stack ships for macos.',
    );
    expect(
      tester.widget<FilledButton>(find.byKey(ChatModelKeys.apply)).onPressed,
      isNull,
    );
    await tearDownSection(tester);
  });

  testWidgets("a failed load shows the plugin's reason as it is: a "
      'BackendInitException with LiteRT-LM\'s own message; no tower line, and '
      'for a build compiled for the NPU no Run on GPU / CPU (it runs nowhere '
      'else)', (tester) async {
    await pumpSection(tester, npu: sm8750, inPlace: FakeBundle.npuWithTowers);
    // Another SoC than the bundle's: the plugin names the stamped one in the
    // engine's message, beside LiteRT-LM's own reason (a string LiteRT's
    // Qualcomm dispatch library carries).
    llm.loadError = engineCreateFailure(
      modelName: 'in_place',
      backend: PreferredBackend.npu,
      nativeMessage: 'Failed to retrieve graph',
      modelPath: '/models/in_place.litertlm',
      compiledFor: kTestNpuTarget,
    );

    await tester.runAsync(() => vm.apply.execute());
    await tester.pump();

    expect(
      tester.widget<Text>(find.byKey(ChatModelKeys.loadError)).data,
      'in_place did not load on npu: npu: Failed to create engine on npu: '
      'Failed to retrieve graph. Model: /models/in_place.litertlm. The bundle '
      'is compiled for the Qualcomm SM8750 NPU; if this device is not a '
      'Qualcomm SM8750, that is the cause.',
    );
    expect(llm.models, hasLength(1), reason: 'no retry, no fallback');
    expect([
      for (final action in ChatModelAction.values)
        if (find.byKey(ChatModelKeys.action(action)).evaluate().isNotEmpty)
          action,
    ], isEmpty);
    await tearDownSection(tester);
  });

  testWidgets('a generic file: GPU and CPU are offered, no reason line', (
    tester,
  ) async {
    await pumpSection(tester, inPlace: FakeBundle.gemma4E2b);

    final backend = control(tester, ChatModelKeys.backend);
    expect(
      backend.segments
          .where((s) => s.value != PreferredBackend.npu)
          .every((s) => s.enabled && s.tooltip == null),
      isTrue,
    );
    expect(find.byKey(ChatModelKeys.gpuCpuReason), findsNothing);
    await tearDownSection(tester);
  });

  testWidgets("a file whose header the plugin cannot read: the Towers block "
      "says so with the plugin's message, no tower rows; Apply is on", (
    tester,
  ) async {
    await pumpSection(tester);
    final path = '${root.path}/notes.litertlm';
    await tester.runAsync(() async {
      File(path).writeAsStringSync('hello');
      await vm.useLocal.execute(path);
    });
    await tester.pump();

    expect(
      tester.widget<Text>(find.byKey(ChatModelKeys.headerNotRead)).data,
      'The plugin could not read its header (notLitertlm): $path is not a '
      '.litertlm container (no LITERTLM magic). The plugin loads such a file '
      'without reading its header; the app asks for no tower (none is known), '
      'so it loads text only.',
    );
    expect(find.byKey(ChatModelKeys.towerVision), findsNothing);
    expect(find.byKey(ChatModelKeys.towerAudio), findsNothing);
    expect(
      tester.widget<FilledButton>(find.byKey(ChatModelKeys.apply)).onPressed,
      isNotNull,
    );
    await tearDownSection(tester);
  });

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

  testWidgets('a file picked and not applied: the card shows it (its towers, '
      'its backend from its own file) and says it is not applied; the '
      'reload, the self-test and a restart use the applied file; Apply saves '
      'and loads it', (tester) async {
    await pumpSection(tester, npu: sm8750, inPlace: FakeBundle.gemma4E2b);
    final gemmaPath = '${root.path}/in_place.litertlm';
    await tester.runAsync(() => vm.apply.execute());
    await tester.pump();
    expect(llm.installs.last, gemmaPath);
    expect(llm.models.last.llm.backend, PreferredBackend.gpu);
    // Apply started the saved file's background SHA-256, which saves its
    // hash: done before the settings are compared.
    await whileRunning(
      tester,
      () => chatModels.state.value.custom?.file?.sha256 == null,
    );
    expect(chatModels.state.value.custom?.file?.sha256, isNotNull);
    final savedBefore = Map.of(prefs.values);

    // An NPU-only build, picked just to see its towers.
    final npuOnly = File('${root.path}/npu_only.litertlm')
      ..writeAsBytesSync(
        fakeLitertlmBytes(FakeBundle.textOnlyNpu, filler: 4096, seed: 5),
      );
    await tester.runAsync(() => vm.useLocal.execute(npuOnly.path));
    await tester.pump();

    expect(
      tester.widget<Text>(find.byKey(ChatModelKeys.candidate)).data,
      'Not applied yet: press "Apply and reload" to use this file. Until '
      'then the chat model stays in_place.',
    );
    expect(
      tester.widget<Text>(find.byKey(ChatModelKeys.fileLine)).data,
      contains('in place: ${npuOnly.path}'),
    );
    expect(
      tester.widget<Text>(find.byKey(ChatModelKeys.towerText)).data,
      'Text model: compiled for the Qualcomm SM8750 NPU',
    );
    expect(
      row(tester, ChatModelKeys.towerVision),
      'Vision | not in this file: photos in Voice chat and detailed answers '
      'in Live camera are off',
    );
    expect(vm.draftContext, '4096', reason: 'its own, not Gemma\'s 8192');
    expect(prefs.values, savedBefore, reason: 'a pick saves nothing');

    // What the self-test runs on, and the reload after it: the applied file.
    final selfTest = await tester.runAsync(
      () => resolveSelfTestChatModel(
        argument: null,
        define: '',
        plan: chatModels.plan,
      ),
    );
    expect(selfTest!.file.path, gemmaPath);
    final loads = llm.models.length;
    await tester.runAsync(models.reloadChatModel);
    expect(llm.models, hasLength(loads + 1));
    expect(llm.installs.last, gemmaPath);
    expect(llm.models.last.llm.backend, PreferredBackend.gpu);

    // A restart reads the same settings: the applied file.
    final relaunched = ChatModelRepository(
      settings: TypedSettings(store: prefs),
      store: store,
      npu: () => sm8750,
      readBundle: fakeReadBundle,
      folders: [
        ModelsFolder(
          directory: () async => Directory('${root.path}/models_folder'),
        ),
      ],
    );
    await tester.runAsync(relaunched.load);
    expect((relaunched.plan as CustomChatPlan).path, gemmaPath);
    expect(relaunched.state.value.candidate, isNull);
    relaunched.dispose();

    // Apply, on the NPU (its default): saved and loaded.
    expect(vm.draftBackend, PreferredBackend.npu);
    expect(find.byKey(ChatModelKeys.draftProblem), findsNothing);
    await tester.runAsync(() => vm.apply.execute());
    await tester.pump();
    expect(llm.installs.last, npuOnly.path);
    expect(llm.models.last.llm.backend, PreferredBackend.npu);
    expect(find.byKey(ChatModelKeys.candidate), findsNothing);
    expect(prefs.values['chat.custom'], contains(npuOnly.path));
    await tearDownSection(tester);
  });

  /// The Models screen closes (its view model goes) and opens again: a new
  /// card on the same repositories.
  Future<void> reopen(WidgetTester tester) async {
    await tester.pumpWidget(const SizedBox.shrink());
    vm.dispose();
    await tester.runAsync(pumpEventQueue);
    vm = ChatModelViewModel(
      chatModels: chatModels,
      models: models,
      switcher: ChatModelSwitcher(
        conversation: conversation,
        reloadChatModel: models.reloadChatModel,
        unloadChatModel: models.unloadChatModel,
        refuseChatModelLoads: models.refuseChatModelLoads,
      ),
      picker: FakeModelFilePicker(),
    );
    await whileRunning(tester, () => vm.rescan.running);
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: ListView(children: [ChatModelSection(viewModel: vm)]),
        ),
      ),
    );
  }

  testWidgets('an imported file not applied survives the Models screen '
      'closing, for this app session: the next card shows it, still to '
      'apply', (tester) async {
    await pumpSection(tester, npu: sm8750, withFile: true);
    expect(find.byKey(ChatModelKeys.candidate), findsOneWidget);

    await reopen(tester);

    expect(chatModels.state.value.candidate?.model.file?.name, fileName);
    expect(find.byKey(ChatModelKeys.candidate), findsOneWidget);
    expect(
      tester.widget<Text>(find.byKey(ChatModelKeys.fileLine)).data,
      contains(fileName),
    );
    expect(llm.installs, isEmpty, reason: 'nothing loads it');
    await tearDownSection(tester);
  });

  testWidgets('leaving the Models screen drops a file picked in place and not '
      'applied: the next card shows the applied model', (tester) async {
    await pumpSection(tester, npu: sm8750, inPlace: FakeBundle.gemma4E2b);
    await tester.runAsync(() => vm.apply.execute());
    await tester.pump();
    final npuOnly = File('${root.path}/npu_only.litertlm')
      ..writeAsBytesSync(
        fakeLitertlmBytes(FakeBundle.textOnlyNpu, filler: 4096, seed: 5),
      );
    await tester.runAsync(() => vm.useLocal.execute(npuOnly.path));
    await tester.pump();
    expect(find.byKey(ChatModelKeys.candidate), findsOneWidget);

    await reopen(tester);

    expect(chatModels.state.value.candidate, isNull);
    expect(find.byKey(ChatModelKeys.candidate), findsNothing);
    expect(
      tester.widget<Text>(find.byKey(ChatModelKeys.fileLine)).data,
      contains('in place: ${root.path}/in_place.litertlm'),
    );
    expect(find.byKey(ChatModelKeys.towerText), findsNothing);
    await tearDownSection(tester);
  });

  testWidgets('a failed load: the reason and Run on GPU / Run on CPU; Run on '
      'CPU reloads on the CPU', (tester) async {
    // A text-only file not compiled for an NPU, asked for the NPU.
    await pumpSection(
      tester,
      npu: const NpuAvailable(soc: 'QTI SM8750'),
      inPlace: FakeBundle.textOnly,
    );
    vm.setBackend(PreferredBackend.npu);
    await tester.pump();
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
    for (final action in [ChatModelAction.runOnGpu, ChatModelAction.runOnCpu]) {
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
