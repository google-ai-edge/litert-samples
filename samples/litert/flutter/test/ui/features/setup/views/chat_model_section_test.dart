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
import '../../../../support/litertlm_header.dart';
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
    List<HeaderSection>? header,
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
        // A text-only NPU build: an import whose header cannot be read is
        // refused.
        final source = File('${root.path}/$fileName')
          ..writeAsBytesSync([
            ...litertlmHeaderBytes(kTextOnlyNpuSections),
            ...testBytes(4096),
          ]);
        await chatModels.importFile(source.path);
      }
      if (header != null) {
        // A file with this header, used in place.
        final file = File('${root.path}/in_place.litertlm')
          ..writeAsBytesSync([
            ...litertlmHeaderBytes(header),
            ...testBytes(4096, seed: 3),
          ]);
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

  /// Every text in the row [key], in order: `Vision | CPU · the file sets
  /// it`, or `Vision | CPU | NPU` for a row with a control.
  String row(WidgetTester tester, Key key) => tester
      .widgetList<Text>(
        find.descendant(of: find.byKey(key), matching: find.byType(Text)),
      )
      .map((t) => t.data)
      .join(' | ');

  SegmentedButton<PreferredBackend> control(WidgetTester tester, Key key) =>
      tester.widget<SegmentedButton<PreferredBackend>>(find.byKey(key));

  Future<void> selectBackend(
    WidgetTester tester,
    PreferredBackend backend,
  ) async {
    control(tester, ChatModelKeys.backend).onSelectionChanged!({backend});
    await tester.pump();
  }

  const choiceCaption =
      'This file does not say where its towers run: NPU only if they are '
      'compiled for this chip, CPU otherwise (for example the official '
      'qualcomm_sm8750 build)';
  const sm8750 = NpuAvailable(soc: 'QTI SM8750');

  testWidgets('no Images, Audio or Encoders controls: a read-only Towers '
      'block under Context, before Tools; an imported text-only build: '
      'neither tower in the file', (tester) async {
    await pumpSection(tester, withFile: true);

    expect(find.byKey(ChatModelKeys.towers), findsOneWidget);
    expect(find.byType(SwitchListTile), findsOneWidget, reason: 'Tools only');
    expect(find.byKey(ChatModelKeys.tools), findsOneWidget);
    expect(find.text('Images'), findsNothing);
    expect(find.text('Encoders'), findsNothing);
    final contextY = tester.getTopLeft(find.byKey(ChatModelKeys.context)).dy;
    final towersY = tester.getTopLeft(find.byKey(ChatModelKeys.towers)).dy;
    final toolsY = tester.getTopLeft(find.byKey(ChatModelKeys.tools)).dy;
    expect(contextY < towersY && towersY < toolsY, isTrue);

    expect(row(tester, ChatModelKeys.towerText), 'Text | GPU');
    expect(
      row(tester, ChatModelKeys.towerVision),
      'Vision | not in this file: photos in Voice chat and detailed answers '
      'in Live camera are off',
    );
    expect(
      row(tester, ChatModelKeys.towerAudio),
      'Audio | not in this file: voice questions are transcribed first',
    );
    expect(find.text(choiceCaption), findsNothing);
    await tearDownSection(tester);
  });

  testWidgets('towers declared by the file (the official qcs8275 build) on '
      'the NPU: Text NPU, Vision CPU, Audio CPU, set by the file; no '
      'control', (tester) async {
    await pumpSection(tester, npu: sm8750, header: kQualcommQcs8275Sections);

    expect(vm.draftBackend, PreferredBackend.npu);
    expect(row(tester, ChatModelKeys.towerText), 'Text | NPU');
    expect(
      row(tester, ChatModelKeys.towerVision),
      'Vision | CPU · the file sets it',
    );
    expect(
      row(tester, ChatModelKeys.towerAudio),
      'Audio | CPU · the file sets it',
    );
    expect(find.byKey(ChatModelKeys.visionBackend), findsNothing);
    expect(find.byKey(ChatModelKeys.audioBackend), findsNothing);
    expect(find.text(choiceCaption), findsNothing);
    await tearDownSection(tester);
  });

  testWidgets('undeclared towers (the official sm8750 build): on the NPU a '
      'CPU | NPU control in each tower row, the NPU by default, with the '
      'caption once; a choice is a draft change; on the GPU both CPU, '
      'read-only', (tester) async {
    await pumpSection(tester, npu: sm8750, header: kQualcommSm8750Sections);
    expect(vm.draftBackend, PreferredBackend.gpu, reason: 'not marked NPU');
    expect(row(tester, ChatModelKeys.towerText), 'Text | GPU');
    expect(row(tester, ChatModelKeys.towerVision), 'Vision | CPU');
    expect(row(tester, ChatModelKeys.towerAudio), 'Audio | CPU');
    expect(find.text(choiceCaption), findsNothing);

    await selectBackend(tester, PreferredBackend.npu);

    expect(row(tester, ChatModelKeys.towerText), 'Text | NPU');
    for (final key in [
      ChatModelKeys.visionBackend,
      ChatModelKeys.audioBackend,
    ]) {
      final segmented = control(tester, key);
      expect(segmented.selected, {PreferredBackend.npu}, reason: '$key');
      expect(segmented.segments.map((s) => s.value), [
        PreferredBackend.cpu,
        PreferredBackend.npu,
      ]);
    }
    expect(row(tester, ChatModelKeys.towerVision), 'Vision | CPU | NPU');
    expect(find.text(choiceCaption), findsOneWidget);

    control(tester, ChatModelKeys.visionBackend).onSelectionChanged!({
      PreferredBackend.cpu,
    });
    await tester.pump();
    expect(control(tester, ChatModelKeys.visionBackend).selected, {
      PreferredBackend.cpu,
    });
    expect(control(tester, ChatModelKeys.audioBackend).selected, {
      PreferredBackend.npu,
    });
    expect(vm.draftTowers?.visionBackend, PreferredBackend.cpu);
    expect(
      tester.widget<FilledButton>(find.byKey(ChatModelKeys.apply)).onPressed,
      isNotNull,
    );

    await selectBackend(tester, PreferredBackend.gpu);
    expect(find.byKey(ChatModelKeys.visionBackend), findsNothing);
    expect(row(tester, ChatModelKeys.towerVision), 'Vision | CPU');
    expect(find.text(choiceCaption), findsNothing);
    await tearDownSection(tester);
  });

  testWidgets('the standard Gemma 4 E2B on the GPU: Text GPU, Vision CPU, '
      'Audio CPU, set by the file', (tester) async {
    await pumpSection(tester, header: kGemma4E2bSections);

    expect(row(tester, ChatModelKeys.towerText), 'Text | GPU');
    expect(
      row(tester, ChatModelKeys.towerVision),
      'Vision | CPU · the file sets it',
    );
    expect(
      row(tester, ChatModelKeys.towerAudio),
      'Audio | CPU · the file sets it',
    );
    expect(find.text(choiceCaption), findsNothing);
    await tearDownSection(tester);
  });

  testWidgets('a text-only file (the older SM8850 build): no vision, no '
      'audio, and what is off without them', (tester) async {
    await pumpSection(tester, npu: sm8750, header: kTextOnlyNpuSections);

    expect(row(tester, ChatModelKeys.towerText), 'Text | NPU');
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

  testWidgets('the official qcs8275 build on the NPU: no control; the '
      'official sm8750 build on the NPU: two controls, the NPU by default', (
    tester,
  ) async {
    await pumpSection(tester, npu: sm8750, header: kQualcommQcs8275Sections);
    expect(vm.draftBackend, PreferredBackend.npu);
    expect(
      row(tester, ChatModelKeys.towerVision),
      'Vision | CPU · the file sets it',
    );
    expect(
      row(tester, ChatModelKeys.towerAudio),
      'Audio | CPU · the file sets it',
    );
    expect(find.byKey(ChatModelKeys.visionBackend), findsNothing);
    expect(find.byKey(ChatModelKeys.audioBackend), findsNothing);
    await tearDownSection(tester);

    await pumpSection(tester, npu: sm8750, header: kQualcommSm8750Sections);
    await selectBackend(tester, PreferredBackend.npu);
    expect(control(tester, ChatModelKeys.visionBackend).selected, {
      PreferredBackend.npu,
    });
    expect(control(tester, ChatModelKeys.audioBackend).selected, {
      PreferredBackend.npu,
    });
    expect(find.text(choiceCaption), findsOneWidget);
    await tearDownSection(tester);
  });

  testWidgets('the official sm8750 build with its towers left on the NPU '
      'fails in the warm-up: the engine\'s reason and a line on what to do; '
      'no tower button (only Run on GPU / CPU); both towers set to CPU in '
      'their rows, Apply, and it loads', (tester) async {
    await pumpSection(tester, npu: sm8750, header: kQualcommSm8750Sections);
    await selectBackend(tester, PreferredBackend.npu);
    llm.warmUpError = Exception('Failed to create conversation');

    await tester.runAsync(() => vm.apply.execute());
    await tester.pump();

    final message = tester
        .widget<Text>(find.byKey(ChatModelKeys.loadError))
        .data!;
    expect(message, contains('Failed to create conversation'));
    expect(
      message,
      endsWith(
        "\nIf this build's towers are not compiled for the NPU, set "
        'Vision/Audio to CPU on the card.',
      ),
    );
    expect(llm.models, hasLength(1), reason: 'no retry, no fallback');
    expect(
      [
        for (final action in ChatModelAction.values)
          if (find.byKey(ChatModelKeys.action(action)).evaluate().isNotEmpty)
            action,
      ],
      [ChatModelAction.runOnGpu, ChatModelAction.runOnCpu],
    );

    for (final key in [
      ChatModelKeys.visionBackend,
      ChatModelKeys.audioBackend,
    ]) {
      control(tester, key).onSelectionChanged!({PreferredBackend.cpu});
      await tester.pump();
    }
    await tester.runAsync(() => vm.apply.execute());
    await tester.pump();

    expect(llm.models, hasLength(2));
    expect(llm.models.last.llm.backend, PreferredBackend.npu);
    expect(llm.models.last.llm.visionBackend, PreferredBackend.cpu);
    expect(llm.models.last.llm.audioBackend, PreferredBackend.cpu);
    expect(find.byKey(ChatModelKeys.loadError), findsNothing);
    expect(control(tester, ChatModelKeys.visionBackend).selected, {
      PreferredBackend.cpu,
    });
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
