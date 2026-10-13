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

import 'dart:async';
import 'dart:io';

import 'package:flutter/foundation.dart' show debugPrint;
import 'package:flutter_edge_ai/flutter_edge_ai.dart'
    show ModelType, PreferredBackend;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/repositories/chat_model_repository.dart';
import 'package:litert_edge_demos/data/repositories/conversation_repository.dart';
import 'package:litert_edge_demos/data/repositories/model_repository.dart';
import 'package:litert_edge_demos/data/services/knowledge/embedder_service.dart';
import 'package:litert_edge_demos/data/services/llm/llm_service.dart';
import 'package:litert_edge_demos/data/services/model_store/model_store.dart';
import 'package:litert_edge_demos/data/services/model_store/models_folder.dart';
import 'package:litert_edge_demos/data/services/settings/typed_settings.dart';
import 'package:litert_edge_demos/domain/models/chat_model.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_source_resolver.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/domain/models/npu_availability.dart';
import 'package:litert_edge_demos/domain/ports/model_file_picker.dart';
import 'package:litert_edge_demos/domain/use_cases/chat_model_switcher.dart';
import 'package:litert_edge_demos/ui/features/setup/view_models/chat_model_view_model.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../../../fakes/fake_bundled_files.dart';
import '../../../../fakes/fake_conversation_repository.dart';
import '../../../../fakes/fake_detector_runtime.dart';
import '../../../../fakes/fake_llm_service.dart';
import '../../../../fakes/fake_model_file_picker.dart';
import '../../../../fakes/fake_settings_store.dart';
import '../../../../fakes/fake_speech.dart';
import '../../../../support/fake_model_server.dart';
import '../../../../support/test_bytes.dart';
import '../../../../support/until.dart';

void main() {
  late Directory root;
  late Directory picked;
  late FakeModelServer server;
  late ModelStore store;
  late FakeLlmService llm;
  late FakeConversationRepository conversation;
  late FakeModelFilePicker picker;
  late ModelRepository models;
  late ChatModelRepository chatModels;
  late ChatModelSwitcher switcher;
  late ChatModelViewModel vm;

  const fileName = 'gemma3_ekv1280_sm8750.litertlm';
  final bytes = testBytes(48 * 1024, seed: 9);

  Future<void> build({
    NpuAvailability npu = const NpuAvailable(soc: 'QTI SM8750'),
    ImportSupport support = const ImportFromFolder(),
    String gemmaModelPath = '',
    bool prepare = true,
    Future<void> Function(TypedSettings settings)? seed,
  }) async {
    store = ModelStore(root: () async => root);
    final settings = TypedSettings(store: InMemorySettingsStore());
    await seed?.call(settings);
    chatModels = ChatModelRepository(
      settings: settings,
      store: store,
      npu: () => npu,
      folders: [
        ModelsFolder(
          directory: () async => Directory('${root.path}/models_folder'),
        ),
        ModelsFolder(
          directory: () async => Directory('${root.path}/tmp_models'),
          create: false,
          label: 'Or, readable in any order',
        ),
      ],
    );
    await chatModels.load();
    models = ModelRepository(
      bundled: FakeBundledFiles(),
      detector: fakeDetectorService(),
      bundledDetector: fakeBundledDetector,
      gemmaModelPath: gemmaModelPath,
      llm: llm,
      stt: fakeSttService(),
      tts: fakeTtsService(),
      chatModels: chatModels,
      embedder: EmbedderService(),
    );
    picker = FakeModelFilePicker(support: support);
    switcher = ChatModelSwitcher(
      conversation: conversation,
      reloadChatModel: models.reloadChatModel,
      unloadChatModel: models.unloadChatModel,
      refuseChatModelLoads: models.refuseChatModelLoads,
    );
    vm = ChatModelViewModel(
      chatModels: chatModels,
      models: models,
      switcher: switcher,
      picker: picker,
      sources: ModelSourceResolver(gemmaModelPath: gemmaModelPath),
    );
    if (!prepare) return;
    // Setup ran once with no chat model: everything else is ready.
    expect(await models.prepareAll(), isA<Error<void>>());
    expect(models.states.value[ModelId.chat], isA<ModelUnavailable>());
  }

  setUp(() async {
    root = Directory.systemTemp.createTempSync('chat_vm_root');
    picked = Directory.systemTemp.createTempSync('chat_vm_picked');
    server = await FakeModelServer.start();
    llm = FakeLlmService()..followRequested = true;
    conversation = FakeConversationRepository();
  });

  tearDown(() async {
    vm.dispose();
    await models.close();
    chatModels.dispose();
    await store.close();
    await conversation.close();
    await server.close();
    root.deleteSync(recursive: true);
    picked.deleteSync(recursive: true);
  });

  String pick([String name = fileName, List<int>? content]) =>
      (File('${picked.path}/$name')..writeAsBytesSync(content ?? bytes)).path;

  test('starts with no chat model (the app ships none): the card says how to '
      'choose one and nothing can be applied yet', () async {
    await build();

    expect(vm.active, ChatModelKind.none);
    expect(vm.noModel, isTrue);
    expect(vm.activeLine, startsWith('No chat model yet'));
    expect(vm.saved, isNull);
    expect(vm.draftProblem, contains('Import or download'));
    expect(vm.canApply, isFalse);
    expect(llm.models, isEmpty, reason: 'nothing loaded');
  });

  test('none chosen and GEMMA_MODEL_PATH set (a developer build): the card '
      'names what loads, by the rule setup loads by', () async {
    await build(gemmaModelPath: pick('define.litertlm'), prepare: false);
    llm.loadGate = Completer<void>();

    final preparing = models.prepareAll();
    await pumpEventQueue();
    expect(models.states.value[ModelId.chat], isA<ModelLoading>());
    expect(vm.activeLine, 'Loading ${kDefineChatModel.name} on gpu…');

    llm.loadGate!.complete();
    expect(await preparing, isA<Ok<void>>());
    expect(
      vm.activeLine,
      startsWith('Loaded: ${kDefineChatModel.name} · gpu → gpu'),
    );
  });

  test('import → defaults in the editor → Use this model: the chat is '
      'released, the model reloaded with exactly these settings', () async {
    await build();
    picker.modelFile = pick();

    await vm.importFile.execute();

    expect(vm.importFile.completed, isTrue);
    expect(vm.fileLine, contains(fileName));
    expect(vm.fileLine, contains('(computed)'));
    expect(vm.draftName, 'gemma3_ekv1280_sm8750');
    expect(vm.draftBackend, PreferredBackend.npu);
    expect(vm.draftContext, '1280');
    expect(vm.contextHint, 1280);
    expect(vm.active, ChatModelKind.none, reason: 'not switched yet');
    expect(vm.applyLabel, 'Use this model');

    vm
      ..setName('My NPU Gemma')
      ..setType(ModelType.gemmaIt)
      ..setContext('896');
    expect(vm.canApply, isTrue);
    await vm.apply.execute();

    expect(vm.apply.completed, isTrue, reason: '${vm.apply.result}');
    expect(conversation.releaseCalls, 1);
    final loaded = llm.models.last;
    expect(loaded.name, 'My NPU Gemma');
    expect(loaded.modelType, ModelType.gemmaIt);
    expect(loaded.llm.backend, PreferredBackend.npu);
    expect(loaded.llm.maxTokens, 896);
    expect(loaded.llm.supportImage, isFalse);
    expect(loaded.tools, isFalse);
    expect(llm.installs.last, '${root.path}/custom/$fileName');
    expect(vm.active, ChatModelKind.custom);
    expect(vm.activeLine, contains('My NPU Gemma · npu → npu (activeBackend)'));
    expect(vm.applyLabel, 'Apply and reload');
    expect(vm.canApply, isFalse, reason: 'nothing changed since');
  });

  test('a replacement file while the custom model runs: Apply is on and the '
      'card says the old file still runs; the old file is kept until the '
      'reload, then deleted', () async {
    await build(npu: const NpuUnavailable('no FastRPC'));
    picker.modelFile = pick('first.litertlm');
    await vm.importFile.execute();
    await vm.apply.execute();
    expect(vm.canApply, isFalse);
    expect(vm.replacementPending, isFalse);
    final first = File('${root.path}/custom/first.litertlm');

    picker.modelFile = pick('second.litertlm', testBytes(48 * 1024, seed: 10));
    await vm.importFile.execute();

    expect(vm.replacementPending, isTrue);
    expect(vm.canApply, isTrue, reason: 'the saved file is not what runs');
    expect(first.existsSync(), isTrue, reason: 'the engine still runs it');
    final loadsBefore = llm.models.length;

    await vm.apply.execute();

    expect(llm.models, hasLength(loadsBefore + 1));
    expect(llm.installs.last, '${root.path}/custom/second.litertlm');
    expect(vm.replacementPending, isFalse);
    expect(vm.canApply, isFalse);
    expect(first.existsSync(), isFalse, reason: 'pruned after the reload');

    // The same name with other bytes (a file re-exported under its old
    // name): the checksum, not the name, decides.
    picker.modelFile = pick('second.litertlm', testBytes(48 * 1024, seed: 11));
    await vm.importFile.execute();
    expect(vm.replacementPending, isTrue);
    expect(vm.canApply, isTrue);
  });

  test('Apply while the previous reply is still stopping: the card says so, '
      'the engine keeps running the old file, which is not pruned; Apply '
      'stays on and works once the reply has ended', () async {
    await build(npu: const NpuUnavailable('no FastRPC'));
    picker.modelFile = pick('first.litertlm');
    await vm.importFile.execute();
    await vm.apply.execute();
    final first = File('${root.path}/custom/first.litertlm');
    picker.modelFile = pick('second.litertlm', testBytes(48 * 1024, seed: 10));
    await vm.importFile.execute();
    final loadsBefore = llm.models.length;
    conversation.releaseError = const ConversationNotReadyException(
      'The previous reply did not finish within 5s of being stopped',
    );

    await vm.apply.execute();

    expect(
      vm.actionError,
      'The previous reply is still stopping; try again in a moment',
    );
    expect(llm.models, hasLength(loadsBefore), reason: 'nothing reloaded');
    expect(first.existsSync(), isTrue, reason: 'the engine still runs it');
    expect(vm.replacementPending, isTrue);
    expect(vm.canApply, isTrue);

    conversation.releaseError = null;
    await vm.apply.execute();

    expect(vm.actionError, isNull);
    expect(llm.installs.last, '${root.path}/custom/second.litertlm');
    expect(first.existsSync(), isFalse, reason: 'pruned after the reload');
  });

  test(
    'an invalid file is refused with the reason; nothing is saved',
    () async {
      await build();
      picker.modelFile = pick('model.tflite');

      await vm.importFile.execute();

      expect(vm.transferError, contains('Import failed'));
      expect(vm.transferError, contains('.litertlm'));
      expect(vm.saved, isNull);
    },
  );

  test('while the self-test holds the chat model, nothing here can switch '
      'it; Apply holds it from the tap', () async {
    await build(npu: const NpuUnavailable('no FastRPC'));
    picker.modelFile = pick();
    await vm.importFile.execute();
    expect(vm.canApply, isTrue);

    final gate = Completer<void>();
    final selfTest = switcher.exclusive((_) => gate.future);

    expect(vm.busy, isTrue);
    expect(vm.canApply, isFalse);
    expect(vm.canImport, isFalse);
    expect(vm.canDownload, isFalse);
    gate.complete();
    await selfTest;
    expect(vm.canApply, isTrue);

    final applying = vm.apply.execute();
    expect(switcher.busy.value, isTrue, reason: 'no window after the tap');
    await applying;
    expect(switcher.busy.value, isFalse);
    expect(vm.apply.completed, isTrue);
    expect(llm.models.last.name, 'gemma3_ekv1280_sm8750');
  });

  test('a cancelled picker is not an error', () async {
    await build();
    picker.modelFile = null;

    await vm.importFile.execute();

    expect(vm.transferError, isNull);
  });

  test('Download from URL: the link is checked, then the file downloaded '
      'with its checksum', () async {
    await build();
    server.files[fileName] = ServedFile(bytes);

    final bad = ModelUrlRequest.parse('ftp://x/y.litertlm', '', '');
    expect((bad as Error<ModelUrlRequest>).error.toString(), contains('https'));
    expect(
      ModelUrlRequest.parse('https://x/y.litertlm', 'abc', ''),
      isA<Error<ModelUrlRequest>>(),
    );
    final request = (ModelUrlRequest.parse(
      '${server.url(fileName)}',
      sha256Hex(bytes).toUpperCase(),
      '${bytes.length}',
    ) as Ok<ModelUrlRequest>).value;
    expect(request.sha256, sha256Hex(bytes), reason: 'lower-cased');

    await vm.download.execute(request);

    expect(vm.download.completed, isTrue, reason: '${vm.download.result}');
    expect(vm.saved, isNotNull);
    expect(vm.fileLine, contains('(matches the one entered)'));
    expect(vm.fileLine, contains('downloaded from ${server.url(fileName)}'));
  });

  test('a link with a user name or password is refused, at entry and by the '
      'repository: nothing is requested', () async {
    await build();
    server.files[fileName] = ServedFile(bytes);
    final withUser = server.url(fileName).replace(userInfo: 'me:hf_secret');

    final entry = ModelUrlRequest.parse('$withUser', '', '');
    expect(
      (entry as Error<ModelUrlRequest>).error.toString(),
      contains('user name or password'),
    );
    final direct = await chatModels.download(withUser);
    expect(
      (direct as Error<CustomChatModel>).error,
      isA<InvalidChatModelException>(),
    );
    expect(server.requests, isEmpty);
    expect(vm.saved, isNull);
  });

  test("a signed link downloads with its query, which is never shown or "
      'logged', () async {
    await build();
    server.files[fileName] = ServedFile(bytes);
    final signed = server
        .url(fileName)
        .replace(queryParameters: {'sig': 'hf_secret'}, fragment: 'f');
    final logs = <String>[];
    final print = debugPrint;
    debugPrint = (message, {wrapWidth}) => logs.add(message ?? '');
    addTearDown(() => debugPrint = print);
    final request = (ModelUrlRequest.parse(
      '$signed',
      '',
      '${bytes.length}',
    ) as Ok<ModelUrlRequest>).value;
    expect(request.url, signed, reason: 'the download gets the whole link');

    await vm.download.execute(request);

    expect(vm.download.completed, isTrue, reason: '${vm.download.result}');
    expect(server.requestsFor(fileName), isNotEmpty);
    expect(vm.fileLine, contains('downloaded from ${server.url(fileName)}'));
    expect(vm.fileLine, isNot(contains('hf_secret')));
    expect((vm.saved!.source as UrlModelSource).url, server.url(fileName));
    expect(logs, isNotEmpty);
    expect(logs.where((line) => line.contains('hf_secret')), isEmpty);
  });

  test(
    'NPU unavailable here: the option is off with flutter_edge_ai\'s reason, '
    'and an NPU draft cannot be applied',
    () async {
      await build(
        npu: const NpuUnavailable('no NPU dispatch stack ships for macos'),
      );
      picker.modelFile = pick();
      await vm.importFile.execute();

      expect(vm.npuOffered, isFalse);
      expect(vm.npuUnavailableReason, contains('no NPU dispatch stack ships'));
      expect(vm.draftBackend, PreferredBackend.gpu, reason: 'the default here');

      vm.setBackend(PreferredBackend.npu);
      expect(
        vm.draftProblem,
        contains('no NPU dispatch stack ships for macos'),
      );
      expect(vm.canApply, isFalse);
    },
  );

  test('a GPU context below 1024 is explained, not silently raised', () async {
    await build();
    picker.modelFile = pick();
    await vm.importFile.execute();

    vm
      ..setBackend(PreferredBackend.gpu)
      ..setContext('896');

    expect(vm.draftProblem, contains('at least 1024'));
    vm.setContext('abc');
    expect(vm.draftProblem, contains('whole number'));
  });

  test('a failed load shows the engine\'s reason and the explicit ways out; '
      'Run on GPU reloads on the GPU', () async {
    await build();
    picker.modelFile = pick();
    await vm.importFile.execute();
    llm.loadError = ChatModelLoadException(
      modelName: 'gemma3_ekv1280_sm8750',
      requested: PreferredBackend.npu,
      cause: Exception('Failed to create engine'),
      nativeReasons: const ['npu backend failed: wrong SoC'],
    );

    await vm.apply.execute();

    expect(models.states.value[ModelId.chat], isA<ModelFailed>());
    expect(vm.loadError, contains('did not load on npu'));
    expect(vm.loadError, contains('wrong SoC'));
    expect(vm.failureActions, [
      ChatModelAction.runOnGpu,
      ChatModelAction.runOnCpu,
    ]);
    expect(
      llm.models.where((m) => m.name == kDefineChatModel.name),
      isEmpty,
      reason: 'no other model is loaded in its place',
    );

    await vm.runOn.execute(PreferredBackend.gpu);
    expect(vm.runOn.completed, isTrue, reason: '${vm.runOn.result}');
    expect(llm.models.last.llm.backend, PreferredBackend.gpu);
    expect(llm.models.last.llm.maxTokens, 1280);
    expect(vm.loadError, isNull);
    expect(vm.failureActions, isEmpty);
    expect(vm.draftBackend, PreferredBackend.gpu);
  });

  /// The models folders listed: the rescan the view model starts with
  /// has ended.
  Future<void> listed() => untilNotified(
    vm.rescan,
    () => !vm.rescan.running,
    what: 'the models folders to be listed',
  );

  test('Android: import is off; both models folders (adb push) and the link '
      'are the ways, each folder with its order of steps', () async {
    await build(
      support: const ImportUnsupported('Importing files is not available'),
    );
    await listed();
    final folder = '${root.path}/models_folder';
    final tmp = '${root.path}/tmp_models';

    expect(vm.folderPath, folder);
    expect(vm.folders.map((f) => f.path), [folder, tmp]);
    expect(vm.importUnavailableReason, contains('adb push the file'));
    expect(vm.importUnavailableReason, contains('Download from URL'));
    expect(vm.folders[0].hint, contains('Launch the app once'));
    expect(vm.folders[0].hint, contains('adb push model.litertlm $folder/'));
    expect(vm.folders[1].hint, contains('adb shell mkdir -p $tmp'));
    expect(vm.folders[1].hint, contains('adb push model.litertlm $tmp/'));
    expect(vm.folders[1].files, isEmpty, reason: 'not there: empty');
    expect(vm.canImport, isFalse);
    expect(vm.canDownload, isTrue);
  });

  test(
    'the models folder: listed with Rescan; a file picked there is used '
    'in place and becomes selected; Use this model loads that path',
    () async {
      await build(npu: const NpuAvailable(soc: 'QTI SM8850'));
      await listed();
      expect(vm.localFiles, isEmpty);
      final folder = Directory(vm.folderPath!);
      final header = File(
        'test_assets/litertlm/gemma4_2b_sm8850_npu.header.bin',
      ).readAsBytesSync();
      final file = File('${folder.path}/gemma4_2b_SM8850.litertlm')
        ..writeAsBytesSync([...header, ...bytes]);

      await vm.rescan.execute();
      expect(vm.localFiles.map((e) => e.name), ['gemma4_2b_SM8850.litertlm']);
      expect(
        ChatModelViewModel.entryLine(vm.localFiles.single),
        matches(RegExp(r'^0\.\d MB · \d{4}-\d\d-\d\d \d\d:\d\d$')),
      );
      await vm.useLocal.execute(file.path);

      expect(vm.useLocal.completed, isTrue, reason: '${vm.localError}');
      expect(vm.isSelected(vm.localFiles.single), isTrue);
      expect(vm.fileLine, contains('in place: ${file.path}'));
      expect(vm.draftBackend, PreferredBackend.npu);
      expect(vm.draftImages, isFalse);
      expect(vm.draftContext, '4096');

      await vm.apply.execute();
      expect(vm.apply.completed, isTrue, reason: '${vm.apply.result}');
      expect(llm.installs.last, file.path, reason: 'loaded where it is');
      expect(llm.models.last.llm.backend, PreferredBackend.npu);
    },
  );

  group('nothing chosen and one file already in the models folder', () {
    late List<int> npuFile;

    setUp(() {
      final header = File(
        'test_assets/litertlm/gemma4_2b_sm8850_npu.header.bin',
      ).readAsBytesSync();
      npuFile = [...header, ...bytes];
    });

    File putInFolder(String name) {
      final folder = Directory('${root.path}/models_folder')
        ..createSync(recursive: true);
      return File('${folder.path}/$name')..writeAsBytesSync(npuFile);
    }

    test('it is selected at start with its own settings; nothing loads '
        'before Use this model', () async {
      final file = putInFolder('gemma4_2b_SM8850.litertlm');
      await build(npu: const NpuAvailable(soc: 'QTI SM8850'));
      await listed();

      expect(vm.useLocal.completed, isTrue, reason: '${vm.localError}');
      expect(vm.isSelected(vm.localFiles.single), isTrue);
      expect(vm.fileLine, contains('in place: ${file.path}'));
      expect(vm.draftBackend, PreferredBackend.npu);
      expect(vm.draftContext, '4096');
      expect(vm.active, ChatModelKind.none, reason: 'selected, not switched');
      expect(vm.canApply, isTrue);
      expect(llm.models, isEmpty, reason: 'loading waits for the user');

      await vm.apply.execute();
      expect(vm.apply.completed, isTrue, reason: '${vm.apply.result}');
      expect(llm.installs.last, file.path);
    });

    test('a file added later is selected by Rescan', () async {
      await build();
      await listed();
      expect(vm.saved, isNull);

      final file = putInFolder('gemma4_2b_SM8850.litertlm');
      await vm.rescan.execute();

      expect(vm.fileLine, contains('in place: ${file.path}'));
    });

    test('two files: none is selected, the user picks', () async {
      putInFolder('a.litertlm');
      putInFolder('b.litertlm');
      await build();
      await listed();

      expect(vm.localFiles, hasLength(2));
      expect(vm.saved, isNull);
      expect(vm.useLocal.result, isNull, reason: 'never tried');
    });

    test('a developer define (GEMMA_MODEL_PATH) keeps its model', () async {
      putInFolder('gemma4_2b_SM8850.litertlm');
      await build(gemmaModelPath: pick('define.litertlm'), prepare: false);
      await listed();

      expect(vm.saved, isNull);
      expect(vm.useLocal.result, isNull);
    });

    test('selected mid-copy: the row can be picked again, and Rescan after '
        'the copy picks it at its full size', () async {
      final folder = Directory('${root.path}/models_folder')
        ..createSync(recursive: true);
      final file = File('${folder.path}/gemma4_2b_SM8850.litertlm')
        ..writeAsBytesSync(npuFile.sublist(0, npuFile.length ~/ 2));
      await build(npu: const NpuAvailable(soc: 'QTI SM8850'));
      await listed();
      expect(vm.saved?.file?.sizeBytes, npuFile.length ~/ 2);

      file.writeAsBytesSync(npuFile);
      await vm.rescan.execute();

      expect(vm.saved?.file?.sizeBytes, npuFile.length);
      expect(vm.isSelected(vm.localFiles.single), isTrue);
      expect(vm.canPick(vm.localFiles.single), isFalse, reason: 'same size');
      expect(vm.canApply, isTrue, reason: '${vm.draftProblem}');
    });

    test('Rescan while a download runs selects nothing: the download keeps '
        'its own settings', () async {
      await build(npu: const NpuAvailable(soc: 'QTI SM8850'));
      await listed();
      server.files[fileName] = ServedFile(
        bytes,
        chunkDelay: const Duration(milliseconds: 40),
        chunkBytes: 4096,
      );
      final request = switch (ModelUrlRequest.parse(
        '${server.url(fileName)}',
        '',
        '',
      )) {
        Ok(:final value) => value,
        final other => throw StateError('$other'),
      };
      final downloading = vm.download.execute(request);
      await untilNotified(
        chatModels.busy,
        () => chatModels.busy.value,
        what: 'the download to start',
      );

      putInFolder('gemma4_2b_SM8850.litertlm');
      await vm.rescan.execute();
      expect(vm.useLocal.result, isNull, reason: 'not while downloading');

      await downloading;
      expect(vm.download.completed, isTrue, reason: '${vm.transferError}');
      expect(vm.saved?.file?.name, fileName);
      expect(vm.saved?.maxTokens, 1280, reason: 'from its own name');
    });

    test('a saved choice that cannot load (blocked) is not replaced, and '
        'nothing loads by itself', () async {
      putInFolder('gemma4_2b_SM8850.litertlm');
      await build(
        prepare: false,
        seed: (settings) => settings.write(Settings.chatModel, 'custom'),
      );
      await listed();

      expect(vm.saved, isNull);
      expect(vm.useLocal.result, isNull);
      expect(llm.installs, isEmpty);
    });

    test('unreadable saved settings: the problem stays to be read', () async {
      putInFolder('gemma4_2b_SM8850.litertlm');
      await build(
        prepare: false,
        seed: (settings) =>
            settings.write(Settings.customChatModel, '{not json'),
      );
      await listed();

      expect(vm.problem, isNotNull);
      expect(vm.useLocal.result, isNull);
    });

    test('the only file cannot be used: its reason shows, nothing is '
        'selected', () async {
      final folder = Directory('${root.path}/models_folder')
        ..createSync(recursive: true);
      File('${folder.path}/notes.litertlm').writeAsStringSync('hello');
      await build();
      await listed();

      expect(vm.localError, contains('not a .litertlm'));
      expect(vm.saved, isNull);
    });

    test('a model chosen before stays chosen', () async {
      await build();
      picker.modelFile = pick();
      await vm.importFile.execute();
      expect(vm.importFile.completed, isTrue);

      putInFolder('gemma4_2b_SM8850.litertlm');
      await vm.rescan.execute();

      expect(vm.fileLine, contains(fileName));
      expect(vm.useLocal.result, isNull);
    });
  });

  test('a path that is not a .litertlm is refused with the reason', () async {
    await build();
    await listed();
    final bogus = File('${picked.path}/notes.litertlm')
      ..writeAsStringSync('hello');

    await vm.useLocal.execute(bogus.path);

    expect(vm.localError, contains('not a .litertlm'));
    expect(vm.saved, isNull);
  });
}
