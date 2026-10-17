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
import 'dart:typed_data';

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
import '../../../../support/fake_bundles.dart';
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

  /// A text-only NPU build (compiled for the NPU, as the name says).
  Uint8List litertlm(int seed) =>
      fakeLitertlmBytes(FakeBundle.textOnlyNpu, filler: 48 * 1024, seed: seed);
  final bytes = litertlm(9);

  Future<void> build({
    NpuAvailability npu = const NpuAvailable(soc: 'QTI SM8750'),
    ImportSupport support = const ImportFromFolder(),
    String gemmaModelPath = '',
    bool prepare = true,
    bool autoApplyTheOnlyFile = true,
    Future<void> Function(TypedSettings settings)? seed,
  }) async {
    store = ModelStore(root: () async => root);
    final settings = TypedSettings(store: InMemorySettingsStore());
    await seed?.call(settings);
    chatModels = ChatModelRepository(
      settings: settings,
      store: store,
      npu: () => npu,
      readBundle: fakeReadBundle,
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
      autoApplyTheOnlyFile: autoApplyTheOnlyFile,
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
    llm = FakeLlmService();
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

  /// A test `.litertlm` of [kind], picked in place.
  Future<void> useInPlace(String name, FakeBundle kind) async {
    await vm.useLocal.execute(
      pick(name, fakeLitertlmBytes(kind, filler: 4096)),
    );
    expect(vm.useLocal.completed, isTrue, reason: '${vm.useLocal.result}');
  }

  /// [tower]'s row as the card shows it; `-` for none.
  String row(Tower tower) => vm.towerRow(tower)?.text ?? '-';

  test('towers: every one the file has loads, nothing to choose; before the '
      'load each row says it is in the file, after it where the engine put '
      'it (activeVisionBackend / activeAudioBackend)', () async {
    await build();
    await useInPlace('gemma-4-E2B-it.litertlm', FakeBundle.gemma4E2b);

    expect(vm.draftBackend, PreferredBackend.gpu);
    expect(vm.towersShown, isTrue);
    expect(vm.npuBuildLine, isNull, reason: 'not compiled for the NPU');
    expect(
      row(Tower.vision),
      'in the file (where it runs shows once it is loaded)',
    );
    expect(row(Tower.audio), row(Tower.vision));

    await vm.apply.execute();
    expect(vm.apply.completed, isTrue, reason: '${vm.apply.result}');
    final loaded = llm.models.last.llm;
    expect(loaded.supportImage && loaded.supportAudio, isTrue);
    expect(row(Tower.vision), 'in the file · runs on the CPU');
    expect(row(Tower.audio), 'in the file · runs on the CPU');
    expect(vm.activeLine, contains('vision cpu · audio cpu'));
    expect(vm.canApply, isFalse);
  });

  test('a build compiled for the NPU: the NPU by default, the block says '
      'which NPU its text model is compiled for; after the load each tower '
      'shows where the engine put it, nothing the app guessed', () async {
    await build();
    llm
      ..visionBackend = PreferredBackend.npu
      ..audioBackend = PreferredBackend.npu;
    await useInPlace(
      'gemma-4-E2B-it_qualcomm_sm8750.litertlm',
      FakeBundle.npuWithTowers,
    );

    expect(vm.draftBackend, PreferredBackend.npu);
    expect(vm.draftTools, isFalse);
    expect(vm.npuBuildLine, 'Text model: compiled for the Qualcomm SM8750 NPU');
    expect(
      row(Tower.vision),
      'in the file (where it runs shows once it is loaded)',
    );

    await vm.apply.execute();
    expect(vm.apply.completed, isTrue, reason: '${vm.apply.result}');
    expect(row(Tower.vision), 'in the file · runs on the NPU');
    expect(row(Tower.audio), 'in the file · runs on the NPU');
    expect(vm.activeLine, contains('vision npu · audio npu'));
  });

  test('the official per-SoC NPU build (its encoders carry no NPU stamp): '
      'the NPU for the text model, and after the load the towers where the '
      'engine put them, the CPU', () async {
    await build();
    await useInPlace(
      'gemma-4-E2B-it_qualcomm_sm8750.litertlm',
      FakeBundle.officialNpu,
    );

    expect(vm.draftBackend, PreferredBackend.npu);
    expect(vm.npuBuildLine, isNotNull);

    await vm.apply.execute();
    expect(vm.apply.completed, isTrue, reason: '${vm.apply.result}');
    expect(row(Tower.vision), 'in the file · runs on the CPU');
    expect(row(Tower.audio), 'in the file · runs on the CPU');
    expect(vm.activeLine, contains('npu → npu'));
  });

  test('a build for an NPU this package has no dispatch for (Google '
      'Tensor): the NPU only; where it is offered, its load is the '
      "plugin's unsupportedNpu refusal, and no backend is offered", () async {
    await build();
    await useInPlace('gemma-4-E2B-it_tensor_g5.litertlm', FakeBundle.tensorNpu);
    expect(vm.draftBackend, PreferredBackend.npu);
    expect(
      vm.gpuCpuUnavailableReason,
      'GPU and CPU: not for this file (compiled for the Google Tensor_G5 NPU: '
      'runs only on the NPU).',
    );
    llm.loadError = const ChatModelLoadException(
      modelName: 'gemma-4-E2B-it_tensor_g5',
      requested: PreferredBackend.npu,
      cause: LiteRtLmBundleException(
        LiteRtLmBundleProblem.unsupportedNpu,
        '/m/gemma-4-E2B-it_tensor_g5.litertlm is compiled for the Google '
        'Tensor_G5 NPU. flutter_edge_ai_litertlm runs Qualcomm and Intel NPU '
        'bundles only; install the generic bundle of this model to run it on '
        'the GPU or the CPU.',
      ),
    );

    await vm.apply.execute();

    expect(vm.loadError, endsWith('(unsupportedNpu)'));
    expect(vm.failureActions, isEmpty);
  });

  test("another file picked while one runs shows its own towers, never the "
      "running model's backends: also another build pushed over the same "
      'path', () async {
    await build();
    await useInPlace('gemma-4-E2B-it.litertlm', FakeBundle.gemma4E2b);
    await vm.apply.execute();
    expect(row(Tower.vision), 'in the file · runs on the CPU');

    await useInPlace('other.litertlm', FakeBundle.gemma4E2b);

    expect(vm.candidate, isNotNull);
    expect(
      row(Tower.vision),
      'in the file (where it runs shows once it is loaded)',
    );

    // An NPU build written over the running file's path, then picked.
    await useInPlace('gemma-4-E2B-it.litertlm', FakeBundle.npuWithTowers);

    expect(vm.candidate?.backend, PreferredBackend.npu);
    expect(
      row(Tower.vision),
      'in the file (where it runs shows once it is loaded)',
      reason: 'the engine runs the old file',
    );
    expect(
      row(Tower.audio),
      'in the file (where it runs shows once it is loaded)',
    );

    // Dropped (the Models screen closes): the saved file at that path is
    // not the one that runs either (it changed since it was chosen).
    await chatModels.dropInPlaceCandidate();
    expect(vm.candidate, isNull);
    expect(vm.localProblem, contains('changed since it was chosen'));
    expect(
      row(Tower.vision),
      'in the file (where it runs shows once it is loaded)',
    );
  });

  test('a candidate with the very bytes that run (the same file imported '
      'again from elsewhere) shows no tower backend: a candidate is never '
      'what runs, whatever its checksum', () async {
    await build();
    final same = fakeLitertlmBytes(FakeBundle.gemma4E2b, filler: 4096);
    picker.modelFile = pick('gemma-4-E2B-it.litertlm', same);
    await vm.importFile.execute();
    await vm.apply.execute();
    expect(vm.apply.completed, isTrue, reason: '${vm.apply.result}');
    expect(row(Tower.vision), 'in the file · runs on the CPU');

    final copy = File('${picked.path}/copy/gemma-4-E2B-it.litertlm')
      ..createSync(recursive: true)
      ..writeAsBytesSync(same);
    picker.modelFile = copy.path;
    await vm.importFile.execute();

    expect(vm.candidate, isNotNull, reason: 'another source: a candidate');
    expect(vm.candidate?.file?.sha256, vm.saved?.file?.sha256);
    expect(vm.localProblem, isNull, reason: 'nothing else hides the rows');
    expect(
      row(Tower.vision),
      'in the file (where it runs shows once it is loaded)',
    );
    expect(
      row(Tower.audio),
      'in the file (where it runs shows once it is loaded)',
    );
  });

  test(
    'a text-only file: Vision and Audio say what is off without them',
    () async {
      await build();
      await useInPlace('gemma4_2b_npu.litertlm', FakeBundle.textOnlyNpu);

      expect(
        row(Tower.vision),
        'not in this file: photos in Voice chat and detailed answers in Live '
        'camera are off',
      );
      expect(
        row(Tower.audio),
        'not in this file: voice questions are transcribed first',
      );
      expect(vm.towerRow(Tower.vision)?.inFile, isFalse);
    },
  );

  test('an imported file whose header the plugin cannot read is adopted as '
      "a file in place is: the plugin's reason in the Towers block, no tower "
      'rows', () async {
    await build();
    picker.modelFile = pick(fileName, testBytes(48 * 1024, seed: 9));
    await vm.importFile.execute();

    expect(vm.importFile.completed, isTrue);
    expect(vm.transferError, isNull);
    expect(vm.candidate, isNotNull);
    expect(vm.headerNotRead, contains('is not a .litertlm container'));
    expect(vm.towerRow(Tower.vision), isNull);
    expect(vm.canApply, isTrue);
  });

  test('an incomplete file picked in place is shown as unusable with the '
      'lengths the plugin reports: no candidate, nothing to apply', () async {
    await build();
    final cut = pick(
      'cut.litertlm',
      fakeLitertlmBytes(FakeBundle.incomplete, filler: 4096),
    );
    final length = File(cut).lengthSync();

    await vm.useLocal.execute(cut);

    expect(vm.useLocal.error, isTrue);
    expect(
      vm.localError,
      '$cut is incomplete: it is $length bytes and its sections end at '
      '${length * 2} (a copy or download that stopped early). Copy or '
      'download it again.',
    );
    expect(vm.candidate, isNull);
    expect(vm.towersShown, isFalse);
    expect(vm.canApply, isFalse);
  });

  test('a build compiled for the NPU: GPU and CPU are not offered (the '
      'plugin refuses it there), and Apply and Run on GPU / CPU refuse them '
      'before the running model is unloaded', () async {
    await build();
    await useInPlace(
      'gemma-4-E2B-it_qualcomm_sm8750.litertlm',
      FakeBundle.npuWithTowers,
    );
    await vm.apply.execute();
    expect(vm.apply.completed, isTrue, reason: '${vm.apply.result}');
    expect(llm.models, hasLength(1));
    final unloads = llm.unloadCalls;

    expect(
      vm.gpuCpuUnavailableReason,
      'GPU and CPU: not for this file (compiled for the Qualcomm SM8750 NPU: '
      'runs only on the NPU).',
    );
    vm.setBackend(PreferredBackend.gpu);
    expect(
      vm.draftProblem,
      'GPU: not for this file (compiled for the Qualcomm SM8750 NPU: runs '
      'only on the NPU). Choose NPU.',
    );
    expect(vm.canApply, isFalse);

    await vm.apply.execute();
    expect(vm.apply.error, isTrue);
    expect(vm.actionError, vm.draftProblem);

    await vm.runOn.execute(PreferredBackend.cpu);
    expect(vm.runOn.error, isTrue);
    expect(
      '${(vm.runOn.result! as Error<void>).error}',
      'CPU: not for this file (compiled for the Qualcomm SM8750 NPU: runs '
          'only on the NPU). Choose NPU.',
    );

    expect(llm.models, hasLength(1), reason: 'nothing loaded again');
    expect(llm.unloadCalls, unloads, reason: 'the running model kept');
    expect(models.states.value[ModelId.chat], isA<ModelReady>());
    expect(vm.saved?.backend, PreferredBackend.npu, reason: 'nothing saved');

    vm.setBackend(PreferredBackend.npu);
    expect(vm.draftProblem, isNull);
  });

  test('a generic file: GPU and CPU are offered', () async {
    await build();
    await useInPlace('gemma-4-E2B-it.litertlm', FakeBundle.gemma4E2b);

    expect(vm.gpuCpuUnavailableReason, isNull);
    vm.setBackend(PreferredBackend.cpu);
    expect(vm.draftProblem, isNull);
  });

  /// A second card on the same repositories, as when the Models screen is
  /// opened again: its draft starts from what is saved. Returned once its
  /// first rescan of the models folders has ended.
  Future<ChatModelViewModel> reopened() async {
    vm.dispose();
    vm = ChatModelViewModel(
      chatModels: chatModels,
      models: models,
      switcher: switcher,
      picker: picker,
    );
    await untilNotified(
      vm.rescan,
      () => !vm.rescan.running,
      what: 'the reopened card\'s rescan',
    );
    return vm;
  }

  test('a file picked while the custom model runs: the card says it is not '
      'applied yet and Apply is on; the old file is kept until the reload, '
      'then deleted', () async {
    await build();
    picker.modelFile = pick('first.litertlm');
    await vm.importFile.execute();
    await vm.apply.execute();
    expect(vm.canApply, isFalse);
    expect(vm.candidateNotice, isNull);
    final first = File('${root.path}/custom/first.litertlm');

    picker.modelFile = pick('second.litertlm', litertlm(10));
    await vm.importFile.execute();

    expect(
      vm.candidateNotice,
      'Not applied yet: press "Apply and reload" to use this file. Until '
      'then the chat model stays first.',
    );
    expect(vm.replacementPending, isFalse, reason: 'nothing saved');
    expect(vm.canApply, isTrue, reason: 'a candidate waits');
    expect(first.existsSync(), isTrue, reason: 'the engine still runs it');
    final loadsBefore = llm.models.length;

    await vm.apply.execute();

    expect(llm.models, hasLength(loadsBefore + 1));
    expect(llm.installs.last, '${root.path}/custom/second.litertlm');
    expect(vm.candidateNotice, isNull);
    expect(vm.canApply, isFalse);
    expect(first.existsSync(), isFalse, reason: 'pruned after the reload');

    // The same name with other bytes (a file re-exported under its old
    // name): the checksum, not the name, decides. It is stored beside the
    // saved file, which still loads.
    final reexported = litertlm(11);
    picker.modelFile = pick('second.litertlm', reexported);
    await vm.importFile.execute();
    expect(vm.candidateNotice, isNotNull);
    expect(vm.canApply, isTrue);
    expect(
      vm.candidate?.file?.name,
      ModelStore.renamedFor('second.litertlm', sha256Hex(reexported)),
    );
    expect(vm.draftName, 'second');
    expect(
      (chatModels.plan as CustomChatPlan).path,
      '${root.path}/custom/second.litertlm',
    );
  });

  test('Apply while the previous reply is still stopping: the card says so, '
      'the engine keeps running the old file, which is not pruned; Apply '
      'stays on and works once the reply has ended', () async {
    await build();
    picker.modelFile = pick('first.litertlm');
    await vm.importFile.execute();
    await vm.apply.execute();
    final first = File('${root.path}/custom/first.litertlm');
    picker.modelFile = pick('second.litertlm', litertlm(10));
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
    await build();
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
    expect(vm.candidate, isNotNull);
    expect(vm.saved, isNull, reason: 'not applied yet');
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
    expect((vm.candidate!.source as UrlModelSource).url, server.url(fileName));
    expect(logs, isNotEmpty);
    expect(logs.where((line) => line.contains('hf_secret')), isEmpty);
  });

  test("NPU unavailable here: the option is off with the app's NPU check's "
      'reason, and an NPU draft cannot be applied', () async {
    await build(
      npu: const NpuUnavailable('no NPU dispatch stack ships for macos'),
    );
    picker.modelFile = pick();
    await vm.importFile.execute();

    expect(vm.npuOffered, isFalse);
    expect(vm.npuUnavailableReason, contains('no NPU dispatch stack ships'));
    expect(
      vm.draftBackend,
      PreferredBackend.npu,
      reason: 'the file is compiled for the NPU',
    );
    expect(vm.draftProblem, contains('no NPU dispatch stack ships for macos'));
    expect(vm.canApply, isFalse);

    vm.setBackend(PreferredBackend.gpu);
    expect(
      vm.draftProblem,
      contains('GPU: not for this file'),
      reason: 'it runs only on the NPU: no way to apply it here',
    );
    expect(vm.canApply, isFalse);
  });

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
    // A file not compiled for an NPU, asked for the NPU.
    picker.modelFile = pick(
      fileName,
      fakeLitertlmBytes(FakeBundle.textOnly, filler: 48 * 1024, seed: 9),
    );
    await vm.importFile.execute();
    vm.setBackend(PreferredBackend.npu);
    llm.loadError = engineCreateFailure(
      modelName: 'gemma3_ekv1280_sm8750',
      backend: PreferredBackend.npu,
      // A string LiteRT's Qualcomm dispatch library carries.
      nativeMessage: 'Failed to set up QNN manager',
    );

    await vm.apply.execute();

    expect(models.states.value[ModelId.chat], isA<ModelFailed>());
    expect(vm.loadError, contains('did not load on npu'));
    expect(
      vm.loadError,
      contains('Failed to create engine on npu: Failed to set up QNN manager'),
    );
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
    expect(
      llm.models.last.llm.supportImage || llm.models.last.llm.supportAudio,
      isFalse,
      reason: 'a text-only build: no towers on any backend',
    );
    expect(vm.loadError, isNull);
    expect(vm.failureActions, isEmpty);
    expect(vm.draftBackend, PreferredBackend.gpu);
  });

  group('an NPU build an earlier build saved on the GPU (its old default)', () {
    const npuFile = 'gemma-4-E2B-it_qualcomm_sm8750.litertlm';

    /// The saved record of [version] for [path] on the GPU.
    String record(int version, String path, int size) {
      final towers = version == 3
          ? '"supportImage":true,"supportAudio":true,"encoderBackend":"cpu",'
          : '"visionChoice":"cpu","audioChoice":"cpu",';
      return '{"v":$version,"displayName":"sm8750","modelType":"gemma4",'
          '"backend":"gpu","maxTokens":4096,$towers"tools":false,'
          '"source":{"kind":"local","path":"$path"},'
          '"file":{"name":"$npuFile","sizeBytes":$size,'
          '"checksumMatched":false}}';
    }

    /// The plugin's refusal of an NPU build asked for the GPU
    /// (`planLiteRtLmLoad`'s wrongBackend message).
    ChatModelLoadException wrongBackend(String path) => ChatModelLoadException(
      modelName: 'sm8750',
      requested: PreferredBackend.gpu,
      cause: LiteRtLmBundleException(
        LiteRtLmBundleProblem.wrongBackend,
        '$path is compiled for the Qualcomm SM8750 NPU and runs only there, '
        'not on gpu. Load it with preferredBackend: PreferredBackend.npu, or '
        'omit preferredBackend; to run on the gpu, install the generic bundle '
        'of this model.',
        path: path,
      ),
    );

    /// Builds the card over [version]'s record and runs setup once: the
    /// load the plugin refuses.
    Future<String> launch(int version, {NpuAvailability? npu}) async {
      final file = File(pick(npuFile, litertlm(4)));
      final path = file.path;
      // The official SM8750 build's shape: both towers, the encoders not
      // compiled for the NPU.
      file.writeAsBytesSync(
        fakeLitertlmBytes(FakeBundle.officialNpu, filler: 4096),
      );
      await build(
        npu: npu ?? const NpuAvailable(soc: 'QTI SM8750'),
        prepare: false,
        seed: (settings) async {
          await settings.write(Settings.chatModel, ChatModelKind.custom.name);
          await settings.write(
            Settings.customChatModel,
            record(version, path, file.lengthSync()),
          );
        },
      );
      llm.loadError = wrongBackend(path);
      expect(await models.prepareAll(), isA<Error<void>>());
      return path;
    }

    for (final version in [3, 4]) {
      test('version $version: the load fails with wrongBackend, not '
          'retryable; the card offers Run on NPU (what the plugin '
          'advises), which loads it', () async {
        final path = await launch(version);

        final failed = models.states.value[ModelId.chat]! as ModelFailed;
        expect(failed.bundleProblem, LiteRtLmBundleProblem.wrongBackend);
        expect(failed.retryable, isFalse, reason: 'the same load: refused');
        expect(
          vm.loadError,
          'sm8750 did not load on gpu: $path is compiled for the Qualcomm '
          'SM8750 NPU and runs only there, not on gpu. Load it with '
          'preferredBackend: PreferredBackend.npu, or omit preferredBackend; '
          'to run on the gpu, install the generic bundle of this model. '
          '(wrongBackend)',
        );
        expect(vm.failureActions, [ChatModelAction.runOnNpu]);
        expect(vm.draftBackend, PreferredBackend.gpu, reason: 'as saved');
        expect(vm.draftProblem, contains('GPU: not for this file'));

        vm.runAction(ChatModelAction.runOnNpu);
        await untilNotified(
          vm.runOn,
          () => vm.runOn.completed || vm.runOn.error,
          what: 'Run on NPU',
        );

        expect(vm.runOn.completed, isTrue, reason: '${vm.runOn.result}');
        final loaded = llm.models.last.llm;
        expect(loaded.backend, PreferredBackend.npu);
        expect(loaded.maxTokens, 4096, reason: 'the saved context kept');
        expect(loaded.supportImage && loaded.supportAudio, isTrue);
        expect(models.states.value[ModelId.chat], isA<ModelReady>());
        expect(vm.loadError, isNull);
        expect(vm.failureActions, isEmpty);
        expect(vm.saved?.backend, PreferredBackend.npu, reason: 'saved');
        expect(vm.draftBackend, PreferredBackend.npu);
        expect(vm.draftProblem, isNull);
      });
    }

    test('where the NPU is not offered: no action (no backend here runs '
        'it), and the card says why', () async {
      await launch(
        4,
        npu: const NpuUnavailable('no NPU dispatch stack ships for macos'),
      );

      expect(vm.loadError, contains('(wrongBackend)'));
      expect(vm.failureActions, isEmpty);
      expect(vm.npuUnavailableReason, contains('no NPU dispatch stack'));
      expect(vm.gpuCpuUnavailableReason, contains('runs only on the NPU'));
    });
  });

  for (final problem in [
    LiteRtLmBundleProblem.incomplete,
    LiteRtLmBundleProblem.unsupportedNpu,
  ]) {
    test("the plugin's refusal before an engine as ${problem.name} is not "
        'retryable and offers no backend', () async {
      await build();
      await useInPlace('gemma-4-E2B-it.litertlm', FakeBundle.gemma4E2b);
      llm.loadError = ChatModelLoadException(
        modelName: 'gemma-4-E2B-it',
        requested: PreferredBackend.gpu,
        cause: LiteRtLmBundleException(problem, 'refused (${problem.name})'),
      );

      await vm.apply.execute();

      final failed = models.states.value[ModelId.chat]! as ModelFailed;
      expect(failed.bundleProblem, problem);
      expect(failed.retryable, isFalse);
      expect(vm.failureActions, isEmpty);
    });
  }

  test('the draft follows the file shown: a candidate picked elsewhere (a '
      'download that ended after its screen closed) starts from its own '
      'settings; dropped, the draft is the saved model again', () async {
    await build();
    await useInPlace('gemma-4-E2B-it.litertlm', FakeBundle.gemma4E2b);
    await vm.apply.execute();
    expect(vm.draftBackend, PreferredBackend.gpu);
    expect(vm.draftContext, '8192');

    final picked = await chatModels.useLocalFile(
      pick('npu.litertlm', litertlm(12)),
    );
    expect(picked, isA<Ok<CustomChatModel>>());
    expect(vm.draftBackend, PreferredBackend.npu);
    expect(vm.draftContext, '4096');
    expect(vm.canApply, isTrue);

    await chatModels.dropInPlaceCandidate();
    expect(vm.candidate, isNull);
    expect(vm.draftBackend, PreferredBackend.gpu);
    expect(vm.draftContext, '8192');
    expect(vm.canApply, isFalse);
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
    'the models folder: listed with Rescan; a file tapped there is used '
    'in place and becomes selected; Use this model loads that path',
    () async {
      await build(npu: const NpuAvailable(soc: 'QTI SM8750'));
      await listed();
      expect(vm.localFiles, isEmpty);
      final folder = Directory(vm.folderPath!);
      final file = File('${folder.path}/gemma4_2b_npu.litertlm')
        ..writeAsBytesSync(bytes);
      // Two files: none is used by itself, the user taps one.
      File('${folder.path}/other.litertlm').writeAsBytesSync(litertlm(3));

      await vm.rescan.execute();
      expect(vm.localFiles.map((e) => e.name).toSet(), {
        'gemma4_2b_npu.litertlm',
        'other.litertlm',
      });
      final entry = vm.localFiles.firstWhere((e) => e.path == file.path);
      expect(
        ChatModelViewModel.entryLine(entry),
        matches(RegExp(r'^0\.\d MB · \d{4}-\d\d-\d\d \d\d:\d\d$')),
      );
      await vm.useLocal.execute(file.path);

      expect(vm.useLocal.completed, isTrue, reason: '${vm.localError}');
      expect(vm.isSelected(entry), isTrue);
      expect(vm.candidate, isNotNull, reason: 'tapped: Apply loads it');
      expect(llm.models, isEmpty);
      expect(vm.fileLine, contains('in place: ${file.path}'));
      expect(vm.draftBackend, PreferredBackend.npu);
      expect(vm.towerRow(Tower.vision)?.inFile, isFalse);
      expect(vm.draftContext, '4096');

      await vm.apply.execute();
      expect(vm.apply.completed, isTrue, reason: '${vm.apply.result}');
      expect(llm.installs.last, file.path, reason: 'loaded where it is');
      expect(llm.models.last.llm.backend, PreferredBackend.npu);
    },
  );

  group('nothing chosen and one file already in the models folder', () {
    late List<int> npuFile;

    setUp(() => npuFile = bytes);

    File putInFolder(String name) {
      final folder = Directory('${root.path}/models_folder')
        ..createSync(recursive: true);
      return File('${folder.path}/$name')..writeAsBytesSync(npuFile);
    }

    test(
      'a desktop or a board (auto-apply off): the file is listed, '
      'nothing is picked or loaded until the user taps it and applies',
      () async {
        putInFolder('gemma4_2b_npu.litertlm');
        await build(
          npu: const NpuAvailable(soc: 'QTI SM8750'),
          autoApplyTheOnlyFile: false,
        );
        await listed();

        expect(vm.localFiles, hasLength(1));
        expect(vm.useLocal.result, isNull, reason: 'not picked by itself');
        expect(vm.candidate, isNull);
        expect(vm.active, ChatModelKind.none);
        expect(llm.installs, isEmpty);
      },
    );

    test('nothing applied yet: it is used at start with its own settings, '
        'saved and loaded (an adb push is all a tester does); the next visit '
        'neither picks nor hashes it again', () async {
      final file = putInFolder('gemma4_2b_npu.litertlm');
      await build(npu: const NpuAvailable(soc: 'QTI SM8750'));
      await listed();

      expect(vm.useLocal.completed, isTrue, reason: '${vm.localError}');
      expect(vm.isSelected(vm.localFiles.single), isTrue);
      expect(vm.candidate, isNull, reason: 'applied at once');
      expect(vm.active, ChatModelKind.custom);
      expect(llm.installs.last, file.path);
      expect(llm.models.last.llm.backend, PreferredBackend.npu);
      expect(llm.models.last.llm.maxTokens, 4096);
      expect(vm.canApply, isFalse);
      await untilNotified(
        chatModels.state,
        () => chatModels.state.value.custom?.file?.sha256 != null,
        what: 'the saved file\'s SHA-256',
      );

      final card = await reopened();
      expect(card.useLocal.result, isNull, reason: 'not picked again');
      expect(chatModels.hashing.value, isFalse, reason: 'not hashed again');
      expect(llm.installs, hasLength(1));
    });

    test('nothing applied yet and no NPU here: a build compiled for the NPU '
        'starts on the NPU (the plugin runs it nowhere else), so it is picked '
        'but not applied, the card says why, and nothing loads', () async {
      final file = putInFolder('gemma4_2b_npu.litertlm');
      await build(npu: const NpuUnavailable('no FastRPC'));
      await listed();

      expect(vm.isSelected(vm.localFiles.single), isTrue);
      expect(vm.candidate?.sourceLine, 'in place: ${file.path}');
      expect(vm.draftBackend, PreferredBackend.npu);
      expect(vm.draftProblem, 'The NPU is not available here: no FastRPC.');
      expect(vm.active, ChatModelKind.none);
      expect(llm.installs, isEmpty);
    });

    test('nothing applied yet and no NPU here: a file not compiled for the '
        'NPU is used with its defaults (the GPU)', () async {
      final folder = Directory('${root.path}/models_folder')
        ..createSync(recursive: true);
      final file = writeFakeLitertlm(
        '${folder.path}/gemma-4-E2B-it.litertlm',
        FakeBundle.gemma4E2b,
      );
      await build(npu: const NpuUnavailable('no FastRPC'));
      await listed();

      expect(vm.candidate, isNull, reason: 'applied at once');
      expect(vm.active, ChatModelKind.custom);
      expect(llm.installs.last, file.path);
      expect(llm.models.last.llm.backend, PreferredBackend.gpu);
    });

    test('a file added later is selected by Rescan', () async {
      await build();
      await listed();
      expect(vm.saved, isNull);

      final file = putInFolder('gemma4_2b_npu.litertlm');
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
      putInFolder('gemma4_2b_npu.litertlm');
      await build(gemmaModelPath: pick('define.litertlm'), prepare: false);
      await listed();

      expect(vm.saved, isNull);
      expect(vm.useLocal.result, isNull);
    });

    test('a saved choice that cannot load, its file now another size: the '
        'only file is not used by itself (a re-pushed build keeps its '
        'settings until the user applies)', () async {
      final file = putInFolder('gemma4_2b_npu.litertlm');
      await build(npu: const NpuAvailable(soc: 'QTI SM8750'));
      await listed();
      expect(vm.active, ChatModelKind.custom, reason: 'used at first');
      final chosen = vm.saved!.copyWith(displayName: 'Mine', maxTokens: 2048);
      expect(await chatModels.apply(chosen), isA<Ok<void>>());
      final installs = llm.installs.length;

      file.writeAsBytesSync([...npuFile, 0, 0]);
      expect(chatModels.plan, isA<ChatPlanBlocked>(), reason: 'changed');
      await vm.rescan.execute();

      expect(vm.candidate, isNull, reason: 'not picked by itself');
      expect(vm.saved!.file!.sizeBytes, npuFile.length, reason: 'as saved');
      expect(vm.saved!.displayName, 'Mine', reason: 'kept');
      expect(vm.saved!.maxTokens, 2048);
      expect(llm.installs, hasLength(installs));
    });

    test('Rescan while a download runs selects nothing: the download keeps '
        'its own settings', () async {
      await build(npu: const NpuAvailable(soc: 'QTI SM8750'));
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

      putInFolder('gemma4_2b_npu.litertlm');
      await vm.rescan.execute();
      expect(vm.useLocal.result, isNull, reason: 'not while downloading');

      await downloading;
      expect(vm.download.completed, isTrue, reason: '${vm.transferError}');
      expect(vm.candidate?.file?.name, fileName);
      expect(vm.candidate?.maxTokens, 1280, reason: 'from its own name');
    });

    test('a saved choice that cannot load (blocked) is not replaced, and '
        'nothing loads by itself', () async {
      putInFolder('gemma4_2b_npu.litertlm');
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
      putInFolder('gemma4_2b_npu.litertlm');
      await build(
        prepare: false,
        seed: (settings) =>
            settings.write(Settings.customChatModel, '{not json'),
      );
      await listed();

      expect(vm.problem, isNotNull);
      expect(vm.useLocal.result, isNull);
    });

    test('the only file cannot be used (a load refuses it: incomplete): its '
        'reason shows, nothing is selected', () async {
      final folder = Directory('${root.path}/models_folder')
        ..createSync(recursive: true);
      File('${folder.path}/cut.litertlm')
          .writeAsBytesSync(fakeLitertlmBytes(FakeBundle.incomplete));
      await build();
      await listed();

      expect(vm.localError, contains('is incomplete'));
      expect(vm.saved, isNull);
      expect(vm.candidate, isNull);
    });

    test('a model chosen before stays chosen', () async {
      await build();
      picker.modelFile = pick();
      await vm.importFile.execute();
      expect(vm.importFile.completed, isTrue);

      putInFolder('gemma4_2b_npu.litertlm');
      await vm.rescan.execute();

      expect(vm.fileLine, contains(fileName));
      expect(vm.useLocal.result, isNull);
    });
  });

  test('a path whose header the plugin cannot read is no refusal (the engine '
      "loads such a file): the card shows the plugin's reason, no tower "
      'rows, and it can be applied, text only', () async {
    await build();
    await listed();
    final bogus = File('${picked.path}/notes.litertlm')
      ..writeAsStringSync('hello');

    await vm.useLocal.execute(bogus.path);

    expect(vm.localError, isNull);
    expect(vm.candidate, isNotNull);
    expect(vm.towersShown, isTrue);
    expect(
      vm.headerNotRead,
      'The plugin could not read its header (notLitertlm): ${bogus.path} is '
      'not a .litertlm container (no LITERTLM magic). The plugin loads such a '
      'file without reading its header; the app asks for no tower (none is '
      'known), so it loads text only.',
    );
    expect(vm.towerRow(Tower.vision), isNull);
    expect(vm.npuBuildLine, isNull);
    expect(vm.draftBackend, PreferredBackend.gpu);
    expect(vm.canApply, isTrue);

    await vm.apply.execute();

    expect(vm.apply.completed, isTrue, reason: '${vm.apply.result}');
    final loaded = llm.models.last.llm;
    expect(loaded.supportImage || loaded.supportAudio, isFalse);
  });
}
