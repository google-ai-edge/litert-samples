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

import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/repositories/chat_model_repository.dart';
import 'package:litert_edge_demos/data/repositories/model_repository.dart';
import 'package:litert_edge_demos/data/repositories/provisioning_repository.dart';
import 'package:litert_edge_demos/data/services/hardware/native_log_tap.dart';
import 'package:litert_edge_demos/data/services/knowledge/embedder_service.dart';
import 'package:litert_edge_demos/data/services/model_store/model_store.dart';
import 'package:litert_edge_demos/data/services/model_store/models_folder.dart';
import 'package:litert_edge_demos/data/services/settings/typed_settings.dart';
import 'package:litert_edge_demos/domain/models/chat_model.dart';
import 'package:litert_edge_demos/domain/models/detector_spec.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/domain/models/provisioning.dart';
import 'package:litert_edge_demos/domain/models/self_test.dart';
import 'package:litert_edge_demos/domain/ports/chat_model_planner.dart';
import 'package:litert_edge_demos/domain/use_cases/chat_model_switcher.dart';
import 'package:litert_edge_demos/selftest/self_test_in_app.dart';
import 'package:litert_edge_demos/ui/features/setup/view_models/setup_view_model.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../../../fakes/fake_audio_repository.dart';
import '../../../../fakes/fake_bundled_files.dart';
import '../../../../fakes/fake_conversation_repository.dart';
import '../../../../fakes/fake_detector_runtime.dart';
import '../../../../fakes/fake_llm_service.dart';
import '../../../../fakes/fake_settings_store.dart';
import '../../../../fakes/fake_speech.dart';
import '../../../../support/fake_model_server.dart';
import '../../../../support/test_bytes.dart';
import '../../../../support/until.dart';

/// The self-test's steps are replaced; its planner is never asked.
final class _NoChatPlan implements ChatModelPlanner {
  @override
  ChatModelPlan get plan => const NoChatModelPlan();
}

/// The Models screen's view model over a real provisioning repository and
/// model store (temp folder, local HTTP server) and the fake model services.
void main() {
  late Directory root;
  late FakeModelServer server;
  late FakeLlmService llm;

  /// GEMMA_MODEL_PATH (a developer define): a chat model present without
  /// the Chat model card, for the tests about the other models.
  const gemmaPath = '/dev/gemma-4-E2B-it.litertlm';

  setUp(() async {
    root = Directory.systemTemp.createTempSync('setup_vm_test');
    server = await FakeModelServer.start();
    llm = FakeLlmService();
  });

  tearDown(() async {
    await server.close();
    root.deleteSync(recursive: true);
  });

  ({SetupViewModel vm, ModelRepository models, ProvisioningRepository repo})
  build({
    SetupMode mode = SetupMode.firstRun,
    ModelRepository? models,
    ChatModelRepository? chatModels,
    ChatModelSwitcher? switcher,
    String gemmaModelPath = gemmaPath,
    ModelStore? store,
  }) {
    final modelStore =
        store ??
        ModelStore(
          root: () async => root,
          retryDelay: (_) => Duration.zero,
          progressInterval: Duration.zero,
        );
    final repo = ProvisioningRepository(
      store: modelStore,
      gemmaModelPath: gemmaModelPath,
      chatModels: chatModels,
    );
    final modelRepo =
        models ??
        ModelRepository(
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
    final vm = SetupViewModel(
      models: modelRepo,
      prepareModels: modelRepo.prepareAll,
      provisioning: repo,
      mode: mode,
      chatModels: chatModels,
      switcher: switcher,
    );
    addTearDown(() async {
      vm.dispose();
      await modelRepo.close();
      await modelStore.close();
    });
    return (vm: vm, models: modelRepo, repo: repo);
  }

  /// Waits until [dir] is gone. The setup prunes old folders in the
  /// background (fire and forget) with nothing to listen to, so this polls
  /// the file system, bounded.
  Future<void> untilDeleted(Directory dir) async {
    final watch = Stopwatch()..start();
    while (dir.existsSync()) {
      if (watch.elapsed > const Duration(seconds: 10)) {
        fail('${dir.path} is still there after 10 s');
      }
      await Future<void>.delayed(const Duration(milliseconds: 10));
    }
  }

  ModelRow rowOf(SetupViewModel vm, ModelId id) =>
      vm.rows.firstWhere((r) => r.spec.id == id);

  /// A `.litertlm` with a real header (Gemma 4 E2B for the GPU), in [dir].
  File litertlm(Directory dir, String name) => File('${dir.path}/$name')
    ..writeAsBytesSync([
      ...File('test_assets/litertlm/gemma4_e2b_gpu.header.bin')
          .readAsBytesSync(),
      ...testBytes(4096, seed: 21),
    ]);

  test('no chat model chosen (the app ships none): every built-in model '
      'loads, the chat row says how to choose one; choosing one by a reload '
      'finishes the first run', () async {
    final chatStore = ModelStore(root: () async => root);
    addTearDown(chatStore.close);
    final chatModels = ChatModelRepository(
      settings: TypedSettings(store: InMemorySettingsStore()),
      store: chatStore,
      folders: [ModelsFolder(directory: () async => root)],
    );
    addTearDown(chatModels.dispose);
    await chatModels.load();
    final (:vm, :models, :repo) = build(
      chatModels: chatModels,
      gemmaModelPath: '',
    );

    await untilNotified(
      vm,
      () => vm.prepare.result != null,
      what: 'the built-in models',
    );

    final chat = rowOf(vm, ModelId.chat);
    expect(chat.title, 'Chat model');
    expect(chat.status, 'No chat model yet');
    expect(chat.detail, contains('Chat model card'));
    expect(chat.tone, Tone.warning);
    for (final id in [
      ModelId.whisperBase,
      ModelId.inflectNano,
      ModelId.moonshineTiny,
      ModelId.yolo26n,
    ]) {
      final row = rowOf(vm, id);
      expect(row.status, startsWith('Built in · Ready'), reason: '$id');
      expect(row.source, 'Part of the app: nothing to download');
      expect(row.bytes, greaterThan(0), reason: '$id: its size');
    }
    expect(rowOf(vm, ModelId.yolo26n).bytes, kDetModelBytes);
    expect(vm.allReady, isFalse);
    expect(vm.canContinue, isFalse, reason: 'the setup ran');
    expect(llm.installs, isEmpty);

    // The Chat model card: a file in place, applied, then its reload.
    final file = litertlm(root, 'gemma-4-E2B-it.litertlm');
    final picked = await chatModels.useLocalFile(file.path);
    expect(picked, isA<Ok<CustomChatModel>>());
    expect(
      await chatModels.apply((picked as Ok<CustomChatModel>).value),
      isA<Ok<void>>(),
    );
    expect(await models.reloadChatModel(), isA<Ok<void>>());

    await untilNotified(vm, () => vm.allReady, what: 'the hand-over');
    expect(rowOf(vm, ModelId.chat).title, 'Chat model · gemma-4-E2B-it');
    expect(llm.installs, [file.path]);
    expect(llm.loads, hasLength(1), reason: 'only the chat model loaded');
  });

  test('your own chat model chosen but without a file: the chat row says '
      'why it cannot load; no other model is loaded instead', () async {
    final prefs = InMemorySettingsStore()..values['chat.model'] = 'custom';
    final chatStore = ModelStore(root: () async => root);
    addTearDown(chatStore.close);
    final chatModels = ChatModelRepository(
      settings: TypedSettings(store: prefs),
      store: chatStore,
      folders: [ModelsFolder(directory: () async => root)],
    );
    addTearDown(chatModels.dispose);
    await chatModels.load();
    final (:vm, :models, :repo) = build(chatModels: chatModels);

    await untilNotified(
      vm,
      () => vm.prepare.result != null,
      what: 'the setup attempt',
    );

    final row = rowOf(vm, ModelId.chat);
    expect(row.title, 'Chat model · your own .litertlm');
    expect(row.status, 'Failed');
    expect(row.detail, contains('has no file'));
    expect(llm.installs, isEmpty, reason: 'nothing in its place');
    expect(vm.allReady, isFalse);
  });

  test('a chat model present (GEMMA_MODEL_PATH) and everything else built '
      'in: loads at once and is ready; then the folders earlier builds '
      'downloaded into go', () async {
    for (final name in [kRetiredGemmaFolder, 'whisperBase', 'custom']) {
      File('${root.path}/$name/f')
        ..createSync(recursive: true)
        ..writeAsStringSync('x');
    }
    final (:vm, :models, :repo) = build();
    // Before the setup starts (after the screen's first build): built-in
    // rows say so, and no Continue flashes up.
    expect(vm.start.running, isTrue);
    expect(vm.prepare.running, isFalse);
    expect(rowOf(vm, ModelId.yolo26n).status, 'Built in');
    expect(vm.canContinue, isFalse);

    await untilNotified(vm, () => vm.allReady, what: 'setup');
    await untilDeleted(Directory('${root.path}/$kRetiredGemmaFolder'));
    expect(Directory('${root.path}/whisperBase').existsSync(), isFalse);
    expect(Directory('${root.path}/custom').existsSync(), isTrue);

    expect(llm.installs, [gemmaPath]);
    expect(rowOf(vm, ModelId.chat).title, 'Chat model · Gemma 4 E2B');
    expect(rowOf(vm, ModelId.chat).source, contains('GEMMA_MODEL_PATH'));
    expect(rowOf(vm, ModelId.chat).status, startsWith('Ready · gpu'));
    final detector = rowOf(vm, ModelId.yolo26n);
    expect(detector.status, startsWith('Built in · Ready'));
    expect(detector.bytes, kDetModelBytes, reason: 'its size, not 0.0 MB');
    expect(models.states.value[ModelId.moonshineTiny], isA<ModelReady>());
  });

  test('leaving the screen does not stop the chat model card\'s download: it '
      'goes on and is adopted', () async {
    final served = Directory.systemTemp.createTempSync('setup_vm_served');
    addTearDown(() => served.deleteSync(recursive: true));
    final bytes = litertlm(served, 'mine.litertlm').readAsBytesSync();
    server.files['c'] = ServedFile(
      bytes,
      chunkBytes: 256,
      chunkDelay: const Duration(milliseconds: 5),
    );
    final store = ModelStore(
      root: () async => root,
      retryDelay: (_) => Duration.zero,
      progressInterval: Duration.zero,
    );
    final chatModels = ChatModelRepository(
      settings: TypedSettings(store: InMemorySettingsStore()),
      store: store,
      folders: [ModelsFolder(directory: () async => root)],
    );
    final models = ModelRepository(
      bundled: FakeBundledFiles(),
      detector: fakeDetectorService(),
      bundledDetector: fakeBundledDetector,
      gemmaModelPath: gemmaPath,
      llm: llm,
      stt: fakeSttService(),
      tts: fakeTtsService(),
      embedder: EmbedderService(),
    );
    addTearDown(() async {
      chatModels.dispose();
      await models.close();
      await store.close();
    });
    await chatModels.load();
    final vm = SetupViewModel(
      models: models,
      prepareModels: models.prepareAll,
      provisioning: ProvisioningRepository(
        store: store,
        gemmaModelPath: gemmaPath,
      ),
      mode: SetupMode.manage,
    );
    await untilNotified(
      vm,
      () => !vm.start.running,
      what: 'the screen to start',
    );

    final downloading = chatModels.download(
      server.url('c'),
      sizeBytes: bytes.length,
    );
    await untilNotified(
      store.customFile,
      () => switch (store.customFile.value) {
        StoreFileDownloading(received: > 0) => true,
        _ => false,
      },
      what: 'the custom download to start',
    );
    vm.dispose();
    final result = await downloading;

    expect(result, isA<Ok<CustomChatModel>>());
    expect(store.customFile.value, isA<StoreFileReady>());
  });

  test('while the self-test holds the chat model: opening the screen starts '
      'no setup, Continue comes once it ends; the setup itself runs as one '
      'exclusive operation', () async {
    final conversation = FakeConversationRepository();
    addTearDown(conversation.close);
    late final ModelRepository repoModels;
    final switcher = ChatModelSwitcher(
      conversation: conversation,
      reloadChatModel: () => repoModels.reloadChatModel(),
      unloadChatModel: () => repoModels.unloadChatModel(),
      refuseChatModelLoads: (reason) => repoModels.refuseChatModelLoads(reason),
    );
    final (:vm, :models, :repo) = build(switcher: switcher);
    repoModels = models;
    // Before the screen starts its setup (a microtask later).
    final gate = Completer<void>();
    final selfTest = switcher.exclusive((_) => gate.future);
    await untilNotified(
      vm,
      () => !vm.start.running,
      what: 'the screen to start',
    );
    // The screen's start decided against a setup synchronously (the chat
    // model is in use); whatever it scheduled has run once the event queue
    // is drained.
    await pumpEventQueue();
    expect(vm.prepare.running, isFalse);
    expect(vm.prepare.result, isNull);
    expect(llm.loads, isEmpty, reason: 'no engine work beside the self-test');
    expect(vm.canContinue, isFalse, reason: 'the chat model is in use');

    gate.complete();
    await selfTest;
    expect(vm.canContinue, isTrue);

    final preparing = vm.prepare.execute();
    expect(switcher.busy.value, isTrue, reason: 'setup holds the chat model');
    await preparing;
    expect(switcher.busy.value, isFalse);
    expect(models.states.value[ModelId.chat], isA<ModelReady>());
  });

  test('a self-test that never ended: re-entering the Models screen loads no '
      'chat model, its row says to restart without a Retry, and neither a '
      'reload nor a setup run loads it', () async {
    final conversation = FakeConversationRepository();
    addTearDown(conversation.close);
    late final ModelRepository repoModels;
    final switcher = ChatModelSwitcher(
      conversation: conversation,
      reloadChatModel: () => repoModels.reloadChatModel(),
      unloadChatModel: () => repoModels.unloadChatModel(),
      refuseChatModelLoads: (reason) => repoModels.refuseChatModelLoads(reason),
    );
    final (:vm, :models, :repo) = build(switcher: switcher);
    repoModels = models;
    await untilNotified(vm, () => vm.allReady, what: 'the first setup');
    expect(llm.loads, hasLength(1));

    final selfTest = InAppSelfTest(
      switcher: switcher,
      chatModels: _NoChatPlan(),
      models: models.states,
      logTap: const NoNativeLogTap(),
      audio: FakeAudioRepository(),
      steps: (_, _) async => const SelfTestOutcome(
        text: 'SELFTEST',
        passed: false,
        stillRunning: true,
      ),
    );
    final run = await selfTest.run(progress: (_) {});
    expect((run as Ok<SelfTestOutcome>).value.stillRunning, isTrue);

    // Back to home and into the Models screen again.
    final again = SetupViewModel(
      models: models,
      prepareModels: models.prepareAll,
      provisioning: repo,
      mode: SetupMode.manage,
      switcher: switcher,
    );
    addTearDown(again.dispose);
    await untilNotified(
      again,
      () => !again.start.running && !again.prepare.running,
      what: 'the screen to settle',
    );
    expect(llm.loads, hasLength(1), reason: 'the self-test\'s engine may live');
    final row = rowOf(again, ModelId.chat);
    expect(row.status, 'Failed');
    expect(row.detail, kSelfTestStillRunning);
    expect(row.canRetryLoad, isFalse);

    // Apply and reload (or Run on GPU/CPU), and a setup run: refused.
    expect(await switcher.reload(), isA<Error<void>>());
    expect(
      await switcher.exclusive((_) => models.prepareAll()),
      isA<Error<void>>(),
    );
    expect(llm.loads, hasLength(1));
    expect(rowOf(again, ModelId.chat).detail, kSelfTestStillRunning);
  });

  test('a load failure keeps its Retry and is not retried by itself', () async {
    llm.activeBackend = null; // the engine reports no backend
    final (:vm, :models, :repo) = build();

    await untilNotified(
      vm,
      () => vm.prepare.result != null,
      what: 'setup to fail',
    );
    expect(vm.prepare.result, isA<Error<void>>());
    final row = rowOf(vm, ModelId.chat);
    expect(row.status, 'Failed');
    expect(row.canRetryLoad, isTrue);
    expect(row.detail, contains('Fallback is disabled'));
    expect(llm.loads, hasLength(1));
  });
}
