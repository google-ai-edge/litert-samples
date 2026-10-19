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

import 'package:flutter_edge_ai/flutter_edge_ai.dart'
    show ModelType, PreferredBackend;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/repositories/chat_model/custom_chat_model_codec.dart';
import 'package:litert_edge_demos/data/repositories/chat_model_repository.dart';
import 'package:litert_edge_demos/data/repositories/provisioning_repository.dart';
import 'package:litert_edge_demos/data/services/model_store/model_file_ops.dart';
import 'package:litert_edge_demos/data/services/model_store/model_store.dart';
import 'package:litert_edge_demos/data/services/model_store/models_folder.dart';
import 'package:litert_edge_demos/data/services/settings/settings_store.dart';
import 'package:litert_edge_demos/data/services/settings/typed_settings.dart';
import 'package:litert_edge_demos/domain/models/chat_model.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/npu_availability.dart';
import 'package:litert_edge_demos/domain/models/provisioning.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../fakes/fake_settings_store.dart';
import '../../support/fake_bundles.dart';
import '../../support/fake_model_server.dart';
import '../../support/test_bytes.dart';
import '../../support/until.dart';

/// A SHA-256 that ends when the test completes [gate].
final class _GatedHashOps extends ModelFileOps {
  _GatedHashOps(this.gate);

  final Completer<void> gate;
  final Completer<void> started = Completer();
  static final hex = 'ab' * 32;

  @override
  Future<String> sha256OfFile(
    String path, {
    required ByteProgress onProgress,
    Future<void>? cancel,
  }) async {
    if (!started.isCompleted) started.complete();
    await gate.future;
    return hex;
  }
}

/// [InMemorySettingsStore] whose next write of [holdKey] waits for [gate]
/// ([held] completes when it does).
final class _HeldWrites implements SettingsStore {
  _HeldWrites(this._inner);

  final InMemorySettingsStore _inner;
  String? holdKey;
  Completer<void>? gate;
  final Completer<void> held = Completer();

  @override
  Future<String?> getString(String key) => _inner.getString(key);

  @override
  Future<bool?> getBool(String key) => _inner.getBool(key);

  @override
  Future<void> setString(String key, String value) async {
    if (key == holdKey && gate != null) {
      final wait = gate!.future;
      gate = null;
      held.complete();
      await wait;
    }
    await _inner.setString(key, value);
  }

  @override
  Future<void> setBool(String key, {required bool value}) =>
      _inner.setBool(key, value: value);

  @override
  Future<void> remove(String key) => _inner.remove(key);
}

void main() {
  late Directory root;
  late Directory picked;
  late InMemorySettingsStore prefs;
  final stores = <ModelStore>[];
  final repos = <ChatModelRepository>[];

  const fileName = 'Gemma3-1B-IT_q4_ekv1280_sm8750.litertlm';
  // A text-only NPU build: compiled for the NPU (as the name says), no
  // towers.
  final bytes = fakeLitertlmBytes(
    FakeBundle.textOnlyNpu,
    filler: 64 * 1024,
    seed: 5,
  );

  setUp(() {
    root = Directory.systemTemp.createTempSync('chat_model_root');
    picked = Directory.systemTemp.createTempSync('chat_model_picked');
    prefs = InMemorySettingsStore();
  });

  tearDown(() async {
    for (final repo in repos) {
      repo.dispose();
    }
    repos.clear();
    for (final store in stores) {
      await store.close();
    }
    stores.clear();
    root.deleteSync(recursive: true);
    picked.deleteSync(recursive: true);
  });

  ChatModelRepository newRepo({
    NpuAvailability npu = const NpuUnavailable(
      'no NPU dispatch stack ships for macos',
    ),
    bool persistMigration = true,
  }) {
    final store = ModelStore(root: () async => root);
    stores.add(store);
    final repo = ChatModelRepository(
      settings: TypedSettings(store: prefs),
      readBundle: fakeReadBundle,
      store: store,
      persistMigration: persistMigration,
      npu: () => npu,
      folders: [
        ModelsFolder(
          directory: () async => Directory('${root.path}/m'),
          permissionAdvice: 'Push after the first launch.',
        ),
        ModelsFolder(
          directory: () async => Directory('${root.path}/tmp-models'),
          create: false,
        ),
      ],
    );
    repos.add(repo);
    return repo;
  }

  String pick([String name = fileName]) =>
      (File('${picked.path}/$name')..writeAsBytesSync(bytes)).path;

  /// The file picked and not applied, with the settings it starts from.
  CustomChatModel candidateOf(ChatModelRepository repo) =>
      repo.state.value.candidate!.model;

  /// What its file says about itself, read when it was picked.
  LiteRtLmBundleInfo candidateBundle(ChatModelRepository repo) =>
      repo.state.value.candidate!.bundle.info!;

  group('a .litertlm used in place (the models folder)', () {
    File inPlace(String name, FakeBundle kind) =>
        writeFakeLitertlm('${picked.path}/$name', kind);

    /// The background SHA-256 of the saved model's file in place (a
    /// candidate is never hashed).
    Future<void> hashed(ChatModelRepository repo) => untilNotified(
      repo.state,
      () => repo.state.value.custom?.file?.sha256 != null,
      what: 'the background SHA-256',
    );

    test('used where it is: no copy into the store; a build compiled for the '
        'NPU defaults to the NPU, with no towers in the file; the SHA-256 '
        'follows in the background', () async {
      final repo = newRepo(npu: const NpuAvailable(soc: 'QTI SM8750'));
      await repo.load();
      final file = inPlace('gemma4_2b_npu.litertlm', FakeBundle.textOnlyNpu);

      final result = await repo.useLocalFile(file.path);

      final model = (result as Ok<CustomChatModel>).value;
      expect(model.source, isA<LocalModelSource>());
      expect((model.source as LocalModelSource).path, file.path);
      expect(model.backend, PreferredBackend.npu);
      final bundle = candidateBundle(repo);
      expect(bundle.isCompiledForNpu, isTrue);
      expect(
        bundle.visionEncoder ?? bundle.audioEncoder,
        isNull,
        reason: 'no vision or audio encoder',
      );
      expect(model.maxTokens, 4096);
      expect(model.tools, isFalse, reason: 'an NPU build: off until known');
      expect(model.file?.sha256, isNull, reason: 'never blocks');
      expect(
        Directory('${root.path}/custom').existsSync()
            ? Directory('${root.path}/custom').listSync()
            : const <FileSystemEntity>[],
        isEmpty,
        reason: 'nothing copied',
      );
      await pumpEventQueue();
      expect(
        candidateOf(repo).file!.sha256,
        isNull,
        reason: 'a file only looked at is not read whole',
      );
      expect(repo.hashing.value, isFalse);
      expect(prefs.values, isEmpty, reason: 'a pick saves nothing');

      expect(await repo.apply(candidateOf(repo)), isA<Ok<void>>());
      expect(repo.state.value.candidate, isNull, reason: 'applied');
      await hashed(repo);
      expect(
        repo.state.value.custom!.file!.sha256,
        sha256Hex(file.readAsBytesSync()),
      );
      final plan = repo.plan as CustomChatPlan;
      expect(plan.path, file.path);
      expect(plan.config.llm.backend, PreferredBackend.npu);
      expect(plan.config.llm.supportImage, isFalse);
      expect(plan.config.llm.supportAudio, isFalse);
    });

    group('the saved file is gone from its path', () {
      Future<File> chosenThenMoved({required List<String> into}) async {
        final repo = newRepo();
        await repo.load();
        final file = inPlace('gemma-4-E2B-it.litertlm', FakeBundle.gemma4E2b);
        await repo.useLocalFile(file.path);
        expect(await repo.apply(candidateOf(repo)), isA<Ok<void>>());
        await hashed(repo);
        File? last;
        for (final dir in into) {
          Directory(dir).createSync(recursive: true);
          last = file.copySync('$dir/gemma-4-E2B-it.litertlm');
        }
        file.deleteSync();
        return last!;
      }

      test(
        'the same name and size in a models folder (iOS gives a '
        'reinstalled app a new folder path): used there, and saved',
        () async {
          final moved = await chosenThenMoved(into: ['${root.path}/m']);

          final relaunched = newRepo();
          expect(await relaunched.load(), isA<Ok<void>>());

          expect((relaunched.plan as CustomChatPlan).path, moved.path);
          final stored = CustomChatModelCodec.decode(
            prefs.values[Settings.customChatModel.key]! as String,
          );
          expect((stored.source as LocalModelSource).path, moved.path);
        },
      );

      test('two candidates (both folders): none is guessed', () async {
        await chosenThenMoved(
          into: ['${root.path}/m', '${root.path}/tmp-models'],
        );

        final relaunched = newRepo();
        await relaunched.load();

        expect(
          (relaunched.plan as ChatPlanBlocked).reason,
          contains('is not there any more'),
        );
      });

      test('the same name at another size is another file', () async {
        final moved = await chosenThenMoved(into: ['${root.path}/m']);
        moved.writeAsBytesSync([...moved.readAsBytesSync(), 0]);

        final relaunched = newRepo();
        await relaunched.load();

        expect(
          (relaunched.plan as ChatPlanBlocked).reason,
          contains('is not there any more'),
        );
      });
    });

    test('a file with towers not compiled for the NPU: the GPU (also where '
        'the NPU is offered), both towers loaded, tools on', () async {
      final repo = newRepo(npu: const NpuAvailable());
      await repo.load();

      final result = await repo.useLocalFile(
        inPlace('gemma-4-E2B-it.litertlm', FakeBundle.gemma4E2b).path,
      );

      final model = (result as Ok<CustomChatModel>).value;
      expect(model.backend, PreferredBackend.gpu);
      final bundle = candidateBundle(repo);
      expect(bundle.visionEncoder, isNotNull);
      expect(bundle.audioEncoder, isNotNull);
      expect(model.maxTokens, 8192, reason: 'the app\'s GPU context');
      expect(model.tools, isTrue, reason: 'not an NPU build: skills work');
      expect(await repo.apply(model), isA<Ok<void>>());
      final llm = (repo.plan as CustomChatPlan).config.llm;
      expect(llm.supportImage && llm.supportAudio, isTrue);
    });

    test('the towers follow the file: another file starts from its own '
        'defaults, never from the settings of the previous one; one without '
        'towers loads none', () async {
      final repo = newRepo(npu: const NpuAvailable(soc: 'QTI SM8750'));
      await repo.load();
      final gemma = (await repo.useLocalFile(
        inPlace('gemma-4-E2B-it.litertlm', FakeBundle.gemma4E2b).path,
      ) as Ok<CustomChatModel>).value;
      expect(
        await repo.apply(gemma.copyWith(displayName: 'Mine', maxTokens: 2048)),
        isA<Ok<void>>(),
      );
      final before = (repo.plan as CustomChatPlan).config.llm;
      expect(before.supportImage && before.supportAudio, isTrue);

      final npu = (await repo.useLocalFile(
        inPlace('mine_sm8850.litertlm', FakeBundle.textOnlyNpu).path,
      ) as Ok<CustomChatModel>).value;

      expect(npu.backend, PreferredBackend.npu, reason: 'compiled for the NPU');
      expect(npu.maxTokens, 4096, reason: 'not the GPU file\'s 2048');
      expect(npu.displayName, 'mine_sm8850');
      expect(npu.tools, isFalse);
      final still = (repo.plan as CustomChatPlan).config;
      expect(still.name, 'Mine', reason: 'only picked: Gemma still loads');
      expect(still.llm.maxTokens, 2048);
      expect(await repo.apply(npu), isA<Ok<void>>());
      final after = (repo.plan as CustomChatPlan).config.llm;
      expect(after.backend, PreferredBackend.npu);
      expect(after.supportImage || after.supportAudio, isFalse);
    });

    group('a file picked and not applied (the candidate)', () {
      test('nothing is saved and nothing loads it: the plan, the next '
          'reload, Run on GPU and a restart keep the applied model', () async {
        final repo = newRepo(npu: const NpuAvailable(soc: 'QTI SM8750'));
        await repo.load();
        final gemmaFile = inPlace(
          'gemma-4-E2B-it.litertlm',
          FakeBundle.gemma4E2b,
        );
        final gemma = (await repo.useLocalFile(
          gemmaFile.path,
        ) as Ok<CustomChatModel>).value;
        expect(await repo.apply(gemma), isA<Ok<void>>());
        await hashed(repo);
        final savedBefore = Map.of(prefs.values);

        // Picked just to see its towers: an NPU-only build.
        final npuFile = inPlace(
          'gemma4_2b_npu.litertlm',
          FakeBundle.textOnlyNpu,
        );
        final npu = (await repo.useLocalFile(
          npuFile.path,
        ) as Ok<CustomChatModel>).value;

        final state = repo.state.value;
        expect(state.candidate?.model.sourceLine, npu.sourceLine);
        expect(state.candidate?.bundle.info?.isCompiledForNpu, isTrue);
        expect(state.custom?.sourceLine, gemma.sourceLine);
        expect(prefs.values, savedBefore, reason: 'nothing saved');
        final plan = repo.plan as CustomChatPlan;
        expect(plan.path, gemmaFile.path);
        expect(plan.config.llm.backend, PreferredBackend.gpu);
        expect(plan.config.llm.supportAudio, isTrue, reason: "Gemma's file");

        // "Run on CPU" acts on the applied model, the candidate stays.
        expect(await repo.runOn(PreferredBackend.cpu), isA<Ok<void>>());
        expect((repo.plan as CustomChatPlan).path, gemmaFile.path);
        expect(repo.state.value.candidate?.model.sourceLine, npu.sourceLine);

        final relaunched = newRepo(npu: const NpuAvailable(soc: 'QTI SM8750'));
        await relaunched.load();
        expect((relaunched.plan as CustomChatPlan).path, gemmaFile.path);
        expect(relaunched.state.value.candidate, isNull);
      });

      test('Apply saves it and makes it the plan; dropping it (leaving the '
          'Models screen) shows the applied model again', () async {
        final repo = newRepo(npu: const NpuAvailable(soc: 'QTI SM8750'));
        await repo.load();
        final gemmaFile = inPlace(
          'gemma-4-E2B-it.litertlm',
          FakeBundle.gemma4E2b,
        );
        await repo.useLocalFile(gemmaFile.path);
        expect(await repo.apply(candidateOf(repo)), isA<Ok<void>>());

        final npuFile = inPlace(
          'gemma4_2b_npu.litertlm',
          FakeBundle.textOnlyNpu,
        );
        await repo.useLocalFile(npuFile.path);
        await repo.dropInPlaceCandidate();
        expect(repo.state.value.candidate, isNull);
        expect((repo.plan as CustomChatPlan).path, gemmaFile.path);

        final again = (await repo.useLocalFile(
          npuFile.path,
        ) as Ok<CustomChatModel>).value;
        expect(await repo.apply(again), isA<Ok<void>>());
        expect(repo.state.value.candidate, isNull);
        expect(repo.state.value.custom?.sourceLine, again.sourceLine);
        final plan = repo.plan as CustomChatPlan;
        expect(plan.path, npuFile.path);
        expect(plan.config.llm.backend, PreferredBackend.npu);
        final stored = CustomChatModelCodec.decode(
          prefs.values[Settings.customChatModel.key]! as String,
        );
        expect((stored.source as LocalModelSource).path, npuFile.path);
        expect(prefs.values[Settings.chatModel.key], 'custom');

        final relaunched = newRepo(npu: const NpuAvailable(soc: 'QTI SM8750'));
        await relaunched.load();
        expect((relaunched.plan as CustomChatPlan).path, npuFile.path);
      });

      test('without the NPU here, a build compiled for the NPU still starts '
          'on the NPU (the plugin refuses it anywhere else), tools off; it '
          'cannot be applied, and the card says why', () async {
        final repo = newRepo();
        await repo.load();
        final npu = (await repo.useLocalFile(
          inPlace('gemma4_2b_npu.litertlm', FakeBundle.npuWithTowers).path,
        ) as Ok<CustomChatModel>).value;

        expect(npu.backend, PreferredBackend.npu, reason: 'the file says so');
        expect(npu.tools, isFalse, reason: 'an NPU build: off until known');
        expect(
          repo.problemWith(npu),
          'The NPU is not available here: no NPU dispatch stack ships for '
          'macos.',
        );
        expect(
          (await repo.apply(npu) as Error<void>).error,
          isA<InvalidChatModelException>(),
        );
      });

      test('an imported candidate stays when the Models screen closes (one in '
          'place does not) and its copy through every prune while it is the '
          'candidate; the next pick replaces it and its copy goes', () async {
        final repo = newRepo(npu: const NpuAvailable());
        await repo.load();
        final first =
            (await repo.importFile(pick()) as Ok<CustomChatModel>).value;
        expect(await repo.apply(first), isA<Ok<void>>());
        final saved = File('${root.path}/custom/$fileName');
        final other = File('${picked.path}/other_ekv2048.litertlm')
          ..writeAsBytesSync(
            fakeLitertlmBytes(
              FakeBundle.textOnlyNpu,
              filler: 32 * 1024,
              seed: 11,
            ),
          );

        await repo.importFile(other.path);
        final candidate = File('${root.path}/custom/other_ekv2048.litertlm');
        expect(candidate.existsSync(), isTrue);
        expect(
          (repo.plan as CustomChatPlan).path,
          saved.path,
          reason: 'the plan follows the saved file, not the store\'s slot',
        );

        await repo.pruneUnused();
        expect(candidate.existsSync(), isTrue, reason: 'the candidate');
        expect(saved.existsSync(), isTrue, reason: 'the saved model');

        await repo.dropInPlaceCandidate(); // the Models screen closes
        expect(
          candidateOf(repo).file?.name,
          'other_ekv2048.litertlm',
          reason: 'a stored candidate stays for the app session',
        );
        await repo.pruneUnused();
        expect(candidate.existsSync(), isTrue);

        await repo.useLocalFile(
          inPlace('in_place.litertlm', FakeBundle.textOnlyNpu).path,
        );
        expect(candidate.existsSync(), isFalse, reason: 'replaced by a pick');
        expect(saved.existsSync(), isTrue);
        expect((repo.plan as CustomChatPlan).path, saved.path);

        await repo.dropInPlaceCandidate();
        expect(repo.state.value.candidate, isNull, reason: 'in place: dropped');
      });

      test('the plan and the setup\'s chat row stay on the saved model while '
          'a candidate downloads', () async {
        final server = await FakeModelServer.start();
        addTearDown(server.close);
        const name = 'slow.litertlm';
        server.files[name] = ServedFile(
          fakeLitertlmBytes(
            FakeBundle.npuWithTowers,
            filler: 64 * 1024,
            seed: 12,
          ),
          stallAfter: 1024,
        );
        final repo = newRepo(npu: const NpuAvailable());
        await repo.load();
        final first =
            (await repo.importFile(pick()) as Ok<CustomChatModel>).value;
        expect(await repo.apply(first), isA<Ok<void>>());
        final savedPath = '${root.path}/custom/$fileName';
        final provisioning = ProvisioningRepository(
          store: stores.last,
          gemmaModelPath: '',
          chatModels: repo,
        );

        final downloading = repo.download(server.url(name));
        await untilNotified(
          repo.customFile,
          () => repo.customFile.value is StoreFileDownloading,
          what: 'the candidate\'s download to start',
        );

        expect((repo.plan as CustomChatPlan).path, savedPath);
        expect(
          provisioning.presenceOf(ModelId.chat),
          isA<PresentAsCustomChatModel>().having(
            (p) => p.where,
            'where',
            'model store (custom/) · verified',
          ),
        );
        repo.cancel();
        expect(await downloading, isA<Error<CustomChatModel>>());
        expect((repo.plan as CustomChatPlan).path, savedPath);
      });
    });

    group('Retry checks the saved file again', () {
      test('a file put back after "Copy it back" loads without a restart; '
          'one still missing stays blocked', () async {
        final first = newRepo(npu: const NpuAvailable());
        await first.load();
        final file = inPlace('mine.litertlm', FakeBundle.textOnlyNpu);
        await first.useLocalFile(file.path);
        expect(await first.apply(candidateOf(first)), isA<Ok<void>>());
        await hashed(first);
        final content = file.readAsBytesSync();
        file.deleteSync();

        final relaunched = newRepo(npu: const NpuAvailable());
        await relaunched.load();
        expect(
          (relaunched.plan as ChatPlanBlocked).reason,
          contains('Copy it back'),
        );
        await relaunched.recheck();
        expect(relaunched.plan, isA<ChatPlanBlocked>(), reason: 'still gone');

        file.writeAsBytesSync(content);
        expect(
          relaunched.plan,
          isA<ChatPlanBlocked>(),
          reason: 'the last check stands until the next one',
        );
        await relaunched.recheck();

        final plan = relaunched.plan as CustomChatPlan;
        expect(plan.path, file.path);
        expect(plan.config.llm.backend, PreferredBackend.npu);
        expect(relaunched.state.value.bundle, isNotNull);
      });

      test('a saved file whose header the plugin cannot read at launch is '
          'planned text-only by the app (the engine loads such a file without '
          "reading it, with the towers it is asked for), with the plugin's "
          'reason; fixed, Use this model (an apply of the saved file) reads it '
          'again and its towers load', () async {
        final first = newRepo(npu: const NpuAvailable());
        await first.load();
        final file = inPlace('mine.litertlm', FakeBundle.npuWithTowers);
        await first.useLocalFile(file.path);
        expect(await first.apply(candidateOf(first)), isA<Ok<void>>());
        await hashed(first);
        final good = file.readAsBytesSync();
        file.writeAsBytesSync(List.filled(good.length, 0));

        final relaunched = newRepo(npu: const NpuAvailable());
        await relaunched.load();
        final unread = relaunched.plan as CustomChatPlan;
        expect(
          unread.bundle,
          isA<BundleNotRead>().having(
            (b) => b.reason.problem,
            'problem',
            LiteRtLmBundleProblem.notLitertlm,
          ),
        );
        expect(unread.config.llm.supportImage, isFalse);
        expect(unread.config.llm.supportAudio, isFalse);
        expect(relaunched.state.value.bundle, isA<BundleNotRead>());

        file.writeAsBytesSync(good);
        expect(
          await relaunched.apply(relaunched.state.value.custom!),
          isA<Ok<void>>(),
        );
        final read = (relaunched.plan as CustomChatPlan).config.llm;
        expect(read.supportImage && read.supportAudio, isTrue);
        expect(relaunched.state.value.bundle, isA<BundleRead>());
      });
    });

    group("towers and defaults: what the plugin reads (LiteRtLmBundleInfo)", () {
      const npu = NpuAvailable(soc: 'QTI SM8750');

      test('a per-SoC build compiled for the NPU, with towers: the NPU, tools '
          'off; the plan loads both towers and leaves their backends to the '
          'engine', () async {
        final repo = newRepo(npu: npu);
        await repo.load();

        final model = (await repo.useLocalFile(
          inPlace(
            'gemma-4-E2B-it_qualcomm_sm8750.litertlm',
            FakeBundle.npuWithTowers,
          ).path,
        ) as Ok<CustomChatModel>).value;

        expect(model.backend, PreferredBackend.npu, reason: 'isCompiledForNpu');
        expect(model.tools, isFalse);
        expect(model.maxTokens, 4096);
        expect(await repo.apply(model), isA<Ok<void>>());
        final llm = (repo.plan as CustomChatPlan).config.llm;
        expect(llm.supportImage && llm.supportAudio, isTrue);
        expect(llm.backend, PreferredBackend.npu);
      });

      test('the file is read again at launch: the plan has it after a '
          'restart', () async {
        final first = newRepo(npu: npu);
        await first.load();
        final model = (await first.useLocalFile(
          inPlace('towers.litertlm', FakeBundle.npuWithTowers).path,
        ) as Ok<CustomChatModel>).value;
        expect(await first.apply(model), isA<Ok<void>>());

        final relaunched = newRepo(npu: npu);
        await relaunched.load();

        final llm = (relaunched.plan as CustomChatPlan).config.llm;
        expect(llm.supportImage && llm.supportAudio, isTrue);
        expect(relaunched.state.value.bundle?.info?.isCompiledForNpu, isTrue);
      });

      test('the CPU | NPU tower choices an earlier build saved are ignored: '
          'the record loads, every tower the file has loads, and nothing '
          'about the towers is written again', () async {
        final file = inPlace('towers.litertlm', FakeBundle.npuWithTowers);
        final size = file.lengthSync();
        prefs.values['chat.model'] = ChatModelKind.custom.name;
        final v4 =
            '{"v":4,"displayName":"towers","modelType":"gemma4",'
            '"backend":"npu","maxTokens":4096,"tools":false,'
            '"visionChoice":"cpu","audioChoice":"npu",'
            '"source":{"kind":"local","path":"${file.path}"},'
            '"file":{"name":"towers.litertlm","sizeBytes":$size,'
            '"checksumMatched":false}}';
        prefs.values['chat.custom'] = v4;
        final repo = newRepo(npu: npu, persistMigration: false);

        expect(await repo.load(), isA<Ok<void>>());

        final llm = (repo.plan as CustomChatPlan).config.llm;
        expect(llm.supportImage && llm.supportAudio, isTrue);
        expect(prefs.values['chat.custom'], v4, reason: 'read only');

        expect(
          await repo.apply(
            repo.state.value.custom!.copyWith(displayName: 'renamed'),
          ),
          isA<Ok<void>>(),
        );
        final saved = prefs.values['chat.custom']! as String;
        expect(saved, contains('"renamed"'));
        expect(saved, isNot(contains('Choice')));
      });

      test('a version 3 record with images off and the encoders on the NPU '
          'loads, and the file wins: both towers load; nothing is '
          'repaired', () async {
        final file = inPlace('towers.litertlm', FakeBundle.npuWithTowers);
        prefs.values['chat.model'] = ChatModelKind.custom.name;
        final v3 =
            '{"v":3,"displayName":"towers","modelType":"gemma4",'
            '"backend":"npu","maxTokens":4096,"supportImage":false,'
            '"supportAudio":true,"encoderBackend":"npu","tools":false,'
            '"source":{"kind":"local","path":"${file.path}"},'
            '"file":{"name":"towers.litertlm","sizeBytes":${file.lengthSync()},'
            '"checksumMatched":false}}';
        prefs.values['chat.custom'] = v3;
        final repo = newRepo(npu: npu, persistMigration: false);

        expect(await repo.load(), isA<Ok<void>>());

        final llm = (repo.plan as CustomChatPlan).config.llm;
        expect(llm.supportImage, isTrue, reason: 'the file has vision');
        expect(llm.supportAudio, isTrue);
        expect(prefs.values['chat.custom'], v3, reason: 'read only');
      });

      test('an incomplete file (LiteRtLmBundleInfo.isComplete false) is '
          'unusable at pick time, with the lengths the plugin reports: in '
          'place, imported and downloaded alike, never stored', () async {
        final repo = newRepo(npu: npu);
        await repo.load();
        final cut = inPlace('cut.litertlm', FakeBundle.incomplete);
        final length = cut.lengthSync();

        final inPlaceResult = await repo.useLocalFile(cut.path);

        final error =
            (inPlaceResult as Error<CustomChatModel>).error
                as LocalModelException;
        expect(
          error.message,
          '${cut.path} is incomplete: it is $length bytes and its sections '
          'end at ${length * 2} (a copy or download that stopped early). '
          'Copy or download it again.',
        );
        expect(repo.state.value.candidate, isNull, reason: 'nothing picked');

        expect(
          (await repo.importFile(cut.path) as Error<CustomChatModel>).error,
          isA<LocalModelException>().having(
            (e) => e.message,
            'message',
            contains('is incomplete'),
          ),
        );
        expect(
          File('${root.path}/custom/cut.litertlm').existsSync(),
          isFalse,
          reason: 'never stored',
        );

        final server = await FakeModelServer.start();
        addTearDown(server.close);
        server.files['cut.litertlm'] = ServedFile(cut.readAsBytesSync());
        expect(
          (await repo.download(
            server.url('cut.litertlm'),
          ) as Error<CustomChatModel>).error.toString(),
          allOf(
            contains('is not a usable .litertlm'),
            contains('the downloaded file is incomplete'),
          ),
        );
        expect(repo.state.value.candidate, isNull);
      });

      test('an imported or downloaded file whose header the plugin cannot '
          'read is not refused (the engine loads it without reading it): it '
          "is stored and becomes the candidate with the plugin's reason, the "
          'GPU and no tower; the chosen model and its file are kept, also '
          'under the same name', () async {
        final repo = newRepo(npu: npu);
        await repo.load();
        final first =
            (await repo.importFile(pick()) as Ok<CustomChatModel>).value;
        expect(await repo.apply(first), isA<Ok<void>>());
        final plan = repo.plan as CustomChatPlan;
        final chosen = File(plan.path).readAsBytesSync();
        final noHeader = File('${picked.path}/no_header.litertlm')
          ..writeAsBytesSync(testBytes(8192, seed: 4));

        final imported = await repo.importFile(noHeader.path);

        final model = (imported as Ok<CustomChatModel>).value;
        expect(model.backend, PreferredBackend.gpu);
        final stored = File('${root.path}/custom/no_header.litertlm');
        expect(stored.existsSync(), isTrue);
        expect(
          repo.state.value.candidate?.bundle,
          isA<BundleNotRead>()
              .having(
                (b) => b.reason.problem,
                'problem',
                LiteRtLmBundleProblem.notLitertlm,
              )
              .having(
                (b) => b.reason.message,
                'message',
                '${stored.path} is not a .litertlm container (no LITERTLM '
                    'magic)',
              ),
        );
        expect(
          towersLine(repo.state.value.candidate!.bundle),
          'towers not read',
        );

        // The same name as the chosen model's file: stored beside it, that
        // file is untouched.
        final sameName = File('${picked.path}/other/$fileName')
          ..createSync(recursive: true)
          ..writeAsBytesSync(testBytes(8192, seed: 7));
        expect(
          await repo.importFile(sameName.path),
          isA<Ok<CustomChatModel>>(),
        );
        expect(File(plan.path).readAsBytesSync(), chosen);

        final server = await FakeModelServer.start();
        addTearDown(server.close);
        server.files['page.litertlm'] = ServedFile(testBytes(4096, seed: 6));
        final downloaded = await repo.download(server.url('page.litertlm'));
        expect(downloaded, isA<Ok<CustomChatModel>>());
        expect(repo.state.value.candidate?.bundle, isA<BundleNotRead>());
        expect(File(plan.path).readAsBytesSync(), chosen);

        expect(repo.state.value.custom?.file?.name, fileName);
        expect(prefs.values['chat.custom'], contains(fileName));
        expect((repo.plan as CustomChatPlan).path, plan.path);
      });

      test('a stored file whose header the plugin cannot read at launch is '
          'planned text-only by the app (the engine loads such a file without '
          "reading it), with the plugin's reason; picked again, its settings "
          'stay', () async {
        final first = newRepo(npu: npu);
        await first.load();
        final model =
            (await first.importFile(pick()) as Ok<CustomChatModel>).value;
        expect(await first.apply(model), isA<Ok<void>>());
        // The file turns unreadable; size and checksum record unchanged.
        final stored = File('${root.path}/custom/$fileName');
        final damaged = stored.readAsBytesSync()..fillRange(0, 8, 0);
        stored.writeAsBytesSync(damaged);

        final relaunched = newRepo(npu: npu);
        final saved = prefs.values['chat.custom'];

        await relaunched.load();
        expect(
          relaunched.plan,
          isA<CustomChatPlan>().having(
            (p) => p.bundle,
            'bundle',
            isA<BundleNotRead>().having(
              (b) => b.reason.message,
              'message',
              contains('is not a .litertlm container'),
            ),
          ),
        );
        expect(
          (relaunched.plan as CustomChatPlan).config.llm.supportImage,
          isFalse,
        );
        expect(relaunched.state.value.problem, isNull, reason: 'settings ok');
        expect(prefs.values['chat.custom'], saved, reason: 'settings kept');

        // Picked again: the settings stay (not a repair to defaults).
        final again =
            (await relaunched.importFile(pick()) as Ok<CustomChatModel>).value;
        expect(again.backend, model.backend);
        expect(again.maxTokens, model.maxTokens);
        expect(relaunched.plan, isA<CustomChatPlan>());
      });

      test('an imported build with towers: read from the store\'s copy (and '
          'again at launch), so its towers load like the same file in '
          'place', () async {
        final repo = newRepo(npu: npu);
        await repo.load();
        final towers = inPlace(
          'gemma-4-E2B-it_qualcomm_sm8750.litertlm',
          FakeBundle.npuWithTowers,
        );

        final model =
            (await repo.importFile(towers.path) as Ok<CustomChatModel>).value;

        expect(model.source, isA<ImportedModelSource>());
        expect(model.backend, PreferredBackend.npu);
        expect(await repo.apply(model), isA<Ok<void>>());
        final llm = (repo.plan as CustomChatPlan).config.llm;
        expect(llm.supportImage && llm.supportAudio, isTrue);

        final relaunched = newRepo(npu: npu);
        await relaunched.load();
        final again = (relaunched.plan as CustomChatPlan).config.llm;
        expect(again.supportImage && again.supportAudio, isTrue);
      });

      test('a downloaded build with towers: the same, from the store\'s '
          'copy', () async {
        final server = await FakeModelServer.start();
        addTearDown(server.close);
        const name = 'gemma-4-E2B-it_qualcomm_sm8750.litertlm';
        server.files[name] = ServedFile(
          fakeLitertlmBytes(FakeBundle.npuWithTowers, seed: 9),
        );
        final repo = newRepo(npu: npu);
        await repo.load();

        final model = (await repo.download(
          server.url(name),
        ) as Ok<CustomChatModel>).value;

        expect(model.source, isA<UrlModelSource>());
        expect(model.backend, PreferredBackend.npu);
        final bundle = candidateBundle(repo);
        expect(bundle.visionEncoder, isNotNull);
        expect(bundle.audioEncoder, isNotNull, reason: 'the file has audio');
      });
    });

    test('another family by its name (Qwen 3): its own type and tools off; '
        'a Gemma file in place: tools on', () async {
      final repo = newRepo(npu: const NpuAvailable());
      await repo.load();

      final qwen = (await repo.useLocalFile(
        inPlace('Qwen3-0.6B.litertlm', FakeBundle.gemma4E2b).path,
      ) as Ok<CustomChatModel>).value;
      expect(qwen.modelType, ModelType.qwen3);
      expect(qwen.tools, isFalse);
    });

    test('a SHA-256 that ends in the middle of an apply patches the applied '
        'settings: the old ones are never saved back', () async {
      final hashing = _GatedHashOps(Completer<void>());
      final writes = _HeldWrites(prefs);
      final store = ModelStore(root: () async => root);
      stores.add(store);
      final repo = ChatModelRepository(
        settings: TypedSettings(store: writes),
        readBundle: fakeReadBundle,
        store: store,
        npu: () => const NpuUnavailable('none here'),
        folders: [ModelsFolder(directory: () async => root)],
        ops: hashing,
      );
      repos.add(repo);
      await repo.load();
      final file = inPlace('gemma-4-E2B-it.litertlm', FakeBundle.gemma4E2b);
      final picked =
          (await repo.useLocalFile(file.path) as Ok<CustomChatModel>).value;
      expect(hashing.started.isCompleted, isFalse, reason: 'only picked');
      expect(await repo.apply(picked), isA<Ok<void>>());
      await hashing.started.future; // the saved file is hashed

      // Another apply saves the custom settings, then the choice; the hash
      // ends between the two.
      final hold = Completer<void>();
      writes
        ..holdKey = Settings.chatModel.key
        ..gate = hold;
      final applying = repo.apply(
        picked.copyWith(displayName: 'Applied', maxTokens: 2048),
      );
      await writes.held.future;
      hashing.gate.complete();
      await pumpEventQueue();
      hold.complete();
      expect(await applying, isA<Ok<void>>());
      await pumpEventQueue();

      final stored = CustomChatModelCodec.decode(
        prefs.values[Settings.customChatModel.key]! as String,
      );
      expect(stored.displayName, 'Applied');
      expect(stored.maxTokens, 2048);
      expect(stored.file?.sha256, _GatedHashOps.hex);
      expect(
        CustomChatModelCodec.encode(repo.state.value.custom!),
        CustomChatModelCodec.encode(stored),
      );

      final relaunched = newRepo();
      await relaunched.load();
      final plan = relaunched.plan as CustomChatPlan;
      expect(plan.config.name, 'Applied');
      expect(plan.config.llm.maxTokens, 2048);
    });

    test('a SHA-256 of a file replaced while it was hashed (another size '
        'now) is never saved', () async {
      final hashing = _GatedHashOps(Completer<void>());
      final store = ModelStore(root: () async => root);
      stores.add(store);
      final repo = ChatModelRepository(
        settings: TypedSettings(store: prefs),
        readBundle: fakeReadBundle,
        store: store,
        npu: () => const NpuUnavailable('none here'),
        folders: [ModelsFolder(directory: () async => root)],
        ops: hashing,
      );
      repos.add(repo);
      await repo.load();
      final file = inPlace('gemma-4-E2B-it.litertlm', FakeBundle.gemma4E2b);
      final picked =
          (await repo.useLocalFile(file.path) as Ok<CustomChatModel>).value;
      expect(await repo.apply(picked), isA<Ok<void>>());
      await hashing.started.future;

      file.writeAsBytesSync([
        ...fakeLitertlmBytes(FakeBundle.gemma4E2b, filler: 9000, seed: 4),
      ]);
      hashing.gate.complete();
      await untilNotified(
        repo.hashing,
        () => !repo.hashing.value,
        what: 'the hash to end',
      );

      final stored = CustomChatModelCodec.decode(
        prefs.values[Settings.customChatModel.key]! as String,
      );
      expect(stored.file?.sha256, isNull, reason: 'the bytes it hashed went');
      expect(repo.state.value.custom!.file!.sha256, isNull);
    });

    test('the app quitting in the middle of a pick: no background SHA-256 '
        'starts on the disposed repository', () async {
      final hashing = _GatedHashOps(Completer<void>());
      final store = ModelStore(root: () async => root);
      stores.add(store);
      final repo = ChatModelRepository(
        settings: TypedSettings(store: prefs),
        readBundle: fakeReadBundle,
        store: store,
        npu: () => const NpuUnavailable('none here'),
        folders: [ModelsFolder(directory: () async => root)],
        ops: hashing,
      );
      await repo.load();
      final file = inPlace('gemma-4-E2B-it.litertlm', FakeBundle.gemma4E2b);
      final picking = repo.useLocalFile(file.path);

      repo.dispose(); // the app quits while the header is read

      await expectLater(picking, completes);
      await pumpEventQueue();
      expect(hashing.started.isCompleted, isFalse);
      expect(prefs.values, isEmpty);
    });

    test('the choice persists across a restart', () async {
      final first = newRepo(npu: const NpuAvailable());
      await first.load();
      final file = inPlace('mine.litertlm', FakeBundle.textOnlyNpu);
      await first.useLocalFile(file.path);
      await first.apply(candidateOf(first));
      await hashed(first);

      final second = newRepo(npu: const NpuAvailable());
      expect(await second.load(), isA<Ok<void>>());

      expect(second.state.value.kind, ChatModelKind.custom);
      expect(second.state.value.custom!.sourceLine, 'in place: ${file.path}');
      expect((second.plan as CustomChatPlan).path, file.path);
    });

    test('a file that disappeared blocks the slot with its path; never '
        'another model', () async {
      final first = newRepo(npu: const NpuAvailable());
      await first.load();
      final file = inPlace('mine.litertlm', FakeBundle.textOnlyNpu);
      await first.useLocalFile(file.path);
      await first.apply(candidateOf(first));
      await hashed(first);
      file.deleteSync();

      final second = newRepo(npu: const NpuAvailable());
      await second.load();

      final plan = second.plan;
      expect(plan, isA<ChatPlanBlocked>());
      expect((plan as ChatPlanBlocked).reason, contains(file.path));
      expect(plan.reason, contains('not there any more'));
      expect(first.plan, isA<ChatPlanBlocked>(), reason: 'checked on use');
    });

    test(
      'a file the app cannot read in its own folder: the error says why '
      'and what to do (an Android push made before the first launch)',
      () async {
        final repo = newRepo();
        await repo.load();
        final dir = Directory('${root.path}/m')..createSync(recursive: true);
        final locked = File('${dir.path}/locked.litertlm')
          ..writeAsBytesSync(
            fakeLitertlmBytes(FakeBundle.gemma4E2b, filler: 64),
          );
        Process.runSync('chmod', ['000', locked.path]);
        addTearDown(() => Process.runSync('chmod', ['644', locked.path]));

        final result = await repo.useLocalFile(locked.path);

        final message = '${(result as Error).error}';
        expect(message, contains('cannot be read'));
        expect(message, contains('Push after the first launch.'));
        expect(repo.state.value.custom, isNull);
      },
    );

    test('the chosen file turned unreadable (a folder the app may not search): '
        'the slot says it cannot be read, with the folder\'s advice, not '
        '"moved, renamed or deleted?"', () async {
      final first = newRepo(npu: const NpuAvailable());
      await first.load();
      final dir = Directory('${root.path}/m')..createSync(recursive: true);
      final file = File('${dir.path}/mine.litertlm')
        ..writeAsBytesSync(
          fakeLitertlmBytes(FakeBundle.textOnlyNpu, filler: 64),
        );
      await first.useLocalFile(file.path);
      await first.apply(candidateOf(first));
      await hashed(first);

      final folder = ModelsFolder(
        directory: () async => dir,
        permissionAdvice: 'Push after the first launch.',
      );
      final denied = ChatModelRepository(
        settings: TypedSettings(store: prefs),
        readBundle: fakeReadBundle,
        store: ModelStore(root: () async => root),
        npu: () => const NpuAvailable(),
        folders: [folder],
        length: (path) => throw FileSystemException(
          'Cannot retrieve length of file',
          path,
          const OSError('Permission denied', 13),
        ),
      );
      repos.add(denied);
      await folder.directory();
      await denied.load();

      final plan = denied.plan as ChatPlanBlocked;
      expect(plan.reason, contains('cannot be read (Permission denied)'));
      expect(plan.reason, contains('Push after the first launch.'));
      expect(plan.reason, isNot(contains('moved, renamed')));
    });

    test('the second folder is listed when it exists and is empty when not, '
        'never created by the app', () async {
      final repo = newRepo();
      final missing = (await repo.listFolders())[1];
      expect(missing.path, '${root.path}/tmp-models');
      expect(missing.error, isNull);
      expect(missing.files, isEmpty);
      expect(Directory('${root.path}/tmp-models').existsSync(), isFalse);

      Directory('${root.path}/tmp-models').createSync();
      inPlace(
        'x.litertlm',
        FakeBundle.textOnlyNpu,
      ).copySync('${root.path}/tmp-models/x.litertlm');
      final listed = (await repo.listFolders())[1].files;
      expect(listed.single.name, 'x.litertlm');
    });

    test('the folders in order, the app\'s own created; one that cannot be '
        'resolved has only its error', () async {
      final store = ModelStore(root: () async => root);
      stores.add(store);
      final repo = ChatModelRepository(
        settings: TypedSettings(store: prefs),
        readBundle: fakeReadBundle,
        store: store,
        folders: [
          ModelsFolder(directory: () async => Directory('${root.path}/m')),
          ModelsFolder(
            directory: () async => throw const FileSystemException('no home'),
            label: 'Broken',
          ),
        ],
      );
      repos.add(repo);

      final [own, broken] = await repo.listFolders();

      expect(own.label, 'Models folder');
      expect(own.path, '${root.path}/m');
      expect(own.error, isNull);
      expect(Directory('${root.path}/m').existsSync(), isTrue);
      expect(broken.label, 'Broken');
      expect(broken.path, isNull);
      expect(broken.error, contains('cannot be created'));
      expect(broken.files, isEmpty);
    });

    test('missing: an error naming the path, nothing saved; not a .litertlm: '
        "the candidate with the plugin's reason, nothing saved", () async {
      final repo = newRepo();
      await repo.load();
      final bogus = File('${picked.path}/bogus.litertlm')
        ..writeAsBytesSync(List.filled(128, 0));

      final notModel = await repo.useLocalFile(bogus.path);
      final missing = await repo.useLocalFile('${picked.path}/none.litertlm');

      expect(notModel, isA<Ok<CustomChatModel>>());
      expect(
        '${(repo.state.value.candidate?.bundle as BundleNotRead).reason}',
        contains('not a .litertlm'),
      );
      expect('${(missing as Error).error}', contains('none.litertlm'));
      expect(repo.state.value.custom, isNull);
    });

    test('a stat that fails after the header was read is a '
        'LocalModelException naming the path, nothing saved', () async {
      final repo = newRepo();
      await repo.load();
      final file = inPlace('gemma4_e2b.litertlm', FakeBundle.gemma4E2b);

      final result = await IOOverrides.runZoned(
        () => repo.useLocalFile(file.path),
        stat: (path) async => path == file.path
            ? throw FileSystemException(
                'Cannot stat',
                path,
                const OSError('Input/output error', 5),
              )
            : FileStat.statSync(path),
      );

      final error = (result as Error<CustomChatModel>).error;
      expect(error, isA<LocalModelException>());
      expect('$error', contains(file.path));
      expect('$error', contains('Input/output error'));
      expect(repo.state.value.custom, isNull);
      expect(await prefs.getString(Settings.customChatModel.key), isNull);
    });

    test('a file gone between its header and its size (stat says notFound, '
        'size -1) is an error, never saved with that size', () async {
      final repo = newRepo();
      await repo.load();
      final file = inPlace('gemma4_e2b.litertlm', FakeBundle.gemma4E2b);

      final result = await IOOverrides.runZoned(
        () => repo.useLocalFile(file.path),
        stat: (path) async => FileStat.statSync(
          path == file.path ? '${picked.path}/gone.litertlm' : path,
        ),
      );

      final error = (result as Error<CustomChatModel>).error;
      expect(error, isA<LocalModelException>());
      expect('$error', contains('is gone'));
      expect(repo.state.value.custom, isNull);
    });
  });

  test("a download an earlier build saved with its link's token: load drops "
      'the token from memory and from the settings', () async {
    const signed = 'https://cdn.example.com/g3.litertlm?X-Signature=secret#f';
    final saved = CustomChatModelCodec.encode(
      CustomChatModel(
        displayName: 'Gemma 3 1B',
        source: UrlModelSource(
          Uri.parse('https://cdn.example.com/g3.litertlm'),
        ),
        backend: PreferredBackend.gpu,
        maxTokens: 4096,
      ),
    ).replaceFirst('https://cdn.example.com/g3.litertlm', signed);
    prefs.values[Settings.customChatModel.key] = saved;

    final repo = newRepo();
    await repo.load();

    final source = repo.state.value.custom!.source as UrlModelSource;
    expect(source.url, Uri.parse('https://cdn.example.com/g3.litertlm'));
    final rewritten = prefs.values[Settings.customChatModel.key]! as String;
    expect(rewritten, isNot(contains('secret')));
    expect(rewritten, contains('https://cdn.example.com/g3.litertlm'));
  });

  test('nothing saved: no chat model (the app ships none), and nothing loads '
      'before load()', () async {
    final repo = newRepo();
    expect(repo.plan, isA<ChatPlanBlocked>(), reason: 'choice not read yet');

    expect(await repo.load(), isA<Ok<void>>());

    expect(repo.state.value.kind, ChatModelKind.none);
    expect(repo.state.value.custom, isNull);
    expect(repo.plan, isA<NoChatModelPlan>());
    expect((repo.plan as NoChatModelPlan).note, isNull);
  });

  group('an upgrade over a Gemma 4 E2B the app downloaded earlier', () {
    /// A verified Gemma 4 E2B download in the retired folder (a small
    /// stand-in: its length is reported as the real one's).
    String earlierGemma() {
      final dir = Directory('${root.path}/$kRetiredGemmaFolder')..createSync();
      final file = File('${dir.path}/$kRetiredGemmaFile')
        ..writeAsBytesSync(fakeLitertlmBytes(FakeBundle.gemma4E2b, filler: 64));
      File('${file.path}.sha256')
          .writeAsStringSync('$kRetiredGemmaSha256  $kRetiredGemmaFile\n');
      return file.path;
    }

    ChatModelRepository repoWith({bool persist = true}) {
      final store = ModelStore(root: () async => root);
      stores.add(store);
      final repo = ChatModelRepository(
        settings: TypedSettings(store: prefs),
        readBundle: fakeReadBundle,
        store: store,
        npu: () =>
            const NpuUnavailable('no NPU dispatch stack ships for macos'),
        folders: [
          ModelsFolder(directory: () async => Directory('${root.path}/m')),
        ],
        length: (p) => p.endsWith(kRetiredGemmaFile)
            ? kRetiredGemmaBytes
            : File(p).lengthSync(),
        persistMigration: persist,
      );
      repos.add(repo);
      return repo;
    }

    test(
      'the retired choice adopts it in place with Gemma 4 E2B\'s settings '
      'and a one-time note; saved, so the next launch just uses it',
      () async {
        final path = earlierGemma();
        prefs.values['chat.model'] = kRetiredBundledChoice;
        final repo = repoWith();

        expect(await repo.load(), isA<Ok<void>>());

        final plan = repo.plan as CustomChatPlan;
        expect(plan.path, path);
        expect(plan.config.name, 'Gemma 4 E2B');
        expect(plan.config.llm.backend, PreferredBackend.gpu);
        expect(plan.config.llm.maxTokens, 8192);
        expect(plan.config.llm.supportImage, isTrue);
        expect(
          plan.config.llm.supportAudio,
          isTrue,
          reason: 'its file has the audio encoder',
        );
        expect(plan.config.tools, isTrue);
        expect(repo.state.value.note, kEarlierGemmaAdoptedNote);
        expect(prefs.values['chat.model'], 'custom');

        final relaunched = repoWith();
        await relaunched.load();
        expect((relaunched.plan as CustomChatPlan).path, path);
        expect(relaunched.state.value.note, isNull);
      },
    );

    test('a file that does not verify is not adopted: the note asks for a '
        'model', () async {
      final path = earlierGemma();
      File('$path.sha256').writeAsStringSync('${'00' * 32}  x\n');
      prefs.values['chat.model'] = kRetiredBundledChoice;
      final repo = repoWith();

      await repo.load();

      expect(repo.plan, isA<NoChatModelPlan>());
      expect(repo.state.value.note, kBundledChoiceRetiredNote);
    });

    test(
      'the headless self-test reads it the same way but writes nothing',
      () async {
        earlierGemma();
        prefs.values['chat.model'] = kRetiredBundledChoice;
        final readOnly = repoWith(persist: false);

        await readOnly.load();

        expect(readOnly.plan, isA<CustomChatPlan>());
        expect(prefs.values['chat.model'], kRetiredBundledChoice);
        expect(prefs.values['chat.custom'], isNull);

        File('${root.path}/$kRetiredGemmaFolder/$kRetiredGemmaFile')
            .deleteSync();
        final none = repoWith(persist: false);
        await none.load();
        expect(none.plan, isA<NoChatModelPlan>());
        expect(prefs.values['chat.model'], kRetiredBundledChoice);
      },
    );
  });

  test('an earlier build\'s saved Gemma 4 E2B choice reads as no model, with '
      'a note once, and is saved as none', () async {
    prefs.values['chat.model'] = kRetiredBundledChoice;
    final repo = newRepo();

    expect(await repo.load(), isA<Ok<void>>());

    expect(repo.state.value.kind, ChatModelKind.none);
    expect(repo.state.value.problem, isNull);
    expect((repo.plan as NoChatModelPlan).note, kBundledChoiceRetiredNote);
    expect(prefs.values['chat.model'], 'none');

    final relaunched = newRepo();
    await relaunched.load();
    expect((relaunched.plan as NoChatModelPlan).note, isNull);
  });

  test('an import gets defaults from the file and its name, is the '
      'candidate, saves nothing and does not switch the chat model by '
      'itself', () async {
    final repo = newRepo(npu: const NpuAvailable(soc: 'QTI SM8750'));
    await repo.load();

    final result = await repo.importFile(pick());

    final model = (result as Ok<CustomChatModel>).value;
    expect(model.displayName, 'Gemma3-1B-IT_q4_ekv1280_sm8750');
    expect(model.backend, PreferredBackend.npu, reason: 'compiled for the NPU');
    expect(model.maxTokens, 1280, reason: 'from ekv1280 in the name');
    expect(model.modelType, ModelType.gemmaIt, reason: 'Gemma3 in the name');
    expect(
      towersLine(repo.state.value.candidate!.bundle),
      'towers none',
      reason: 'a text-only build',
    );
    expect(model.tools, isFalse, reason: 'an NPU build');
    expect(model.file?.sha256, sha256Hex(bytes));
    expect(model.file?.checksumMatched, isFalse);
    expect(model.source, isA<ImportedModelSource>());
    expect(candidateOf(repo).sourceLine, model.sourceLine);
    expect(repo.state.value.custom, isNull);
    expect(repo.state.value.kind, ChatModelKind.none);
    expect(repo.plan, isA<NoChatModelPlan>());
    expect(prefs.values['chat.custom'], isNull);
  });

  test('a file not compiled for the NPU defaults to the GPU, with at least '
      '1024 tokens (an ekv512 name or not)', () async {
    final repo = newRepo(npu: const NpuAvailable());
    await repo.load();
    final tiny = File('${picked.path}/tiny_ekv512.litertlm')
      ..writeAsBytesSync(fakeLitertlmBytes(FakeBundle.textOnly));

    final model =
        (await repo.importFile(tiny.path) as Ok<CustomChatModel>).value;

    expect(model.backend, PreferredBackend.gpu);
    expect(model.maxTokens, 1024);
  });

  test('apply saves the settings and the choice; a new launch reads them and '
      'finds the verified file without hashing', () async {
    final repo = newRepo(npu: const NpuAvailable());
    await repo.load();
    final imported =
        (await repo.importFile(pick()) as Ok<CustomChatModel>).value;

    final applied = await repo.apply(
      imported.copyWith(
        displayName: 'My NPU Gemma',
        maxTokens: 896,
        tools: true,
        modelType: ModelType.gemmaIt,
      ),
    );
    expect(applied, isA<Ok<void>>());
    expect(prefs.values['chat.model'], 'custom');

    final relaunched = newRepo(npu: const NpuAvailable());
    expect(await relaunched.load(), isA<Ok<void>>());
    final plan = relaunched.plan as CustomChatPlan;
    expect(plan.path, '${root.path}/custom/$fileName');
    expect(plan.config.name, 'My NPU Gemma');
    expect(plan.config.llm.backend, PreferredBackend.npu);
    expect(plan.config.llm.maxTokens, 896);
    expect(plan.config.llm.supportImage, isFalse);
    expect(plan.config.tools, isTrue);
    expect(plan.config.modelType, ModelType.gemmaIt);
  });

  group('a choice that cannot be saved', () {
    test('first apply: nothing is left half saved; the candidate stays to '
        'apply again; a restart loads what was saved before', () async {
      final repo = newRepo(npu: const NpuAvailable());
      await repo.load();
      final imported =
          (await repo.importFile(pick()) as Ok<CustomChatModel>).value;
      prefs.failWritesOf.add('chat.model');

      final result = await repo.apply(
        imported.copyWith(displayName: 'Renamed', tools: true),
      );

      expect(result, isA<Error<void>>());
      expect(prefs.values.containsKey('chat.custom'), isFalse);
      expect(prefs.values.containsKey('chat.model'), isFalse);
      expect(repo.state.value.kind, isNot(ChatModelKind.custom));
      expect(repo.state.value.custom, isNull);
      expect(candidateOf(repo).displayName, imported.displayName);

      prefs.failWritesOf.clear();
      final relaunched = newRepo(npu: const NpuAvailable());
      await relaunched.load();
      expect(relaunched.state.value.kind, ChatModelKind.none);
      expect(relaunched.state.value.custom, isNull);
    });

    test('custom chosen already: a failed save keeps the running '
        'settings, not the edited ones', () async {
      final repo = newRepo(npu: const NpuAvailable());
      await repo.load();
      final imported =
          (await repo.importFile(pick()) as Ok<CustomChatModel>).value;
      expect(await repo.apply(imported), isA<Ok<void>>());
      final savedBefore = prefs.values['chat.custom'];
      prefs.failWritesOf.add('chat.model');

      final result = await repo.apply(imported.copyWith(maxTokens: 4096));

      expect(result, isA<Error<void>>());
      expect(prefs.values['chat.custom'], savedBefore);
      prefs.failWritesOf.clear();
      final relaunched = newRepo(npu: const NpuAvailable());
      await relaunched.load();
      expect(
        (relaunched.plan as CustomChatPlan).config.llm.maxTokens,
        imported.maxTokens,
      );
    });
  });

  test("NPU is refused where the app's copy of the plugin's NPU check "
      'finds none, with the reason', () async {
    final repo = newRepo();
    await repo.load();
    final imported =
        (await repo.importFile(pick()) as Ok<CustomChatModel>).value;

    final result = await repo.apply(
      imported.copyWith(backend: PreferredBackend.npu),
    );

    final error = (result as Error<void>).error;
    expect(error, isA<InvalidChatModelException>());
    expect(error.toString(), contains('no NPU dispatch stack ships for macos'));
    expect(prefs.values['chat.model'], isNot('custom'));
  });

  test('Run on GPU raises a small NPU context to 1024 (a generic file asked '
      'for the NPU)', () async {
    final repo = newRepo(npu: const NpuAvailable());
    await repo.load();
    final generic = File('${picked.path}/qwen3_0.6b.litertlm')
      ..writeAsBytesSync(fakeLitertlmBytes(FakeBundle.textOnly));
    final picked0 =
        (await repo.useLocalFile(generic.path) as Ok<CustomChatModel>).value;
    expect(
      await repo.apply(
        picked0.copyWith(backend: PreferredBackend.npu, maxTokens: 896),
      ),
      isA<Ok<void>>(),
    );

    expect(await repo.runOn(PreferredBackend.gpu), isA<Ok<void>>());
    final onGpu = (repo.plan as CustomChatPlan).config.llm;
    expect(onGpu.backend, PreferredBackend.gpu);
    expect(onGpu.maxTokens, 1024);
    expect(prefs.values['chat.model'], 'custom');
  });

  test('a build compiled for the NPU: Apply and Run on GPU / CPU refuse the '
      'GPU and the CPU in the plugin\'s terms (its load would refuse them '
      'after the running model was unloaded); nothing is saved', () async {
    final repo = newRepo(npu: const NpuAvailable());
    await repo.load();
    final imported =
        (await repo.importFile(pick()) as Ok<CustomChatModel>).value;
    expect(
      repo.problemWith(imported.copyWith(backend: PreferredBackend.gpu)),
      'GPU: not for this file (compiled for the Qualcomm SM8750 NPU: runs '
      'only on the NPU). Choose NPU.',
      reason: 'the candidate\'s file',
    );
    expect(await repo.apply(imported), isA<Ok<void>>());
    final saved = prefs.values['chat.custom'];

    for (final backend in [PreferredBackend.gpu, PreferredBackend.cpu]) {
      final result = await repo.runOn(backend);
      expect(
        '${(result as Error<void>).error}',
        '${backend.name.toUpperCase()}: not for this file (compiled for the '
            'Qualcomm SM8750 NPU: runs only on the NPU). Choose NPU.',
        reason: 'the saved file',
      );
    }
    expect(prefs.values['chat.custom'], saved);
    expect(
      (repo.plan as CustomChatPlan).config.llm.backend,
      PreferredBackend.npu,
    );
  });

  test(
    'a custom choice whose file is gone is blocked, never another model',
    () async {
      final repo = newRepo(npu: const NpuAvailable());
      await repo.load();
      final imported =
          (await repo.importFile(pick()) as Ok<CustomChatModel>).value;
      expect(await repo.apply(imported), isA<Ok<void>>());
      File('${root.path}/custom/$fileName').deleteSync();

      expect(
        (repo.plan as ChatPlanBlocked).reason,
        contains('not in the model store any more'),
        reason: 'its size is checked on every use',
      );

      final relaunched = newRepo(npu: const NpuAvailable());
      await relaunched.load();

      final plan = relaunched.plan as ChatPlanBlocked;
      expect(plan.reason, contains('not in the model store any more'));
      expect(plan.reason, contains('choose another .litertlm'));
    },
  );

  test('unreadable saved settings are a visible problem and block the slot; '
      'a file picked is the candidate, and Apply repairs it', () async {
    prefs.values['chat.model'] = 'custom';
    prefs.values['chat.custom'] = '{"v":1}';
    final repo = newRepo(npu: const NpuAvailable());

    final result = await repo.load();

    expect(result, isA<Error<void>>());
    expect(
      repo.state.value.problem,
      contains('the saved custom model is invalid'),
    );
    expect(repo.plan, isA<ChatPlanBlocked>());

    final imported =
        (await repo.importFile(pick()) as Ok<CustomChatModel>).value;
    expect(repo.state.value.problem, isNotNull, reason: 'not applied yet');
    expect(repo.plan, isA<ChatPlanBlocked>());
    expect(prefs.values['chat.custom'], '{"v":1}', reason: 'nothing saved');

    expect(await repo.apply(imported), isA<Ok<void>>());
    expect(repo.state.value.problem, isNull);
    expect(repo.state.value.kind, ChatModelKind.custom);
    expect(
      (repo.plan as CustomChatPlan).model.displayName,
      imported.displayName,
    );
  });

  test('a storage failure on load is shown, not defaulted', () async {
    prefs.failWith = Exception('disk');
    final repo = newRepo();

    expect(await repo.load(), isA<Error<void>>());
    expect(repo.state.value.problem, contains('cannot be read'));
    expect(repo.plan, isA<ChatPlanBlocked>());
  });

  test('another file starts from its own defaults, not the saved settings; '
      'the old file stays while it may run (the custom model is chosen) and '
      'goes after the reload that applies the new one', () async {
    final repo = newRepo(npu: const NpuAvailable());
    await repo.load();
    final first = (await repo.importFile(pick()) as Ok<CustomChatModel>).value;
    await repo.apply(first.copyWith(displayName: 'Mine', tools: true));
    final other = File('${picked.path}/other_ekv2048.litertlm')
      ..writeAsBytesSync(
        fakeLitertlmBytes(FakeBundle.textOnlyNpu, filler: 32 * 1024, seed: 11),
      );

    final second =
        (await repo.importFile(other.path) as Ok<CustomChatModel>).value;

    expect(second.displayName, 'other_ekv2048');
    expect(second.tools, isFalse, reason: 'an NPU build, by its file');
    expect(second.maxTokens, 2048, reason: 'its name');
    expect(second.file?.name, 'other_ekv2048.litertlm');
    final old = File('${root.path}/custom/$fileName');
    expect(old.existsSync(), isTrue, reason: 'the engine may run it');

    await repo.pruneUnused();
    expect(old.existsSync(), isTrue, reason: 'still the saved model');

    expect(await repo.apply(second), isA<Ok<void>>());
    await repo.pruneUnused();
    expect(old.existsSync(), isFalse);
    expect(
      File('${root.path}/custom/other_ekv2048.litertlm').existsSync(),
      isTrue,
    );
  });

  test('a candidate of the saved file\'s store name with other bytes (the '
      'same size) is stored beside it as <stem>-<sha8>.litertlm: the saved '
      'model keeps its file and still loads; Apply makes the copy the chat '
      'model; the same bytes keep the name (the saved file again)', () async {
    final repo = newRepo(npu: const NpuAvailable());
    await repo.load();
    final first = (await repo.importFile(pick()) as Ok<CustomChatModel>).value;
    expect(await repo.apply(first), isA<Ok<void>>());
    final savedPath = '${root.path}/custom/$fileName';
    final sameName = File('${picked.path}/other/$fileName')
      ..createSync(recursive: true)
      ..writeAsBytesSync(
        fakeLitertlmBytes(FakeBundle.textOnlyNpu, filler: 64 * 1024, seed: 99),
      );
    expect(sameName.lengthSync(), bytes.length);
    final otherSha = sha256Hex(sameName.readAsBytesSync());

    final candidate =
        (await repo.importFile(sameName.path) as Ok<CustomChatModel>).value;

    final beside = ModelStore.renamedFor(fileName, otherSha);
    expect(
      beside,
      'Gemma3-1B-IT_q4_ekv1280_sm8750-${otherSha.substring(0, 8)}'
      '.litertlm',
    );
    expect(candidate.file?.name, beside);
    expect(candidate.displayName, 'Gemma3-1B-IT_q4_ekv1280_sm8750');
    expect(File(savedPath).readAsBytesSync(), bytes, reason: 'untouched');
    expect(
      File('$savedPath.sha256').readAsStringSync(),
      startsWith(sha256Hex(bytes)),
    );
    expect((repo.plan as CustomChatPlan).path, savedPath);

    final relaunched = newRepo(npu: const NpuAvailable());
    await relaunched.load();
    expect((relaunched.plan as CustomChatPlan).path, savedPath);
    relaunched.dispose();

    expect(await repo.apply(candidate), isA<Ok<void>>());
    final plan = repo.plan as CustomChatPlan;
    expect(plan.path, '${root.path}/custom/$beside');
    expect(plan.model.file?.sha256, otherSha);

    // The applied copy's own bytes again: no copy beside it.
    final same = File('${picked.path}/same/$beside')
      ..createSync(recursive: true)
      ..writeAsBytesSync(sameName.readAsBytesSync());
    final again =
        (await repo.importFile(same.path) as Ok<CustomChatModel>).value;
    expect(again.file?.name, beside, reason: 'the same bytes keep the name');
    expect(
      Directory('${root.path}/custom')
          .listSync()
          .map((e) => e.uri.pathSegments.last)
          .where((n) => n.endsWith('.litertlm')),
      [beside],
      reason: 'no copy beside the applied one; the first file pruned',
    );
    expect((repo.plan as CustomChatPlan).path, '${root.path}/custom/$beside');
  });

  test('with no model chosen, a replaced custom file is not running and goes '
      'at once', () async {
    final repo = newRepo(npu: const NpuAvailable());
    await repo.load();
    await repo.importFile(pick());

    await repo.importFile(pick('other_ekv2048.litertlm'));

    expect(File('${root.path}/custom/$fileName').existsSync(), isFalse);
  });
}
