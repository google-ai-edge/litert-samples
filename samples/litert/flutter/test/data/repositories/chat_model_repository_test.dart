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
import 'package:litert_edge_demos/data/services/model_store/model_file_ops.dart';
import 'package:litert_edge_demos/data/services/model_store/model_store.dart';
import 'package:litert_edge_demos/data/services/model_store/models_folder.dart';
import 'package:litert_edge_demos/data/services/settings/settings_store.dart';
import 'package:litert_edge_demos/data/services/settings/typed_settings.dart';
import 'package:litert_edge_demos/domain/models/chat_model.dart';
import 'package:litert_edge_demos/domain/models/npu_availability.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../fakes/fake_settings_store.dart';
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
  final bytes = testBytes(64 * 1024, seed: 5);

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
  }) {
    final store = ModelStore(root: () async => root);
    stores.add(store);
    final repo = ChatModelRepository(
      settings: TypedSettings(store: prefs),
      store: store,
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

  group('a .litertlm used in place (the models folder)', () {
    final npuHeader = File(
      'test_assets/litertlm/gemma4_2b_sm8850_npu.header.bin',
    ).readAsBytesSync();
    final gpuHeader = File('test_assets/litertlm/gemma4_e2b_gpu.header.bin')
        .readAsBytesSync();

    File inPlace(String name, List<int> header) =>
        File('${picked.path}/$name')
          ..writeAsBytesSync([...header, ...testBytes(8192, seed: 3)]);

    Future<void> hashed(ChatModelRepository repo) => untilNotified(
      repo.state,
      () => repo.state.value.custom?.file?.sha256 != null,
      what: 'the background SHA-256',
    );

    test('used where it is: no copy into the store; an NPU-only build '
        'defaults to the NPU with images off; the SHA-256 follows in the '
        'background', () async {
      final repo = newRepo(npu: const NpuAvailable(soc: 'QTI SM8850'));
      await repo.load();
      final file = inPlace('gemma4_2b_SM8850.litertlm', npuHeader);

      final result = await repo.useLocalFile(file.path);

      final model = (result as Ok<CustomChatModel>).value;
      expect(model.source, isA<LocalModelSource>());
      expect((model.source as LocalModelSource).path, file.path);
      expect(model.backend, PreferredBackend.npu);
      expect(model.supportImage, isFalse, reason: 'no vision section');
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
      await hashed(repo);
      expect(
        repo.state.value.custom!.file!.sha256,
        sha256Hex(file.readAsBytesSync()),
      );

      expect(await repo.apply(repo.state.value.custom!), isA<Ok<void>>());
      final plan = repo.plan as CustomChatPlan;
      expect(plan.path, file.path);
      expect(plan.config.llm.backend, PreferredBackend.npu);
    });

    group('the saved file is gone from its path', () {
      Future<File> chosenThenMoved({required List<String> into}) async {
        final repo = newRepo();
        await repo.load();
        final file = inPlace('gemma-4-E2B-it.litertlm', gpuHeader);
        await repo.useLocalFile(file.path);
        await hashed(repo);
        expect(await repo.apply(repo.state.value.custom!), isA<Ok<void>>());
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

    test(
      'a GPU file with a vision section defaults images on and the GPU',
      () async {
        final repo = newRepo(npu: const NpuAvailable());
        await repo.load();

        final result = await repo.useLocalFile(
          inPlace('gemma-4-E2B-it.litertlm', gpuHeader).path,
        );

        final model = (result as Ok<CustomChatModel>).value;
        expect(model.backend, PreferredBackend.gpu);
        expect(model.supportImage, isTrue);
        expect(model.maxTokens, 8192, reason: 'the app\'s GPU context');
        expect(model.tools, isTrue, reason: 'not an NPU build: skills work');
      },
    );

    test('another family by its name (Qwen 3): its own type and tools off; '
        'a Gemma file in place: tools on', () async {
      final repo = newRepo(npu: const NpuAvailable());
      await repo.load();

      final qwen = (await repo.useLocalFile(
        inPlace('Qwen3-0.6B.litertlm', gpuHeader).path,
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
        store: store,
        npu: () => const NpuUnavailable('none here'),
        folders: [ModelsFolder(directory: () async => root)],
        ops: hashing,
      );
      repos.add(repo);
      await repo.load();
      final file = inPlace('gemma-4-E2B-it.litertlm', gpuHeader);
      final picked =
          (await repo.useLocalFile(file.path) as Ok<CustomChatModel>).value;
      await hashing.started.future;

      // Apply saves the custom settings, then the choice; the hash ends
      // between the two.
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

    test('the app quitting in the middle of a pick: no background SHA-256 '
        'starts on the disposed repository', () async {
      final hashing = _GatedHashOps(Completer<void>());
      final writes = _HeldWrites(prefs);
      final store = ModelStore(root: () async => root);
      stores.add(store);
      final repo = ChatModelRepository(
        settings: TypedSettings(store: writes),
        store: store,
        npu: () => const NpuUnavailable('none here'),
        folders: [ModelsFolder(directory: () async => root)],
        ops: hashing,
      );
      await repo.load();
      final file = inPlace('gemma-4-E2B-it.litertlm', gpuHeader);
      final hold = Completer<void>();
      writes
        ..holdKey = Settings.customChatModel.key
        ..gate = hold;
      final picking = repo.useLocalFile(file.path);
      await writes.held.future;

      repo.dispose(); // the app quits while the pick saves
      hold.complete();

      await expectLater(picking, completes);
      await pumpEventQueue();
      expect(hashing.started.isCompleted, isFalse);
    });

    test('the choice persists across a restart', () async {
      final first = newRepo(npu: const NpuAvailable());
      await first.load();
      final file = inPlace('mine.litertlm', npuHeader);
      await first.useLocalFile(file.path);
      await hashed(first);
      await first.apply(first.state.value.custom!);

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
      final file = inPlace('mine.litertlm', npuHeader);
      await first.useLocalFile(file.path);
      await hashed(first);
      await first.apply(first.state.value.custom!);
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
          ..writeAsBytesSync([...gpuHeader, ...testBytes(64)]);
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
        ..writeAsBytesSync([...npuHeader, ...testBytes(64)]);
      await first.useLocalFile(file.path);
      await hashed(first);
      await first.apply(first.state.value.custom!);

      final folder = ModelsFolder(
        directory: () async => dir,
        permissionAdvice: 'Push after the first launch.',
      );
      final denied = ChatModelRepository(
        settings: TypedSettings(store: prefs),
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
        npuHeader,
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

    test('not a .litertlm, or missing: an error naming the path, nothing '
        'saved', () async {
      final repo = newRepo();
      await repo.load();
      final bogus = File('${picked.path}/bogus.litertlm')
        ..writeAsBytesSync(List.filled(128, 0));

      final notModel = await repo.useLocalFile(bogus.path);
      final missing = await repo.useLocalFile('${picked.path}/none.litertlm');

      expect('${(notModel as Error).error}', contains('not a .litertlm'));
      expect('${(missing as Error).error}', contains('none.litertlm'));
      expect(repo.state.value.custom, isNull);
    });

    test('a stat that fails after the header was read is a '
        'LocalModelException naming the path, nothing saved', () async {
      final repo = newRepo();
      await repo.load();
      final file = inPlace('gemma4_e2b.litertlm', gpuHeader);

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
      final file = inPlace('gemma4_e2b.litertlm', gpuHeader);

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
    final gpuHeader = File('test_assets/litertlm/gemma4_e2b_gpu.header.bin')
        .readAsBytesSync();

    /// The earlier build's verified download (a small stand-in: its length
    /// is reported as the real one's).
    String earlierGemma() {
      final dir = Directory('${root.path}/$kRetiredGemmaFolder')..createSync();
      final file = File('${dir.path}/$kRetiredGemmaFile')
        ..writeAsBytesSync([...gpuHeader, ...testBytes(64)]);
      File('${file.path}.sha256')
          .writeAsStringSync('$kRetiredGemmaSha256  $kRetiredGemmaFile\n');
      return file.path;
    }

    ChatModelRepository repoWith({bool persist = true}) {
      final store = ModelStore(root: () async => root);
      stores.add(store);
      final repo = ChatModelRepository(
        settings: TypedSettings(store: prefs),
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

  test('an import gets defaults from the device and the file name, is saved, '
      'and does not switch the chat model by itself', () async {
    final repo = newRepo(npu: const NpuAvailable(soc: 'QTI SM8750'));
    await repo.load();

    final result = await repo.importFile(pick());

    final model = (result as Ok<CustomChatModel>).value;
    expect(model.displayName, 'Gemma3-1B-IT_q4_ekv1280_sm8750');
    expect(model.backend, PreferredBackend.npu, reason: 'the NPU is offered');
    expect(model.maxTokens, 1280, reason: 'from ekv1280 in the name');
    expect(model.modelType, ModelType.gemmaIt, reason: 'Gemma3 in the name');
    expect(model.supportImage, isFalse);
    expect(model.tools, isFalse);
    expect(model.file?.sha256, sha256Hex(bytes));
    expect(model.file?.checksumMatched, isFalse);
    expect(model.source, isA<ImportedModelSource>());
    expect(repo.state.value.kind, ChatModelKind.none);
    expect(repo.plan, isA<NoChatModelPlan>());
    expect(prefs.values['chat.custom'], isA<String>());
  });

  test(
    'without an NPU the default is the GPU, with at least 1024 tokens',
    () async {
      final repo = newRepo();
      await repo.load();

      final model = (await repo.importFile(
        pick('tiny_ekv512.litertlm'),
      ) as Ok<CustomChatModel>).value;

      expect(model.backend, PreferredBackend.gpu);
      expect(model.maxTokens, 1024);
    },
  );

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
    test('first apply: the custom settings are put back; a restart loads '
        'what the screen still shows', () async {
      final repo = newRepo();
      await repo.load();
      final imported =
          (await repo.importFile(pick()) as Ok<CustomChatModel>).value;
      final savedBefore = prefs.values['chat.custom'];
      prefs.failWritesOf.add('chat.model');

      final result = await repo.apply(
        imported.copyWith(displayName: 'Renamed', tools: true),
      );

      expect(result, isA<Error<void>>());
      expect(prefs.values['chat.custom'], savedBefore);
      expect(prefs.values.containsKey('chat.model'), isFalse);
      expect(repo.state.value.kind, isNot(ChatModelKind.custom));
      expect(repo.state.value.custom?.displayName, imported.displayName);

      prefs.failWritesOf.clear();
      final relaunched = newRepo();
      await relaunched.load();
      expect(relaunched.state.value.kind, repo.state.value.kind);
      expect(relaunched.state.value.custom?.displayName, imported.displayName);
    });

    test('custom chosen already: a failed save keeps the running '
        'settings, not the edited ones', () async {
      final repo = newRepo();
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
      final relaunched = newRepo();
      await relaunched.load();
      expect(
        (relaunched.plan as CustomChatPlan).config.llm.maxTokens,
        imported.maxTokens,
      );
    });
  });

  test('NPU is refused where flutter_edge_ai would not offer it, with the '
      'reason', () async {
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

  test('Run on GPU raises a small NPU context to 1024', () async {
    final repo = newRepo(npu: const NpuAvailable());
    await repo.load();
    final imported =
        (await repo.importFile(pick()) as Ok<CustomChatModel>).value;
    await repo.apply(imported.copyWith(maxTokens: 896));

    expect(await repo.runOn(PreferredBackend.gpu), isA<Ok<void>>());
    final onGpu = (repo.plan as CustomChatPlan).config.llm;
    expect(onGpu.backend, PreferredBackend.gpu);
    expect(onGpu.maxTokens, 1024);
    expect(prefs.values['chat.model'], 'custom');
  });

  test(
    'a custom choice whose file is gone is blocked, never another model',
    () async {
      final repo = newRepo();
      await repo.load();
      final imported =
          (await repo.importFile(pick()) as Ok<CustomChatModel>).value;
      await repo.apply(imported);
      File('${root.path}/custom/$fileName').deleteSync();

      final relaunched = newRepo();
      await relaunched.load();

      final plan = relaunched.plan as ChatPlanBlocked;
      expect(plan.reason, contains('not in the model store any more'));
      expect(plan.reason, contains('choose another .litertlm'));
    },
  );

  test(
    'unreadable saved settings are a visible problem and block the slot',
    () async {
      prefs.values['chat.model'] = 'custom';
      prefs.values['chat.custom'] = '{"v":1}';
      final repo = newRepo();

      final result = await repo.load();

      expect(result, isA<Error<void>>());
      expect(
        repo.state.value.problem,
        contains('the saved custom model is invalid'),
      );
      expect(repo.plan, isA<ChatPlanBlocked>());

      // Importing again repairs it; the choice stays custom.
      final imported =
          (await repo.importFile(pick()) as Ok<CustomChatModel>).value;
      expect(repo.state.value.problem, isNull);
      expect(repo.state.value.kind, ChatModelKind.custom);
      expect(
        (repo.plan as CustomChatPlan).model.displayName,
        imported.displayName,
      );
    },
  );

  test('a storage failure on load is shown, not defaulted', () async {
    prefs.failWith = Exception('disk');
    final repo = newRepo();

    expect(await repo.load(), isA<Error<void>>());
    expect(repo.state.value.problem, contains('cannot be read'));
    expect(repo.plan, isA<ChatPlanBlocked>());
  });

  test('a replacement file keeps the settings; the old file stays while it may '
      'run (the custom model is chosen) and goes after a reload', () async {
    final repo = newRepo(npu: const NpuAvailable());
    await repo.load();
    final first = (await repo.importFile(pick()) as Ok<CustomChatModel>).value;
    await repo.apply(first.copyWith(displayName: 'Mine', tools: true));

    final second = (await repo.importFile(
      pick('other_ekv2048.litertlm'),
    ) as Ok<CustomChatModel>).value;

    expect(second.displayName, 'Mine');
    expect(second.tools, isTrue);
    expect(second.file?.name, 'other_ekv2048.litertlm');
    final old = File('${root.path}/custom/$fileName');
    expect(old.existsSync(), isTrue, reason: 'the engine may run it');

    await repo.pruneUnused();
    expect(old.existsSync(), isFalse);
    expect(
      File('${root.path}/custom/other_ekv2048.litertlm').existsSync(),
      isTrue,
    );
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
