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

import 'package:flutter/foundation.dart';
import 'package:flutter_edge_ai/flutter_edge_ai.dart' show PreferredBackend;
import 'package:flutter_edge_ai_litertlm/flutter_edge_ai_litertlm.dart'
    show readLiteRtLmBundleInfo;

import '../../config/model_catalog.dart'
    show
        kDefineChatModel,
        kLlmConfig,
        kRetiredGemmaBytes,
        kRetiredGemmaFile,
        kRetiredGemmaFolder,
        kRetiredGemmaSha256;
import '../../domain/models/chat_model.dart';
import '../../domain/models/npu_availability.dart';
import '../../domain/models/provisioning.dart';
import '../../domain/ports/chat_model_planner.dart';
import '../../utils/redact_url.dart';
import '../../utils/result.dart';
import '../../utils/serial_queue.dart';
import '../services/llm/npu_availability.dart';
import '../services/model_store/checksum_record.dart';
import '../services/model_store/model_file_ops.dart';
import '../services/model_store/model_store.dart';
import '../services/model_store/models_folder.dart';
import '../services/settings/typed_settings.dart';
import 'chat_model/custom_chat_model_codec.dart';

/// The chat model choice as saved, for the Models screen, and the file picked
/// on it but not applied yet ([candidate]).
final class const ChatModelState({
  required final ChatModelKind kind,

  /// The user's own model as saved (the applied one when [kind] is custom);
  /// kept while [kind] is none.
  final CustomChatModel? custom,

  /// The saved choice or custom settings could not be read; shown with the
  /// way out, never replaced by a default.
  final String? problem,

  /// Something to tell once, not a problem (a retired saved choice).
  final String? note,

  /// What `readLiteRtLmBundleInfo` last made of [custom]'s file (the file in
  /// place, or the store's copy): what it says about itself, or why its
  /// header could not be read ([BundleNotRead]: it still loads); null when
  /// it cannot be opened, is incomplete, or there is no file. The towers
  /// that load are the ones it has; nothing about them is saved.
  final BundleReading? bundle,

  /// A file picked on the Models screen and not applied: only the card shows
  /// it. Never saved, never planned ([ChatModelRepository.plan]).
  final ChatModelCandidate? candidate,

  /// [ChatModelRepository.load] has run.
  final bool loaded = false,
}) {
  /// This state with [candidate] in place of its own.
  ChatModelState withCandidate(ChatModelCandidate? candidate) => ChatModelState(
    kind: kind,
    custom: custom,
    problem: problem,
    note: note,
    bundle: bundle,
    candidate: candidate,
    loaded: loaded,
  );
}

/// A `.litertlm` picked on the Models screen (a models folder, Path…, an
/// import or a download) and not applied yet: the card shows its towers,
/// backend, context and tools, and Apply ([ChatModelRepository.apply]) saves
/// it and makes it the chat model. Until then nothing of it is saved: the
/// self-test, every reload, "Run on GPU/CPU", the next launch and the demos
/// use the applied model. Picking another file replaces it; leaving the
/// Models screen drops one used in place, while an imported or downloaded
/// one stays for the app session ([ChatModelRepository.dropInPlaceCandidate]).
/// Its bytes never go onto the saved model's store file (another file of
/// that name is kept beside it). It is never hashed: only the saved model's
/// file in place is, in the background.
final class const ChatModelCandidate({
  /// The file and the settings it starts from: the same file picked again
  /// keeps the settings it had (the applied model's or the previous
  /// candidate's), another file starts from what it says about itself
  /// ([ChatModelRepository.defaultsFor]).
  required final CustomChatModel model,

  /// What `readLiteRtLmBundleInfo` made of its file when it was picked.
  required final BundleReading bundle,

  /// Where the file is: its path in place, or the store's copy of an
  /// imported or downloaded file (kept by every prune while it is the
  /// candidate).
  required final String path,
});

/// A custom setting the Models screen should not have let through.
final class InvalidChatModelException implements Exception {
  const InvalidChatModelException(this.message);

  final String message;

  @override
  String toString() => message;
}

/// Which chat model the demos use: the user's own `.litertlm` (the app ships
/// none). The choice and its settings live in [TypedSettings], the file in the
/// model store's `custom/` folder or used in place (a models folder, or any
/// path: [useLocalFile]). `ModelRepository` asks it what to load ([plan]); it
/// never substitutes another model: none chosen is [NoChatModelPlan], one that
/// cannot load is a [ChatPlanBlocked] the Models screen shows.
///
/// A picked file is only a [ChatModelCandidate] until [apply] saves it: the
/// plan, and so every load, the self-test and the next launch, follow the
/// applied model alone.
class ChatModelRepository implements ChatModelPlanner {
  ChatModelRepository({
    required this._settings,
    required this._store,
    this._npu = probeNpu,
    required this._folders,
    this._ops = const ModelFileOps(),
    this._length = _fileLength,
    this._readBundle = readLiteRtLmBundleInfo,
    this._persistMigration = true,
  });

  /// Reads what a `.litertlm` says about itself: flutter_edge_ai_litertlm's
  /// [readLiteRtLmBundleInfo] (a fake in tests).
  final BundleReader _readBundle;

  /// [inspectLitertlm] through [_readBundle].
  Future<Result<BundleReading>> _inspect(String path) =>
      inspectLitertlm(path, read: _readBundle);

  /// Writes what [load] migrates (a retired saved choice) back to the
  /// settings. Off for the headless self-test, which reads only.
  final bool _persistMigration;

  static int _fileLength(String path) => File(path).lengthSync();

  /// A file's length; throws [FileSystemException] (a seam for tests).
  final int Function(String path) _length;

  final TypedSettings _settings;
  final ModelStore _store;
  final NpuAvailability Function() _npu;
  final ModelFileOps _ops;

  /// Where users copy their `.litertlm` to use it in place: the first is
  /// the app's own (created on first run); Android adds
  /// [kAndroidTmpModelsDir].
  final List<ModelsFolder> _folders;

  /// Why the in-place file of the saved model cannot be used (missing,
  /// unreadable, not a `.litertlm`, changed); null when it can. Checked at
  /// [load], again by [recheck] (Retry, Rescan) and when the same file is
  /// picked again.
  String? _localProblem;

  /// Why the store's copy of the saved model cannot be used although it
  /// verified (it cannot be opened, or is incomplete, at launch); null when
  /// it can. Like
  /// [_localProblem] it blocks the plan, not the saved settings: picking the
  /// file again keeps them.
  String? _storedProblem;

  /// What `readLiteRtLmBundleInfo` last made of the saved model's file: the
  /// file in place or the store's copy of an imported or downloaded file,
  /// read at [load], by [recheck] and when it was applied; null when it
  /// cannot be opened or is incomplete. Published with every state
  /// ([ChatModelState.bundle]): the towers that load are the ones it has.
  BundleReading? _appliedBundle;

  /// What was last found for the saved model's store copy (an imported or
  /// downloaded file): [StoreFileReady] with its path once it verified. Kept
  /// apart from the store's slot ([customFile]), which shows a candidate's
  /// transfer: the plan never follows it.
  StoreFileState _savedStore = const StoreFileMissing();

  /// The background SHA-256 of a file used in place runs.
  final ValueNotifier<bool> _hashing = ValueNotifier(false);
  ValueListenable<bool> get hashing => _hashing;
  Completer<void>? _hashCancel;

  /// The file the background SHA-256 reads.
  String? _hashPath;

  /// The writes of the chat model settings, each with the state it
  /// publishes, one after another: one never lands between another's read
  /// and its save (the background SHA-256 never saves the old settings over
  /// an apply's). With none queued a write starts at once.
  final SerialQueue _settingsWrites = SerialQueue.atOnceWhenIdle();

  final ValueNotifier<ChatModelState> _state = ValueNotifier(
    const ChatModelState(kind: ChatModelKind.none),
  );
  bool _closed = false;

  ValueListenable<ChatModelState> get state => _state;

  /// The store's slot: a custom file's transfer (progress while it is
  /// stored), then what was last found or stored.
  ValueListenable<StoreFileState> get customFile => _store.customFile;

  /// What was last found for the saved model's store copy (missing, changed
  /// on disk, verified); [StoreFileMissing] for a file in place or none.
  /// Changes with a published [state].
  StoreFileState get savedFile => _savedStore;

  /// A download or import (of any model) runs.
  ValueListenable<bool> get busy => _store.busy;

  /// Whether the NPU can be reached on this device, by the app's copy of
  /// flutter_edge_ai_litertlm's private NPU check (the plugin is not asked:
  /// it has no public API for it): only for the Models screen, which offers
  /// the NPU by it.
  NpuAvailability get npu => _npu();

  /// Each models folder, in order (the app's own first, created when
  /// missing), with the `.litertlm` files in it: a folder that cannot be
  /// resolved has only an error, one that cannot be read its path and the
  /// error.
  Future<List<ModelsFolderListing>> listFolders() async => [
    for (final folder in _folders) await _list(folder),
  ];

  static Future<ModelsFolderListing> _list(ModelsFolder folder) async {
    switch (await folder.directory()) {
      case Error(:final error):
        return ModelsFolderListing(label: folder.label, error: '$error');
      case Ok(value: final dir):
        return switch (await folder.list()) {
          Ok(:final value) => ModelsFolderListing(
            label: folder.label,
            path: dir.path,
            files: value,
          ),
          Error(:final error) => ModelsFolderListing(
            label: folder.label,
            path: dir.path,
            error: '$error',
          ),
        };
    }
  }

  /// Reads the saved choice and custom model and looks for its file.
  Future<Result<void>> load() async {
    final problems = <String>[];
    var kind = ChatModelKind.none;
    String? note;
    switch (await _settings.read(Settings.chatModel)) {
      case Ok(value: kRetiredBundledChoice):
        // The retired choice "Gemma 4 E2B, downloaded by the app": its
        // verified download is adopted in place; without one it is said
        // once, then saved as no choice.
        if (await _adoptEarlierGemma() case final adopted?) return adopted;
        note = kBundledChoiceRetiredNote;
        debugPrint('[ChatModel] the saved Gemma 4 E2B choice is retired');
        if (_persistMigration) {
          await _settingsWrites.run(
            () => _settings.write(Settings.chatModel, ChatModelKind.none.name),
          );
        }
      case Ok(:final value):
        if (value != null) {
          final parsed = ChatModelKind.values
              .where((k) => k.name == value)
              .firstOrNull;
          if (parsed == null) {
            problems.add('the saved chat model choice "$value" is unknown');
            kind = ChatModelKind.custom; // blocked until fixed
          } else {
            kind = parsed;
          }
        }
      case Error(:final error):
        problems.add('the saved chat model choice cannot be read ($error)');
        kind = ChatModelKind.custom;
    }
    CustomChatModel? custom;
    switch (await _settings.read(Settings.customChatModel)) {
      case Ok(:final value?):
        try {
          custom = CustomChatModelCodec.decode(value);
        } on FormatException catch (e) {
          problems.add('the saved custom model is invalid (${e.message})');
        }
        // A download's link saved whole: decoding dropped its query and
        // fragment (a signed link's token); so does the file.
        if (custom case final CustomChatModel model?
            when model.source is UrlModelSource && _persistMigration) {
          final redacted = CustomChatModelCodec.encode(model);
          if (redacted != value) {
            await _settingsWrites.run(
              () => _settings.write(Settings.customChatModel, redacted),
            );
          }
        }
      case Ok():
        break;
      case Error(:final error):
        problems.add('the saved custom model cannot be read ($error)');
    }
    if (custom != null) {
      final (checked, unchecked) = await _checkSaved(custom);
      custom = checked;
      if (unchecked != null) {
        problems.add('its file could not be checked ($unchecked)');
      }
    }
    final problem = problems.isEmpty
        ? null
        : 'Your own chat model: ${problems.join('; ')}. Choose the '
              '.litertlm again.';
    if (problem != null) debugPrint('[ChatModel] $problem');
    _publish(
      ChatModelState(
        kind: kind,
        custom: custom,
        problem: problem,
        note: note,
        loaded: true,
        bundle: _appliedBundle,
      ),
    );
    debugPrint(
      '[ChatModel] ${kind.name}'
      '${custom == null ? '' : ' · custom ${custom.displayName} '
                '(${_settingsLineOf(custom)}) '
                '${custom.sourceLine}'}',
    );
    if (custom
        case CustomChatModel(
          source: LocalModelSource(:final path),
          file: CustomModelFile(sha256: null, :final sizeBytes),
        )
        when _localProblem == null) {
      _hashInBackground(path, sizeBytes);
    }
    return problem == null
        ? const Result.ok(null)
        : Result.error(InvalidChatModelException(problem));
  }

  /// The Gemma 4 E2B the retired choice downloaded and verified
  /// (`<store>/gemma4E2b/`, its `.sha256` record matching), made the chat
  /// model in place with Gemma 4 E2B's settings (GPU, 8192, tools on; the
  /// towers the file has) and a one-time note. Null when there is none.
  Future<Result<void>?> _adoptEarlierGemma() async {
    final String path;
    try {
      path =
          '${(await _store.root()).path}/$kRetiredGemmaFolder/'
          '$kRetiredGemmaFile';
      if (_length(path) != kRetiredGemmaBytes ||
          await ChecksumRecord.read(File(path)) != kRetiredGemmaSha256) {
        return null;
      }
    } on FileSystemException {
      return null;
    }
    final BundleReading bundle;
    switch (await _inspect(path)) {
      case Ok(:final value):
        bundle = value;
      case Error(:final error):
        debugPrint('[ChatModel] the earlier Gemma 4 E2B at $path: $error');
        return null;
    }
    final model = CustomChatModel(
      displayName: kDefineChatModel.name,
      source: LocalModelSource(path),
      file: const CustomModelFile(
        name: kRetiredGemmaFile,
        sizeBytes: kRetiredGemmaBytes,
        sha256: kRetiredGemmaSha256,
        checksumMatched: false,
      ),
      modelType: kDefineChatModel.modelType,
      backend: PreferredBackend.gpu,
      maxTokens: kLlmConfig.maxTokens,
      tools: true,
    );
    final adopted = await _settingsWrites.run(() async {
      if (_persistMigration) {
        if (await _saveCustomThenChoice(model, ChatModelKind.custom) case Error(
          :final error,
        )) {
          return Result<void>.error(error);
        }
      }
      _localProblem = null;
      _appliedBundle = bundle;
      _publish(
        ChatModelState(
          kind: ChatModelKind.custom,
          custom: model,
          note: kEarlierGemmaAdoptedNote,
          loaded: true,
          bundle: bundle,
        ),
      );
      return const Result<void>.ok(null);
    });
    if (adopted is Error<void>) return adopted;
    debugPrint(
      '[ChatModel] the saved Gemma 4 E2B choice: its earlier download is '
      'used in place ($path)',
    );
    return const Result.ok(null);
  }

  @override
  ChatModelPlan get plan {
    final s = _state.value;
    // Before load() the saved choice is unknown: loading anything now could
    // be the wrong model.
    if (!s.loaded) {
      return const ChatPlanBlocked(
        'The saved chat model choice has not been read yet.',
      );
    }
    if (s.kind == ChatModelKind.none) return NoChatModelPlan(note: s.note);
    if (s.problem case final problem?) return ChatPlanBlocked(problem);
    final custom = s.custom;
    final file = custom?.file;
    if (custom == null || file == null) {
      return const ChatPlanBlocked(
        'Your own chat model is chosen but has no file: choose a .litertlm.',
      );
    }
    // The saved model only, never the candidate. Its file is read at launch,
    // by a recheck and when it was applied; one that cannot be opened, or an
    // incomplete one, is refused or a problem before this.
    final bundle = s.bundle;
    const unread = 'the file has not been read: choose the .litertlm again.';
    if (custom.source case LocalModelSource(:final path)) {
      // The size now first: a launch's verdict may be stale either way.
      final problem = _quickLocalProblem(path, file) ?? _localProblem;
      return switch ((problem, bundle)) {
        (final problem?, _) => ChatPlanBlocked(
          '${custom.displayName}: $problem',
        ),
        (null, final bundle?) => CustomChatPlan(
          path: path,
          model: custom,
          bundle: bundle,
        ),
        (null, null) => ChatPlanBlocked('${custom.displayName}: $unread'),
      };
    }
    if (_storedProblem case final problem?) {
      return ChatPlanBlocked('${custom.displayName}: $problem');
    }
    // What was found for the saved file (not the store's slot, which follows
    // a candidate's transfer), and its size now: a candidate of the same name
    // stored over it is another file.
    return switch (_savedStore) {
      StoreFileReady(:final path) => switch ((
        _quickStoredProblem(path, file),
        bundle,
      )) {
        (final problem?, _) => ChatPlanBlocked(
          '${custom.displayName}: $problem',
        ),
        (null, final bundle?) => CustomChatPlan(
          path: path,
          model: custom,
          bundle: bundle,
        ),
        (null, null) => ChatPlanBlocked('${custom.displayName}: $unread'),
      },
      StoreFileUnverified() => ChatPlanBlocked(
        '${custom.displayName}: ${_storeChanged(file)}',
      ),
      StoreFileMissing() => ChatPlanBlocked(
        '${custom.displayName}: ${_storeMissing(file)}',
      ),
      _ => ChatPlanBlocked(
        '${custom.displayName}: ${file.name} is still being stored.',
      ),
    };
  }

  static String _storeChanged(CustomModelFile file) =>
      '${file.name} changed on disk since it was stored (its size or checksum '
      'record differs). Import or download it again, or choose another '
      '.litertlm.';

  static String _storeMissing(CustomModelFile file) =>
      '${file.name} is not in the model store any more. Import or download it '
      'again, or choose another .litertlm.';

  /// The cheap check of a store copy at [path] (its length) for [plan] and
  /// [problemWith]: gone, or another size than [file] had when it was
  /// stored.
  String? _quickStoredProblem(String path, CustomModelFile file) {
    try {
      if (_length(path) == file.sizeBytes) return null;
    } on FileSystemException catch (e) {
      if (!isNotFound(e)) {
        return '$path cannot be read (${e.osError?.message ?? e.message}).';
      }
      return _storeMissing(file);
    }
    return _storeChanged(file);
  }

  /// Imports the `.litertlm` at [path] (see `ModelStore.importCustomFile`)
  /// into the store and makes it the [ChatModelCandidate] ([_adoptStored]:
  /// refused only as a load refuses it, [inspectLitertlm]). Nothing
  /// is saved: [apply] makes it the chat model. Other bytes under the saved
  /// model's store name are stored beside its file, never over it
  /// ([_reservedStoreName]).
  Future<Result<CustomChatModel>> importFile(
    String path, {
    String? moveFrom,
  }) async {
    // Read before the store copies it: a file a load refuses (it cannot be
    // opened, or is incomplete) is never stored.
    if (await _inspect(path) case Error(:final error)) {
      debugPrint('[ChatModel] import of $path refused: $error');
      return Result.error(error);
    }
    final stored = await _store.importCustomFile(
      path,
      moveFrom: moveFrom,
      reserved: _reservedStoreName,
    );
    return switch (stored) {
      Ok(:final value) => _adoptStored(
        value,
        ImportedModelSource(path),
        knownAs: path.split(RegExp(r'[/\\]')).last,
      ),
      Error(:final error) => Result.error(error),
    };
  }

  /// The saved model's file name in the store: no picked file goes onto it
  /// before it is applied (another file of that name is stored beside it,
  /// `ModelStore.placementFor`); null for a file in place or none.
  String? get _reservedStoreName => switch (_state.value.custom) {
    CustomChatModel(:final file?, :final source)
        when source is! LocalModelSource =>
      file.name,
    _ => null,
  };

  /// Adopts the store's copy [file] (just imported or downloaded) from
  /// [source] as the candidate, its defaults and towers from what it says
  /// about itself like a file in place; a first name from [knownAs], the file
  /// name the user knows (the store may keep it under another, beside the
  /// saved model's). The picked file and the downloaded bytes were checked
  /// before they were stored; should the copy still be refused (an I/O
  /// error), it is refused with the inspection error: the chosen model and
  /// the candidate stay as they were, and the copy goes with the next prune
  /// of the store ([pruneUnused], the next adopt).
  Future<Result<CustomChatModel>> _adoptStored(
    CustomModelFile file,
    CustomModelSource source, {
    required String knownAs,
  }) async {
    final path = await _store.customPathOf(file.name);
    switch (await _inspect(path)) {
      case Ok(:final value):
        return _adopt(
          file,
          source,
          bundle: value,
          path: path,
          knownAs: knownAs,
        );
      case Error(:final error):
        debugPrint('[ChatModel] ${file.name} refused: $error');
        // The store's slot points at the refused copy now: back to the
        // saved model's file, checked again (a copy of the same name went
        // over it).
        await _settingsWrites.run(() async {
          await _lookAtSavedStoreFile();
          _republish();
        });
        return Result.error(error);
    }
  }

  /// Looks for the saved model's store copy again (`useCustomFile`): what is
  /// found becomes [_savedStore] and the store's slot shows it; none for a
  /// file in place or no model. A transfer running (the store is busy)
  /// leaves both as they were. Publishes nothing; runs in the settings
  /// queue.
  Future<void> _lookAtSavedStoreFile() async {
    final saved = _state.value.custom;
    final file = saved == null || saved.source is LocalModelSource
        ? null
        : saved.file;
    switch (await _store.useCustomFile(file)) {
      case Ok(:final value?):
        _savedStore = StoreFileReady(value);
      case Ok():
        _savedStore = file == null
            ? const StoreFileMissing()
            : _store.customFile.value;
      case Error(:final error):
        debugPrint('[ChatModel] the saved model\'s file not checked: $error');
    }
  }

  /// Downloads the custom model's file from [url] (see
  /// `ModelStore.downloadCustomFile`); otherwise as [importFile]. A link
  /// with a user name or password in it is refused
  /// ([kUrlCredentialsMessage]). The whole link goes only to the download;
  /// what is saved, logged and shown is [redactUrl]'s form of it.
  Future<Result<CustomChatModel>> download(
    Uri url, {
    String? sha256,
    int? sizeBytes,
  }) async {
    if (url.userInfo.isNotEmpty) {
      return const Result.error(
        InvalidChatModelException(kUrlCredentialsMessage),
      );
    }
    final stored = await _store.downloadCustomFile(
      url,
      sha256: sha256,
      sizeBytes: sizeBytes,
      reserved: _reservedStoreName,
      // Read before it is stored: a file a load refuses (incomplete) never
      // is. One whose header the plugin cannot read is stored and shown so:
      // the plugin's header check does not refuse it (its load goes on to
      // create the engine without reading the header, and a file that is no
      // model can still fail there), and the app plans no tower for it.
      accept: (path) async => switch (await _inspect(path)) {
        Ok() => null,
        Error(:final error) =>
          'The download from ${redactUrl(url)} is not a usable .litertlm '
              '(${'$error'.replaceAll(path, 'the downloaded file')})',
      },
    );
    return switch (stored) {
      Ok(:final value) => _adoptStored(
        value,
        UrlModelSource(
          Uri.parse(redactUrl(url)),
          sha256: sha256,
          sizeBytes: sizeBytes,
        ),
        knownAs: ModelStore.customFileNameFor(url),
      ),
      Error(:final error) => Result.error(error),
    };
  }

  /// Uses the `.litertlm` at [path] where it is (no copy): it must exist, be
  /// readable, and not be one a load refuses ([inspectLitertlm]: incomplete).
  /// It becomes the [ChatModelCandidate]: another file
  /// than the saved one starts from its defaults ([defaultsFor]: a build
  /// compiled for the NPU on the NPU), the same file picked again keeps its
  /// settings. Its towers are the ones it has ([ChatModelCandidate.bundle]).
  /// Nothing is saved: [apply] makes it the chat model. Its SHA-256 is
  /// computed in the background once it is the saved model, never blocking
  /// (a file only looked at is not read whole).
  Future<Result<CustomChatModel>> useLocalFile(String path) async {
    final absolute = File(path).absolute.path;
    final BundleReading info;
    switch (await _inspect(absolute)) {
      case Ok(:final value):
        info = value;
      case Error(error: LocalModelException(permissionDenied: true) && final e):
        return Result.error(await _withAdvice(absolute, e));
      case Error(:final error):
        return Result.error(error);
    }
    final FileStat stat;
    try {
      stat = await File(absolute).stat();
    } on FileSystemException catch (e) {
      return Result.error(
        LocalModelException(
          '$absolute cannot be read (${e.osError?.message ?? e.message}).',
        ),
      );
    }
    // stat() reports a file that went away after it was read as notFound
    // (size -1) rather than throwing: never saved as a model.
    if (stat.type == FileSystemEntityType.notFound || stat.size <= 0) {
      return Result.error(
        LocalModelException('$absolute is gone or empty: choose it again.'),
      );
    }
    final file = CustomModelFile(
      name: absolute.split('/').last,
      sizeBytes: stat.size,
      checksumMatched: false,
    );
    final adopted = await _adopt(
      file,
      LocalModelSource(absolute),
      bundle: info,
      path: absolute,
    );
    if (adopted is Ok<CustomChatModel>) {
      debugPrint(
        '[ChatModel] in place: $absolute (${switch (info) {
          BundleRead(:final bundle) => '$bundle',
          BundleNotRead(:final reason) => 'header not read: $reason',
        }}, ${towersLine(info)}'
        '${info.info?.isCompiledForNpu ?? false ? ', compiled for the NPU' : ''})',
      );
    }
    return adopted;
  }

  /// [error] with the advice of the models folder holding [path] (an
  /// Android push the app cannot read).
  Future<LocalModelException> _withAdvice(
    String path,
    LocalModelException error,
  ) async {
    for (final folder in _folders) {
      if (folder.permissionAdvice case final advice?
          when await folder.contains(path)) {
        return LocalModelException(
          '${error.message} $advice',
          permissionDenied: true,
        );
      }
    }
    return error;
  }

  /// Why the in-place [file] at [path] cannot be used, or null: it must be
  /// there with the size it had when chosen and still not be one a load
  /// refuses ([inspectLitertlm]). What the plugin makes of it becomes
  /// [_appliedBundle].
  Future<String?> _checkLocal(String path, CustomModelFile file) async {
    _appliedBundle = null;
    if (_quickLocalProblem(path, file) case final problem?) return problem;
    switch (await _inspect(path)) {
      case Ok(:final value):
        _appliedBundle = value;
        return null;
      case Error(error: LocalModelException(permissionDenied: true) && final e):
        return '${await _withAdvice(path, e)}';
      case Error(:final error):
        return '$error Pick the file again, or choose another .litertlm.';
    }
  }

  /// Checks the saved model [custom]'s file and reads what it says about
  /// itself ([_appliedBundle]). In place: a file gone from its path but found
  /// once in a models folder (same name and size) is used there (saved,
  /// unless [_persistMigration] is off), then it must be there, unchanged and
  /// not refused as a load refuses it ([_localProblem]). In the store: its
  /// copy must verify ([_savedStore]) and open ([_storedProblem]). Returns
  /// [custom] (with its new path when it was found elsewhere) and why the
  /// store could not be looked at (null when it could; nothing changed then).
  Future<(CustomChatModel, String?)> _checkSaved(CustomChatModel custom) async {
    if (custom case CustomChatModel(
      source: LocalModelSource(:final path),
      :final file?,
    )) {
      var model = custom;
      final found = await _foundInFolders(path, file);
      if (found != null) {
        model = custom.copyWith(source: LocalModelSource(found));
        debugPrint('[ChatModel] $path is gone; the same file is at $found');
        if (_persistMigration) {
          await _settings.write(
            Settings.customChatModel,
            CustomChatModelCodec.encode(model),
          );
        }
      }
      _storedProblem = null;
      _localProblem = await _checkLocal(found ?? path, file);
      return (model, null);
    }
    final file = custom.file;
    if (file == null) return (custom, null);
    switch (await _store.useCustomFile(file)) {
      case Error(:final error):
        return (custom, '$error');
      case Ok(value: final stored?):
        _savedStore = StoreFileReady(stored);
        _localProblem = null;
        // As for a file in place: refused only as a load refuses it.
        switch (await _inspect(stored)) {
          case Ok(:final value):
            _appliedBundle = value;
            _storedProblem = null;
          case Error(:final error):
            _appliedBundle = null;
            _storedProblem =
                '${file.name} in the model store cannot be used ($error). '
                'Import or download it again, or choose another .litertlm.';
            debugPrint('[ChatModel] $_storedProblem');
        }
      case Ok():
        // Missing or changed: the plan says so.
        _savedStore = _store.customFile.value;
        _localProblem = null;
        _storedProblem = null;
        _appliedBundle = null;
    }
    return (custom, null);
  }

  /// Checks the saved model's file again when its last check found a
  /// problem or could not read its header (Retry, Rescan): a file copied
  /// back to where it was, or into a models folder, loads without a restart.
  /// A saved file that was fine costs nothing here: [plan] checks its size
  /// on every use. Not while a transfer runs (the store's copy is being
  /// written); never touches the candidate.
  Future<void> recheck() => _settingsWrites.run(_recheckNow);

  /// [recheck], in the settings queue.
  Future<void> _recheckNow() async {
    final s = _state.value;
    final custom = s.custom;
    if (_closed || !s.loaded || custom == null || custom.file == null) return;
    final stale =
        _appliedBundle is! BundleRead ||
        switch (custom.source) {
          LocalModelSource(:final path) =>
            _localProblem != null ||
                _quickLocalProblem(path, custom.file!) != null,
          _ => _storedProblem != null || _savedStore is! StoreFileReady,
        };
    if (!stale || _store.busy.value) return;
    final (checked, unchecked) = await _checkSaved(custom);
    if (unchecked != null) {
      debugPrint(
        '[ChatModel] recheck: the file could not be looked at '
        '($unchecked)',
      );
      return;
    }
    final now = _state.value;
    _publish(
      ChatModelState(
        kind: now.kind,
        custom: checked,
        problem: now.problem,
        note: now.note,
        bundle: _appliedBundle,
        candidate: now.candidate,
        loaded: true,
      ),
    );
    final problem = switch (checked.source) {
      LocalModelSource() => _localProblem,
      _ => _storedProblem,
    };
    debugPrint(
      '[ChatModel] checked ${checked.displayName} again: '
      '${problem ?? (_appliedBundle == null ? 'not usable' : 'usable')}',
    );
    _hashSavedIfNeeded();
  }

  /// [path] (a saved file in place) when it is gone and exactly one file of
  /// its name and size is in the models folders: that file. iOS moves the
  /// app's folders when the app is reinstalled, so the saved absolute path
  /// goes stale while the file stayed in the app's models folder.
  Future<String?> _foundInFolders(String path, CustomModelFile file) async {
    try {
      _length(path);
      return null;
    } on FileSystemException catch (e) {
      if (!isNotFound(e)) return null;
    }
    final matches = [
      for (final folder in _folders)
        if (await folder.list() case Ok(:final value))
          for (final entry in value)
            if (entry.name == file.name && entry.sizeBytes == file.sizeBytes)
              entry.path,
    ];
    return matches.length == 1 ? matches.single : null;
  }

  /// The cheap part of [_checkLocal] (the file's length), for [plan]. Not
  /// `existsSync()`: it says false for a file the app may not search, which
  /// would read as "moved, renamed or deleted?" — the length's own error
  /// tells ENOENT from EACCES/EPERM.
  String? _quickLocalProblem(String path, CustomModelFile file) {
    final int size;
    try {
      size = _length(path);
    } on FileSystemException catch (e) {
      if (isPermissionDenied(e)) {
        final advice = _folders
            .where(
              (f) => f.permissionAdvice != null && f.containsResolved(path),
            )
            .map((f) => f.permissionAdvice)
            .firstOrNull;
        return '$path cannot be read (${e.osError?.message ?? e.message}).'
            '${advice == null ? '' : ' $advice'}';
      }
      if (!isNotFound(e)) {
        return '$path cannot be read (${e.osError?.message ?? e.message}).';
      }
      return '$path is not there any more (moved, renamed or deleted?). '
          'Copy it back, or pick another file in the models folder.';
    }
    if (size != file.sizeBytes) {
      return '$path changed since it was chosen (${file.sizeBytes} bytes '
          'then, $size now). Pick it again to use the new file.';
    }
    return null;
  }

  /// Computes the SHA-256 of the in-place file at [path] ([sizeBytes] long
  /// when it was chosen) in a worker isolate and puts it into the saved
  /// model if it is still that file ([_saveSha256]). Not after [dispose]
  /// (the app quit in the middle of an apply).
  void _hashInBackground(String path, int sizeBytes) {
    if (_closed) return;
    _cancelHash();
    final cancel = _hashCancel = Completer<void>();
    _hashPath = path;
    _hashing.value = true;
    unawaited(() async {
      final watch = Stopwatch()..start();
      try {
        final hex = await _ops.sha256OfFile(
          path,
          onProgress: (_, _) {},
          cancel: cancel.future,
        );
        if (await _settingsWrites.run(
          () => _saveSha256(path, sizeBytes, hex, cancel),
        )) {
          debugPrint(
            '[ChatModel] sha256 $hex of $path '
            '(${watch.elapsedMilliseconds} ms, in the background)',
          );
        }
      } on OperationCancelledException {
        // Another file was applied, or the app closes.
      } catch (e) {
        debugPrint('[ChatModel] hashing $path failed: $e');
      } finally {
        if (identical(_hashCancel, cancel)) {
          _hashCancel = null;
          _hashPath = null;
          if (!_closed) _hashing.value = false;
        }
      }
    }());
  }

  /// Stops the background SHA-256 (once: it may be stopped by a new hash
  /// before the app closes).
  void _cancelHash() {
    final cancel = _hashCancel;
    if (cancel != null && !cancel.isCompleted) cancel.complete();
  }

  /// Starts the background SHA-256 of the saved model's file in place when
  /// it has none yet, it is usable and it is not being hashed already (a
  /// running hash is never started over).
  void _hashSavedIfNeeded() {
    if (_state.value.custom
        case CustomChatModel(
          source: LocalModelSource(:final path),
          file: CustomModelFile(sha256: null, :final sizeBytes),
        )
        when _localProblem == null && _hashPath != path) {
      _hashInBackground(path, sizeBytes);
    }
  }

  /// Puts [hex] into the saved model as stored now, when that is still the
  /// file in place at [path] with [sizeBytes] (the file there now too: one
  /// replaced while it was hashed is another file) and the hash was not
  /// cancelled: only `file.sha256` changes (settings applied while it ran
  /// are kept). True when saved. Runs in the settings queue.
  Future<bool> _saveSha256(
    String path,
    int sizeBytes,
    String hex,
    Completer<void> cancel,
  ) async {
    if (_closed || cancel.isCompleted) return false;
    try {
      if (_length(path) != sizeBytes) return false;
    } on FileSystemException {
      return false;
    }
    final CustomChatModel stored;
    switch (await _settings.read(Settings.customChatModel)) {
      case Ok(:final value?):
        try {
          stored = CustomChatModelCodec.decode(value);
        } on FormatException catch (e) {
          debugPrint('[ChatModel] not saving the SHA-256 of $path: $e');
          return false;
        }
      case Ok():
        return false;
      case Error(:final error):
        debugPrint('[ChatModel] not saving the SHA-256 of $path: $error');
        return false;
    }
    if (stored
        case CustomChatModel(
          source: LocalModelSource(path: final storedPath),
          :final file?,
        )
        when storedPath == path && file.sizeBytes == sizeBytes) {
      final saved = stored.copyWith(
        file: CustomModelFile(
          name: file.name,
          sizeBytes: file.sizeBytes,
          sha256: hex,
          checksumMatched: false,
        ),
      );
      if (await _settings.write(
            Settings.customChatModel,
            CustomChatModelCodec.encode(saved),
          )
          case Error(:final error)) {
        debugPrint('[ChatModel] saving the SHA-256 of $path: $error');
        return false;
      }
      final now = _state.value;
      _publish(
        ChatModelState(
          kind: now.kind,
          custom: saved,
          problem: now.problem,
          note: now.note,
          loaded: true,
          bundle: _appliedBundle,
          candidate: now.candidate,
        ),
      );
      return true;
    }
    return false;
  }

  /// Settings for a first file, from what the file name and [bundle] (what
  /// `readLiteRtLmBundleInfo` made of it) say: the NPU for a build compiled
  /// for one ([LiteRtLmBundleInfo.isCompiledForNpu]: its text model runs only
  /// there, and the plugin refuses it on the GPU and the CPU), else the GPU
  /// (also for a header the plugin could not read: its load starts on the
  /// GPU too); the model type the name announces
  /// ([modelTypeFromFileName]); the context an NPU build's name announces
  /// (`…_ekv1280_…`), else 4096 on the NPU (the published Gemma 4 E2B NPU
  /// builds are benchmarked at a 4096 context on their model card) and the
  /// app's 8192 on GPU/CPU (agent turns, excerpts and a picture soon outgrow
  /// 4096); tools on for a Gemma-family file that is not compiled for the
  /// NPU (what Gemma 4 E2B gets: its skills work), off for other types and
  /// NPU builds, until the user knows the file has them. Nothing about the
  /// towers: the engine places them. A file in place, an imported and a
  /// downloaded one alike (the last two read from the store's copy). An NPU
  /// build starts on the NPU also where this device offers none: the card
  /// then says why it cannot be applied.
  CustomChatModel defaultsFor(
    CustomModelFile file,
    CustomModelSource source, {
    required BundleReading bundle,
  }) {
    final npuBuild = bundle.info?.isCompiledForNpu ?? false;
    final backend = npuBuild ? PreferredBackend.npu : PreferredBackend.gpu;
    final hint = contextHintFromFileName(file.name);
    var context =
        hint ?? (backend == PreferredBackend.npu ? 4096 : kLlmConfig.maxTokens);
    if (backend != PreferredBackend.npu && context < kMinGpuCpuContextTokens) {
      context = kMinGpuCpuContextTokens;
    }
    final type = modelTypeFromFileName(file.name);
    return CustomChatModel(
      displayName: _stem(file.name),
      source: source,
      file: file,
      modelType: type,
      backend: backend,
      maxTokens: context,
      tools: !npuBuild && kToolsByDefaultTypes.contains(type),
    );
  }

  Future<Result<CustomChatModel>> _adopt(
    CustomModelFile file,
    CustomModelSource source, {
    required BundleReading bundle,
    required String path,
    String? knownAs,
  }) => _settingsWrites.run(
    () => _adoptNow(file, source, bundle: bundle, path: path, knownAs: knownAs),
  );

  /// Makes [file] from [source] (at [path]) the candidate; nothing is saved.
  /// The same file as the candidate's or the saved model's keeps its
  /// settings; another file starts from its own [bundle] ([defaultsFor]),
  /// never from settings made for another file. The saved model's own file
  /// picked again with nothing to change is no candidate: the pick checked it
  /// again (a file put back where it was loads at the next Retry or reload).
  /// [knownAs]: the file name the user knows, for a first name (the store may
  /// keep the file under another).
  Future<Result<CustomChatModel>> _adoptNow(
    CustomModelFile file,
    CustomModelSource source, {
    required BundleReading bundle,
    required String path,
    String? knownAs,
  }) async {
    final s = _state.value;
    final previous = s.candidate?.model;
    // Unreadable saved settings are no settings to keep.
    final saved = s.problem == null ? s.custom : null;
    final model = switch ((previous, saved)) {
      (final p?, _) when _sameFile(p, file, source) => _refiled(
        p,
        file,
        source,
      ),
      (_, final c?) when _sameFile(c, file, source) => _refiled(
        c,
        file,
        source,
      ),
      _ => switch (defaultsFor(file, source, bundle: bundle)) {
        final defaults when knownAs != null && knownAs != file.name =>
          defaults.copyWith(displayName: _stem(knownAs)),
        final defaults => defaults,
      },
    };
    final again =
        saved != null &&
        _sameFile(saved, file, source) &&
        sameSaved(model.copyWith(file: saved.file), saved);
    // The candidate's copy and the saved model's (the engine may run it) are
    // kept; an earlier candidate's copy, abandoned, goes. A file in place is
    // not in the store: nothing of it is pruned.
    await _store.pruneCustom(
      keep: {
        if (source is! LocalModelSource) file.name,
        if (s.custom
            case CustomChatModel(
              file: final savedFile?,
              source: final savedSource,
            )
            when savedSource is! LocalModelSource)
          savedFile.name,
      },
    );
    // The store's slot shows this copy now; one of the same name went over
    // the saved model's file: that file is looked at again (another file
    // there blocks the plan, never loads with the saved settings).
    if (source is! LocalModelSource) await _lookAtSavedStoreFile();
    if (again) {
      _appliedBundle = bundle;
      if (source is LocalModelSource) {
        _localProblem = null;
      } else {
        _storedProblem = null;
      }
      _publish(
        ChatModelState(
          kind: s.kind,
          custom: s.custom,
          problem: s.problem,
          bundle: _appliedBundle,
          loaded: true,
        ),
      );
      debugPrint(
        '[ChatModel] ${file.name} is the saved model\'s file: checked again, '
        'nothing to apply',
      );
      _hashSavedIfNeeded();
      return Result.ok(saved);
    }
    _publish(
      ChatModelState(
        kind: s.kind,
        custom: s.custom,
        problem: s.problem,
        bundle: _appliedBundle,
        candidate: ChatModelCandidate(model: model, bundle: bundle, path: path),
        loaded: true,
      ),
    );
    debugPrint(
      '[ChatModel] picked ${file.name} (${file.sizeBytes} B, sha256 '
      '${file.sha256 ?? 'pending'}, '
      '${file.checksumMatched ? 'matched' : 'recorded'}) ${model.sourceLine}, '
      'not applied: ${model.settingsLine(bundle)}',
    );
    return Result.ok(model);
  }

  /// [model] with [file] from [source], the same file picked again: its
  /// settings kept; a name that was the old file's stem follows the new
  /// one's.
  static CustomChatModel _refiled(
    CustomChatModel model,
    CustomModelFile file,
    CustomModelSource source,
  ) => model.copyWith(
    file: file,
    source: source,
    displayName:
        model.file != null && model.displayName == _stem(model.file!.name)
        ? _stem(file.name)
        : null,
  );

  /// [file] from [source] is [current]'s file again: the same SHA-256 and
  /// size when both are known, else the same path and size in place.
  static bool _sameFile(
    CustomChatModel current,
    CustomModelFile file,
    CustomModelSource source,
  ) {
    final old = current.file;
    if (old == null || old.sizeBytes != file.sizeBytes) return false;
    if (old.sha256 != null && file.sha256 != null) {
      return old.sha256 == file.sha256;
    }
    return switch ((current.source, source)) {
      (LocalModelSource(path: final a), LocalModelSource(path: final b)) =>
        a == b,
      _ => false,
    };
  }

  /// [a] and [b] name the same file the same way: its source, name and size
  /// (a SHA-256 that came in since does not count). The settings may differ.
  static bool _sameRef(CustomChatModel a, CustomChatModel b) =>
      a.sourceLine == b.sourceLine &&
      a.file?.name == b.file?.name &&
      a.file?.sizeBytes == b.file?.sizeBytes;

  /// The candidate when [model] is its file (with any settings); null
  /// otherwise.
  ChatModelCandidate? _candidateFor(CustomChatModel model) =>
      switch (_state.value.candidate) {
        final c? when _sameRef(c.model, model) => c,
        _ => null,
      };

  /// [model] is the saved model's file (with any settings).
  bool _isSavedFile(CustomChatModel model) => switch (_state.value.custom) {
    final saved? => _sameRef(saved, model),
    null => false,
  };

  /// [a] and [b] save as the same settings (the persisted form compared).
  bool sameSaved(CustomChatModel a, CustomChatModel b) =>
      CustomChatModelCodec.encode(a) == CustomChatModelCodec.encode(b);

  /// Why [model] cannot be the chat model as it is; null when it can. The
  /// file must still be there; a build compiled for an NPU
  /// ([LiteRtLmBundleInfo.isCompiledForNpu], its file as last read) only on
  /// the NPU, as the plugin's load would refuse it elsewhere
  /// (flutter_edge_ai_litertlm's README, "Where a model runs") after the
  /// running model was unloaded.
  String? problemWith(CustomChatModel model) {
    if (model.displayName.trim().isEmpty) return 'Give the model a name.';
    final file = model.file;
    if (file == null) {
      return 'Import or download the .litertlm first.';
    }
    final candidate = _candidateFor(model);
    if (model.source case LocalModelSource(:final path)) {
      if (_quickLocalProblem(path, file) case final problem?) return problem;
    } else if (candidate != null) {
      if (_quickStoredProblem(candidate.path, file) case final problem?) {
        return problem;
      }
    } else if (_savedStore is! StoreFileReady) {
      return 'The file is not verified in the model store.';
    }
    if (CustomChatModel.contextProblem(model.maxTokens, model.backend)
        case final problem?) {
      return problem;
    }
    // A build compiled for an NPU runs only on the NPU: the plugin refuses it
    // on the GPU and the CPU (wrongBackend), and only once the running model
    // was unloaded. The candidate's file, else the saved one as last read.
    final bundle = (candidate?.bundle ?? _appliedBundle)?.info;
    if (model.backend != PreferredBackend.npu) {
      if (npuOnlyReason(bundle) case final reason?) {
        return '${model.backend.name.toUpperCase()}: not for this file '
            '($reason). Choose NPU.';
      }
    }
    // The NPU only where the Models screen offers it ([npu]).
    if (model.backend == PreferredBackend.npu) {
      if (_npu() case NpuUnavailable(:final reason)) {
        return 'The NPU is not available here: $reason.';
      }
    }
    return null;
  }

  /// Saves [model]'s settings and makes it the chat model: the candidate's
  /// file (then no candidate is left), or the saved model's own file with
  /// other settings (a candidate stays). Refused for any other file, and as
  /// [problemWith] says. The caller reloads the chat model.
  Future<Result<void>> apply(CustomChatModel model) =>
      _settingsWrites.run(() => _applyNow(model));

  Future<Result<void>> _applyNow(CustomChatModel model) async {
    final candidate = _candidateFor(model);
    if (candidate == null) {
      if (!_isSavedFile(model)) {
        return const Result.error(
          InvalidChatModelException(
            'Choose the .litertlm again: this file is neither the one picked '
            'nor the saved one.',
          ),
        );
      }
      // Its last check found a problem or could not read its header: looked
      // at again (a file put back).
      if (_appliedBundle is! BundleRead ||
          _localProblem != null ||
          _storedProblem != null) {
        await _recheckNow();
      }
    }
    // The file's facts as they are now (a SHA-256 that came in meanwhile).
    final latest = candidate?.model ?? _state.value.custom;
    final applying = latest != null && _sameRef(latest, model)
        ? model.copyWith(file: latest.file)
        : model;
    if (problemWith(applying) case final problem?) {
      return Result.error(InvalidChatModelException(problem));
    }
    if (candidate == null && _appliedBundle == null) {
      final reason =
          _localProblem ??
          _storedProblem ??
          'the file has not been read: choose the .litertlm again.';
      return Result.error(
        InvalidChatModelException('${applying.displayName}: $reason'),
      );
    }
    var slotMoved = false;
    if (candidate != null && applying.source is! LocalModelSource) {
      // The store's slot to the candidate's copy, verified again.
      switch (await _store.useCustomFile(applying.file)) {
        case Ok(value: _?):
          slotMoved = true;
        case Ok():
          final reason = _store.customFile.value is StoreFileUnverified
              ? _storeChanged(applying.file!)
              : _storeMissing(applying.file!);
          await _lookAtSavedStoreFile();
          return Result.error(InvalidChatModelException(reason));
        case Error(:final error):
          return Result.error(error);
      }
    }
    if (await _saveCustomThenChoice(applying, ChatModelKind.custom) case Error(
      :final error,
    )) {
      // Back to the saved model's copy.
      if (slotMoved) await _lookAtSavedStoreFile();
      return Result.error(error);
    }
    if (candidate != null) {
      _appliedBundle = candidate.bundle;
      _localProblem = null;
      _storedProblem = null;
      _savedStore = applying.source is LocalModelSource
          ? const StoreFileMissing()
          : StoreFileReady(candidate.path);
    }
    _publish(
      ChatModelState(
        kind: ChatModelKind.custom,
        custom: applying,
        loaded: true,
        bundle: _appliedBundle,
        candidate: candidate != null ? null : _state.value.candidate,
      ),
    );
    debugPrint(
      '[ChatModel] using ${applying.displayName} '
      '(${_settingsLineOf(applying)})',
    );
    _hashSavedIfNeeded();
    return const Result.ok(null);
  }

  /// [model]'s settings line with the towers its file has; a file not read
  /// yet (a problem at launch) says so instead.
  String _settingsLineOf(CustomChatModel model) => switch (_appliedBundle) {
    final bundle? => model.settingsLine(bundle),
    null => '${model.settingsLine()}, file not read',
  };

  /// Saves [custom] as the custom model, then [kind] as the choice. In this
  /// order every step is a configuration a launch can load: the custom
  /// model alone is kept while another choice stands, or is the new chat
  /// model when custom is chosen already. When the choice cannot be saved,
  /// the custom model is put back as it was, so a restart loads what the
  /// screen still shows (the keys and their format are unchanged).
  Future<Result<void>> _saveCustomThenChoice(
    CustomChatModel custom,
    ChatModelKind kind,
  ) => _settings.writePair(
    (Settings.customChatModel, CustomChatModelCodec.encode(custom)),
    (Settings.chatModel, kind.name),
  );

  /// "Run on GPU" / "Run on CPU" / "Run on NPU" after a failed load: the
  /// saved model (never the candidate) on [backend], saved. GPU and CPU need
  /// at least 1024 tokens of context (the engine raises a smaller one
  /// anyway), so a smaller NPU context is raised to it, visibly. Refused as
  /// [problemWith] says: a build compiled for the NPU on the GPU or the CPU.
  Future<Result<void>> runOn(PreferredBackend backend) =>
      _settingsWrites.run(() async {
        final custom = _state.value.custom;
        if (custom == null) {
          return const Result.error(
            InvalidChatModelException('No custom chat model is set up.'),
          );
        }
        final context =
            backend != PreferredBackend.npu &&
                custom.maxTokens < kMinGpuCpuContextTokens
            ? kMinGpuCpuContextTokens
            : custom.maxTokens;
        return _applyNow(custom.copyWith(backend: backend, maxTokens: context));
      });

  /// The Models screen closes: a candidate used in place is dropped (the
  /// card shows the saved model next time; the folder may have changed by
  /// then). One in the model store (imported, downloaded) stays for this app
  /// session, until the next pick, Apply or a restart: a download is not
  /// thrown away by leaving the screen. Candidates are never hashed, so no
  /// background SHA-256 of the saved file was stopped by the pick; one
  /// missing is started again.
  Future<void> dropInPlaceCandidate() => _settingsWrites.run(() async {
    if (_closed) return;
    final candidate = _state.value.candidate;
    if (candidate == null) return;
    if (candidate.model.source is LocalModelSource) {
      _publish(_state.value.withCandidate(null));
      debugPrint(
        '[ChatModel] ${candidate.model.file?.name} was not applied: dropped',
      );
    }
    _hashSavedIfNeeded();
  });

  /// After a reload: deletes the custom files nothing refers to any more (a
  /// file replaced while the engine still ran it, a dropped candidate's).
  /// Keeps the saved custom model's file, chosen or not, and the
  /// candidate's.
  Future<void> pruneUnused() async {
    final s = _state.value;
    await _store.pruneCustom(
      keep: {
        ?s.custom?.file?.name,
        if (s.candidate?.model case CustomChatModel(:final file?, :final source)
            when source is! LocalModelSource)
          file.name,
      },
    );
  }

  /// Stops a running custom download or import (its partial file is kept).
  void cancel() => _store.cancel();

  static String _stem(String name) =>
      name.toLowerCase().endsWith(kCustomModelExtension)
      ? name.substring(0, name.length - kCustomModelExtension.length)
      : name;

  void _publish(ChatModelState state) {
    if (!_closed) _state.value = state;
  }

  /// The same state again, for what changed beside it ([savedFile]).
  void _republish() =>
      _publish(_state.value.withCandidate(_state.value.candidate));

  void dispose() {
    if (_closed) return;
    _closed = true;
    _cancelHash();
    _state.dispose();
    _hashing.dispose();
  }
}
