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

/// The chat model choice as saved, for the Models screen.
final class const ChatModelState({
  required final ChatModelKind kind,

  /// The user's own model; kept while [kind] is none.
  final CustomChatModel? custom,

  /// The saved choice or custom settings could not be read; shown with the
  /// way out, never replaced by a default.
  final String? problem,

  /// Something to tell once, not a problem (a retired saved choice).
  final String? note,

  /// [ChatModelRepository.load] has run.
  final bool loaded = false,
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
class ChatModelRepository implements ChatModelPlanner {
  ChatModelRepository({
    required this._settings,
    required this._store,
    this._npu = probeNpu,
    required this._folders,
    this._ops = const ModelFileOps(),
    this._length = _fileLength,
    this._persistMigration = true,
  });

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
  /// unreadable, not a `.litertlm`, changed); null when it can.
  String? _localProblem;

  /// The background SHA-256 of a file used in place runs.
  final ValueNotifier<bool> _hashing = ValueNotifier(false);
  ValueListenable<bool> get hashing => _hashing;
  Completer<void>? _hashCancel;

  /// The writes of the chat model settings, each with the state it
  /// publishes, one after another: one never lands between another's read
  /// and its save (the background SHA-256 used to save the old settings over
  /// an apply's). With none queued a write starts at once.
  final SerialQueue _settingsWrites = SerialQueue.atOnceWhenIdle();

  final ValueNotifier<ChatModelState> _state = ValueNotifier(
    const ChatModelState(kind: ChatModelKind.none),
  );
  bool _closed = false;

  ValueListenable<ChatModelState> get state => _state;

  /// The custom model's file in the store (progress while it is stored).
  ValueListenable<StoreFileState> get customFile => _store.customFile;

  /// A download or import (of any model) runs.
  ValueListenable<bool> get busy => _store.busy;

  /// Whether flutter_edge_ai would accept the NPU on this device.
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
        // An earlier build's "Gemma 4 E2B, downloaded by the app": its
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
        // An earlier build saved a download's whole link: decoding dropped
        // its query and fragment (a signed link's token); so does the file.
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
    if (custom case CustomChatModel(
      source: LocalModelSource(:final path),
      :final file?,
    )) {
      final found = await _foundInFolders(path, file);
      if (found != null) {
        custom = custom.copyWith(source: LocalModelSource(found));
        debugPrint('[ChatModel] $path is gone; the same file is at $found');
        if (_persistMigration) {
          await _settings.write(
            Settings.customChatModel,
            CustomChatModelCodec.encode(custom),
          );
        }
      }
      _localProblem = await _checkLocal(found ?? path, file);
    } else if (custom?.file case final file?) {
      if (await _store.useCustomFile(file) case Error(:final error)) {
        problems.add('its file could not be checked ($error)');
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
      ),
    );
    debugPrint(
      '[ChatModel] ${kind.name}'
      '${custom == null ? '' : ' · custom ${custom.displayName} '
                '(${custom.settingsLine}) ${custom.sourceLine}'}',
    );
    if (custom
        case CustomChatModel(
          source: LocalModelSource(:final path),
          file: CustomModelFile(sha256: null),
        )
        when _localProblem == null) {
      _hashInBackground(path);
    }
    return problem == null
        ? const Result.ok(null)
        : Result.error(InvalidChatModelException(problem));
  }

  /// After an upgrade: the Gemma 4 E2B an earlier build downloaded and
  /// verified (`<store>/gemma4E2b/`, its `.sha256` record matching), made
  /// the chat model in place with Gemma 4 E2B's settings (GPU, 8192,
  /// images and tools on) and a one-time note. Null when there is none.
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
    if (await inspectLitertlm(path) case Error(:final error)) {
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
      supportImage: true,
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
      _publish(
        ChatModelState(
          kind: ChatModelKind.custom,
          custom: model,
          note: kEarlierGemmaAdoptedNote,
          loaded: true,
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
    if (custom.source case LocalModelSource(:final path)) {
      final problem = _localProblem ?? _quickLocalProblem(path, file);
      return problem == null
          ? CustomChatPlan(path: path, model: custom)
          : ChatPlanBlocked('${custom.displayName}: $problem');
    }
    return switch (_store.customFile.value) {
      StoreFileReady(:final path) => CustomChatPlan(path: path, model: custom),
      StoreFileUnverified() => ChatPlanBlocked(
        '${custom.displayName}: ${file.name} changed on disk since it was '
        'stored (its size or checksum record differs). Import or download it '
        'again, or choose another .litertlm.',
      ),
      StoreFileMissing() => ChatPlanBlocked(
        '${custom.displayName}: ${file.name} is not in the model store any '
        'more. Import or download it again, or choose another .litertlm.',
      ),
      _ => ChatPlanBlocked(
        '${custom.displayName}: ${file.name} is still being stored.',
      ),
    };
  }

  /// Imports the `.litertlm` at [path] (see `ModelStore.importCustomFile`)
  /// as the custom model's file: a first one gets default settings, a
  /// replacement keeps the current ones. Saved, but the choice is not
  /// changed: [apply] makes it the chat model.
  Future<Result<CustomChatModel>> importFile(
    String path, {
    String? moveFrom,
  }) async {
    final stored = await _store.importCustomFile(path, moveFrom: moveFrom);
    return switch (stored) {
      Ok(:final value) => _adopt(value, ImportedModelSource(path)),
      Error(:final error) => Result.error(error),
    };
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
    );
    return switch (stored) {
      Ok(:final value) => _adopt(
        value,
        UrlModelSource(
          Uri.parse(redactUrl(url)),
          sha256: sha256,
          sizeBytes: sizeBytes,
        ),
      ),
      Error(:final error) => Result.error(error),
    };
  }

  /// Uses the `.litertlm` at [path] where it is (no copy): it must exist, be
  /// readable and start with the `LITERTLM` header. A first file gets
  /// default settings, a replacement keeps the current ones; images are
  /// off when the header has no vision section, and an NPU-only build
  /// (`backend_constraint: npu`) defaults to the NPU where it is offered.
  /// Saved, but the choice is not changed: [apply] makes it the chat model.
  /// The SHA-256 is computed in the background afterwards, never blocking.
  Future<Result<CustomChatModel>> useLocalFile(String path) async {
    final absolute = File(path).absolute.path;
    final LitertlmInfo info;
    switch (await inspectLitertlm(absolute)) {
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
    // stat() reports a file that went away after its header was read as
    // notFound (size -1) rather than throwing: never saved as a model.
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
      header: info,
    );
    if (adopted is Ok<CustomChatModel>) {
      _localProblem = null;
      debugPrint(
        '[ChatModel] in place: $absolute (${stat.size} B, litertlm '
        '${info.version}, sections ${info.modelTypes.join(',')}'
        '${info.npuOnly ? ', npu-only' : ''})',
      );
      _hashInBackground(absolute);
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
  /// there with the size it had when chosen and still be a `.litertlm`.
  Future<String?> _checkLocal(String path, CustomModelFile file) async {
    if (_quickLocalProblem(path, file) case final problem?) return problem;
    return switch (await inspectLitertlm(path)) {
      Ok() => null,
      Error(error: LocalModelException(permissionDenied: true) && final e) =>
        '${await _withAdvice(path, e)}',
      Error(:final error) =>
        '$error Pick the file again, or choose another .litertlm.',
    };
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

  /// Computes the SHA-256 of the in-place file at [path] in a worker
  /// isolate and saves it into the custom model if it is still that file.
  /// Not after [dispose] (the app quit in the middle of a pick).
  void _hashInBackground(String path) {
    if (_closed) return;
    _hashCancel?.complete();
    final cancel = _hashCancel = Completer<void>();
    _hashing.value = true;
    unawaited(() async {
      final watch = Stopwatch()..start();
      try {
        final hex = await _ops.sha256OfFile(
          path,
          onProgress: (_, _) {},
          cancel: cancel.future,
        );
        if (await _settingsWrites.run(() => _saveSha256(path, hex, cancel))) {
          debugPrint(
            '[ChatModel] sha256 $hex of $path '
            '(${watch.elapsedMilliseconds} ms, in the background)',
          );
        }
      } on OperationCancelledException {
        // Another file was chosen, or the app closes.
      } catch (e) {
        debugPrint('[ChatModel] hashing $path failed: $e');
      } finally {
        if (identical(_hashCancel, cancel)) {
          _hashCancel = null;
          if (!_closed) _hashing.value = false;
        }
      }
    }());
  }

  /// Puts [hex] into the custom model as stored now, when that is still the
  /// file in place at [path] and the hash was not cancelled: only
  /// `file.sha256` changes (settings applied while it ran are kept). True
  /// when saved. Runs in the settings queue.
  Future<bool> _saveSha256(
    String path,
    String hex,
    Completer<void> cancel,
  ) async {
    if (_closed || cancel.isCompleted) return false;
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
        when storedPath == path) {
      final hashed = stored.copyWith(
        file: CustomModelFile(
          name: file.name,
          sizeBytes: file.sizeBytes,
          sha256: hex,
          checksumMatched: false,
        ),
      );
      if (await _settings.write(
            Settings.customChatModel,
            CustomChatModelCodec.encode(hashed),
          )
          case Error(:final error)) {
        debugPrint('[ChatModel] saving the SHA-256 of $path: $error');
        return false;
      }
      _publish(
        ChatModelState(
          kind: _state.value.kind,
          custom: hashed,
          problem: _state.value.problem,
          loaded: true,
        ),
      );
      return true;
    }
    return false;
  }

  /// Settings for a first file, from what this device, the file name and
  /// (for a file in place) its header say: the NPU where flutter_edge_ai
  /// offers it — for a file in place only when the header marks an NPU
  /// build (`backend_constraint: npu`) — else the GPU; the model type the
  /// name announces ([modelTypeFromFileName]); the context an NPU build's
  /// name announces (`…_ekv1280_…`), else 4096 on the NPU (the official
  /// Gemma 4 Qualcomm NPU builds are compiled for 4096: their KV-cache
  /// inputs are 4096 long) and the app's 8192 on GPU/CPU (agent turns,
  /// excerpts and a picture outgrow 4096 in two or three turns); images on
  /// only when the header has a vision section (off when unknown); tools on
  /// for a Gemma-family file in place that is not an NPU build (what Gemma
  /// 4 E2B gets: its skills work as with the old built-in setting), off for
  /// other types, NPU builds and files whose header was not read, until the
  /// user knows the file has them.
  CustomChatModel defaultsFor(
    CustomModelFile file,
    CustomModelSource source, {
    LitertlmInfo? header,
  }) {
    final npuOffered = _npu() is NpuAvailable;
    final backend = (header == null ? npuOffered : header.npuOnly && npuOffered)
        ? PreferredBackend.npu
        : PreferredBackend.gpu;
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
      supportImage: header?.hasVision ?? false,
      tools:
          header != null &&
          !header.npuOnly &&
          kToolsByDefaultTypes.contains(type),
    );
  }

  Future<Result<CustomChatModel>> _adopt(
    CustomModelFile file,
    CustomModelSource source, {
    LitertlmInfo? header,
  }) => _settingsWrites.run(() => _adoptNow(file, source, header: header));

  Future<Result<CustomChatModel>> _adoptNow(
    CustomModelFile file,
    CustomModelSource source, {
    LitertlmInfo? header,
  }) async {
    final current = _state.value.custom;
    var updated = switch (current) {
      final c? when _state.value.problem == null => c.copyWith(
        file: file,
        source: source,
        displayName: c.file != null && c.displayName == _stem(c.file!.name)
            ? _stem(file.name)
            : null,
      ),
      _ => defaultsFor(file, source, header: header),
    };
    // What the header knows wins over a kept setting that cannot work.
    if (header != null && !header.hasVision && updated.supportImage) {
      updated = updated.copyWith(supportImage: false);
    }
    final repairing = _state.value.problem != null;
    final kind = _state.value.kind;
    // Repairing: the unreadable choice is saved again as what it was taken
    // for.
    final saved = repairing
        ? await _saveCustomThenChoice(updated, kind)
        : await _settings.write(
            Settings.customChatModel,
            CustomChatModelCodec.encode(updated),
          );
    if (saved case Error(:final error)) return Result.error(error);
    // The previous file may be what the engine runs (the custom model is
    // the chosen one): kept until a reload replaces it ([pruneUnused]). A
    // file in place is not in the store: nothing of it is pruned.
    final inUse = kind == ChatModelKind.custom ? current?.file?.name : null;
    await _store.pruneCustom(
      keep: {if (source is! LocalModelSource) file.name, ?inUse},
    );
    _publish(ChatModelState(kind: kind, custom: updated, loaded: true));
    debugPrint(
      '[ChatModel] custom file ${file.name} (${file.sizeBytes} B, sha256 '
      '${file.sha256 ?? 'pending'}, '
      '${file.checksumMatched ? 'matched' : 'recorded'}) ${updated.sourceLine}',
    );
    return Result.ok(updated);
  }

  /// [a] and [b] save as the same settings (the persisted form compared).
  bool sameSaved(CustomChatModel a, CustomChatModel b) =>
      CustomChatModelCodec.encode(a) == CustomChatModelCodec.encode(b);

  /// Why [model] cannot be the chat model as it is; null when it can.
  String? problemWith(CustomChatModel model) {
    if (model.displayName.trim().isEmpty) return 'Give the model a name.';
    final file = model.file;
    if (file == null) {
      return 'Import or download the .litertlm first.';
    }
    if (model.source case LocalModelSource(:final path)) {
      if (_quickLocalProblem(path, file) case final problem?) return problem;
    } else if (_store.customFile.value is! StoreFileReady) {
      return 'The file is not verified in the model store.';
    }
    if (CustomChatModel.contextProblem(model.maxTokens, model.backend)
        case final problem?) {
      return problem;
    }
    if (model.backend == PreferredBackend.npu) {
      if (_npu() case NpuUnavailable(:final reason)) {
        return 'The NPU is not available here: $reason.';
      }
    }
    return null;
  }

  /// Saves [model]'s settings and makes it the chat model. The caller
  /// reloads the chat model.
  Future<Result<void>> apply(CustomChatModel model) =>
      _settingsWrites.run(() => _applyNow(model));

  Future<Result<void>> _applyNow(CustomChatModel model) async {
    if (problemWith(model) case final problem?) {
      return Result.error(InvalidChatModelException(problem));
    }
    if (await _saveCustomThenChoice(model, ChatModelKind.custom) case Error(
      :final error,
    )) {
      return Result.error(error);
    }
    _publish(
      ChatModelState(kind: ChatModelKind.custom, custom: model, loaded: true),
    );
    debugPrint(
      '[ChatModel] using ${model.displayName} (${model.settingsLine})',
    );
    return const Result.ok(null);
  }

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

  /// "Run on GPU" / "Run on CPU" after a failed load: the same model on
  /// [backend], saved. GPU and CPU need at least 1024 tokens of context (the
  /// engine raises a smaller one anyway), so a smaller NPU context is raised
  /// to it, visibly.
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

  /// After a reload: deletes the custom files nothing refers to any more
  /// (a file replaced while the engine still ran it). Keeps the saved
  /// custom model's file, chosen or not.
  Future<void> pruneUnused() async {
    final file = _state.value.custom?.file;
    await _store.pruneCustom(keep: {?file?.name});
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

  void dispose() {
    if (_closed) return;
    _closed = true;
    _hashCancel?.complete();
    _state.dispose();
    _hashing.dispose();
  }
}
