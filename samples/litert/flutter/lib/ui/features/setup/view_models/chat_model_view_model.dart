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

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart' show PlatformException;
import 'package:flutter_edge_ai/flutter_edge_ai.dart'
    show ModelType, PreferredBackend;

import '../../../../data/repositories/chat_model_repository.dart';
import '../../../../domain/models/chat_model.dart';
import '../../../../domain/models/model_id.dart';
import '../../../../domain/models/model_source_resolver.dart';
import '../../../../domain/models/model_state.dart';
import '../../../../domain/models/npu_availability.dart';
import '../../../../domain/models/provisioning.dart';
import '../../../../domain/ports/model_file_picker.dart';
import '../../../../domain/ports/model_states.dart';
import '../../../../domain/use_cases/chat_model_switcher.dart';
import '../../../../utils/command.dart';
import '../../../../utils/result.dart';

/// "Download from URL…": what the user typed, checked.
final class const ModelUrlRequest({
  required final Uri url,
  final String? sha256,
  final int? sizeBytes,
}) {
  /// [url] must be an absolute https link (http only for a local server)
  /// without a user name or password in it ([kUrlCredentialsMessage]);
  /// [sha256] empty or 64 hex digits (any case); [size] empty or a positive
  /// whole number of bytes. Ok, or the first problem.
  static Result<ModelUrlRequest> parse(String url, String sha256, String size) {
    final uri = Uri.tryParse(url.trim());
    final loopback =
        uri != null && (uri.host == 'localhost' || uri.host == '127.0.0.1');
    if (uri == null ||
        !uri.hasAuthority ||
        !(uri.scheme == 'https' || (uri.scheme == 'http' && loopback))) {
      return const Result.error(
        FormatException('Enter an absolute https:// link to the file.'),
      );
    }
    if (uri.userInfo.isNotEmpty) {
      return const Result.error(FormatException(kUrlCredentialsMessage));
    }
    final sha = sha256.trim().toLowerCase();
    if (sha.isNotEmpty && !isSha256Hex(sha)) {
      return const Result.error(
        FormatException(
          'The SHA-256 must be 64 hex digits (or leave it empty).',
        ),
      );
    }
    final sizeText = size.trim().replaceAll(RegExp(r'[\s_,]'), '');
    final bytes = sizeText.isEmpty ? null : int.tryParse(sizeText);
    if (sizeText.isNotEmpty && (bytes == null || bytes <= 0)) {
      return const Result.error(
        FormatException(
          'The size is a whole number of bytes (or leave it empty).',
        ),
      );
    }
    return Result.ok(
      ModelUrlRequest(
        url: uri,
        sha256: sha.isEmpty ? null : sha,
        sizeBytes: bytes,
      ),
    );
  }
}

/// What the user can do after the chat model failed to load.
enum ChatModelAction { runOnGpu, runOnCpu }

/// One models folder as the card shows it: its path (or why it has none),
/// the `.litertlm` files in it and how to get one there.
final class const ModelsFolderView({
  required final String label,
  final String? path,
  final String? error,
  final List<LocalModelEntry> files = const [],
  final String? hint,
});

/// The Models screen's "Chat model" section. The app ships no chat model:
/// the user chooses a `.litertlm` — from a models folder, a path, an import
/// or a link — and sets its backend, context, images and tools; when a load
/// fails, the explicit ways out.
/// Never switches models by itself.
class ChatModelViewModel extends ChangeNotifier {
  ChatModelViewModel({
    required this._chatModels,
    required this._models,
    required this._switcher,
    required this._picker,
    this._sources = const ModelSourceResolver(),
  }) {
    importFile = Command0<CustomChatModel>(_importFile);
    download = Command1<CustomChatModel, ModelUrlRequest>(_download);
    apply = Command0<void>(_apply);
    runOn = Command1<void, PreferredBackend>(_runOn);
    useLocal = Command1<CustomChatModel, String>(_useLocal);
    rescan = Command0<void>(_rescan);
    for (final command in _commands) {
      command.addListener(notifyListeners);
    }
    for (final listenable in _listenables) {
      listenable.addListener(_onChanged);
    }
    _resetDraft();
    unawaited(rescan.execute());
  }

  final ChatModelRepository _chatModels;
  final ModelStates _models;
  final ChatModelSwitcher _switcher;
  final ModelFilePicker _picker;

  /// What the chat slot loads for the saved choice: the rule setup loads by
  /// (the build's defines by default, like `ModelRepository`).
  final ModelSourceResolver _sources;
  bool _disposed = false;

  /// Picks and imports a `.litertlm` (desktop, iOS).
  late final Command0<CustomChatModel> importFile;

  /// Downloads a `.litertlm` from a link.
  late final Command1<CustomChatModel, ModelUrlRequest> download;

  /// Saves the draft and makes it the chat model, then reloads it.
  late final Command0<void> apply;

  /// "Run on GPU" / "Run on CPU" after a failed load.
  late final Command1<void, PreferredBackend> runOn;

  /// Uses a `.litertlm` in place: a file of the models folder, or the path
  /// typed in "Path…".
  late final Command1<CustomChatModel, String> useLocal;

  /// Lists the models folder again; with nothing chosen, selects the only
  /// file found.
  late final Command0<void> rescan;

  List<Command<Object?>> get _commands => [
    importFile,
    download,
    apply,
    runOn,
    useLocal,
    rescan,
  ];

  List<Listenable> get _listenables => [
    _chatModels.state,
    _chatModels.customFile,
    _chatModels.busy,
    _models.states,
    _models.preparing,
    _switcher.busy,
    _chatModels.hashing,
  ];

  ChatModelState get _state => _chatModels.state.value;

  /// The saved choice.
  ChatModelKind get active => _state.kind;

  /// The saved custom settings could not be read.
  String? get problem => _state.problem;

  /// Said once (a retired saved choice: Gemma 4 E2B is no longer
  /// downloaded).
  String? get note => _state.note;

  /// No chat model chosen and none running: the card leads with how to
  /// choose one.
  bool get noModel => active == ChatModelKind.none && _slot is! ModelReady;

  /// A command or a store operation runs, the models load, or the chat
  /// model is in use by an exclusive operation (a reload, the self-test).
  bool get busy =>
      _commands.any((c) => c.running) ||
      _chatModels.busy.value ||
      _models.preparing.value ||
      _switcher.busy.value;

  // ---- The file ----

  CustomChatModel? get saved => _state.custom;

  /// `mine.litertlm · 1.21 GB · sha256 1a2b3c4d… (computed) · imported from …`
  /// (a file in place: `in place: /path`, the SHA-256 once computed).
  String? get fileLine {
    final model = saved;
    final file = model?.file;
    if (model == null || file == null) return null;
    final state = model.source is LocalModelSource
        ? null
        : switch (_chatModels.customFile.value) {
            StoreFileReady() => null,
            StoreFileUnverified() => 'CHANGED ON DISK',
            StoreFileMissing() => 'MISSING',
            _ => null,
          };
    final sha = switch (file.sha256) {
      final hex? =>
        'sha256 ${hex.substring(0, 12)}… '
            '(${file.checksumMatched ? 'matches the one entered' : 'computed'})',
      null when _chatModels.hashing.value => 'sha256 computing…',
      null => 'sha256 not computed',
    };
    return [
      file.name,
      _size(file.sizeBytes),
      sha,
      model.sourceLine,
      ?state,
    ].join(' · ');
  }

  // ---- The models folders (files used in place) ----

  /// Each models folder: the app's own first (created on first run), then,
  /// on Android, `/data/local/tmp/litert-models`.
  List<ModelsFolderView> get folders => _folders;
  List<ModelsFolderView> _folders = const [];

  /// The app's own folder's path; null until listed or when it cannot be
  /// created.
  String? get folderPath => _folders.firstOrNull?.path;

  /// Every `.litertlm` found, folder by folder.
  @visibleForTesting
  List<LocalModelEntry> get localFiles => [
    for (final folder in _folders) ...folder.files,
  ];

  /// [entry] is the saved custom model's file.
  bool isSelected(LocalModelEntry entry) => switch (saved?.source) {
    LocalModelSource(:final path) => path == entry.path,
    _ => false,
  };

  /// [entry] can be picked: another file than the selected one, or the
  /// selected one at another size than when it was chosen (chosen while it
  /// was still being copied).
  bool canPick(LocalModelEntry entry) =>
      !isSelected(entry) || entry.sizeBytes != saved?.file?.sizeBytes;

  /// `2.59 GB · 2026-10-06 21:42`.
  static String entryLine(LocalModelEntry e) {
    final m = e.modified.toLocal();
    String two(int n) => n.toString().padLeft(2, '0');
    return '${_size(e.sizeBytes)} · ${m.year}-${two(m.month)}-${two(m.day)} '
        '${two(m.hour)}:${two(m.minute)}';
  }

  /// How to get a file into the folder at [index] ([path]) from a computer.
  String _hintFor(int index, String path) => switch ((_picker.support, index)) {
    (ImportUnsupported(), 0) =>
      'Launch the app once (it creates this folder), then copy the '
          '.litertlm here and Rescan:\nadb push model.litertlm $path/',
    (ImportUnsupported(), _) =>
      'Readable whatever the order:\nadb shell mkdir -p $path\n'
          'adb push model.litertlm $path/',
    _ => 'Copy a .litertlm into this folder, then Rescan.',
  };

  /// The saved file used in place cannot be used (missing, changed); null
  /// otherwise.
  String? get localProblem => switch (_chatModels.plan) {
    ChatPlanBlocked(:final reason) when saved?.source is LocalModelSource =>
      reason,
    _ => null,
  };

  /// The custom file's transfer, as a label and a fraction; null when idle.
  (String, double)? get fileProgress => switch (_chatModels.customFile.value) {
    StoreFileDownloading(:final received, :final total, :final attempt) => (
      '${attempt > 1 ? 'Reconnecting (attempt $attempt)' : 'Downloading'} '
          '${_size(received)} of ${_size(total)}',
      total == 0 ? 0 : received / total,
    ),
    StoreFileVerifying(:final processed, :final total) => (
      'Verifying (SHA-256) ${_percent(processed, total)}',
      total == 0 ? 0 : processed / total,
    ),
    StoreFileCopying(:final copied, :final total) => (
      'Copying ${_size(copied)} of ${_size(total)}',
      total == 0 ? 0 : copied / total,
    ),
    _ => null,
  };

  /// How this platform picks the file; Android cannot (file_selector).
  ImportSupport get importSupport => _picker.support;

  /// Why "Import file…" is not available here; null when it is. On
  /// Android the models folder replaces it (the system picker reads whole
  /// files into memory and breaks above 2 GB).
  String? get importUnavailableReason => switch (_picker.support) {
    ImportUnsupported() =>
      'No import on Android: adb push the file into one of the folders '
          'above and pick it, or use "Download from URL…".',
    _ => null,
  };

  bool get canImport => importUnavailableReason == null && !busy;

  bool get canDownload => !busy;

  /// The last in-place pick's error (missing, unreadable, not a .litertlm).
  String? get localError => switch (useLocal.result) {
    Error(:final error) => '$error',
    _ => null,
  };

  /// The last import's or download's error.
  String? get transferError => switch ((importFile.result, download.result)) {
    (Error(error: _PickCancelled()), _) => null,
    (Error(error: OperationCancelledException()), _) ||
    (
      _,
      Error(error: OperationCancelledException()),
    ) => 'Cancelled. A download resumes where it stopped.',
    (Error(:final error), _) => 'Import failed: ${_reason(error)}',
    (_, Error(:final error)) => 'Download failed: ${_reason(error)}',
    _ => null,
  };

  /// Stops a running download or import.
  void cancel() => _chatModels.cancel();

  // ---- The draft (the custom editor's fields) ----

  String _name = '';
  ModelType _type = ModelType.gemma4;
  PreferredBackend _backend = PreferredBackend.gpu;
  String _context = '';
  bool _images = false;
  bool _tools = false;

  String get draftName => _name;
  ModelType get draftType => _type;
  PreferredBackend get draftBackend => _backend;
  String get draftContext => _context;
  bool get draftImages => _images;
  bool get draftTools => _tools;

  /// The context an NPU build's file name announces, for the hint.
  int? get contextHint =>
      saved?.file == null ? null : contextHintFromFileName(saved!.file!.name);

  void setName(String value) => _edit(() => _name = value);
  void setType(ModelType value) => _edit(() => _type = value);
  void setBackend(PreferredBackend value) => _edit(() => _backend = value);
  void setContext(String value) => _edit(() => _context = value);
  void setImages({required bool enabled}) => _edit(() => _images = enabled);
  void setTools({required bool enabled}) => _edit(() => _tools = enabled);

  void _edit(void Function() change) {
    change();
    _notify();
  }

  void _resetDraft() {
    final model = saved;
    if (model == null) return;
    _name = model.displayName;
    _type = model.modelType;
    _backend = model.backend;
    _context = '${model.maxTokens}';
    _images = model.supportImage;
    _tools = model.tools;
  }

  /// The draft as a model; null without a saved custom model or with a
  /// context that is not a number.
  CustomChatModel? get _draft {
    final model = saved;
    final context = int.tryParse(_context.trim());
    if (model == null || context == null) return null;
    return model.copyWith(
      displayName: _name.trim(),
      modelType: _type,
      backend: _backend,
      maxTokens: context,
      supportImage: _images,
      tools: _tools,
    );
  }

  /// Why the draft cannot be applied; null when it can.
  String? get draftProblem {
    if (saved == null) return 'Import or download a .litertlm first.';
    if (int.tryParse(_context.trim()) == null) {
      return 'The context length is a whole number of tokens.';
    }
    return _chatModels.problemWith(_draft!);
  }

  /// The draft differs from what is saved, or from what the chat slot runs:
  /// a replacement file adopted while the old one runs is saved already, so
  /// comparing with the saved model alone would leave Apply off.
  bool get _changed {
    final model = saved;
    final draft = _draft;
    if (model == null || draft == null) return false;
    return !_chatModels.sameSaved(draft, model) ||
        active != ChatModelKind.custom ||
        !_runs(draft);
  }

  /// [model] is what the chat slot runs: the same file (by SHA-256: every
  /// nameless link used to land under one name; a file in place by its
  /// path, its hash may still be computing) and the same settings.
  bool _runs(CustomChatModel model) => switch (_slot) {
    ModelReady(info: LoadedModelInfo(chat: final chat?)) when chat.custom =>
      _sameFile(chat, model) &&
          chat.name == model.displayName &&
          chat.requestedBackend == model.backend.name &&
          chat.requestedContext == model.maxTokens &&
          chat.images == model.supportImage &&
          chat.tools == model.tools &&
          chat.modelType == model.modelType.name,
    _ => false,
  };

  /// A new file is saved while the previous one still runs: it runs only
  /// after Apply and reload.
  bool get replacementPending => switch (_slot) {
    ModelReady(info: LoadedModelInfo(chat: final chat?))
        when chat.custom && active == ChatModelKind.custom =>
      saved?.file != null && !_sameFile(chat, saved!),
    _ => false,
  };

  static bool _sameFile(ChatModelFacts chat, CustomChatModel model) =>
      switch (model.source) {
        LocalModelSource() => chat.source == model.sourceLine,
        _ => chat.sha256 == model.file?.sha256,
      };

  bool get canApply => !busy && draftProblem == null && _changed;

  String get applyLabel =>
      active == ChatModelKind.custom ? 'Apply and reload' : 'Use this model';

  // ---- The NPU ----

  /// Whether flutter_edge_ai would accept the NPU here.
  bool get npuOffered => _chatModels.npu is NpuAvailable;

  /// `available (SoC QTI SM8750): …` or `unavailable: <reason>`.
  String get npuLine => describeNpu(_chatModels.npu);

  /// Why the NPU option is disabled; null when it is offered.
  String? get npuUnavailableReason => switch (_chatModels.npu) {
    NpuUnavailable(:final reason) => 'NPU unavailable: $reason.',
    NpuAvailable() => null,
  };

  // ---- The chat slot's state ----

  ModelState get _slot =>
      _models.states.value[ModelId.chat] ?? const ModelPending();

  /// What runs now: `Gemma 3 1B NPU · npu → npu · ctx 1280 · images off …`.
  String get activeLine => switch (_slot) {
    _ when noModel && _planned is NoChatSource =>
      'No chat model yet: choose a .litertlm below',
    ModelReady(:final info) => [
      'Loaded: ${info.chat?.name ?? info.modelId}',
      if (info.chat case final chat?)
        '${chat.requestedBackend} → ${info.backend} (activeBackend)'
      else
        info.backend,
      ?info.chat?.capabilityLine,
      'load ${(info.loadTime.inMilliseconds / 1000).toStringAsFixed(1)} s',
    ].join(' · '),
    ModelInstalling() ||
    ModelLoading() ||
    ModelWarmingUp() => switch (_plannedBackend()) {
      final backend? => 'Loading ${_plannedName()} on $backend…',
      null => 'Loading ${_plannedName()}…',
    },
    ModelPending() => 'Not loaded yet (${_plannedName()})',
    ModelUnavailable(:final reason) => 'Not available: $reason',
    ModelFailed() => 'Failed to load ${_plannedName()}',
  };

  /// The load failure, as the engine and flutter_edge_ai said it.
  String? get loadError => switch (_slot) {
    ModelFailed(:final message) => message,
    _ => null,
  };

  /// The ways out after a failed load of the user's model: never taken
  /// on its own. None when no load can help (a self-test that did not end).
  List<ChatModelAction> get failureActions {
    final retryable = switch (_slot) {
      ModelFailed(:final retryable) => retryable,
      _ => false,
    };
    if (!retryable || active != ChatModelKind.custom) return const [];
    final backend = saved?.backend;
    return [
      if (backend != PreferredBackend.gpu) ChatModelAction.runOnGpu,
      if (backend != PreferredBackend.cpu) ChatModelAction.runOnCpu,
    ];
  }

  void runAction(ChatModelAction action) => unawaited(switch (action) {
    ChatModelAction.runOnGpu => runOn.execute(PreferredBackend.gpu),
    ChatModelAction.runOnCpu => runOn.execute(PreferredBackend.cpu),
  });

  /// What the chat slot loads for the saved choice. There is no default
  /// model: with none chosen, only `GEMMA_MODEL_PATH` (a `--dart-define`)
  /// fills the slot.
  ChatModelSource get _planned => _sources.chat(_chatModels.plan);

  String _plannedName() => switch (_planned) {
    CustomChatSource(:final plan) => plan.model.displayName,
    BlockedChatSource() => 'your own model',
    DefineChatSource(:final config) => config.name,
    NoChatSource() => 'no chat model',
  };

  /// The backend the planned load asks for; null when nothing loads.
  String? _plannedBackend() => switch (_planned) {
    CustomChatSource(:final plan) => plan.model.backend.name,
    DefineChatSource(:final config) => config.llm.backend.name,
    BlockedChatSource() || NoChatSource() => null,
  };

  // ---- Actions ----

  Future<Result<CustomChatModel>> _importFile() async {
    final String? path;
    final String? temp;
    try {
      path = await _picker.pickModelFile();
      if (path == null) return const Result.error(_PickCancelled());
      temp = await _picker.temporaryCopiesDirectory();
    } on PlatformException catch (e) {
      return Result.error(e);
    }
    final result = await _chatModels.importFile(path, moveFrom: temp);
    if (result is Ok<CustomChatModel>) _adopted();
    return result;
  }

  Future<Result<CustomChatModel>> _useLocal(String path) async {
    final trimmed = path.trim();
    if (trimmed.isEmpty) {
      return const Result.error(
        LocalModelException('Enter the path of a .litertlm file.'),
      );
    }
    final result = await _chatModels.useLocalFile(trimmed);
    if (result is Ok<CustomChatModel>) _adopted();
    return result;
  }

  Future<Result<void>> _rescan() async {
    final listings = await _chatModels.listFolders();
    _folders = List.unmodifiable([
      for (final (index, folder) in listings.indexed)
        ModelsFolderView(
          label: folder.label,
          path: folder.path,
          error: folder.error,
          files: folder.files,
          hint: switch (folder.path) {
            final path? => _hintFor(index, path),
            null => null,
          },
        ),
    ]);
    _notify();
    await _selectTheOnlyFile();
    return const Result.ok(null);
  }

  /// Nothing chosen yet (no saved model, no `GEMMA_MODEL_PATH`, nothing said
  /// once still to read) and exactly one `.litertlm` in the models folders:
  /// it is selected as if tapped, so the card opens on its settings. Loading
  /// still waits for "Use this model"; a file that cannot be used shows its
  /// reason like a tapped one. A selection made mid-copy (not applied yet) is
  /// made again once the file's size changed. Never while an import or a
  /// download runs: its own file is about to be adopted.
  Future<void> _selectTheOnlyFile() async {
    if (importFile.running || download.running || _chatModels.busy.value) {
      return;
    }
    if (problem != null || note != null) return;
    final files = localFiles;
    if (files.length != 1) return;
    final only = files.single;
    final fresh = saved == null && _planned is NoChatSource;
    final copiedSince =
        active == ChatModelKind.none && isSelected(only) && canPick(only);
    if (!fresh && !copiedSince) return;
    await useLocal.execute(only.path);
  }

  Future<Result<CustomChatModel>> _download(ModelUrlRequest request) async {
    final result = await _chatModels.download(
      request.url,
      sha256: request.sha256,
      sizeBytes: request.sizeBytes,
    );
    if (result is Ok<CustomChatModel>) _adopted();
    return result;
  }

  /// A new file is in: the editor shows its settings.
  void _adopted() {
    _resetDraft();
    _notify();
  }

  // The choice and the reload are one exclusive operation: the self-test
  // cannot start between saving the choice and the reload.

  Future<Result<void>> _apply() {
    final draft = _draft;
    if (draft == null) {
      return Future.value(
        Result.error(
          InvalidChatModelException(draftProblem ?? 'Nothing to apply.'),
        ),
      );
    }
    return _switcher.exclusive((ops) async {
      if (await _chatModels.apply(draft) case Error(:final error)) {
        return Result.error(error);
      }
      return _reloadAndPrune(ops);
    });
  }

  Future<Result<void>> _runOn(PreferredBackend backend) =>
      _switcher.exclusive((ops) async {
        if (await _chatModels.runOn(backend) case Error(:final error)) {
          return Result.error(error);
        }
        _resetDraft();
        return _reloadAndPrune(ops);
      });

  /// Reloads, then deletes a replaced file: the engine that ran it is
  /// closed now, whether the new load worked or not. The result is the chat
  /// model's: another model's failure in the same setup run shows on its
  /// own row, never as this Apply's error.
  Future<Result<void>> _reloadAndPrune(ChatModelOps ops) async {
    final result = await ops.reload();
    // Refused before the engine was touched: it still runs the old file.
    if (result case Error(error: ReplyStillStoppingException())) return result;
    await _chatModels.pruneUnused();
    if (_slot is ModelReady) return const Result.ok(null);
    return result;
  }

  /// The last apply's or action's error (a reload's own failure shows as
  /// [loadError]), or why its reload was refused (the previous reply is
  /// still stopping).
  String? get actionError => switch ((apply.result, runOn.result)) {
    (Error(error: final InvalidChatModelException e), _) ||
    (_, Error(error: final InvalidChatModelException e)) => e.message,
    (Error(error: final ReplyStillStoppingException e), _) ||
    (_, Error(error: final ReplyStillStoppingException e)) => '$e',
    _ => null,
  };

  void _onChanged() {
    // The saved model changed under the editor (an import elsewhere):
    // follow it unless the user is mid-edit of the same file.
    _notify();
  }

  static String _reason(Exception error) => switch (error) {
    _PickCancelled() => 'no file chosen',
    PlatformException(:final message?) => message,
    _ => '$error',
  };

  static String _percent(int done, int total) =>
      total == 0 ? '0%' : '${(done * 100 / total).floor()}%';

  static String _size(int bytes) => bytes >= 1e9
      ? '${(bytes / 1e9).toStringAsFixed(2)} GB'
      : '${(bytes / 1e6).toStringAsFixed(1)} MB';

  void _notify() {
    if (!_disposed) notifyListeners();
  }

  @override
  void dispose() {
    _disposed = true;
    for (final listenable in _listenables) {
      listenable.removeListener(_onChanged);
    }
    for (final command in _commands) {
      command
        ..removeListener(notifyListeners)
        ..dispose();
    }
    super.dispose();
  }
}

/// The picker was closed without a file.
final class _PickCancelled implements Exception {
  const _PickCancelled();

  @override
  String toString() => 'No file chosen';
}
