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

/// What the user can do after the chat model failed to load: run it on the
/// NPU (a build compiled for one, saved on another backend), the GPU or the
/// CPU.
enum ChatModelAction { runOnNpu, runOnGpu, runOnCpu }

/// The towers a chat model file can have beside its text model.
enum Tower { vision, audio }

/// One tower's row in the card's Towers block (`ChatModelViewModel.towerRow`):
/// whether the shown file has the tower's encoder ([inFile]:
/// `LiteRtLmBundleInfo.visionEncoder` / `audioEncoder`) and, once that file
/// is loaded, where the engine put it ([runsOn]:
/// `InferenceModel.activeVisionBackend` / `activeAudioBackend`, a
/// `PreferredBackend` name). The app never says where a tower will run: the
/// engine places it at the load.
final class const TowerRow(
  final Tower tower, {
  required final bool inFile,
  final String? runsOn,
}) {
  /// `in the file · runs on the CPU`, `in the file (where it runs shows once
  /// it is loaded)`, or what is off without it.
  String get text => switch ((inFile, runsOn)) {
    (false, _) => switch (tower) {
      Tower.vision =>
        'not in this file: photos in Voice chat and detailed answers in '
            'Live camera are off',
      Tower.audio => 'not in this file: voice questions are transcribed first',
    },
    (true, final backend?) =>
      'in the file · runs on the ${backend.toUpperCase()}',
    (true, null) => 'in the file (where it runs shows once it is loaded)',
  };
}

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
/// or a link — and sets its backend, context and tools. Its towers (vision,
/// audio) are the ones the file has (`readLiteRtLmBundleInfo`), all of them
/// loaded; where the engine put each is shown after the load. When a load
/// fails, the plugin's reason and the explicit ways out. Never switches
/// models by itself.
class ChatModelViewModel extends ChangeNotifier {
  ChatModelViewModel({
    required this._chatModels,
    required this._models,
    required this._switcher,
    required this._picker,
    this._sources = const ModelSourceResolver(),
    this._autoApplyTheOnlyFile = true,
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

  /// Whether the only file of the models folders is applied by itself when
  /// nothing is chosen yet ([_selectTheOnlyFile]). The app turns it on for
  /// Android and iOS only: there a tester pushes one file and the defaults
  /// fit (the GPU, or the NPU for a build compiled for it); on a
  /// desktop or a board the guide picks the backend first (the Raspberry Pi
  /// guide runs Gemma on the CPU).
  final bool _autoApplyTheOnlyFile;
  bool _disposed = false;

  /// Picks and imports a `.litertlm` (desktop, iOS). Like every pick it only
  /// shows the file ([candidate]); [apply] saves and loads it.
  late final Command0<CustomChatModel> importFile;

  /// Downloads a `.litertlm` from a link.
  late final Command1<CustomChatModel, ModelUrlRequest> download;

  /// Saves the draft (the candidate's file, or the saved one's) and makes it
  /// the chat model, then reloads it.
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

  /// Said once (a saved choice the app does not offer: it downloads no
  /// Gemma 4 E2B).
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

  /// The custom model as saved: the chat model once [active] is custom.
  CustomChatModel? get saved => _state.custom;

  /// The file picked and not applied yet, with the settings it starts from
  /// (`ChatModelCandidate`); null when there is none. Nothing loads it until
  /// Apply.
  CustomChatModel? get candidate => _state.candidate?.model;

  /// What the card shows and edits: the [candidate], else the [saved] model.
  CustomChatModel? get shown => candidate ?? saved;

  /// `mine.litertlm · 1.21 GB · sha256 1a2b3c4d… (computed) · imported from …`
  /// (a file in place: `in place: /path`, the SHA-256 once computed) of the
  /// [shown] file.
  String? get fileLine {
    final model = shown;
    final file = model?.file;
    if (model == null || file == null) return null;
    // The saved model's store copy as last found; a candidate's verified.
    final state = model.source is LocalModelSource || candidate != null
        ? null
        : switch (_chatModels.savedFile) {
            StoreFileReady() => null,
            StoreFileUnverified() => 'CHANGED ON DISK',
            StoreFileMissing() => 'MISSING',
            _ => null,
          };
    final sha = switch (file.sha256) {
      final hex? =>
        'sha256 ${hex.substring(0, 12)}… '
            '(${file.checksumMatched ? 'matches the one entered' : 'computed'})',
      // A file only looked at is not read whole.
      null when candidate != null => 'sha256 once applied',
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

  /// [entry] is the [shown] file: the candidate's, else the saved model's.
  bool isSelected(LocalModelEntry entry) => switch (shown?.source) {
    LocalModelSource(:final path) => path == entry.path,
    _ => false,
  };

  /// [entry] can be picked: another file than the selected one, or the
  /// selected one at another size than when it was chosen (chosen while it
  /// was still being copied).
  bool canPick(LocalModelEntry entry) =>
      !isSelected(entry) || entry.sizeBytes != shown?.file?.sizeBytes;

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
  /// Android the models folder replaces it (file_selector_android 0.5.2+11
  /// reads a picked file into one Java `byte[]` sized by an int: the whole
  /// file goes into memory, and above 2 GB the size overflows).
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
  bool _tools = false;

  String get draftName => _name;
  ModelType get draftType => _type;
  PreferredBackend get draftBackend => _backend;
  String get draftContext => _context;
  bool get draftTools => _tools;

  /// The context an NPU build's file name announces, for the hint.
  int? get contextHint => switch (shown?.file) {
    final file? => contextHintFromFileName(file.name),
    null => null,
  };

  void setName(String value) => _edit(() => _name = value);
  void setType(ModelType value) => _edit(() => _type = value);
  void setBackend(PreferredBackend value) => _edit(() => _backend = value);
  void setContext(String value) => _edit(() => _context = value);
  void setTools({required bool enabled}) => _edit(() => _tools = enabled);

  // ---- The towers (what the file says about itself) ----

  /// What `readLiteRtLmBundleInfo` made of the [shown] file: the
  /// candidate's, read when it was picked, else the saved file's; null when
  /// that has not been read (before the launch's check, or a saved file that
  /// is gone, cannot be opened or is incomplete: the card shows that
  /// problem, and no Towers block).
  BundleReading? get _reading => _state.candidate?.bundle ?? _state.bundle;

  /// What the [shown] file says about itself; null when it is not read or
  /// its header could not be ([headerNotRead]).
  LiteRtLmBundleInfo? get _bundle => _reading?.info;

  /// The Towers block is shown: the file is read, or its header could not
  /// be and [headerNotRead] says so.
  bool get towersShown => _reading != null;

  /// The Towers block's line for a file whose header the plugin could not
  /// read (`BundleNotRead`, the plugin's message): it loads text only; null
  /// for any other file.
  String? get headerNotRead => switch (_reading) {
    BundleNotRead(:final reason) =>
      'The plugin could not read its header (${reason.problem.name}): '
          '${reason.message}. The plugin loads such a file without reading '
          'its header; the app asks for no tower (none is known), so it '
          'loads text only.',
    _ => null,
  };

  /// The Towers block's first line for a build compiled for the NPU
  /// (`LiteRtLmBundleInfo.isCompiledForNpu`): `Text model: compiled for the
  /// Qualcomm SM8750 NPU`; null for any other file or none. It says nothing
  /// about other SoCs: whether the file runs on this one only a load tells.
  String? get npuBuildLine => switch (_bundle) {
    final bundle? when bundle.isCompiledForNpu =>
      'Text model: compiled for the '
          '${bundle.npuTarget == null ? 'NPU' : '${bundle.npuTarget} NPU'}',
    _ => null,
  };

  /// [tower]'s row: whether the shown file has it and, when that file is
  /// the one loaded, where the engine put it; null when the file is not read
  /// ([towersShown]).
  TowerRow? towerRow(Tower tower) {
    final bundle = _bundle;
    if (bundle == null) return null;
    final inFile = switch (tower) {
      Tower.vision => bundle.visionEncoder != null,
      Tower.audio => bundle.audioEncoder != null,
    };
    final loaded = _shownLoaded;
    return TowerRow(
      tower,
      inFile: inFile,
      runsOn: !inFile || loaded == null
          ? null
          : switch (tower) {
              Tower.vision => loaded.visionBackend,
              Tower.audio => loaded.audioBackend,
            },
    );
  }

  /// The chat slot's facts when it runs the [shown] file: the saved model
  /// (a candidate is never what runs, even at the same path: another build
  /// pushed over the file), the same file, unchanged since it was chosen (a
  /// file in place replaced since blocks the plan: [localProblem]); null
  /// otherwise, so a row never says where a file it does not run went.
  ChatModelFacts? get _shownLoaded {
    final model = shown;
    if (candidate != null || model == null || localProblem != null) {
      return null;
    }
    return switch (_slot) {
      ModelReady(info: LoadedModelInfo(chat: final chat?))
          when chat.custom && _sameFile(chat, model) =>
        chat,
      _ => null,
    };
  }

  void _edit(void Function() change) {
    change();
    _notify();
  }

  /// The file the draft was made for ([_fileKey] of [shown]).
  String? _draftFile;

  /// A file by its source, name and size: a SHA-256 that came in since is
  /// the same file.
  static String? _fileKey(CustomChatModel? model) => model == null
      ? null
      : '${model.sourceLine}|${model.file?.name}|${model.file?.sizeBytes}';

  void _resetDraft() {
    final model = shown;
    _draftFile = _fileKey(model);
    if (model == null) return;
    _name = model.displayName;
    _type = model.modelType;
    _backend = model.backend;
    _context = '${model.maxTokens}';
    _tools = model.tools;
  }

  /// The draft as a model (the [shown] file with the editor's settings);
  /// null without a file or with a context that is not a number.
  CustomChatModel? get _draft {
    final model = shown;
    final context = int.tryParse(_context.trim());
    if (model == null || context == null) return null;
    return model.copyWith(
      displayName: _name.trim(),
      modelType: _type,
      backend: _backend,
      maxTokens: context,
      tools: _tools,
    );
  }

  /// Why the draft cannot be applied; null when it can.
  String? get draftProblem {
    if (shown == null) return 'Import or download a .litertlm first.';
    if (int.tryParse(_context.trim()) == null) {
      return 'The context length is a whole number of tokens.';
    }
    return _chatModels.problemWith(_draft!);
  }

  /// A candidate is waiting, or the draft differs from what is saved or
  /// from what the chat slot runs (an apply whose reload was refused saved a
  /// file the old engine does not run).
  bool get _changed {
    final model = saved;
    final draft = _draft;
    if (draft == null) return false;
    if (candidate != null) return true;
    if (model == null) return false;
    return !_chatModels.sameSaved(draft, model) ||
        active != ChatModelKind.custom ||
        !_runs(draft);
  }

  /// [model] (the saved model's file) is what the chat slot runs: the same
  /// file (by SHA-256; a file in place by its path, its hash may still be
  /// computing), read, and the same settings.
  bool _runs(CustomChatModel model) {
    if (_state.bundle == null) return false;
    return switch (_slot) {
      ModelReady(info: LoadedModelInfo(chat: final chat?)) when chat.custom =>
        _sameFile(chat, model) &&
            chat.name == model.displayName &&
            chat.requestedBackend == model.backend.name &&
            chat.requestedContext == model.maxTokens &&
            chat.tools == model.tools &&
            chat.modelType == model.modelType.name,
      _ => false,
    };
  }

  /// Another file is saved than the one the chat slot runs (an apply whose
  /// reload was refused): it runs only after Apply and reload. Not while a
  /// candidate waits ([candidateNotice] says what to do then).
  bool get replacementPending => switch (_slot) {
    ModelReady(info: LoadedModelInfo(chat: final chat?))
        when chat.custom &&
            active == ChatModelKind.custom &&
            candidate == null =>
      saved?.file != null && !_sameFile(chat, saved!),
    _ => false,
  };

  /// What the card says about a picked file that is not applied yet; null
  /// when there is none.
  String? get candidateNotice {
    if (candidate == null) return null;
    final until = _planned is NoChatSource
        ? 'Until then there is no chat model.'
        : 'Until then the chat model stays ${_plannedName()}.';
    return 'Not applied yet: press "$applyLabel" to use this file. $until';
  }

  static bool _sameFile(ChatModelFacts chat, CustomChatModel model) =>
      switch (model.source) {
        LocalModelSource() => chat.source == model.sourceLine,
        _ => chat.sha256 == model.file?.sha256,
      };

  bool get canApply => !busy && draftProblem == null && _changed;

  String get applyLabel =>
      active == ChatModelKind.custom ? 'Apply and reload' : 'Use this model';

  // ---- The NPU ----

  /// Whether the app's copy of flutter_edge_ai_litertlm's private NPU check
  /// finds the NPU reachable here (the plugin is not asked). Whether the
  /// shown file may run there is [npuNotForFileReason].
  bool get npuOffered => _chatModels.npu is NpuAvailable;

  /// `available (SoC QTI SM8750)` or `unavailable: <reason>`.
  String get npuLine => describeNpu(_chatModels.npu);

  /// Why this device offers no NPU; null when it does.
  String? get npuUnavailableReason => switch (_chatModels.npu) {
    NpuUnavailable(:final reason) => 'NPU unavailable: $reason.',
    NpuPending() => 'NPU: still being checked.',
    NpuAvailable() => null,
  };

  /// Why the NPU is not offered for the [shown] file; null when it is: a
  /// file not compiled for an NPU, which the strict engine refuses on the
  /// NPU, or one compiled for an NPU vendor this build does not run here
  /// (`ChatModelRepository.cannotRunHere`: it runs on no backend here).
  /// `ChatModelRepository.problemWith` refuses both. The counterpart of
  /// [gpuCpuUnavailableReason].
  String? get npuNotForFileReason =>
      switch ((notForNpuReason(_bundle), _chatModels.cannotRunHere(_bundle))) {
        (final reason?, _) => 'NPU: not for this file ($reason).',
        (_, final why?) => 'NPU: not on this device (this file is $why).',
        _ => null,
      };

  /// Why the NPU option is disabled ([npuNotForFileReason], else
  /// [npuUnavailableReason]); null when it is offered.
  String? get npuDisabledReason => npuNotForFileReason ?? npuUnavailableReason;

  /// Why the GPU and CPU options are disabled for the [shown] file; null
  /// when they are offered: a build compiled for an NPU
  /// (`LiteRtLmBundleInfo.isCompiledForNpu`), which the plugin refuses on
  /// the GPU and the CPU (`ChatModelRepository.problemWith` refuses them
  /// too).
  String? get gpuCpuUnavailableReason => switch (npuOnlyReason(_bundle)) {
    final reason? => 'GPU and CPU: not for this file ($reason).',
    null => null,
  };

  // ---- The chat slot's state ----

  ModelState get _slot =>
      _models.states.value[ModelId.chat] ?? const ModelPending();

  /// What runs now: `Gemma 3 1B NPU · npu → npu · ctx 1280 · vision cpu …`.
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

  /// The load failure, as the plugin said it (`ChatModelLoadException`: its
  /// `LiteRtLmBundleException` or `BackendInitException` reason).
  String? get loadError => switch (_slot) {
    ModelFailed(:final message) => message,
    _ => null,
  };

  /// The ways out after a failed load of the user's (saved) model: never
  /// taken on its own.
  /// - A build compiled for an NPU (`LiteRtLmBundleInfo.isCompiledForNpu`)
  ///   the plugin refused as `wrongBackend` (saved on the GPU or the CPU):
  ///   Run on NPU, what the plugin's message advises, where the NPU is
  ///   offered.
  /// - Any other refusal before an engine (`ModelFailed.bundleProblem`:
  ///   incomplete, unsupportedNpu): none, no backend changes it; for a
  ///   generic file's `wrongBackend` (saved on the NPU by an earlier app
  ///   version, or a text model that declares its backends) the other
  ///   backends.
  /// - Any other failure: the other backends, except for a build compiled
  ///   for an NPU (it runs nowhere else) and for a failure no load can
  ///   help (a self-test that did not end).
  List<ChatModelAction> get failureActions {
    final failed = _slot;
    final model = saved;
    if (failed is! ModelFailed ||
        active != ChatModelKind.custom ||
        model == null) {
      return const [];
    }
    final npuBuild = _state.bundle?.info?.isCompiledForNpu ?? false;
    final others = [
      if (model.backend != PreferredBackend.gpu) ChatModelAction.runOnGpu,
      if (model.backend != PreferredBackend.cpu) ChatModelAction.runOnCpu,
    ];
    return switch (failed.bundleProblem) {
      LiteRtLmBundleProblem.wrongBackend when npuBuild => [
        if (npuOffered && model.backend != PreferredBackend.npu)
          ChatModelAction.runOnNpu,
      ],
      LiteRtLmBundleProblem.wrongBackend => others,
      null when failed.retryable && !npuBuild => others,
      _ => const [],
    };
  }

  void runAction(ChatModelAction action) => unawaited(switch (action) {
    ChatModelAction.runOnNpu => runOn.execute(PreferredBackend.npu),
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

  /// Lists the folders, after checking the saved file again when its last
  /// check found a problem (a file copied back).
  Future<Result<void>> _rescan() async {
    await _chatModels.recheck();
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

  /// Nothing chosen yet (no choice saved, no candidate, no
  /// `GEMMA_MODEL_PATH`, nothing said once still to read) and exactly one
  /// `.litertlm` in the models folders: it is picked as if tapped and, as no
  /// model is applied that a candidate could protect, saved and loaded at
  /// once (a tester pushes one file with adb and expects it to run). One
  /// that cannot be applied as it is stays on the card with the reason, and
  /// is picked again by a Rescan once its size changed. A file that cannot
  /// be used shows its reason like a tapped one. A saved choice, even one
  /// that cannot load, is never replaced by itself (its settings would go),
  /// and a file tapped by hand stays a candidate until Apply.
  /// Never while an import or a download runs: its own file is about to be
  /// adopted.
  Future<void> _selectTheOnlyFile() async {
    if (!_autoApplyTheOnlyFile) return;
    if (importFile.running || download.running || _chatModels.busy.value) {
      return;
    }
    if (problem != null || note != null) return;
    if (active != ChatModelKind.none || _planned is! NoChatSource) return;
    final files = localFiles;
    if (files.length != 1) return;
    final only = files.single;
    final fresh = shown == null;
    final copiedSince = isSelected(only) && canPick(only);
    if (!fresh && !copiedSince) return;
    await useLocal.execute(only.path);
    if (useLocal.result is! Ok<CustomChatModel>) return;
    if (candidate == null || draftProblem != null) return;
    debugPrint('[ChatModel] the only file in the models folder, used at once');
    await apply.execute();
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

  /// A file was picked: the editor shows its settings (those it starts
  /// from).
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
  String? get actionError {
    final errors = [
      for (final result in [apply.result, runOn.result])
        if (result case Error(:final error)) error,
    ];
    for (final error in errors) {
      if (error is InvalidChatModelException) return error.message;
    }
    for (final error in errors) {
      if (error is ReplyStillStoppingException) return '$error';
    }
    return null;
  }

  void _onChanged() {
    // The shown file changed under the editor (a candidate picked, applied
    // or dropped elsewhere, a download that ended after its screen closed):
    // the draft follows it, unless it is the same file (mid-edit).
    if (_fileKey(shown) != _draftFile) _resetDraft();
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
    // Leaving the Models screen drops a file picked in place and not
    // applied; an imported or downloaded one stays for this app session.
    unawaited(_chatModels.dropInPlaceCandidate());
    super.dispose();
  }
}

/// The picker was closed without a file.
final class _PickCancelled implements Exception {
  const _PickCancelled();

  @override
  String toString() => 'No file chosen';
}
