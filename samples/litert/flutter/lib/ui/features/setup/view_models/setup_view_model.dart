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

import '../../../../config/model_catalog.dart';
import '../../../../data/repositories/chat_model_repository.dart';
import '../../../../data/repositories/hardware_repository.dart';
import '../../../../data/repositories/provisioning_repository.dart';
import '../../../../domain/hardware/device_summary.dart';
import '../../../../domain/models/chat_model.dart';
import '../../../../domain/models/model_id.dart';
import '../../../../domain/models/model_state.dart';
import '../../../../domain/models/provisioning.dart';
import '../../../../domain/ports/model_states.dart';
import '../../../../domain/use_cases/chat_model_switcher.dart';
import '../../../../utils/command.dart';
import '../../../../utils/result.dart';

/// Where the Models screen was opened.
enum SetupMode {
  /// The first route: load the built-in models (and the chat model once one
  /// is chosen), then hand over to home.
  firstRun,

  /// From home's menu: what is present but not loaded is loaded; no
  /// hand-over.
  manage,
}

/// How a line is coloured.
enum Tone { normal, warning, error }

/// One model row, ready to show: the widget only lays it out.
final class const ModelRow({
  required final ModelSpec spec,
  required final ModelState load,

  /// The row's title when it is not the spec's name (the chat slot:
  /// `Chat model · Gemma 4 E2B`).
  final String? title,

  /// The built-in files' size, or the chosen chat model's; 0 when unknown
  /// (`GEMMA_MODEL_PATH`'s file): not shown.
  required final int bytes,
  required final String status,

  /// Where the files come from, when present.
  final String? source,

  /// An error message or what to do, under the status.
  final String? detail,

  /// Shown when non-null; `double.nan` is an indeterminate bar.
  final double? progress,
  final Tone tone = Tone.normal,

  /// Loading failed: Retry runs the model setup again.
  final bool canRetryLoad = false,
});

/// Drives the Models screen: every model's
/// presence and load state; starts the model setup when it opens (first
/// run: then hands over to home once every required model is ready).
class SetupViewModel extends ChangeNotifier {
  SetupViewModel({
    required this._models,
    required this._prepareModels,
    required this._provisioning,
    this.mode = SetupMode.firstRun,
    this._hardware,
    this._chatModels,
    this._switcher,
  }) {
    start = Command0<void>(_start);
    prepare = Command0<void>(_prepare);
    for (final command in _commands) {
      command.addListener(notifyListeners);
    }
    for (final listenable in _listenables) {
      listenable.addListener(notifyListeners);
    }
    _hardwareChanges?.addListener(notifyListeners);
    unawaited(start.execute());
  }

  final ModelStates _models;

  /// Runs the model setup (`ModelRepository.prepareAll`): installs, loads
  /// and warms up every present model not ready yet.
  final Future<Result<void>> Function() _prepareModels;
  final ProvisioningRepository _provisioning;
  final SetupMode mode;

  /// The "This device" card's source; no card without it.
  final HardwareRepository? _hardware;
  late final Listenable? _hardwareChanges = _hardware?.changes;

  /// What the chat slot holds (its row's name, size and backend); null: no
  /// chat model chooser (tests).
  final ChatModelRepository? _chatModels;

  /// The chat model's owner: the model setup runs as one of its exclusive
  /// operations, and nothing here starts while one runs (the self-test, a
  /// reload). Null in tests of the setup alone.
  final ChatModelSwitcher? _switcher;

  bool get _chatModelInUse => _switcher?.busy.value ?? false;

  Future<Result<void>> _prepare() async {
    final result = await switch (_switcher) {
      null => _prepareModels(),
      final switcher => switcher.exclusive((_) => _prepareModels()),
    };
    // A successful start: what earlier builds downloaded goes.
    if (result is Ok<void>) unawaited(_provisioning.pruneOldModelFolders());
    return result;
  }

  /// The "This device" card; null when there is no hardware repository.
  DeviceSummary? get device => _hardware?.summary();

  /// The text "Copy diagnostics" puts on the clipboard.
  String diagnosticsReport() => _hardware?.report() ?? '';
  bool _disposed = false;

  /// Starts the model setup on a first run, or loads what is present but
  /// not loaded (manage).
  late final Command0<void> start;

  /// Installs, loads and warms up every present model; succeeds when every
  /// required model is ready. Also the Retry of a load failure.
  late final Command0<void> prepare;

  List<Command<Object?>> get _commands => [start, prepare];

  List<Listenable> get _listenables => [
    _models.states,
    _models.preparing,
    _provisioning.busy,
    ?_chatModels?.state,
    ?_chatModels?.customFile,
    ?_switcher?.busy,
  ];

  /// The model store runs an import or download (the Chat model card's).
  bool get busy => _provisioning.busy.value;

  bool get _idle =>
      !busy && !start.running && !prepare.running && !_chatModelInUse;

  /// First run: every required model is present but setup did not start
  /// (the chat model was in use when the screen opened): start it
  /// explicitly.
  bool get canContinue =>
      mode == SetupMode.firstRun &&
      _idle &&
      prepare.result == null &&
      _provisioning.requiredPresent;

  /// First run is done: setup ran, every required model is ready and
  /// nothing runs. The screen then hands over to home. Optional models may be
  /// missing. A setup that a chat model reload completed (the model chosen
  /// after the built-in ones loaded, "Run on GPU") counts too: its run
  /// prepared the rest.
  bool get allReady =>
      prepare.result != null &&
      !prepare.running &&
      !_models.preparing.value &&
      _models.requiredReady &&
      !busy;

  /// Rows in catalog order.
  List<ModelRow> get rows => [for (final id in ModelId.values) _row(id)];

  ModelRow _row(ModelId id) {
    final spec = id.spec;
    final load = _models.states.value[id] ?? const ModelPending();
    switch (_provisioning.presenceOf(id)) {
      case PresentAsCustomChatModel(
        :final name,
        :final sizeBytes,
        :final where,
      ):
        return _loadRow(
          spec,
          load,
          sizeBytes,
          'Your own .litertlm · $where',
          title: 'Chat model · $name',
        );
      case CustomChatModelBlocked(:final reason) when load is! ModelFailed:
        return ModelRow(
          spec: spec,
          load: load,
          title: 'Chat model · your own .litertlm',
          bytes: 0,
          status: 'Cannot load',
          detail: reason,
          tone: Tone.error,
        );
      case CustomChatModelBlocked():
        return _loadRow(
          spec,
          load,
          0,
          'Your own .litertlm',
          title: 'Chat model · your own .litertlm',
        );
      case ChatModelNotChosen(:final note):
        return ModelRow(
          spec: spec,
          load: load,
          title: 'Chat model',
          bytes: 0,
          status: 'No chat model yet',
          detail: [
            'The app ships none: choose a .litertlm in the Chat model card '
                'above.',
            ?note,
          ].join(' '),
          tone: Tone.warning,
        );
      case PresentByDefine(:final define, :final value):
        return _loadRow(
          spec,
          load,
          0, // the define's file: its size is not known here
          'Set by the build: $define=$value',
          // GEMMA_MODEL_PATH (a --dart-define) is the only other chat model.
          title: id == ModelId.chat
              ? 'Chat model · ${kDefineChatModel.name}'
              : null,
        );
      case PresentBundled():
        return _loadRow(
          spec,
          load,
          ProvisioningRepository.bundledBytes(id),
          // The status says "Built in" already.
          'Part of the app: nothing to download',
          builtIn: true,
        );
    }
  }

  ModelRow _loadRow(
    ModelSpec spec,
    ModelState load,
    int bytes,
    String source, {
    String? title,
    bool builtIn = false,
  }) {
    var (status, progress, tone) = switch (load) {
      // A built-in model needs nothing: it loads with the setup.
      ModelPending() when builtIn => ('Built in', null, Tone.normal),
      ModelPending() => (
        mode == SetupMode.firstRun ? 'Waiting' : 'Not loaded',
        null,
        Tone.normal,
      ),
      ModelInstalling(percent: null) => (
        'Installing…',
        double.nan,
        Tone.normal,
      ),
      ModelInstalling(:final int percent) => (
        'Installing $percent%',
        percent / 100,
        Tone.normal,
      ),
      ModelLoading() => (
        'Loading on ${_backendOf(spec.id)}…',
        double.nan,
        Tone.normal,
      ),
      ModelWarmingUp() => ('Warming up…', double.nan, Tone.normal),
      ModelReady(:final info) => (
        [
          'Ready',
          info.detail ?? info.backend,
          if (info.chat case final chat?) chat.capabilityLine,
          'load ${_seconds(info.loadTime)}',
        ].join(' · '),
        null,
        info.explicitCpu ? Tone.warning : Tone.normal,
      ),
      ModelUnavailable(:final reason) => (
        'Not available: $reason',
        null,
        Tone.warning,
      ),
      ModelFailed() => ('Failed', null, Tone.error),
    };
    if (builtIn && load is! ModelPending) status = 'Built in · $status';
    return ModelRow(
      spec: spec,
      load: load,
      title: title,
      bytes: bytes,
      status: status,
      source: source,
      progress: progress,
      tone: tone,
      detail: switch (load) {
        ModelFailed(:final message) => message,
        _ => null,
      },
      canRetryLoad:
          load is ModelFailed && load.retryable && !prepare.running && !busy,
    );
  }

  String _backendOf(ModelId id) => switch (id) {
    ModelId.chat => switch (_chatModels?.plan) {
      CustomChatPlan(:final model) => model.backend.name,
      _ => kLlmConfig.backend.name,
    },
    ModelId.whisperBase ||
    ModelId.moonshineTiny => kSttConfigs[id]!.backend.name,
    ModelId.inflectNano => kTtsConfig.backend.name,
    ModelId.embeddingGemma => kEmbedderConfig.backend.name,
    ModelId.yolo26n => 'the detector backend',
  };

  static String _seconds(Duration d) =>
      '${(d.inMilliseconds / 1000).toStringAsFixed(1)} s';

  static String _size(int bytes) => bytes >= 1e9
      ? '${(bytes / 1e9).toStringAsFixed(2)} GB'
      : '${(bytes / 1e6).toStringAsFixed(1)} MB';

  /// Sizes as the screen shows them.
  static String sizeLabel(int bytes) => _size(bytes);

  Future<Result<void>> _start() async {
    // The view model is created while its screen builds, and the setup
    // publishes model states other widgets listen to: it starts after this
    // build, never inside it.
    await Future<void>.value();
    _startSetup();
    return const Result.ok(null);
  }

  /// When the screen opens: first run starts the setup (every model but the
  /// chat model is built in: they load, and the chat model with them when
  /// one is chosen; choosing it later loads only it, and the screen hands
  /// over); manage loads whatever is present but not loaded. Failures are
  /// never retried automatically.
  void _startSetup() {
    if (_disposed || prepare.running || _chatModelInUse) return;
    final run = switch (mode) {
      SetupMode.firstRun => true,
      SetupMode.manage =>
        _provisioning.requiredPresent && ModelId.values.any(_needsLoad),
    };
    if (run) unawaited(prepare.execute());
  }

  /// Present but not loaded (or unavailable before it was present). Failed
  /// loads are not retried behind the user's back: their row has Retry.
  bool _needsLoad(ModelId id) =>
      ProvisioningRepository.isPresent(_provisioning.presenceOf(id)) &&
      switch (_models.states.value[id]) {
        ModelPending() || ModelUnavailable() => true,
        _ => false,
      };

  @override
  void dispose() {
    _disposed = true;
    for (final listenable in _listenables) {
      listenable.removeListener(notifyListeners);
    }
    _hardwareChanges?.removeListener(notifyListeners);
    for (final command in _commands) {
      command
        ..removeListener(notifyListeners)
        ..dispose();
    }
    super.dispose();
  }
}
