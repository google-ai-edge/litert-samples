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

import 'package:flutter/foundation.dart';

import '../../domain/hardware/accelerator_inference.dart';
import '../../domain/hardware/device_summary.dart';
import '../../domain/hardware/diagnostics_report.dart';
import '../../domain/hardware/native_log_parser.dart';
import '../../domain/models/audio_devices.dart';
import '../../domain/models/hardware_profile.dart';
import '../../domain/models/model_id.dart';
import '../../domain/models/model_state.dart';
import '../services/hardware/hardware_info_service.dart';
import '../services/hardware/memory_probe.dart';
import '../services/hardware/native_log_tap.dart';

/// Hardware visibility: probes the device once, logs the `[Hardware]` startup
/// line and one line per loaded model, and builds the "This device" card and
/// the diagnostics report from the probe, the model states, the audio device
/// checks and a memory sample.
class HardwareRepository {
  HardwareRepository({
    required this._service,
    required this._models,
    required this._memory,
    this._logTap = const NoNativeLogTap(),
    this._audio,
    required this._build,
    this._now = DateTime.now,
    void Function(String line)? log,
  }) : _log = log ?? debugPrint {
    _models.addListener(_logReadyModels);
  }

  final HardwareInfoService _service;
  final ValueListenable<Map<ModelId, ModelState>> _models;
  final MemoryProbe _memory;
  final NativeLogTap _logTap;

  /// The voice demos' input and output checks (the card's Mic and Speaker).
  final ValueListenable<AudioDeviceStatus>? _audio;
  final BuildInfo _build;
  final DateTime Function() _now;
  final void Function(String line) _log;

  /// The probe's result, or why there is none yet.
  final ValueNotifier<({HardwareProfile? profile, String? note})> _state =
      ValueNotifier((profile: null, note: 'probing…'));
  Future<void>? _probing;
  final Map<ModelId, LoadedModelInfo> _logged = {};
  bool _disposed = false;

  /// Null until [probe] finished (or when it failed).
  HardwareProfile? get profile => _state.value.profile;

  /// The card and the report change with the probe and the model states.
  Listenable get changes => Listenable.merge([_state, _models, ?_audio]);

  /// Probes once (later calls share the first); logs the startup line. A
  /// probe failure is a bug: logged and shown on the card, never thrown.
  Future<void> probe() => _probing ??= _probe();

  Future<void> _probe() async {
    try {
      final profile = await _service.probe();
      if (_disposed) return;
      _state.value = (profile: profile, note: null);
      _log(hardwareStartupLine(profile, _build));
      for (final note in profile.notes) {
        _log('[Hardware] note: $note');
      }
      _logReadyModels();
    } catch (e, st) {
      _log('[Hardware] probe failed: $e\n$st');
      if (_disposed) return;
      _state.value = (profile: null, note: 'probe failed: $e');
    }
  }

  /// One row per model, in catalog order.
  List<ModelDiagnostics> modelDiagnostics() {
    final states = _models.value;
    return [
      for (final id in ModelId.values)
        diagnosticsFor(
          id,
          states[id] ?? const ModelPending(),
          hardware: profile,
        ),
    ];
  }

  /// The "This device" card.
  DeviceSummary summary() => buildDeviceSummary(
    hardware: profile,
    hardwareNote: _state.value.note,
    models: modelDiagnostics(),
    audio: _audio?.value,
  );

  /// The plain-text report behind "Copy diagnostics".
  String report() => formatDiagnostics(
    DiagnosticsInput(
      generatedAt: _now(),
      build: _build,
      hardware: profile,
      hardwareNote: _state.value.note,
      models: modelDiagnostics(),
      memory: _memory.snapshot(),
      nativeLog: _logTap.description,
      audio: _audio?.value,
    ),
  );

  /// `[Hardware] <model> req=… act=…` once per load, after the probe.
  void _logReadyModels() {
    final hardware = profile;
    if (hardware == null || _disposed) return;
    for (final MapEntry(key: id, value: state) in _models.value.entries) {
      if (state is! ModelReady || identical(_logged[id], state.info)) continue;
      _logged[id] = state.info;
      final evidence = diagnosticsFor(id, state, hardware: hardware).evidence;
      if (evidence != null) _log(evidenceLogLine(id.name, evidence));
    }
  }

  void dispose() {
    if (_disposed) return;
    _disposed = true;
    _models.removeListener(_logReadyModels);
    _state.dispose();
  }
}

/// One model's report row from its state (pure). A ready model's evidence:
/// the backend its runtime reported (or only requested, for speech), the
/// native lines of its load, and what the probe implies.
ModelDiagnostics diagnosticsFor(
  ModelId id,
  ModelState state, {
  required HardwareProfile? hardware,
}) {
  // The chat slot holds the user's own model or the GEMMA_MODEL_PATH define:
  // its rows say which.
  final name = id == ModelId.chat ? 'Chat model' : id.spec.displayName;
  switch (state) {
    case ModelReady(:final info):
      return ModelDiagnostics(
        name: name,
        chat: info.chat,
        state: 'ready',
        evidence: inferEvidence(
          requested: info.backend,
          reported: info.backendReported ? info.backend : null,
          hardware: hardware,
          log: parseNativeLog(info.nativeLog),
        ),
        detail: [
          ?info.detail,
          if (id == ModelId.chat && (info.chat?.images ?? false))
            'vision encoder on the CPU (flutter_edge_ai default, requested)',
        ].join(' · ').nullIfEmpty,
        loadTime: info.loadTime,
        warmUpTime: info.warmUpTime,
        explicitCpu: info.explicitCpu,
      );
    case ModelPending():
      return ModelDiagnostics(name: name, state: 'waiting');
    case ModelInstalling(:final percent):
      return ModelDiagnostics(
        name: name,
        state: 'installing${percent == null ? '' : ' $percent%'}',
      );
    case ModelLoading():
      return ModelDiagnostics(name: name, state: 'loading');
    case ModelWarmingUp():
      return ModelDiagnostics(name: name, state: 'warming up');
    case ModelUnavailable(:final reason):
      return ModelDiagnostics(name: name, state: 'unavailable: $reason');
    case ModelFailed(:final message):
      return ModelDiagnostics(name: name, state: 'failed: $message');
  }
}

extension on String {
  String? get nullIfEmpty => isEmpty ? null : this;
}
