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

import '../../domain/models/assistant_event.dart';
import '../../domain/models/audio_devices.dart';
import '../../domain/models/camera_side_event.dart';
import '../../domain/models/diagnostics_snapshot.dart';
import '../../domain/models/knowledge.dart';
import '../../domain/models/live_state.dart';
import '../../domain/models/llm_image.dart';
import '../../domain/models/model_id.dart';
import '../../domain/models/model_state.dart';
import '../../domain/models/skill_catalog.dart';
import '../../domain/models/voice.dart';
import '../../domain/ports/voice_diagnostics_sink.dart';
import '../services/speech/stt_service.dart' show ActiveStt;

/// Collects what the debug overlay shows and publishes it as one
/// [DiagnosticsSnapshot], coalesced to at most one update per [minInterval]
/// (≤4 Hz by default) so high-rate sources never drive high-rate rebuilds.
class DiagnosticsRepository implements VoiceDiagnosticsSink {
  DiagnosticsRepository({
    required this._models,
    this._liveState,
    this._liveStats,
    this._llmBusy,
    this._knowledge,
    this._activeStt,
    this._sttSwitchError,
    this._skills,
    this._audioDevices,
    this._minInterval = const Duration(milliseconds: 250),
    Duration rssInterval = const Duration(seconds: 1),
    bool sampleRss = true,
  }) {
    _skills?.addListener(_onSkills);
    _onSkills();
    _audioDevices?.addListener(_onAudioDevices);
    _onAudioDevices();
    _models.addListener(_onModels);
    _onModels();
    _liveState?.addListener(_onLive);
    _liveStats?.addListener(_onLive);
    _onLive();
    _llmWasBusy = _llmBusy?.value ?? false;
    _llmBusy?.addListener(_onLlmBusy);
    _knowledge?.addListener(_onKnowledge);
    _onKnowledge();
    _activeStt?.addListener(_onActiveStt);
    _onActiveStt();
    _sttSwitchError?.addListener(_onSttSwitchError);
    _onSttSwitchError();
    if (sampleRss) {
      _sampleRss();
      _rssTimer = Timer.periodic(rssInterval, (_) => _sampleRss());
    }
  }

  final ValueListenable<Map<ModelId, ModelState>> _models;

  /// Live detection's state and stats (≤4 Hz already; coalesced again here).
  final ValueListenable<LiveState>? _liveState;
  final ValueListenable<LiveStats>? _liveStats;

  /// The shared chat's generating flag; each rising edge is one LLM turn.
  final ValueListenable<bool>? _llmBusy;
  bool _llmWasBusy = false;

  /// The knowledge base's state (indexing progress at batch rate).
  final ValueListenable<KnowledgeStatus>? _knowledge;

  /// The active speech recognizer (changes on demo switches) and why the
  /// last switch failed.
  final ValueListenable<ActiveStt?>? _activeStt;
  final ValueListenable<String?>? _sttSwitchError;

  /// The runtime skills' latest scan.
  final ValueListenable<SkillCatalog?>? _skills;

  /// The voice demos' input and output checks (a few changes per session).
  final ValueListenable<AudioDeviceStatus>? _audioDevices;
  final Duration _minInterval;
  final ValueNotifier<DiagnosticsSnapshot> _snapshot = ValueNotifier(
    const DiagnosticsSnapshot(),
  );
  final Stopwatch _clock = Stopwatch()..start();

  DiagnosticsSnapshot _latest = const DiagnosticsSnapshot();
  Duration? _lastFlushAt;
  Timer? _flushTimer;
  Timer? _rssTimer;
  bool _disposed = false;

  /// The coalesced snapshot for the overlay.
  ValueListenable<DiagnosticsSnapshot> get snapshot => _snapshot;

  /// The newest values, not yet coalesced. For tests and log lines.
  DiagnosticsSnapshot get latest => _latest;

  /// Records the timing of a finished turn.
  void recordGeneration(GenerationMetrics metrics) {
    debugPrint(
      '[Diagnostics] turn ttft=${metrics.timeToFirstToken?.inMilliseconds}ms '
      'tokps=${metrics.tokensPerSecond?.toStringAsFixed(1)} '
      '(${metrics.tokensPerSecondSource.name}) chunks=${metrics.chunks} '
      'total=${metrics.total.inMilliseconds}ms stopped=${metrics.stopped}'
      '${metrics.stopLatency == null ? '' : ' stop=${metrics.stopLatency!.inMilliseconds}ms'}'
      ' ctx=${metrics.contextTokens} prefill=${metrics.prefillTokens} '
      'image=${metrics.imageSent ? 'sent' : (metrics.imageAttached ? 'kept' : 'none')}'
      '${metrics.imageResent == null ? '' : ' resent(${metrics.imageResent!.name})'}'
      '${metrics.contextReset ? ' context_reset' : ''}'
      '${metrics.toolRounds == 0 ? '' : ' tools=${metrics.toolRounds} steps=${metrics.skillSteps.map((s) => '$s@${s.at.inMilliseconds}ms').join(' | ')}'}',
    );
    _update(_latest.copyWith(lastGeneration: metrics));
  }

  /// Demo 1's attached picture (null: removed).
  void recordAttachment(LlmImage? image) =>
      _update(_latest.copyWith(attachment: (image: image)));

  @override
  void recordVoicePhase(TurnPhase phase) {
    if (phase == _latest.voicePhase) return;
    _update(_latest.copyWith(voicePhase: phase));
  }

  /// Called after STT, at the first audio, per clause and at the end.
  @override
  void recordVoiceTurn(VoiceTurnMetrics metrics) =>
      _update(_latest.copyWith(lastVoiceTurn: metrics));

  @override
  void recordBargeIn(BargeInMetrics metrics) =>
      _update(_latest.copyWith(lastBargeIn: metrics));

  /// A camera question's route and timings (recorded again as a detailed
  /// turn's facts arrive: snapshot, PNG, TTFT).
  void recordCameraTurn(CameraTurnMetrics metrics) {
    final image = metrics.image;
    debugPrint(
      '[Diagnostics] camq route=${metrics.route.runtimeType} '
      'rule=${metrics.route.rule} '
      'route=${metrics.routeTime.inMicroseconds}µs '
      'snap=${metrics.snapshotLatency?.inMilliseconds}ms '
      'basis="${metrics.basis ?? ''}" llm_turns=${_latest.llmTurns}'
      '${image == null ? '' : ' png=${image.width}x${image.height} '
                '${image.png.length}B enc=${image.encodeTime.inMilliseconds}ms '
                'frame=${image.frameId}'}'
      '${metrics.timeToFirstToken == null ? '' : ' ttft=${metrics.timeToFirstToken!.inMilliseconds}ms'}'
      '${metrics.snapshotError == null ? '' : ' snapshot_error="${metrics.snapshotError}"'}'
      '${metrics.chatError == null ? '' : ' chat_error="${metrics.chatError}"'}',
    );
    _update(_latest.copyWith(lastCameraTurn: metrics));
  }

  /// Demo 3's post-turn chat reset: its time, or its failure.
  void recordCameraReset(Duration elapsed, {String? error}) {
    debugPrint(
      '[Diagnostics] camq reset=${elapsed.inMilliseconds}ms'
      '${error == null ? '' : ' failed: $error'}',
    );
    _update(_latest.copyWith(cameraReset: (elapsed: elapsed, error: error)));
  }

  /// Demo 3's view froze on [frameId] (null: live again).
  void recordFrozen(int? frameId) {
    if (frameId == _latest.frozenFrameId) return;
    _update(_latest.copyWith(frozen: (frameId: frameId)));
  }

  /// A Demo 1 turn's knowledge-base retrieval.
  void recordRetrieval(Retrieval retrieval) {
    debugPrint(
      '[Diagnostics] retrieval ${retrieval.outcome.name} '
      'latency=${retrieval.latency?.inMilliseconds}ms '
      'top=${retrieval.topSimilarity?.toStringAsFixed(3)} '
      'gate=${retrieval.gate} excerpts=${retrieval.passages.length}'
      '${retrieval.detail == null ? '' : ' detail="${retrieval.detail}"'}',
    );
    _update(_latest.copyWith(lastRetrieval: retrieval));
  }

  void _onSttSwitchError() {
    final listenable = _sttSwitchError;
    if (listenable == null) return;
    _update(_latest.copyWith(sttSwitchError: (error: listenable.value)));
  }

  void _onActiveStt() {
    final listenable = _activeStt;
    if (listenable == null) return;
    final active = listenable.value;
    _update(
      _latest.copyWith(
        activeStt: (
          stt: active == null
              ? null
              : ActiveSttInfo(
                  id: active.id,
                  modelId: active.modelId,
                  switchTime: active.switchTime,
                ),
        ),
      ),
    );
  }

  void _onKnowledge() {
    final status = _knowledge?.value;
    if (status == null) return;
    _update(_latest.copyWith(knowledge: status));
  }

  void _onSkills() {
    final catalog = _skills?.value;
    if (catalog == null) return;
    _update(_latest.copyWith(skills: catalog));
  }

  void _onAudioDevices() {
    final status = _audioDevices?.value;
    if (status == null) return;
    _update(_latest.copyWith(audioDevices: status));
  }

  void _onLlmBusy() {
    final busy = _llmBusy!.value;
    if (busy && !_llmWasBusy) {
      _update(_latest.copyWith(llmTurns: _latest.llmTurns + 1));
    }
    _llmWasBusy = busy;
  }

  void _onModels() {
    final loaded = <ModelId, LoadedModelInfo>{
      for (final MapEntry(:key, :value) in _models.value.entries)
        if (value case ModelReady(:final info)) key: info,
    };
    _update(_latest.copyWith(models: Map.unmodifiable(loaded)));
  }

  void _onLive() {
    final state = _liveState?.value;
    final stats = _liveStats?.value;
    if (state == null && stats == null) return;
    _update(_latest.copyWith(liveState: state, liveStats: stats));
  }

  void _sampleRss() => _update(
    _latest.copyWith(
      rssBytes: ProcessInfo.currentRss,
      peakRssBytes: ProcessInfo.maxRss,
    ),
  );

  void _update(DiagnosticsSnapshot next) {
    _latest = next;
    if (_disposed || _flushTimer != null) return;
    final last = _lastFlushAt;
    final wait = last == null
        ? Duration.zero
        : _minInterval - (_clock.elapsed - last);
    _flushTimer = Timer(wait.isNegative ? Duration.zero : wait, _flush);
  }

  void _flush() {
    _flushTimer = null;
    if (_disposed) return;
    _lastFlushAt = _clock.elapsed;
    _snapshot.value = _latest;
  }

  /// Stops sampling. Dispose before the model repository it listens to.
  void dispose() {
    if (_disposed) return;
    _disposed = true;
    _models.removeListener(_onModels);
    _liveState?.removeListener(_onLive);
    _liveStats?.removeListener(_onLive);
    _llmBusy?.removeListener(_onLlmBusy);
    _knowledge?.removeListener(_onKnowledge);
    _activeStt?.removeListener(_onActiveStt);
    _sttSwitchError?.removeListener(_onSttSwitchError);
    _skills?.removeListener(_onSkills);
    _audioDevices?.removeListener(_onAudioDevices);
    _rssTimer?.cancel();
    _flushTimer?.cancel();
    _snapshot.dispose();
  }
}
