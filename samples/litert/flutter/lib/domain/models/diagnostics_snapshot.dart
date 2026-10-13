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

import 'assistant_event.dart';
import 'audio_devices.dart';
import 'camera_side_event.dart';
import 'knowledge.dart';
import 'live_state.dart';
import 'llm_image.dart';
import 'model_id.dart';
import 'model_state.dart';
import 'skill_catalog.dart';
import 'voice.dart';

/// Everything the debug overlay shows, as one immutable value.
final class const DiagnosticsSnapshot({
  /// Loaded models only.
  final Map<ModelId, LoadedModelInfo> models = const {},
  final GenerationMetrics? lastGeneration,

  /// `ProcessInfo.currentRss` / `ProcessInfo.maxRss`.
  final int? rssBytes,
  final int? peakRssBytes,

  /// Live detection (Demo 3): its state and rolling figures.
  final LiveState liveState = const LiveStopped(),
  final LiveStats liveStats = const LiveStats(),

  /// The voice turn state machine's phase, and the newest turn's timings.
  final TurnPhase voicePhase = TurnPhase.idle,
  final VoiceTurnMetrics? lastVoiceTurn,
  final BargeInMetrics? lastBargeIn,

  /// Demo 3: the newest camera question's route and timings.
  final CameraTurnMetrics? lastCameraTurn,

  /// Demo 3: the newest post-turn chat reset and its failure.
  final Duration? lastCameraReset,
  final String? cameraResetError,

  /// Demo 3: the frame the view is frozen on; null while live.
  final int? frozenFrameId,

  /// The active speech recognizer (switched per demo); null while none is.
  final ActiveSttInfo? activeStt,

  /// Why the last recognizer switch failed; null after a successful one.
  final String? sttSwitchError,

  /// Generations the shared chat has started since launch (both demos),
  /// counted from its generating flag: the fast path must leave it alone.
  final int llmTurns = 0,

  /// Demo 1: the picture attached to the next turns; null when none.
  final LlmImage? attachment,

  /// The knowledge base's state (null when none is wired) and the
  /// newest Demo 1 turn's retrieval.
  final KnowledgeStatus? knowledge,
  final Retrieval? lastRetrieval,

  /// The runtime skills' latest scan (null when none is wired or
  /// none finished yet).
  final SkillCatalog? skills,

  /// The voice demos' input and output as last checked.
  final AudioDeviceStatus audioDevices = const AudioDeviceStatus(),
}) {
  DiagnosticsSnapshot copyWith({
    Map<ModelId, LoadedModelInfo>? models,
    GenerationMetrics? lastGeneration,
    int? rssBytes,
    int? peakRssBytes,
    LiveState? liveState,
    LiveStats? liveStats,
    TurnPhase? voicePhase,
    VoiceTurnMetrics? lastVoiceTurn,
    BargeInMetrics? lastBargeIn,
    CameraTurnMetrics? lastCameraTurn,
    ({Duration elapsed, String? error})? cameraReset,

    /// A record so it can be cleared: `(frameId: null)`.
    ({int? frameId})? frozen,

    /// A record so it can be cleared: `(stt: null)`.
    ({ActiveSttInfo? stt})? activeStt,

    /// A record so it can be cleared: `(error: null)`.
    ({String? error})? sttSwitchError,
    int? llmTurns,

    /// A record so the attachment can be cleared: `(image: null)`.
    ({LlmImage? image})? attachment,
    KnowledgeStatus? knowledge,
    Retrieval? lastRetrieval,
    SkillCatalog? skills,
    AudioDeviceStatus? audioDevices,
  }) => DiagnosticsSnapshot(
    models: models ?? this.models,
    lastGeneration: lastGeneration ?? this.lastGeneration,
    rssBytes: rssBytes ?? this.rssBytes,
    peakRssBytes: peakRssBytes ?? this.peakRssBytes,
    liveState: liveState ?? this.liveState,
    liveStats: liveStats ?? this.liveStats,
    voicePhase: voicePhase ?? this.voicePhase,
    lastVoiceTurn: lastVoiceTurn ?? this.lastVoiceTurn,
    lastBargeIn: lastBargeIn ?? this.lastBargeIn,
    lastCameraTurn: lastCameraTurn ?? this.lastCameraTurn,
    lastCameraReset: cameraReset == null
        ? lastCameraReset
        : cameraReset.elapsed,
    cameraResetError: cameraReset == null
        ? cameraResetError
        : cameraReset.error,
    frozenFrameId: frozen == null ? frozenFrameId : frozen.frameId,
    activeStt: activeStt == null ? this.activeStt : activeStt.stt,
    sttSwitchError: sttSwitchError == null
        ? this.sttSwitchError
        : sttSwitchError.error,
    llmTurns: llmTurns ?? this.llmTurns,
    attachment: attachment == null ? this.attachment : attachment.image,
    knowledge: knowledge ?? this.knowledge,
    lastRetrieval: lastRetrieval ?? this.lastRetrieval,
    skills: skills ?? this.skills,
    audioDevices: audioDevices ?? this.audioDevices,
  );
}

/// The active speech recognizer as the overlay shows it.
final class const ActiveSttInfo({
  required final ModelId id,
  required final String modelId,

  /// The last switch on demo entry (null for setup's load).
  final Duration? switchTime,
});
