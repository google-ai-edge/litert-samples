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

import 'package:flutter_edge_ai/flutter_edge_ai.dart' show PreferredBackend;

/// One section of a `.litertlm` as its header describes it: its
/// `model_type` (`tf_lite_prefill_decode`, `tf_lite_vision_encoder`, …) and
/// its `backend_constraint` (`npu`, `cpu`), each null when the section has
/// none.
final class const LitertlmSection({
  final String? modelType,
  final String? backendConstraint,
});

/// The towers a chat model's file can have beside its text decoder.
enum Tower {
  /// `tf_lite_vision_encoder` and `tf_lite_vision_adapter`: photos and
  /// camera frames.
  vision,

  /// `tf_lite_audio_encoder_hw` and `tf_lite_audio_adapter` (every
  /// `tf_lite_audio_*` section): a spoken question as audio.
  audio,
}

/// The tower [modelType] belongs to; null for every other section (the
/// decoder, the embedders, and the `tf_lite_end_of_*` markers, which are
/// not part of a tower).
Tower? towerOf(String? modelType) => switch (modelType) {
  'tf_lite_vision_encoder' || 'tf_lite_vision_adapter' => Tower.vision,
  final type? when type.startsWith('tf_lite_audio') => Tower.audio,
  _ => null,
};

/// The file has [tower]: a section of its encoder (`tf_lite_vision_encoder`,
/// `tf_lite_audio_encoder*`). An adapter alone is no tower.
bool hasTower(List<LitertlmSection> sections, Tower tower) => sections.any((s) {
  final type = s.modelType;
  return type != null &&
      switch (tower) {
        Tower.vision => type == 'tf_lite_vision_encoder',
        Tower.audio => type.startsWith('tf_lite_audio_encoder'),
      };
});

/// Why a tower runs where it does.
enum TowerReason {
  /// A section of the tower declares a `backend_constraint` (`cpu`, `npu`,
  /// `gpu`): the file sets it. Read-only, and it wins over a saved choice.
  declared,

  /// Nothing declared and the chat model runs on the NPU: the user's choice
  /// of CPU or NPU (the NPU by default). The card shows a control for it.
  choice,

  /// Nothing declared and the chat model runs on the GPU or the CPU: the
  /// CPU (flutter_edge_ai's default; GPU towers are not offered).
  offNpu,
}

/// Where one tower runs, and why.
final class const TowerPlacement(
  final PreferredBackend backend, {
  final TowerReason reason = TowerReason.offNpu,
}) {
  @override
  bool operator ==(Object other) =>
      other is TowerPlacement &&
      other.backend == backend &&
      other.reason == reason;

  @override
  int get hashCode => Object.hash(backend, reason);

  @override
  String toString() => '${backend.name} (${reason.name})';
}

/// The towers a chat model loads and where each runs: `getActiveModel`'s
/// `supportImage` / `preferredVisionBackend` and `supportAudio` /
/// `preferredAudioBackend`. Derived from the file's header, the chat
/// backend and the user's choices for the towers the header says nothing
/// about, every time a configuration is built ([resolveTowers]): a replaced
/// file or another backend cannot disagree with the file.
final class const ModelTowers({
  /// Null: the file has no vision tower (nothing loaded).
  final TowerPlacement? vision,

  /// Null: the file has no audio tower (nothing loaded).
  final TowerPlacement? audio,
}) {
  /// A file without towers (text only).
  static const none = ModelTowers();

  /// `supportImage`.
  bool get supportImage => vision != null;

  /// `supportAudio`.
  bool get supportAudio => audio != null;

  /// `preferredVisionBackend`: null without a vision tower.
  PreferredBackend? get visionBackend => vision?.backend;

  /// `preferredAudioBackend`: null without an audio tower.
  PreferredBackend? get audioBackend => audio?.backend;

  /// [tower]'s placement; null when the file does not have it.
  TowerPlacement? of(Tower tower) => switch (tower) {
    Tower.vision => vision,
    Tower.audio => audio,
  };

  /// A tower the header says nothing about runs on the NPU (the user's
  /// choice, or its default): a failed load of this configuration says how
  /// to put it on the CPU.
  bool get undeclaredOnNpu => [vision, audio].any(
    (p) =>
        p?.reason == TowerReason.choice && p?.backend == PreferredBackend.npu,
  );

  /// Each tower that is loaded, with where it runs: `vision cpu`, `audio
  /// npu`; empty for a text-only file.
  List<String> get parts => [
    if (visionBackend case final v?) 'vision ${v.name}',
    if (audioBackend case final a?) 'audio ${a.name}',
  ];

  @override
  bool operator ==(Object other) =>
      other is ModelTowers && other.vision == vision && other.audio == audio;

  @override
  int get hashCode => Object.hash(vision, audio);

  @override
  String toString() => parts.isEmpty ? 'no towers' : parts.join(' · ');
}

/// The towers a file whose header has [sections] loads with its chat model
/// on [backend]: each tower the file has ([hasTower]), placed on its own by
/// the sections of that tower (its encoder and adapter; the
/// `tf_lite_end_of_*` markers do not count):
///
/// 1. a section declares `backend_constraint: cpu` → the CPU, read-only;
/// 2. else a section declares `npu` → the NPU, read-only;
/// 3. else a section declares `gpu` → the GPU, read-only (no known Gemma
///    tower does; flutter_edge_ai_litertlm documents the case, and asking
///    such a tower for the CPU fails the load);
/// 4. else (nothing declared, or a value that names no backend), on the
///    NPU → the user's choice ([vision], [audio]: the CPU or the NPU), the
///    NPU when there is none;
/// 5. else (the GPU or the CPU) → the CPU (flutter_edge_ai's default; an
///    undeclared tower is never put on the GPU).
///
/// A declaration always wins over a choice, and a choice means nothing off
/// the NPU. A file whose header cannot be read is never loaded (it is
/// refused when it is chosen, and a problem at launch), so there is no
/// "unknown" placement.
ModelTowers resolveTowers(
  List<LitertlmSection> sections,
  PreferredBackend backend, {
  PreferredBackend? vision,
  PreferredBackend? audio,
}) {
  TowerPlacement? place(Tower tower, PreferredBackend? choice) {
    if (!hasTower(sections, tower)) return null;
    final declared = {
      for (final s in sections)
        if (towerOf(s.modelType) == tower) ?s.backendConstraint?.toLowerCase(),
    };
    if (declared.contains('cpu')) {
      return const TowerPlacement(
        PreferredBackend.cpu,
        reason: TowerReason.declared,
      );
    }
    if (declared.contains('npu')) {
      return const TowerPlacement(
        PreferredBackend.npu,
        reason: TowerReason.declared,
      );
    }
    if (declared.contains('gpu')) {
      return const TowerPlacement(
        PreferredBackend.gpu,
        reason: TowerReason.declared,
      );
    }
    if (backend == PreferredBackend.npu) {
      return TowerPlacement(
        choice ?? PreferredBackend.npu,
        reason: TowerReason.choice,
      );
    }
    return const TowerPlacement(PreferredBackend.cpu);
  }

  return ModelTowers(
    vision: place(Tower.vision, vision),
    audio: place(Tower.audio, audio),
  );
}

/// The detail line of a loaded vision tower asked to run on [backend]
/// (`cpu`, `npu`: a `PreferredBackend` name), for the Models screen's
/// report and the self-test: where it was asked to run, whoever decided it
/// (the file's header, the user's choice, or the CPU off the NPU).
String visionTowerDetail(String backend) =>
    'vision encoder on the ${backend.toUpperCase()} (requested)';
