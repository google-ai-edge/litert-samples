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

import 'dart:typed_data';

import 'detection.dart';

/// A memory frame as the detector worker made it from a live frame: that
/// frame's detections, its grey thumbnail (the scene-change test), and the
/// PNG when the scene changed.
final class const MemoryShot({
  required final int frameId,

  /// The live frame's own detections (published like any other frame).
  required final DetectionFrame detections,

  /// The frame's grey thumbnail: 128×128 (32×32 cells of 4×4 pixels, 16
  /// KB), the scene-change test's input.
  required final Uint8List grey,

  /// The mean absolute grey difference to the reference thumbnail (0–255),
  /// once the two are matched in overall brightness; null when there was no
  /// reference.
  final double? difference,

  /// How many thumbnail cells differ from the reference's by at least the
  /// cell threshold (`kMemoryCellChange`), after the same brightness match;
  /// null when there was no reference.
  final int? changedCells,

  /// The upright frame at most `kMemoryThumbMaxSide` on its long side,
  /// mirrored back when the source mirrors; null when the scene had not
  /// changed (nothing to embed).
  final Uint8List? png,

  /// The PNG's size (also when [png] was not made).
  required final int width,
  required final int height,

  /// The frame was mirrored back.
  final bool unmirrored = false,

  /// The worker's conversion, downscale, thumbnail and PNG.
  required final Duration workerTime,
});

/// The detector worker could not make a memory frame (downscale, thumbnail
/// or PNG) of the frame it detected: that memory capture fails, the frame's
/// detections are published and the live pipeline goes on. The visual
/// memory counts it like a failed embedding.
final class MemoryFrameException implements Exception {
  const MemoryFrameException(this.message);

  final String message;

  @override
  String toString() => message;
}

/// One remembered frame: when it was seen, its PNG thumbnail, what the
/// detector saw in it, and its EmbeddingGemma 2 vector. Kept in memory only.
final class const MemoryEntry({
  required final int id,
  required final int frameId,

  /// The monotonic clock (µs) when it was last seen: when it was stored, or
  /// the latest time an unchanged scene refreshed it ([seenAgainAt]). Ages
  /// ("seen 12 s ago") and the acceptance's time separation are measured on
  /// it.
  required final int atMicros,

  /// When it was first seen (stored), on the same clock; [atMicros] when not
  /// given. A refresh keeps it: how far back the memory reaches is measured
  /// from the oldest frame's first sighting.
  int? firstSeenMicros,

  /// The PNG thumbnail (what was embedded); its length counts against the
  /// byte cap.
  required final Uint8List png,
  required final int width,
  required final int height,

  /// Class id → boxes in that frame at the counting score, most numerous
  /// first.
  required final Map<int, int> counts,

  /// The image vector.
  required final Float32List vector,

  /// The 128×128 grey thumbnail (the next frame's scene-change reference,
  /// and recall's tie test); 16 KB, not counted in the PNG byte cap (200
  /// frames hold 3.2 MB of them).
  required final Uint8List grey,
}) {
  this : firstSeenMicros = firstSeenMicros ?? atMicros;

  /// When it was first seen (stored); see the constructor.
  final int firstSeenMicros;

  /// This frame, seen again unchanged at [atMicros] (a static scene's
  /// re-store): its last-seen time moves, its first sighting and everything
  /// else are kept.
  MemoryEntry seenAgainAt(int atMicros) => MemoryEntry(
    id: id,
    frameId: frameId,
    atMicros: atMicros,
    firstSeenMicros: firstSeenMicros,
    png: png,
    width: width,
    height: height,
    counts: counts,
    vector: vector,
    grey: grey,
  );
}

/// What the visual memory holds and how it is doing, for the camera screen's
/// chip and banner and the debug overlay. Replaced, never mutated.
final class const VisualMemoryStatus({
  /// The camera demo is open: frames are sampled while it is idle.
  final bool active = false,

  /// The "Visual memory" switch in the camera settings: off, nothing is
  /// sampled, the image encoder is not loaded and the chip is hidden.
  final bool enabled = true,

  /// EmbeddingGemma's image encoder is being loaded (the first time the
  /// camera demo starts its memory): nothing is sampled until it is.
  final bool preparing = false,
  final int frames = 0,

  /// PNG thumbnail bytes held.
  final int bytes = 0,

  /// From the oldest frame's first sighting to the last publish.
  final Duration span = Duration.zero,

  /// The last frame embedding's time.
  final Duration? lastEmbed,

  /// Failed frame embeddings since the last clear, and how many of them in a
  /// row (at `kMemoryFailuresUntilOff` the memory turns off).
  final int failures = 0,
  final int failuresInARow = 0,

  /// Why the visual memory is off (the embedder, or failed embeddings);
  /// null while it works.
  final String? unavailable,

  /// The last memory question: the best frame's cosine (null: none asked
  /// since the last clear, or nothing to compare).
  final double? lastBestCosine,
}) {
  VisualMemoryStatus copyWith({
    bool? active,
    bool? enabled,
    bool? preparing,
    int? frames,
    int? bytes,
    Duration? span,
    Duration? lastEmbed,
    int? failures,
    int? failuresInARow,

    /// A record so it can be cleared: `(reason: null)`.
    ({String? reason})? unavailable,
    double? lastBestCosine,
  }) => VisualMemoryStatus(
    active: active ?? this.active,
    enabled: enabled ?? this.enabled,
    preparing: preparing ?? this.preparing,
    frames: frames ?? this.frames,
    bytes: bytes ?? this.bytes,
    span: span ?? this.span,
    lastEmbed: lastEmbed ?? this.lastEmbed,
    failures: failures ?? this.failures,
    failuresInARow: failuresInARow ?? this.failuresInARow,
    unavailable: unavailable == null ? this.unavailable : unavailable.reason,
    lastBestCosine: lastBestCosine ?? this.lastBestCosine,
  );
}

/// What a memory question found.
sealed class const MemoryRecall();

/// A frame to answer with: [entry], its [cosine], the best different-scene
/// frame more than the separation away ([rival], null when there was
/// none), and how long ago it was seen.
///
/// [ambiguous]: no scene won clearly — a different scene scored within the
/// margin — so the object may have been seen in more than one place;
/// [entry] is then the newest of the competing frames, and the answer says
/// so. Otherwise one scene won, and [entry] is its newest near-identical
/// frame (its latest sighting).
///
/// [cosine] null: the newest stored frame, asked for by a question with
/// nothing to search for ("What did you see?"), nothing scored ([latest]).
final class const MemoryHit({
  required final MemoryEntry entry,
  required final double? cosine,
  final double? rival,
  required final Duration age,
  final bool ambiguous = false,
}) extends MemoryRecall {
  /// The newest stored frame, not a search result.
  bool get latest => cosine == null;
}

/// No frame was accepted: the best [cosine] was under the floor, or under
/// the floor plus the margin with no rival or a rival within the margin.
/// [span] is how far back the memory reaches.
final class const MemoryMiss({
  required final double cosine,
  final double? rival,
  required final Duration span,
  required final int frames,

  /// `below the floor 0.62`, `no rival and below 0.645 (floor + margin)` or
  /// `not distinct and below 0.645 (floor + margin): …`.
  required final String reason,
}) extends MemoryRecall;

/// Nothing is remembered yet.
final class const MemoryEmpty() extends MemoryRecall;

/// The visual memory is off; [reason] says why.
final class const MemoryOff(final String reason) extends MemoryRecall;

/// The visual memory is switched off in the camera settings.
final class const MemoryDisabled() extends MemoryRecall;
