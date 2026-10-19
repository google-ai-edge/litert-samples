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

import '../../config/live_camera_config.dart';

/// How a memory question's scores were judged ([judgeMemory]).
enum MemoryOutcome {
  /// One scene clearly wins: its frame is the answer.
  accepted,

  /// At least the floor plus the margin, but a different scene scores within
  /// the margin: the object may have been seen in more than one place; the
  /// newest of the competing frames is the answer, said as such.
  ambiguous,

  /// Nothing scores high enough: not seen.
  rejected,
}

/// The verdict on a memory question's scores ([judgeMemory]).
final class const MemoryVerdict({
  /// The frame to report, an index into the scores: for [MemoryOutcome.
  /// accepted] the newest near-identical frame that ties with the best (a
  /// static scene's frames score within noise of each other, and "seen 12 s
  /// ago" must come from its latest sighting); for [MemoryOutcome.ambiguous]
  /// the newest of the competing frames; for [MemoryOutcome.rejected] the
  /// best.
  required final int best,

  /// The best score of all frames (what the floor and the margin judge).
  required final double cosine,

  /// The best rival: a frame more than the separation away in time that
  /// shows a different scene; null when there is none.
  final double? rival,
  required final MemoryOutcome outcome,

  /// `accepted`, `below the floor 0.62`, `no rival and below 0.645 (floor +
  /// margin)`, `not distinct and below 0.645 (floor + margin): …` or `not
  /// distinct: …`.
  required final String reason,
}) {
  bool get accepted => outcome == MemoryOutcome.accepted;
}

/// The image vectors' cosine of every frame to the best one, which
/// [judgeMemory]'s caller fills in once it knows the best frame.
typedef SceneCosineOf = double Function(int index, int best);

/// Whether frame `index`'s grey thumbnail shows a change from the best
/// frame's (the sampler's scene-change test: an object put down or taken
/// away), which [judgeMemory]'s caller answers once it knows the best frame.
typedef ViewChangedOf = bool Function(int index, int best);

/// Whether a memory question found its frame.
///
/// The best frame must score at least [floor]. A rival is a frame more than
/// [separation] away in time whose image is not the same scene as the best
/// frame's (their image vectors' cosine under [sameScene]): an object seen
/// for half a minute is in several frames of one scene, and those agree
/// rather than compete.
///
/// - **Accepted**: the best beats its best rival by at least [margin]; with
///   no rival at all (a session that saw one scene) it must clear
///   [floor] + [margin] instead, so the floor alone never decides. The frame
///   reported is the newest near-identical frame (image cosine ≥ [sameScene])
///   whose score is within [tie] of the best and whose thumbnail shows no
///   change from the best's ([viewChanged]): the image vectors of a desk
///   with and without the keys are near-identical, so "the keys, seen 12 s
///   ago" must not come from the frame after they were taken away.
/// - **Ambiguous**: at least [floor] + [margin] (so the rival, within the
///   margin, is at or above the floor: two real sightings), but a rival
///   scores within the margin (an object seen in two places ties with its
///   own second sighting). The frame reported is the newest of the competing
///   frames: those at or above the floor and within the margin of the best.
/// - **Rejected**: under the floor; or under floor + margin with no rival,
///   or with a rival within the margin (a faint best that does not stand out
///   from the rest is not a sighting, whichever side of the floor it is).
///
/// [cosines] and [atMicros] are per frame, in the same order: the order of
/// sightings, oldest first, as the store keeps them (it only refreshes the
/// newest frame in place). "Newest" means the latest in that order, never
/// the largest [atMicros]: those are wall-clock times, which a clock set
/// back can reorder. The time separation is measured on [atMicros].
/// [sceneCosine] gives the image cosine of frame `i` to the best frame,
/// [viewChanged] whether its thumbnail shows a change from the best's
/// (null: never). Null for an empty memory.
///
/// A score that is not a finite number (NaN from an unusable vector) is
/// never the best, a rival or a reported frame; when no score is finite the
/// verdict is rejected (`no score is a number`) with a NaN cosine. Of equal
/// best scores the earliest frame is the best.
MemoryVerdict? judgeMemory({
  required List<double> cosines,
  required List<int> atMicros,
  required SceneCosineOf sceneCosine,
  ViewChangedOf? viewChanged,
  double floor = kMemoryMinCosine,
  double margin = kMemoryMinMargin,
  Duration separation = kMemoryMarginSeparation,
  double sameScene = kMemorySameSceneCosine,
  double tie = kMemoryTieCosine,
}) {
  if (cosines.length != atMicros.length) {
    throw ArgumentError(
      '${cosines.length} cosines for ${atMicros.length} frames',
    );
  }
  if (cosines.isEmpty) return null;
  // A NaN compares false with everything: as the first score it would stay
  // the best and pass every test below. Only finite scores take part.
  var best = -1;
  for (var i = 0; i < cosines.length; i++) {
    if (!cosines[i].isFinite) continue;
    if (best < 0 || cosines[i] > cosines[best]) best = i;
  }
  if (best < 0) {
    return const MemoryVerdict(
      best: 0,
      cosine: double.nan,
      outcome: MemoryOutcome.rejected,
      reason: 'no score is a number',
    );
  }
  final top = cosines[best];
  final apart = separation.inMicroseconds;
  // The image cosines to the best frame, asked once per frame (not for a
  // frame without a finite score, which takes no part).
  final scene = [
    for (var i = 0; i < cosines.length; i++)
      i == best
          ? 1.0
          : cosines[i].isFinite
          ? sceneCosine(i, best)
          : double.nan,
  ];
  double? rival;
  for (var i = 0; i < cosines.length; i++) {
    if (i == best || !cosines[i].isFinite) continue;
    if ((atMicros[i] - atMicros[best]).abs() <= apart) continue;
    if (scene[i] >= sameScene) continue;
    if (rival == null || cosines[i] > rival) rival = cosines[i];
  }
  // The newest frame [keep] admits; the best when none is newer. Newest by
  // index (the order of sightings), not by time: a wall clock set back
  // would make a later sighting look older.
  int newest(bool Function(int i) keep) {
    var pick = best;
    for (var i = 0; i < cosines.length; i++) {
      if (keep(i) && i > pick) pick = i;
    }
    return pick;
  }

  final MemoryOutcome outcome;
  final String reason;
  var report = best;
  if (top < floor) {
    outcome = MemoryOutcome.rejected;
    reason = 'below the floor ${floor.toStringAsFixed(2)}';
  } else if (rival == null && top < floor + margin) {
    outcome = MemoryOutcome.rejected;
    reason =
        'no rival and below ${(floor + margin).toStringAsFixed(3)} (floor + '
        'margin)';
  } else if (rival != null && top - rival < margin) {
    final distance =
        '${(top - rival).toStringAsFixed(3)} over a rival at '
        '${rival.toStringAsFixed(3)} (margin ${margin.toStringAsFixed(3)})';
    if (top < floor + margin) {
      outcome = MemoryOutcome.rejected;
      reason =
          'not distinct and below ${(floor + margin).toStringAsFixed(3)} '
          '(floor + margin): $distance';
    } else {
      outcome = MemoryOutcome.ambiguous;
      reason = 'not distinct: $distance';
      report = newest((i) => cosines[i] >= floor && top - cosines[i] < margin);
    }
  } else {
    outcome = MemoryOutcome.accepted;
    reason = 'accepted';
    report = newest(
      (i) =>
          scene[i] >= sameScene &&
          top - cosines[i] <= tie &&
          !(viewChanged?.call(i, best) ?? false),
    );
  }
  return MemoryVerdict(
    best: report,
    cosine: top,
    rival: rival,
    outcome: outcome,
    reason: reason,
  );
}
