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

import 'detection.dart';

/// What the detector saw "just now": per class, the median
/// count over a window of recent frames, counting only boxes scoring at
/// least `minScore`. The median absorbs one-frame flicker at the threshold.
final class DetectionSummary {
  DetectionSummary._(this.counts, this.frames);

  /// The median count of each class over [window], for classes whose median
  /// is at least 1, most numerous first (then by class id). An even window
  /// takes the lower median: a count that showed in only half the frames is
  /// not reported.
  factory DetectionSummary.fromWindow(
    List<DetectionFrame> window, {
    required double minScore,
  }) {
    final perFrame = [for (final frame in window) _counts(frame, minScore)];
    final classes = {for (final counts in perFrame) ...counts.keys};
    final medians = <int, int>{};
    for (final cls in classes) {
      final values = [for (final counts in perFrame) counts[cls] ?? 0]..sort();
      final median = values[(values.length - 1) ~/ 2];
      if (median > 0) medians[cls] = median;
    }
    final ordered = medians.keys.toList()
      ..sort((a, b) {
        final byCount = medians[b]!.compareTo(medians[a]!);
        return byCount != 0 ? byCount : a.compareTo(b);
      });
    return DetectionSummary._(
      Map.unmodifiable({for (final cls in ordered) cls: medians[cls]!}),
      window.length,
    );
  }

  /// Class id → median count (≥ 1), most numerous first.
  final Map<int, int> counts;

  /// Frames the summary was taken over.
  final int frames;

  bool get isEmpty => counts.isEmpty;

  int countOf(int cls) => counts[cls] ?? 0;

  /// "cat ×2 · remote", at most [max] classes, named by [name].
  String label(String Function(int cls) name, {int max = 4}) {
    if (counts.isEmpty) return 'nothing detected';
    final parts = [
      for (final MapEntry(:key, :value) in counts.entries.take(max))
        value == 1 ? name(key) : '${name(key)} ×$value',
    ];
    final more = counts.length - max;
    return [...parts, if (more > 0) '+$more'].join(' · ');
  }

  /// Boxes per class in [frame] scoring at least [minScore].
  static Map<int, int> _counts(DetectionFrame frame, double minScore) {
    final counts = <int, int>{};
    for (var i = 0; i < frame.count; i++) {
      if (frame.score(i) < minScore) continue;
      counts.update(frame.classId(i), (n) => n + 1, ifAbsent: () => 1);
    }
    return counts;
  }
}
