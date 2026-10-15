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

/// Something on the trail ahead, as the narrator hears about it.
class AheadItem {
  const AheadItem(this.label, this.metres);
  final String label;
  final double metres;
}

/// The facts the narrator gets beside the events each turn: what is coming,
/// how Dash is doing, and the hardware the narrator itself is running on.
///
/// Events alone made the commentary a list of speeds: every event carried
/// one, so every line quoted one. A short situation gives the model
/// something to build suspense from and something to joke about, the chip it
/// is running on, which is the act's own argument.
///
/// Pure data and formatting, so the wording is unit-tested.
class RunSituation {
  const RunSituation({
    this.ahead = const [],
    this.riverIn,
    this.rise = 0,
    this.nextBiome,
    this.nextBiomeIn,
    this.foreshadow = false,
    this.streak = 0,
    this.bestStreak = 0,
    this.stumblesLastMinute = 0,
    this.presenterDriving = false,
    this.backend,
    this.speedShare,
    this.frameBudget,
    this.cpuLoad,
    this.gpuLoad,
    this.npuLoad,
  });

  /// The next obstacles, nearest first.
  final List<AheadItem> ahead;

  /// Metres to the next river crossing beyond [ahead], if there is one soon.
  final double? riverIn;

  /// Height change over the next 30 m of trail, metres.
  final double rise;

  final String? nextBiome;
  final double? nextBiomeIn;

  /// The autopilot has already decided to fumble the next obstacle. The
  /// narrator gets a hint, never the outcome: suspense, not a spoiler.
  final bool foreshadow;

  final int streak;
  final int bestStreak;
  final int stumblesLastMinute;
  final bool presenterDriving;

  /// `CPU`, `GPU` or `NPU`: what the narrator is running on.
  final String? backend;

  /// The narrator's speed as a share of its own speed with the scene idle.
  final double? speedShare;

  /// Share of the frame budget the scene is using (1.0 = all of it).
  final double? frameBudget;

  /// Chip utilisation 0..1 where the platform reports it (Windows sidecar).
  final double? cpuLoad;
  final double? gpuLoad;
  final double? npuLoad;

  static const maxLines = 6;

  List<String> lines() {
    final out = <String>[];
    if (ahead.isNotEmpty) {
      final parts = [for (final a in ahead.take(2)) '${_article(a.label)} in ${a.metres.round()} m'];
      out.add('Ahead: ${parts.join(', then ')}.');
    }
    final river = riverIn;
    if (river != null) out.add('A river crosses the trail in ${river.round()} m.');
    if (rise > 1.5) {
      out.add('The trail climbs ahead.');
    } else if (rise < -1.5) {
      out.add('The trail drops away ahead.');
    }
    final biome = nextBiome, biomeIn = nextBiomeIn;
    if (biome != null && biomeIn != null) out.add('The $biome begins in ${biomeIn.round()} m.');
    if (foreshadow) out.add('Dash looks a little too relaxed about what is coming.');
    if (presenterDriving) out.add('The presenter is steering Dash by hand.');

    final form = <String>[];
    if (streak >= 3) form.add('$streak clean in a row (best $bestStreak)');
    if (stumblesLastMinute >= 2) form.add('$stumblesLastMinute stumbles in the last minute');
    if (form.isNotEmpty) out.add('Form: ${form.join('; ')}.');

    final b = backend;
    if (b != null) {
      final share = speedShare;
      final budget = frameBudget;
      final you = share == null ? 'You are running on the $b' : 'You are running on the $b at ${_pct(share)} of your own speed';
      out.add(budget == null ? '$you.' : '$you; the scene is using ${_pct(budget)} of its frame budget.');
      final loads = [
        if (cpuLoad != null) 'CPU ${_pct(cpuLoad!)}',
        if (gpuLoad != null) 'GPU ${_pct(gpuLoad!)}',
        if (npuLoad != null) 'NPU ${_pct(npuLoad!)}',
      ];
      if (loads.isNotEmpty) out.add('Chip load: ${loads.join(', ')}.');
    }
    // The hardware line goes last and survives the cut: it is the act's
    // point. Trail detail is what gives way.
    if (out.length > maxLines) {
      final keep = out.sublist(out.length - 2);
      return [...out.take(maxLines - 2), ...keep];
    }
    return out;
  }

  String get asText {
    final ls = lines();
    if (ls.isEmpty) return '';
    return 'Situation:\n${ls.map((l) => '- $l').join('\n')}';
  }

  static String _pct(double x) => '${(x * 100).round()}%';

  static String _article(String label) {
    final vowel = label.isNotEmpty && 'aeiou'.contains(label[0].toLowerCase());
    return '${vowel ? 'an' : 'a'} $label';
  }
}
