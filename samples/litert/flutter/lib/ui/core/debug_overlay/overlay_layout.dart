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

import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart' show kToolbarHeight;
import 'package:flutter/rendering.dart';

/// The panel's widest width, in logical pixels.
const double kDebugOverlayMaxWidth = 440;

/// Below this width (Material's compact/medium breakpoint) or this height
/// (Material's compact height: a phone in landscape) the panel starts
/// collapsed to the summary (`debugOverlaySummaryLines`) and uses smaller
/// text.
const double kDebugOverlayCompactWidth = 600;
const double kDebugOverlayCompactHeight = 480;

/// From the top of the safe area: clear of the app bar (and its toggle).
const double _kTop = kToolbarHeight + 4;

/// From the right edge of the safe area, and the least gap on the left.
const double _kSide = 8;

/// Kept free at the bottom for the demos' controls: the phase line, the
/// attachment bar and the composer with the mic (Demo 1), the voice bar
/// (Demo 3).
const double _kBottomFree = 160;

/// The panel's height bound never goes below this (a few lines), unless the
/// window has no room for it at all.
const double _kMinHeight = 72;

/// Below this the panel is not shown: room for a line and the handle.
const Size _kSmallestPanel = Size(160, 56);

/// The panel's share of the safe area's height.
const double _kCompactShare = 0.45;
const double _kWideShare = 0.75;

/// Where the diagnostics panel goes and how large it may be. Pure: the host
/// measures the window and the room inside its safe area above the keyboard,
/// this decides. The panel hangs from the top right corner of that room,
/// [margin] in from it.
@immutable
final class DebugOverlayLayout {
  /// The layout in a [window] (the whole view, which decides [compact])
  /// whose safe area above the keyboard is [safe] (which bounds the panel).
  factory DebugOverlayLayout({required Size window, required Size safe}) {
    final compact =
        window.width < kDebugOverlayCompactWidth ||
        window.height < kDebugOverlayCompactHeight;
    final width = math.max(
      0.0,
      math.min(safe.width - 2 * _kSide, kDebugOverlayMaxWidth),
    );
    final under = math.max(0.0, safe.height - _kTop - _kSide);
    final share = math.min(
      (compact ? _kCompactShare : _kWideShare) * safe.height,
      safe.height - _kTop - _kBottomFree,
    );
    return DebugOverlayLayout._(
      compact: compact,
      width: width,
      maxHeight: math.min(under, math.max(_kMinHeight, share)),
    );
  }

  const DebugOverlayLayout._({
    required this.compact,
    required this.width,
    required this.maxHeight,
  });

  /// The panel's gap to the top of the safe area (clear of the app bar and
  /// its toggle) and to its right edge.
  static const EdgeInsets margin = EdgeInsets.only(top: _kTop, right: _kSide);

  /// A phone-sized window (narrower than [kDebugOverlayCompactWidth] or
  /// shorter than [kDebugOverlayCompactHeight]): the panel starts collapsed
  /// and uses smaller text.
  final bool compact;

  /// `min(safe width - 16, 440)`, never negative.
  final double width;

  /// The panel's height bound: its share of the safe height (45% compact,
  /// 75% wide) with 160 left free under it for the demos' controls, at least
  /// 72, and never past the safe area's bottom (8 from it).
  final double maxHeight;

  /// Room for a line and the handle. A window too short or narrow (a short
  /// macOS window, a split screen) gets no panel.
  bool get showsPanel =>
      maxHeight >= _kSmallestPanel.height && width >= _kSmallestPanel.width;

  /// Exactly [width] wide, at most [maxHeight] tall.
  BoxConstraints get constraints =>
      BoxConstraints(minWidth: width, maxWidth: width, maxHeight: maxHeight);
}
