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

import '../../../config/live_camera_config.dart';
import '../../services/frames/frame_source.dart';

/// The black-frames warning turned on or off.
sealed class const BlackFrameChange();

/// The sampled frames have been dark and flat for [darkMicros]; the newest
/// sample's mean [luma] and its [spread] (0–255).
final class const BlackFramesStarted({
  required final double luma,
  required final double spread,
  required final int darkMicros,
}) extends BlackFrameChange;

/// A bright or textured sample (mean [luma]) ended the warning.
final class const BlackFramesEnded(final double luma) extends BlackFrameChange;

/// The black-frames warning (a covered lens, a camera in a dark box, or
/// macOS zeroing the frames because camera access was attributed to the
/// terminal): a warning, not a failure.
///
/// The luma of every `sampleEvery`-th frame sent to the detector is
/// sampled ([lumaStats]). Samples that stay dark (mean below `darkLuma`)
/// and flat (spread below `flatSpread`) for `after` turn the warning on;
/// any brighter or textured sample turns it off. The first frames are often
/// dark while auto-exposure settles, so a single dark sample proves nothing.
///
/// Pure: each frame carries the caller's monotonic clock time in µs, and
/// the caller shows the [BlackFrameChange]s it hands back.
final class BlackFrameDetector {
  BlackFrameDetector({
    this._sampleEvery = kLumaSampleEvery,
    this._darkLuma = kBlackFrameLuma,
    this._flatSpread = kBlackFrameSpread,
    Duration after = kBlackFramesAfter,
  }) : _afterMicros = after.inMicroseconds;

  final int _sampleEvery;
  final double _darkLuma;
  final double _flatSpread;
  final int _afterMicros;

  int _sentFrames = 0;
  int? _darkSince;
  double? _luma;
  bool _black = false;

  /// The newest sampled mean luma (0–255); null before the first sample.
  double? get luma => _luma;

  /// Whether the warning is on.
  bool get black => _black;

  /// A frame sent to the detector at [nowMicros]. [view] must still be
  /// valid (inside the source's callback). The first frame and every
  /// `sampleEvery`-th after it are sampled.
  BlackFrameChange? frameSent(FrameView view, int nowMicros) {
    if (_sentFrames++ % _sampleEvery != 0) return null;
    final stats = lumaStats(view);
    return sample(mean: stats.mean, spread: stats.spread, nowMicros: nowMicros);
  }

  /// One luma sample at [nowMicros]: its [mean] and [spread] (0–255).
  BlackFrameChange? sample({
    required double mean,
    required double spread,
    required int nowMicros,
  }) {
    _luma = mean;
    if (mean >= _darkLuma || spread >= _flatSpread) {
      _darkSince = null;
      if (!_black) return null;
      _black = false;
      return BlackFramesEnded(mean);
    }
    final since = _darkSince ??= nowMicros;
    if (_black || nowMicros - since < _afterMicros) return null;
    _black = true;
    return BlackFramesStarted(
      luma: mean,
      spread: spread,
      darkMicros: nowMicros - since,
    );
  }

  /// Back to the start (a stopped source): no samples, no warning.
  void reset() {
    _sentFrames = 0;
    _darkSince = null;
    _luma = null;
    _black = false;
  }
}
