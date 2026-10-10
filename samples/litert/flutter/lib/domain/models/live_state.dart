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

/// Whether the detector may take frames. The GPU arbiter pauses it while
/// Gemma generates.
enum DetectorDuty { live, paused }

/// Lifecycle of the live detection pipeline (source → worker → boxes).
sealed class const LiveState();

/// No source is running.
final class const LiveStopped() extends LiveState;

/// The source is starting (fixture images decoding, camera opening).
final class const LiveStarting() extends LiveState;

/// Frames flow to the detector. [source] is the source's label.
final class const LiveRunning(final String source) extends LiveState;

/// The source runs but the gate is closed: no frame goes to the detector.
final class const LivePaused({
  required final String source,
  required final String reason,
  required final DateTime since,
}) extends LiveState;

/// Stopped by an error the UI must show (the source or the detector failed).
final class const LiveFailed(final String message) extends LiveState;

/// Rolling live-detection figures for the debug overlay, published at most
/// every `kLiveStatsInterval`.
final class const LiveStats({
  /// Processed frames per second over the last `kLiveStatsWindow` frames.
  final double fps = 0,

  /// Frames the detector returned since start.
  final int processed = 0,

  /// Frames the source delivered since start.
  final int sourceFrames = 0,

  /// Frames per second the source delivered over the last
  /// `kLiveStatsWindow` frames (before the gate), and the newest frame's
  /// size; 0 before the first frame.
  final double sourceFps = 0,
  final int sourceWidth = 0,
  final int sourceHeight = 0,

  /// Frames the gate dropped since start, by reason.
  final int droppedBusy = 0,
  final int droppedPaused = 0,
  final int droppedRate = 0,

  /// Medians over the window, in milliseconds: main-isolate copy, worker
  /// gather, model run, decode, and send → result on the main isolate.
  final double? copyMs,
  final double? preMs,
  final double? runMs,
  final double? postMs,
  final double? latencyMs,
});
