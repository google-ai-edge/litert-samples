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

import '../vision/stt_corrections.dart';
import 'assistant_event.dart';
import 'route_decision.dart';
import 'scene_snapshot.dart';

/// Demo 3's side channel of a voice turn.
sealed class const CameraSideEvent();

/// The frame the question is about was captured and detected.
final class const SnapshotTaken(final SceneSnapshot snapshot)
    extends CameraSideEvent;

/// No frame could be captured (the camera is not running, or no frame came);
/// the turn speaks "The camera isn't running" and the UI shows [message].
final class const SnapshotFailed(final String message) extends CameraSideEvent;

/// Words the recognizer misheard were replaced before routing
/// ("cop" → "cup"); the caption shows what was heard.
final class const TranscriptCorrected(
  final String text,
  final List<SttCorrection> corrections,
) extends CameraSideEvent;

/// The router's decision and how long it took.
final class const RouteChosen(final RouteDecision route, final Duration elapsed)
    extends CameraSideEvent;

/// A detailed question could not go to Gemma: the camera chat is not open
/// (still opening, or a reset failed). The turn speaks a short notice and
/// the UI shows [message].
final class const CameraChatUnavailable(final String message)
    extends CameraSideEvent;

/// The snapshot went to Gemma as [image]: its frame id is
/// the frozen frame's.
final class const FrameSentToGemma(final EncodedSnapshot image)
    extends CameraSideEvent;

/// Gemma's detailed answer ended (normally or stopped).
final class const DetailedAnswered(final GenerationMetrics metrics)
    extends CameraSideEvent;

/// The post-turn chat reset finished, off the critical path.
/// [error]: it failed; the next detailed question then says the chat isn't
/// ready.
final class const CameraChatReset({
  required final Duration elapsed,
  final String? error,
}) extends CameraSideEvent;

/// A fast answer: what is spoken and what it was based on.
final class const FastAnswered({
  required final String answer,

  /// "cat ×2 · remote".
  required final String basis,
}) extends CameraSideEvent;

/// A detailed question while the chat model has images off: the frame is not
/// sent; [answer] says what the detector sees instead ([basis]:
/// "cat ×2 · remote"), and [reason] why detailed answers are off.
final class const DetailedUnavailable({
  required final String reason,
  required final String answer,
  required final String basis,
}) extends CameraSideEvent;

/// One camera question for the overlay: where it went and what it cost.
final class const CameraTurnMetrics({
  required final RouteDecision route,

  /// The router's time.
  required final Duration routeTime,

  /// Release to the snapshot's detection result.
  final Duration? snapshotLatency,

  /// The fast answer's basis ("cat ×2 · remote").
  final String? basis,

  /// Why no frame could be captured; null when one was.
  final String? snapshotError,

  /// Detailed turns: the PNG sent to Gemma (frame id, size, encode time),
  /// and Gemma's time to first token.
  final EncodedSnapshot? image,
  final Duration? timeToFirstToken,

  /// Detailed turns: why Gemma was not asked (the camera chat not ready).
  final String? chatError,
});
