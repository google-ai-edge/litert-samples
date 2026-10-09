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

// Not a port: its consumer (VoiceAssistant) and both implementers (the chat
// and camera turn responders) are use cases, so it lives next to them.

import 'package:flutter/foundation.dart';
import 'package:flutter_edge_ai_speech/flutter_edge_ai_speech.dart'
    show VoiceResponder;

import '../../utils/waits.dart';

/// Builds the LLM step of each voice turn. Generic over
/// the demo's side events [S] (Demo 1: chat metrics, later citations and
/// skill steps; Demo 3: snapshot and route), so `VoiceAssistant` holds no
/// demo types.
///
/// Two phases, because some work must start at the moment of release (Demo 3
/// snapshots the frame the user was looking at) while the capture is
/// still closing and before anyone knows whether the turn will run.
abstract interface class TurnResponderFactory<S> {
  /// Called once per turn at release — mic up, Send, or a submitted
  /// utterance — before the capture closes, before the silence gate and
  /// before STT, so whatever it starts runs in parallel with them. Must not
  /// block.
  TurnPreparation<S> prepare(TurnRequest request);
}

/// What the caller hands one turn at release. Per turn, never state on the
/// factory: a late turn cannot pick up the next one's attachment.
final class const TurnRequest({
  /// A typed turn (no STT).
  required final bool typed,

  /// An image attached to this turn (Demo 1): JPEG/PNG bytes for the
  /// LLM. Demo 3 takes its own snapshot and ignores it.
  final Uint8List? image,
});

/// One turn's prepared LLM step. Exactly one of [responder] (the turn runs)
/// or [discard] (it does not) is called.
abstract interface class TurnPreparation<S> {
  /// The responder for this turn's `VoiceSession`, built when the turn
  /// starts. Its `stop` must end its `respond` stream and must only stop
  /// what this turn started. [onSide] delivers facts the reply text does not
  /// carry.
  VoiceResponder responder(void Function(S event) onSide);

  /// The turn will not run (not heard, no mic audio, superseded, or it could
  /// not start): drop whatever [TurnResponderFactory.prepare] started.
  void discard();
}

/// Waits, up to [wait], for [isGenerating] to turn false before a responder
/// asks: after a barge-in whose drain was forced, the previous
/// generation can still run, and asking now would fail as busy. Past [wait]
/// it logs (as [tag]) and returns; the ask then fails visibly.
Future<void> whenPreviousReplyEnded(
  ValueListenable<bool> isGenerating, {
  required Duration wait,
  required String tag,
}) async {
  if (await whenFalse(isGenerating, timeout: wait) == WaitEnd.timedOut) {
    debugPrint(
      '[$tag] the previous reply still runs after ${wait.inSeconds}s; '
      'asking anyway',
    );
  }
}
