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

import 'knowledge.dart';
import 'skill_step.dart';

enum ChatRole {
  user,
  assistant,
  error,

  /// Not a message: "Didn't catch that" after a capture with no LLM call.
  notice,
}

/// A user entry's question that went to the chat model as audio (audio
/// mode): [turn] finds the entry when its transcript arrives after the reply.
final class const SpokenQuestion({
  required final int turn,

  /// The recording's length.
  required final Duration duration,

  /// The words, once the transcript job has them.
  final String? transcript,

  /// Why there is no transcript (the recognizer failed or heard nothing).
  final String? failure,

  /// A barge-in replaced the turn: its recording was dropped, so no
  /// transcript will come.
  final bool interrupted = false,
}) {
  /// `🎤 3.2 s`: the entry's text until the words arrive.
  String get label =>
      '🎤 ${(duration.inMilliseconds / 1000).toStringAsFixed(1)} s';

  /// What the entry says: the words, the label with why there are none
  /// (`🎤 3.2 s (interrupted)` after a barge-in), or the label alone while
  /// the transcript is pending.
  String get text => switch ((transcript, failure)) {
    (final String words, _) => words,
    (null, _) when interrupted => '$label (interrupted)',
    (null, final String why) => '$label (no transcript: $why)',
    (null, null) => label,
  };
}

/// One committed line of the conversation. The reply that is still streaming
/// is not an entry; it lives in the view model's `partialReply` notifier.
final class const ChatEntry({
  required final ChatRole role,
  required final String text,

  /// An assistant reply cut short by Stop or a barge-in.
  final bool interrupted = false,

  /// A user entry's attached picture (the PNG sent to the model), shown as a
  /// thumbnail. The same object as the attachment: no copy per entry.
  final Uint8List? image,

  /// An assistant reply's knowledge-base retrieval and citations.
  final ReplyKnowledge? knowledge,

  /// The skill steps behind a reply (or a failed turn), in order.
  final List<SkillStep> steps = const [],

  /// A user entry asked by voice and sent as audio; its [text] follows
  /// [SpokenQuestion.text].
  final SpokenQuestion? spoken,
});
