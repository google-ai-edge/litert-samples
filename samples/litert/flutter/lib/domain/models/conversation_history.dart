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

/// What the text history of the conversation says for a spoken question
/// sent as audio, until its transcript arrives (the transcript job runs
/// after the reply has played).
const kSpokenQuestionPlaceholder = '[voice question]';

/// One exchange of the conversation's text mirror: what the user asked, as
/// text, and what the assistant answered. A rebuilt chat replays these
/// instead of the engine's own history, which has no words for a question
/// that was only audio.
final class const HistoryTurn({
  /// The user's words: the typed or transcribed question, or
  /// [kSpokenQuestionPlaceholder] for an audio turn whose transcript has not
  /// arrived. Never the prompt's knowledge-base excerpts.
  required final String user,

  /// The reply as far as it streamed (an interrupted reply is partial).
  required final String assistant,

  /// The voice turn's number (`TurnRequest.turn`, never reused in the
  /// process, so turns of different visits of a demo on one chat never share
  /// it); the transcript finds its entry by it. Null for a turn asked
  /// without one.
  final int? turn,

  /// The question went to the model as audio.
  final bool audio = false,
}) {
  /// A turn worth replaying: something was answered. A turn stopped or
  /// failed before any text adds nothing but an unanswered line.
  bool get replayable => assistant.trim().isNotEmpty && user.trim().isNotEmpty;

  /// A spoken question whose transcript has not arrived: a replay gives the
  /// model only [kSpokenQuestionPlaceholder] for it.
  bool get wordless => audio && user == kSpokenQuestionPlaceholder;

  @override
  String toString() =>
      'HistoryTurn(#$turn${audio ? ' audio' : ''}: "$user" -> '
      '${assistant.length} chars)';
}
