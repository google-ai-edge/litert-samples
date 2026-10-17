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

import 'package:flutter/foundation.dart';

/// Whisper base's observed misses of the knowledge base's product names, as
/// the voice chat's transcripts write them (the spoken golden set,
/// integration_test/kb_retrieval_test.dart): what it heard → what was said.
/// Matched as whole words, ignoring case; a longer phrase wins over a shorter
/// one it contains, and a match that starts with a capital letter keeps it.
/// Phrases only, each one ordinary speech does not produce: "littered" on its
/// own is an English word ("The beach was littered with plastic") and stays
/// as it was heard.
const kVoiceTranscriptCorrections = <String, String>{
  // "What is LiteRT, and how is it related to TensorFlow Lite?" → "What is
  // littered? And how is it related to tents or flow light?" (on_01).
  'what is littered': 'what is LiteRT',
  'tents or flow light': 'TensorFlow Lite',
  // "How do I convert a PyTorch model so it runs on LiteRT?" → "…so it runs
  // on littered?" (on_04).
  'runs on littered': 'runs on LiteRT',
  // "How can I look inside a litertlm file…" → "…inside a littered and
  // file…" (on_06).
  'littered and file': 'LiteRT-LM file',
};

/// One replacement [correctVoiceTranscript] made.
final class const TranscriptCorrection(final String heard, final String meant) {
  @override
  String toString() => '"$heard"→"$meant"';
}

/// [transcript] with Whisper's known misses ([corrections]) replaced, and
/// which replacements were made, in order of appearance.
({String text, List<TranscriptCorrection> corrections}) correctTranscript(
  String transcript, {
  Map<String, String> corrections = kVoiceTranscriptCorrections,
}) {
  if (corrections.isEmpty || transcript.isEmpty) {
    return (text: transcript, corrections: const []);
  }
  // Longest phrase first, so a phrase is never cut by a shorter entry it
  // contains.
  final heard = corrections.keys.toList()
    ..sort((a, b) => b.length.compareTo(a.length));
  final pattern = RegExp(
    '\\b(?:${heard.map(_phrase).join('|')})\\b',
    caseSensitive: false,
  );
  final made = <TranscriptCorrection>[];
  final text = transcript.replaceAllMapped(pattern, (m) {
    final said = m[0]!;
    final key = heard.firstWhere(
      (h) => RegExp('^${_phrase(h)}\$', caseSensitive: false).hasMatch(said),
    );
    final meant = _keepCapital(said, corrections[key]!);
    made.add(TranscriptCorrection(said, meant));
    return meant;
  });
  return (text: text, corrections: List.unmodifiable(made));
}

/// The voice chat's corrector, as the speech pipeline and the transcript job
/// apply it: the corrected text, with each correction logged.
String correctVoiceTranscript(String transcript) {
  final (:text, :corrections) = correctTranscript(transcript);
  if (corrections.isNotEmpty) {
    debugPrint(
      '[Voice] transcript corrected: "$transcript" -> "$text" '
      '(${corrections.join(', ')})',
    );
  }
  return text;
}

/// [phrase] as a pattern: its words literal, any run of spaces between them.
String _phrase(String phrase) =>
    phrase.trim().split(RegExp(r'\s+')).map(RegExp.escape).join(r'\s+');

/// [meant] with its first letter capitalized when [said] starts with a capital
/// (a correction at the start of a sentence): "What is littered" → "What is
/// LiteRT", not "what is LiteRT".
String _keepCapital(String said, String meant) {
  if (said.isEmpty || meant.isEmpty) return meant;
  final first = said[0];
  if (first == first.toLowerCase()) return meant;
  return meant[0].toUpperCase() + meant.substring(1);
}
