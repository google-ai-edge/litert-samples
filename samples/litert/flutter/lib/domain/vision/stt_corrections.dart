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

import 'coco_vocabulary.dart';

/// A word the recognizer heard instead of a COCO object.
final class const SttCorrection(final String heard, final String meant) {
  @override
  String toString() => '$heard→$meant';
}

/// The transcript after [correctSttHomophones], and what was changed.
final class const CorrectedTranscript(
  final String text,
  final List<SttCorrection> corrections,
);

/// moonshine-tiny's known misses on Demo 3 questions (router golden, live
/// run): the heard word, which is no COCO class, → the object meant. Only
/// exact words: "culture" stays (a real question word), "cultures" is the
/// miss. "cop" is a real word too, but the detector has no such class and
/// cups are the demo's props.
const kSttHomophones = <String, String>{
  'cop': 'cup',
  'cops': 'cups',
  'cultures': 'couches',
  'dob': 'dog',
  'dobs': 'dogs',
};

/// Replaces each whole word of [transcript] that [homophones] maps, when
/// the heard word is not something the detector knows and the meant one
/// is (a real COCO word is never rewritten). Case of the first letter is
/// kept; punctuation and spacing are untouched.
CorrectedTranscript correctSttHomophones(
  String transcript, {
  CocoVocabulary vocabulary = kCocoVocabulary,
  Map<String, String> homophones = kSttHomophones,
}) {
  final corrections = <SttCorrection>[];
  final text = transcript.replaceAllMapped(RegExp(r"[A-Za-z']+"), (m) {
    final word = m[0]!;
    final lower = word.toLowerCase();
    final meant = homophones[lower];
    if (meant == null ||
        vocabulary.resolveNoun(lower) != null ||
        vocabulary.resolveNoun(meant) == null) {
      return word;
    }
    corrections.add(SttCorrection(lower, meant));
    final upper = word[0] != word[0].toLowerCase();
    return upper ? '${meant[0].toUpperCase()}${meant.substring(1)}' : meant;
  });
  return CorrectedTranscript(text, List.unmodifiable(corrections));
}
