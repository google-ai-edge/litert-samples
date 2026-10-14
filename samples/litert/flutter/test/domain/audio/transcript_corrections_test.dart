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

import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/domain/audio/transcript_corrections.dart';

void main() {
  group('Whisper\'s measured misses (the spoken golden set)', () {
    test('on_01: "what is littered" and "tents or flow light"', () {
      final (:text, :corrections) = correctTranscript(
        'What is littered? And how is it related to tents or flow light?',
      );
      expect(text, 'What is LiteRT? And how is it related to TensorFlow Lite?');
      expect(corrections.map((c) => c.meant), [
        'What is LiteRT',
        'TensorFlow Lite',
      ]);
    });

    test('on_04: "runs on littered"', () {
      expect(
        correctTranscript(
          'How do I convert a PyTrick model so it runs on littered?',
        ).text,
        'How do I convert a PyTrick model so it runs on LiteRT?',
      );
    });

    test('on_06: "littered and file"', () {
      final (:text, :corrections) = correctTranscript(
        "How can I look inside a littered and file to see what's in it?",
      );
      expect(
        text,
        "How can I look inside a LiteRT-LM file to see what's in it?",
      );
      expect(corrections, hasLength(1));
      expect(corrections.single.heard, 'littered and file');
    });
  });

  test('"littered" on its own is the English word: ordinary speech stays as '
      'it was heard', () {
    for (final said in [
      'The beach was littered with plastic.',
      'Littered streets are a problem here.',
      'The floor was littered.',
      'Is the park littered and filthy again?',
    ]) {
      expect(correctTranscript(said).text, same(said), reason: said);
    }
    final (:text, :corrections) = correctTranscript(
      'The beach was littered with plastic.',
    );
    expect(text, 'The beach was littered with plastic.');
    expect(corrections, isEmpty);
  });

  test('whole words only, ignoring case, a leading capital kept; the words '
      'around are kept', () {
    expect(correctTranscript('so what is littered').text, 'so what is LiteRT');
    expect(correctTranscript('WHAT IS LITTERED').text, 'What is LiteRT');
    expect(correctTranscript('LITTERED and FILE').text, 'LiteRT-LM file');
    expect(correctTranscript('tents  or   flow light').text, 'TensorFlow Lite');
    expect(correctTranscript('Tents or flow light').text, 'TensorFlow Lite');
    // Inside another word: not a miss.
    expect(
      correctTranscript('what is litteredness').text,
      'what is litteredness',
    );
    expect(correctTranscript('contents or flow').text, 'contents or flow');
    expect(correctTranscript('reruns on littered').text, 'reruns on littered');
  });

  test('a transcript without a known miss is returned as it is', () {
    const said = 'What is the capital of France?';
    final (:text, :corrections) = correctTranscript(said);
    expect(text, same(said));
    expect(corrections, isEmpty);
    expect(correctTranscript('').text, '');
  });

  test('a table of its own (tests, other recognizers)', () {
    final (:text, :corrections) = correctTranscript(
      'the cop is on the table',
      corrections: const {'cop': 'cup'},
    );
    expect(text, 'the cup is on the table');
    expect(corrections.single.toString(), '"cop"→"cup"');
    expect(
      correctTranscript('what is littered', corrections: const {}).text,
      'what is littered',
    );
  });

  test('correctVoiceTranscript is the voice chat\'s table, phrases only', () {
    expect(correctVoiceTranscript('What is littered?'), 'What is LiteRT?');
    expect(
      correctVoiceTranscript('The beach was littered with plastic.'),
      'The beach was littered with plastic.',
    );
    expect(kVoiceTranscriptCorrections.keys, contains('littered and file'));
    expect(
      kVoiceTranscriptCorrections.keys,
      everyElement(contains(' ')),
      reason: 'no single word is corrected unconditionally',
    );
  });
}
