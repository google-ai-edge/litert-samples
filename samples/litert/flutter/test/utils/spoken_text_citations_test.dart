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
import 'package:litert_edge_demos/utils/spoken_text.dart';

/// What a knowledge-base reply adds to the TTS sanitizer.
void main() {
  test('citation ranges and runs are never read aloud', () {
    expect(toSpokenText('All three agree [1-3].'), 'All three agree.');
    expect(toSpokenText('All three agree [1–3].'), 'All three agree.');
    expect(toSpokenText('Two agree [1, 2].'), 'Two agree.');
    expect(toSpokenText('Two agree [1][2].'), 'Two agree.');
    expect(toSpokenText('Mixed [1, 2-3] list.'), 'Mixed list.');
    expect(toSpokenText('[2]'), isEmpty);
  });

  test('an echoed excerpt label reads its › as a comma', () {
    expect(
      toSpokenText('LiteRT overview › What LiteRT is says so [1].'),
      'LiteRT overview, What LiteRT is says so.',
    );
  });

  test('square brackets that are not citations stay', () {
    expect(toSpokenText('An array [a, b] here.'), 'An array [a, b] here.');
  });

  group('tensor shapes are not citations', () {
    test('a bracket group with a number outside 1..3 is read, not removed', () {
      expect(
        toSpokenText(
          'It takes a float32 tensor of shape [1, 3, 640, 640] [1].',
        ),
        'It takes a float32 tensor of shape [1, 3, 640, 640].',
      );
      expect(
        toSpokenText('The output is [1, 300, 6] boxes.'),
        'The output is [1, 300, 6] boxes.',
      );
      expect(
        toSpokenText('Whisper takes [1, 80, 3000] features.'),
        'Whisper takes [1, 80, 3000] features.',
      );
    });

    test('a marker inside backticks is code, not a citation', () {
      expect(
        toSpokenText('Index it with `[1]` here [2].'),
        'Index it with [1] here.',
      );
      expect(
        toSpokenText('The shape is `[1, 3, 640, 640]` [1].'),
        'The shape is [1, 3, 640, 640].',
      );
    });

    test('real citations still go, with the comma between them', () {
      expect(toSpokenText('Both say so [1], [2].'), 'Both say so.');
      expect(toSpokenText('Both say so [1, 2].'), 'Both say so.');
    });

    test('maxCitation sets the range a marker must stay in', () {
      expect(toSpokenText('See [4].'), 'See [4].');
      expect(toSpokenText('See [4].', maxCitation: 5), 'See.');
    });
  });
}
