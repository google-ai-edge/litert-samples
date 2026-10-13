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

void main() {
  test('citations are never read aloud', () {
    expect(
      toSpokenText('LiteRT runs on the GPU [1]. '),
      'LiteRT runs on the GPU.',
    );
    expect(toSpokenText('See both [1, 2].'), 'See both.');
  });

  test('markdown syntax goes, the words stay', () {
    expect(
      toSpokenText('**Paris** is the *capital*.'),
      'Paris is the capital.',
    );
    expect(
      toSpokenText('## Summary\n- one item\n> quoted'),
      'Summary one item quoted',
    );
    expect(toSpokenText('Use `flutter run` now.'), 'Use flutter run now.');
  });

  test('links keep their text, bare URLs are dropped', () {
    expect(
      toSpokenText('Read [the docs](https://ai.google.dev/edge) today.'),
      'Read the docs today.',
    );
    expect(
      toSpokenText('Visit https://example.com for more.'),
      'Visit for more.',
    );
  });

  test('a clause with nothing speakable left is empty', () {
    expect(toSpokenText(' [3] '), isEmpty);
    expect(toSpokenText('**'), isEmpty);
  });

  test('plain text is unchanged apart from whitespace', () {
    expect(toSpokenText('  Hello there.\n'), 'Hello there.');
    expect(toSpokenText('3.14 is pi.'), '3.14 is pi.');
  });
}
