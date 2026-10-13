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
import 'package:litert_edge_demos/utils/spoken_numbers.dart';

void main() {
  test('numbers as words', () {
    expect(numberToWords(0), 'zero');
    expect(numberToWords(7), 'seven');
    expect(numberToWords(17), 'seventeen');
    expect(numberToWords(20), 'twenty');
    expect(numberToWords(25), 'twenty-five');
    expect(numberToWords(90), 'ninety');
    expect(numberToWords(105), 'one hundred five');
    expect(numberToWords(850), 'eight hundred fifty');
    expect(numberToWords(2300), 'two thousand three hundred');
    expect(numberToWords(86400), 'eighty-six thousand four hundred');
    expect(numberToWords(-3), 'minus three');
  });

  test('joinWords', () {
    expect(joinWords(const []), '');
    expect(joinWords(const ['a']), 'a');
    expect(joinWords(const ['a', 'b']), 'a and b');
    expect(joinWords(const ['a', 'b', 'c']), 'a, b and c');
  });
}
