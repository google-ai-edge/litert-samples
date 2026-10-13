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

/// Numbers as words for skill results the model repeats aloud: copying words
/// is safer than copying digits on the fp16 GPU path, and TTS reads them as
/// written.
library;

/// [n] in words: `0` → "zero", `21` → "twenty-one", `105` → "one hundred
/// five", `2300` → "two thousand three hundred". Negative numbers get
/// "minus"; values of a million and more are written with digits.
String numberToWords(int n) {
  if (n < 0) return 'minus ${numberToWords(-n)}';
  if (n >= 1000000) return '$n';
  if (n < 20) return _ones[n];
  if (n < 100) {
    final tens = _tens[n ~/ 10];
    return n % 10 == 0 ? tens : '$tens-${_ones[n % 10]}';
  }
  if (n < 1000) {
    final rest = n % 100;
    final hundreds = '${_ones[n ~/ 100]} hundred';
    return rest == 0 ? hundreds : '$hundreds ${numberToWords(rest)}';
  }
  final rest = n % 1000;
  final thousands = '${numberToWords(n ~/ 1000)} thousand';
  return rest == 0 ? thousands : '$thousands ${numberToWords(rest)}';
}

/// [items] joined as English prose: "a", "a and b", "a, b and c".
String joinWords(List<String> items) => switch (items.length) {
  0 => '',
  1 => items.single,
  _ => '${items.sublist(0, items.length - 1).join(', ')} and ${items.last}',
};

const _ones = [
  'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', //
  'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen',
  'sixteen', 'seventeen', 'eighteen', 'nineteen',
];

const _tens = [
  '', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', //
  'eighty', 'ninety',
];
