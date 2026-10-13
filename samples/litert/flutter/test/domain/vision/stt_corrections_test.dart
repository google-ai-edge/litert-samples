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
import 'package:litert_edge_demos/domain/vision/stt_corrections.dart';

/// moonshine heard "Is there a cup?" as "Is there a cop?" in a live
/// run; the router sent it to Gemma, which answered "No cop visible." Known
/// misses from the router golden: cop→cup, cultures→couches, dobs→dogs.
void main() {
  test('the known misses become the COCO object, case kept', () {
    for (final (heard, meant) in const [
      ('Is there a cop?', 'Is there a cup?'),
      ('How many cops do you see?', 'How many cups do you see?'),
      ('How many cultures are there?', 'How many couches are there?'),
      ('Are there any dobs?', 'Are there any dogs?'),
      ('Cop on the table?', 'Cup on the table?'),
    ]) {
      final corrected = correctSttHomophones(heard);
      expect(corrected.text, meant, reason: heard);
      expect(corrected.corrections, hasLength(1), reason: heard);
    }
    expect(
      correctSttHomophones('Is there a cop?').corrections.single.heard,
      'cop',
    );
  });

  test('real COCO words, other words and partial matches stay', () {
    for (final text in const [
      'Is there a cup?',
      'How many dogs do you see?',
      'Is there a couch?',
      'Describe the culture of this city.',
      'What does the sign say?',
      'Can you cope with this?',
      'Is there a copper pot?',
    ]) {
      expect(correctSttHomophones(text).text, text, reason: text);
    }
  });

  test('a map entry whose heard word is itself a COCO class is ignored', () {
    final corrected = correctSttHomophones(
      'Is there a cat?',
      homophones: const {'cat': 'dog'},
    );
    expect(corrected.text, 'Is there a cat?');
    expect(corrected.corrections, isEmpty);
  });
}
