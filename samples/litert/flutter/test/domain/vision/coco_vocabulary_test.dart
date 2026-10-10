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
import 'package:litert_edge_demos/domain/vision/coco_vocabulary.dart';

int cls(String name) => kCocoNames.indexOf(name);

void main() {
  const v = kCocoVocabulary;

  test('names, plurals and determiners resolve', () {
    expect(v.resolveNoun('cat'), cls('cat'));
    expect(v.resolveNoun('cats'), cls('cat'));
    expect(v.resolveNoun('the cats'), cls('cat'));
    expect(v.resolveNoun('any buses'), cls('bus'));
    expect(v.resolveNoun('benches'), cls('bench'));
    expect(v.resolveNoun('cell phones'), cls('cell phone'));
    expect(v.resolveNoun('knives'), cls('knife'));
    expect(v.resolveNoun('mice'), cls('mouse'));
    expect(v.resolveNoun('sheep'), cls('sheep'));
    expect(v.resolveNoun('skis'), cls('skis'));
    expect(v.resolveNoun('remotes'), cls('remote'));
    expect(v.resolveNoun('oranges'), cls('orange'));
  });

  test('COCO aliases and everyday words', () {
    expect(v.resolveNoun('couch'), cls('sofa'));
    expect(v.resolveNoun('TVs'), cls('tvmonitor'));
    expect(v.resolveNoun('television'), cls('tvmonitor'));
    expect(v.resolveNoun('monitor'), cls('tvmonitor'));
    expect(v.resolveNoun('plants'), cls('pottedplant'));
    expect(v.resolveNoun('table'), cls('diningtable'));
    expect(v.resolveNoun('phone'), cls('cell phone'));
    expect(v.resolveNoun('smartphones'), cls('cell phone'));
    expect(v.resolveNoun('motorcycle'), cls('motorbike'));
    expect(v.resolveNoun('airplanes'), cls('aeroplane'));
    expect(v.resolveNoun('people'), cls('person'));
    expect(v.resolveNoun('anyone'), cls('person'));
    expect(v.resolveNoun('someone'), cls('person'));
    expect(v.resolveNoun('humans'), cls('person'));
    expect(v.resolveNoun('mugs'), cls('cup'));
  });

  test('a kind of thing is not its whole class', () {
    for (final noun in [
      'man',
      'men',
      'woman',
      'women',
      'child',
      'children',
      'kid',
      'kids',
      'boy',
      'girls',
      'guy',
      'adults',
      'kitten',
      'kitty',
      'puppies',
    ]) {
      expect(v.resolveNoun(noun), isNull, reason: noun);
    }
  });

  test('digits are kept, so a counted phrase is not a bare class', () {
    expect(v.resolveNoun('2 cats'), isNull);
    expect(v.resolveNoun('the 3 people'), isNull);
    expect(v.resolveNoun('1 dog'), isNull);
    expect(v.resolveNoun('cats'), cls('cat'));
  });

  test('an extra word or an ambiguous word is not a match', () {
    expect(v.resolveNoun('red cars'), isNull);
    expect(v.resolveNoun('my phone'), isNull);
    expect(v.resolveNoun('glasses'), isNull, reason: 'eyeglasses');
    expect(v.resolveNoun('bag'), isNull);
    expect(v.resolveNoun('unicorns'), isNull);
    expect(v.resolveNoun(''), isNull);
  });

  test('spoken names: plurals, irregulars, the Darknet spellings', () {
    expect(v.spokenName(cls('person'), 1), 'person');
    expect(v.spokenName(cls('person'), 2), 'people');
    expect(v.spokenName(cls('tvmonitor'), 1), 'TV');
    expect(v.spokenName(cls('tvmonitor'), 3), 'TVs');
    expect(v.spokenName(cls('pottedplant'), 2), 'potted plants');
    expect(v.spokenName(cls('diningtable'), 1), 'dining table');
    expect(v.spokenName(cls('aeroplane'), 2), 'airplanes');
    expect(v.spokenName(cls('cell phone'), 2), 'phones');
    expect(v.spokenName(cls('bus'), 2), 'buses');
    expect(v.spokenName(cls('bench'), 2), 'benches');
    expect(v.spokenName(cls('knife'), 2), 'knives');
    expect(v.spokenName(cls('sheep'), 4), 'sheep');
    expect(v.spokenName(cls('cat'), 2), 'cats');
    expect(v.counted(cls('cat'), 2), 'two cats');
    expect(v.counted(cls('skis'), 2), 'two pairs of skis');
    expect(v.counted(cls('chair'), 21), '21 chairs');
  });

  test('articles', () {
    expect(v.withArticle(cls('cup')), 'a cup');
    expect(v.withArticle(cls('apple')), 'an apple');
    expect(v.withArticle(cls('umbrella')), 'an umbrella');
    expect(v.withArticle(cls('aeroplane')), 'an airplane');
    expect(v.withArticle(cls('tvmonitor')), 'a TV');
    expect(v.withArticle(cls('giraffe')), 'a giraffe');
  });
}
