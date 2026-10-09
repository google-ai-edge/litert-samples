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

/// The detector's 80 class names, index = class id, in the **Darknet
/// spellings** of the model package's `assets/class_names.json`:
/// `motorbike, aeroplane, sofa, pottedplant, diningtable, tvmonitor`.
const kCocoNames = <String>[
  'person',
  'bicycle',
  'car',
  'motorbike',
  'aeroplane',
  'bus',
  'train',
  'truck',
  'boat',
  'traffic light',
  'fire hydrant',
  'stop sign',
  'parking meter',
  'bench',
  'bird',
  'cat',
  'dog',
  'horse',
  'sheep',
  'cow',
  'elephant',
  'bear',
  'zebra',
  'giraffe',
  'backpack',
  'umbrella',
  'handbag',
  'tie',
  'suitcase',
  'frisbee',
  'skis',
  'snowboard',
  'sports ball',
  'kite',
  'baseball bat',
  'baseball glove',
  'skateboard',
  'surfboard',
  'tennis racket',
  'bottle',
  'wine glass',
  'cup',
  'fork',
  'knife',
  'spoon',
  'bowl',
  'banana',
  'apple',
  'sandwich',
  'orange',
  'broccoli',
  'carrot',
  'hot dog',
  'pizza',
  'donut',
  'cake',
  'chair',
  'sofa',
  'pottedplant',
  'bed',
  'diningtable',
  'toilet',
  'tvmonitor',
  'laptop',
  'mouse',
  'remote',
  'keyboard',
  'cell phone',
  'microwave',
  'oven',
  'toaster',
  'sink',
  'refrigerator',
  'book',
  'clock',
  'vase',
  'scissors',
  'teddy bear',
  'hair drier',
  'toothbrush',
];

/// The display name of class [cls], or `#cls` for an id outside the list.
String cocoName(int cls) =>
    cls >= 0 && cls < kCocoNames.length ? kCocoNames[cls] : '#$cls';

/// The vocabulary the fast path understands and speaks: class names in
/// either spelling, the everyday names of the Darknet spellings plus common
/// everyday words, plurals, and how each class is said aloud.
const kCocoVocabulary = CocoVocabulary();

final class CocoVocabulary {
  const CocoVocabulary();

  /// The class a noun phrase names exactly ("cats", "cell phones", "a TV",
  /// "any people", "couch"), or null. Leading determiners are ignored; any
  /// other extra word ("red cars", "my phone", "2 cats") is not a match, so a
  /// question about a kind of thing — or about a number of them — never gets
  /// an answer about the whole class. Digits are kept for that reason: the
  /// router keeps them too, and the two must agree.
  int? resolveNoun(String phrase) {
    final words = phrase
        .toLowerCase()
        .replaceAll(RegExp(r"[^a-z0-9' ]"), ' ')
        .split(RegExp(r'\s+'))
        .where((w) => w.isNotEmpty)
        .toList();
    while (words.isNotEmpty && _determiners.contains(words.first)) {
      words.removeAt(0);
    }
    if (words.isEmpty || words.length > 3) return null;
    final head = words.sublist(0, words.length - 1);
    for (final last in _singulars(words.last)) {
      final key = [...head, last].join(' ');
      if (_index[key] case final id?) return id;
    }
    return null;
  }

  /// [cls] as said aloud for [count] of them, without the number:
  /// person → "person" / "people", tvmonitor → "TV" / "TVs".
  String spokenName(int cls, int count) {
    final (singular, plural) = _spoken(cls);
    return count == 1 ? singular : plural;
  }

  /// "a cup", "an apple", "a pair of skis".
  String withArticle(int cls) {
    final name = spokenName(cls, 1);
    final article = RegExp('^[aeiouAEIOU]').hasMatch(name) ? 'an' : 'a';
    return '$article $name';
  }

  /// "one cup", "two people", "21 chairs".
  String counted(int cls, int count) =>
      '${numberWord(count)} ${spokenName(cls, count)}';

  static (String, String) _spoken(int cls) {
    final name = cocoName(cls);
    if (_spokenOverrides[name] case final pair?) return pair;
    return (name, _pluralOf(name));
  }

  static String _pluralOf(String name) {
    if (RegExp(r'(s|sh|ch|x|z)$').hasMatch(name)) return '${name}es';
    if (RegExp(r'[^aeiou]y$').hasMatch(name)) {
      return '${name.substring(0, name.length - 1)}ies';
    }
    return '${name}s';
  }

  /// [word] and the singulars it may be the plural of, most likely first.
  static Iterable<String> _singulars(String word) sync* {
    yield word;
    if (_irregular[word] case final singular?) yield singular;
    if (word.endsWith('ies') && word.length > 4) {
      yield '${word.substring(0, word.length - 3)}y';
    }
    if (word.endsWith('ves')) {
      final stem = word.substring(0, word.length - 3);
      yield '${stem}fe';
      yield '${stem}f';
    }
    if (word.endsWith('es') && word.length > 3) {
      yield word.substring(0, word.length - 2);
    }
    if (word.endsWith('s') && word.length > 2) {
      yield word.substring(0, word.length - 1);
    }
  }

  static const _determiners = {
    'a',
    'an',
    'any',
    'the',
    'some',
    'of',
    'these',
    'those',
  };

  static const _irregular = {
    'people': 'person',
    'persons': 'person',
    'mice': 'mouse',
    'knives': 'knife',
    'sheep': 'sheep',
  };

  /// Everyday words → the class's Darknet name. Deliberately absent:
  /// "glass" (eyeglasses are not a class), "bag" (handbag, backpack or
  /// suitcase), "screen" — guessing would answer the wrong question. So are
  /// kinds of a class (man, woman, child, kid, boy, girl, guy, adult, kitten,
  /// puppy): "How many kids?" counted every person would be wrong, so those
  /// go to Gemma. Words for "any person" stay.
  static const _aliases = {
    // The everyday names of the Darknet spellings.
    'couch': 'sofa',
    'tv': 'tvmonitor',
    'television': 'tvmonitor',
    'monitor': 'tvmonitor',
    'tv monitor': 'tvmonitor',
    'plant': 'pottedplant',
    'potted plant': 'pottedplant',
    'houseplant': 'pottedplant',
    'table': 'diningtable',
    'dining table': 'diningtable',
    'phone': 'cell phone',
    'smartphone': 'cell phone',
    'cellphone': 'cell phone',
    'mobile phone': 'cell phone',
    'motorcycle': 'motorbike',
    'airplane': 'aeroplane',
    'plane': 'aeroplane',
    // People: only words for any person.
    'human': 'person',
    'someone': 'person',
    'somebody': 'person',
    'anyone': 'person',
    'anybody': 'person',
    // Everyday words.
    'bike': 'bicycle',
    'automobile': 'car',
    'lorry': 'truck',
    'rucksack': 'backpack',
    'purse': 'handbag',
    'necktie': 'tie',
    'ski': 'skis',
    'ball': 'sports ball',
    'racket': 'tennis racket',
    'racquet': 'tennis racket',
    'mug': 'cup',
    'hotdog': 'hot dog',
    'doughnut': 'donut',
    'laptop computer': 'laptop',
    'computer mouse': 'mouse',
    'remote control': 'remote',
    'fridge': 'refrigerator',
    'teddy': 'teddy bear',
    'hair dryer': 'hair drier',
    'hairdryer': 'hair drier',
    'hydrant': 'fire hydrant',
    'stoplight': 'traffic light',
    'scissor': 'scissors',
  };

  static const _spokenOverrides = {
    'person': ('person', 'people'),
    'tvmonitor': ('TV', 'TVs'),
    'pottedplant': ('potted plant', 'potted plants'),
    'diningtable': ('dining table', 'dining tables'),
    'aeroplane': ('airplane', 'airplanes'),
    'cell phone': ('phone', 'phones'),
    'sports ball': ('ball', 'balls'),
    'hair drier': ('hair dryer', 'hair dryers'),
    'skis': ('pair of skis', 'pairs of skis'),
    'scissors': ('pair of scissors', 'pairs of scissors'),
    'mouse': ('mouse', 'mice'),
    'knife': ('knife', 'knives'),
    'sheep': ('sheep', 'sheep'),
  };

  /// Every name, Ultralytics spelling and alias → class id.
  static final Map<String, int> _index = {
    for (var i = 0; i < kCocoNames.length; i++) kCocoNames[i]: i,
    for (final MapEntry(:key, :value) in _aliases.entries)
      key: kCocoNames.indexOf(value),
  };
}

/// "one" … "twenty", then digits.
String numberWord(int n) =>
    n >= 0 && n < _numberWords.length ? _numberWords[n] : '$n';

const _numberWords = [
  'zero',
  'one',
  'two',
  'three',
  'four',
  'five',
  'six',
  'seven',
  'eight',
  'nine',
  'ten',
  'eleven',
  'twelve',
  'thirteen',
  'fourteen',
  'fifteen',
  'sixteen',
  'seventeen',
  'eighteen',
  'nineteen',
  'twenty',
];
