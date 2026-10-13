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

import '../models/route_decision.dart';
import '../vision/coco_vocabulary.dart';

/// Decides whether a camera question is answered from the detection list
/// (fast, no LLM) or needs the frame and Gemma. Rules over
/// the normalized transcript, in order:
///
/// 1. A detail cue anywhere goes detailed (`detail:<cue>`).
/// 2. Inventory ("what do you see", "what's in front of me", "what
///    objects") goes fast (`inventory`).
/// 3. "how many X" goes fast (`count`), and "is/are there (a|any) X",
///    "do/can you see X" go fast (`presence`) — only when X names a COCO
///    class exactly; otherwise detailed (`unknown-noun`).
/// 4. Everything else is detailed (`default`).
///
/// Stricter than the rules alone in one way: words around a fast
/// question that are not neutral filler — a qualifier ("sleeping", "on the
/// table", "or a cat") or a prefix ("besides the cat, …") — send it to
/// Gemma (`qualifier`, `prefix`). A fast answer to a different question
/// than the one asked is worse than a slower right one. About 0.1 ms.
final class QuestionRouter {
  const QuestionRouter([this._vocabulary = kCocoVocabulary]);

  final CocoVocabulary _vocabulary;

  RouteDecision classify(String transcript) {
    final words = normalizeQuestion(transcript);
    if (words.isEmpty) return const DetailedRoute('default');
    final padded = ' ${words.join(' ')} ';
    for (final cue in _detailCues) {
      if (padded.contains(' $cue ')) return DetailedRoute('detail:$cue');
    }
    final match = _findTrigger(words);
    if (match == null) return const DetailedRoute('default');
    final (start, trigger) = match;
    final prefix = words.sublist(0, start);
    if (!prefix.every(_neutralPrefix.contains)) {
      return const DetailedRoute('prefix');
    }
    final rest = words.sublist(start + trigger.words.length);
    switch (trigger.kind) {
      case _Kind.inventory:
        return rest.every(_neutralTail.contains)
            ? const FastRoute(FastIntent.inventory, 'inventory')
            : const DetailedRoute('qualifier');
      case _Kind.count:
      case _Kind.presence:
        final (noun, tail) = _nounPhrase(rest);
        if (trigger.kind == _Kind.presence && _anything.contains(noun)) {
          return tail.every(_neutralTail.contains)
              ? const FastRoute(FastIntent.inventory, 'inventory')
              : const DetailedRoute('qualifier');
        }
        final cls = noun.isEmpty ? null : _vocabulary.resolveNoun(noun);
        if (cls == null) return const DetailedRoute('unknown-noun');
        if (!tail.every(_neutralTail.contains)) {
          return const DetailedRoute('qualifier');
        }
        return trigger.kind == _Kind.count
            ? FastRoute(FastIntent.count, 'count', cls: cls)
            : FastRoute(FastIntent.presence, 'presence', cls: cls);
    }
  }

  /// The earliest trigger, the longest one at that position.
  static (int, _Trigger)? _findTrigger(List<String> words) {
    for (var i = 0; i < words.length; i++) {
      _Trigger? best;
      for (final trigger in _triggers) {
        if (_startsWith(words, i, trigger.words) &&
            (best == null || trigger.words.length > best.words.length)) {
          best = trigger;
        }
      }
      if (best != null) return (i, best);
    }
    return null;
  }

  static bool _startsWith(List<String> words, int at, List<String> seq) {
    if (at + seq.length > words.length) return false;
    for (var j = 0; j < seq.length; j++) {
      if (words[at + j] != seq[j]) return false;
    }
    return true;
  }

  /// The noun phrase after a trigger (leading determiners dropped, at most
  /// three words, up to the first stop word) and the words after it.
  static (String, List<String>) _nounPhrase(List<String> rest) {
    var i = 0;
    while (i < rest.length && _determiners.contains(rest[i])) {
      i++;
    }
    final noun = <String>[];
    while (i < rest.length &&
        noun.length < 3 &&
        !_stopWords.contains(rest[i])) {
      noun.add(rest[i]);
      i++;
    }
    return (noun.join(' '), rest.sublist(i));
  }

  static const _triggers = [
    _Trigger(_Kind.inventory, ['what', 'do', 'you', 'see']),
    _Trigger(_Kind.inventory, ['what', 'can', 'you', 'see']),
    _Trigger(_Kind.inventory, ['what', 'you', 'see']),
    _Trigger(_Kind.inventory, ['what', 'you', 'can', 'see']),
    _Trigger(_Kind.inventory, ['what', 'are', 'you', 'seeing']),
    _Trigger(_Kind.inventory, ['what', 'is', 'in', 'front', 'of', 'me']),
    _Trigger(_Kind.inventory, ['what', 'is', 'in', 'front', 'of', 'you']),
    _Trigger(_Kind.inventory, ['what', 'objects']),
    _Trigger(_Kind.inventory, ['what', 'things']),
    _Trigger(_Kind.count, ['how', 'many']),
    _Trigger(_Kind.presence, ['is', 'there']),
    _Trigger(_Kind.presence, ['are', 'there']),
    _Trigger(_Kind.presence, ['do', 'you', 'see']),
    _Trigger(_Kind.presence, ['can', 'you', 'see']),
    _Trigger(_Kind.presence, ['could', 'you', 'see']),
  ];

  /// Rule 1's detail cues, plus a few synonyms of the same kinds (reading,
  /// position, attributes, identity, reasons).
  static const _detailCues = [
    'describe',
    'description',
    'read',
    'reading',
    'say',
    'says',
    'said',
    'text',
    'written',
    'writing',
    'sign',
    'signs',
    'label',
    'labels',
    'color',
    'colour',
    'colors',
    'colours',
    'colored',
    'coloured',
    'wearing',
    'wears',
    'wear',
    'doing',
    'happening',
    'where',
    'left',
    'right',
    'behind',
    'next to',
    'on top',
    'under',
    'underneath',
    'beneath',
    'below',
    'above',
    'between',
    'closest',
    'nearest',
    'farthest',
    'furthest',
    'what kind',
    'what type',
    'what sort',
    'brand',
    'breed',
    'why',
    'explain',
    'tell me about',
    'look closer',
    'are you sure',
    'who',
    'whose',
    'what is this',
    'what is that',
    'what is it',
    'what are these',
    'what are those',
  ];

  static const _determiners = {
    'a',
    'an',
    'any',
    'the',
    'of',
    'these',
    'those',
    'some',
  };

  /// Presence of "anything" is an inventory question.
  static const _anything = {'anything', 'something', 'objects', 'things'};

  static const _stopWords = {
    'am',
    'and',
    'are',
    'at',
    'by',
    'can',
    'could',
    'currently',
    'did',
    'do',
    'does',
    'for',
    'from',
    'has',
    'have',
    'here',
    'i',
    'in',
    'inside',
    'is',
    'near',
    'now',
    'on',
    'or',
    'outside',
    'over',
    'see',
    'that',
    'there',
    'they',
    'to',
    'total',
    'under',
    'visible',
    'was',
    'we',
    'were',
    'which',
    'who',
    'will',
    'with',
    'without',
    'would',
    'you',
  };

  /// Words that may stand before a trigger without changing the question.
  static const _neutralPrefix = {
    'can',
    'could',
    'would',
    'will',
    'you',
    'please',
    'tell',
    'me',
    'hey',
    'hi',
    'so',
    'ok',
    'okay',
    'um',
    'uh',
    'well',
    'and',
    'i',
    'wonder',
    'do',
    'know',
    'want',
    'to',
    'just',
    'now',
  };

  /// Words that may follow a fast question without changing it ("… do you
  /// see in the picture right now", "… are there in total").
  static const _neutralTail = {
    'a',
    'all',
    'altogether',
    'an',
    'are',
    'around',
    'at',
    'camera',
    'can',
    'could',
    'count',
    'currently',
    'do',
    'does',
    'exactly',
    'frame',
    'front',
    'here',
    'i',
    'image',
    'in',
    'is',
    'me',
    'moment',
    'now',
    'of',
    'overall',
    'photo',
    'picture',
    'please',
    'room',
    'scene',
    'see',
    'seeing',
    'shot',
    'that',
    'the',
    'there',
    'these',
    'this',
    'those',
    'today',
    'total',
    'us',
    'view',
    'visible',
    'we',
    'you',
  };
}

/// The transcript as lowercase words: contractions expanded, punctuation
/// dropped, "right now" → "now" (so "right" stays a position cue).
List<String> normalizeQuestion(String transcript) {
  var text = transcript.toLowerCase().replaceAll('’', "'");
  text = text.replaceAllMapped(
    RegExp(r"\b(what|where|who|there|it|that|how|here)'s\b"),
    (m) => '${m[1]} is',
  );
  text = text
      .replaceAll("'re", ' are')
      .replaceAll("n't", ' not')
      .replaceAll("'m", ' am')
      .replaceAll(RegExp('[^a-z0-9 ]'), ' ');
  text = ' ${text.split(RegExp(r'\s+')).where((w) => w.isNotEmpty).join(' ')} '
      .replaceAll(' right now ', ' now ')
      .replaceAll(' all right ', ' ')
      .replaceAll(' alright ', ' ');
  return text.split(' ').where((w) => w.isNotEmpty).toList();
}

enum _Kind { inventory, count, presence }

final class _Trigger {
  const _Trigger(this.kind, this.words);

  final _Kind kind;
  final List<String> words;
}
