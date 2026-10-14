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

import 'dart:math' as math;

import 'package:flutter/foundation.dart' show visibleForTesting;

import '../models/route_decision.dart';
import '../vision/coco_vocabulary.dart';

/// Decides whether a camera question is answered from the detection list
/// (fast, no LLM), needs the frame and Gemma, or asks about the past (the
/// visual memory). Rules over the normalized transcript, in order:
///
/// 0. A memory cue goes to the visual memory (`memory:<cue>`): "where
///    was/were …", "where is/are my …", "when did you (last) see …",
///    "have/did you see(n) …", "do you remember …", "where did I leave/put
///    …". Before rule 1, whose detail cues include "where". The present
///    "where is the dog?" stays a detail question about the frame: only the
///    past tense or a possession ("my keys") is about something no longer in
///    view. The query is the noun phrase after the cue ([memoryPhrase]); with
///    nothing to search for after the cue ("what did you see?", "… what I
///    just put down?") the route asks for the newest stored frame
///    ([MemoryRoute.latest]). The earliest cue decides, and it is not a
///    memory question — the later rules decide — when the words after it
///    - mark the present: "now", "right now" (not "just now", which is the
///      past), "currently", "at the moment", or — unless a past marker
///      ("earlier", "ago", "before", "last", "just now", the past of "be":
///      "was", "were") overrides it —
///      "here" or "in the/this picture/frame/image/photo": "where is my cup
///      right now?" is about the frame, "did you see a dog here earlier?"
///      is not;
///    - name a body part ("where is my hand?", "where is my finger
///      pointing?"): it is in view, or nowhere;
///    - end in an -ing word after "where is/are my X" ("where is my dog
///      going?", "… my phone charging?"): the present progressive, about the
///      frame ("where was my dog going?" stays a memory);
///    - start an embedded wh-question in the present, one with "is", "are",
///      "am" or a third-person present verb ("did you see how many people
///      are here?", "… what color the car is", "… what this sign says", "…
///      where the cat goes"), or with "this"/"that" as its subject and no
///      past verb ("… what this sign displays"): a question about the frame;
///      so is a relative clause in the present ("did you see the cat that is
///      on the table?"). In the past it is a memory
///      ("did you see where I put my keys?", "… who took my cup?"), and so
///      is "where" with a possession and no present marker ("have you seen
///      where my glasses are?"), like "where are my glasses?";
///    - are only the speaker or the listener ("where was I?", "where were
///      we?", "did you see where I was?"): nothing a frame can show.
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
    if (_memoryRoute(words, transcript) case final memory?) return memory;
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

  /// Rule 0: the earliest memory cue (the longest at that position) and the
  /// phrase after it; null when there is none, or when the words after it
  /// make it a question about the frame ([_aboutTheFrame]).
  static MemoryRoute? _memoryRoute(List<String> words, String transcript) {
    for (var i = 0; i < words.length; i++) {
      _MemoryCue? best;
      for (final cue in _memoryCues) {
        if (_startsWith(words, i, cue.words) &&
            (best == null || cue.words.length > best.words.length)) {
          best = cue;
        }
      }
      if (best == null) continue;
      final rest = words.sublist(i + best.words.length);
      if (_aboutTheFrame(best.words, rest)) return null;
      final phrase = memoryPhrase(rest);
      // Nothing to search for ("what did you see?"): an embedding of the
      // question would match a frame by chance; the newest frame answers it.
      return phrase.isEmpty
          ? MemoryRoute(transcript.trim(), 'memory:${best.name}', latest: true)
          : MemoryRoute(phrase, 'memory:${best.name}');
    }
    return null;
  }

  /// Whether [rest], the words after memory cue [cue], make the question
  /// one about the frame: only the speaker or the listener; a present marker
  /// before any later clause ("… when I was here" is still the past), a
  /// place marker only when no past marker (in the cue too: "when did you
  /// last see the remote here?") overrides it; an embedded wh-question in
  /// the present ([_presentEmbedded]) or about the speaker or the listener
  /// alone; a body part as the phrase's noun.
  static bool _aboutTheFrame(List<String> cue, List<String> rest) {
    if (rest.isEmpty) return false;
    if (rest.every(_selfOrListener.contains)) return true;
    var clause = rest.indexWhere(_memoryClauseWords.contains);
    if (clause < 0) clause = rest.length;
    final head = rest.sublist(0, clause);
    final padded = ' ${head.join(' ')} ';
    // "just now" is the past, not a "now".
    final present = padded.replaceAll(' just now ', ' ');
    if (_presentTimeMarkers.any((m) => present.contains(' $m '))) return true;
    final everything = ' ${[...cue, ...rest].join(' ')} ';
    final past = _pastMarkers.any((m) => everything.contains(' $m '));
    if (!past && _presentPlaceMarkers.any((m) => padded.contains(' $m '))) {
      return true;
    }
    if (head.isNotEmpty && _whWords.contains(head.first)) {
      final embedded = head.sublist(1);
      if (_presentEmbedded(head.first, embedded)) return true;
      final others = embedded.where((w) => !_auxiliaries.contains(w));
      if (others.isNotEmpty && others.every(_selfOrListener.contains)) {
        return true;
      }
    }
    // A relative clause in the present: "the cat that is on the table"; not
    // after a possession ("my wallet that has my cards" asks where it is,
    // like "where is my wallet?").
    for (var j = 1; j < head.length - 1; j++) {
      if (_relativePronouns.contains(head[j]) &&
          _isPresentVerb(head[j + 1]) &&
          !head.take(j).contains('my')) {
        return true;
      }
    }
    final phrase = memoryPhrase(rest).split(' ');
    // "where is my dog going?": the present progressive, about the frame.
    if (cue.length > 1 &&
        cue.first == 'where' &&
        _presentVerbs.contains(cue[1]) &&
        phrase.length > 1 &&
        _isIngWord(phrase.last)) {
      return true;
    }
    var end = phrase.indexWhere(_prepositions.contains);
    if (end < 0) end = phrase.length;
    // "finger pointing": the noun is before the -ing word.
    while (end > 0 && _isIngWord(phrase[end - 1])) {
      end--;
    }
    return end > 0 && _bodyParts.contains(phrase[end - 1]);
  }

  /// Whether the embedded question after wh-word [wh] ([clause], its words
  /// after the wh-word) is in the present: it has "is", "are", "am" or a
  /// third-person present verb ("says", "goes"), or its subject is a
  /// demonstrative ("this sign …", "that …") and no verb in it is in the
  /// past ("… what this sign displays", but "… who that was"). Decided:
  /// "where" with "my" before that verb is not ("have you seen where my
  /// glasses are?" asks where a possession is, like "where are my
  /// glasses?", which is a memory cue); a present marker ("… are now")
  /// still makes it the frame's.
  static bool _presentEmbedded(String wh, List<String> clause) {
    final verb = clause.indexWhere(_isPresentVerb);
    if (verb >= 0) {
      return !(wh == 'where' && clause.take(verb).contains('my'));
    }
    return clause.isNotEmpty &&
        _demonstratives.contains(clause.first) &&
        !clause.any(_isPastVerb);
  }

  /// "is", "are", "am", or a third-person present verb ("says", "goes").
  static bool _isPresentVerb(String word) =>
      _presentVerbs.contains(word) || _thirdPersonVerbs.contains(word);

  /// A verb in the past: the past of "be", "did", "had", a known irregular
  /// past ("went", "took") or a regular "-ed" form.
  static bool _isPastVerb(String word) =>
      _pastVerbs.contains(word) ||
      (word.length > 3 && word.endsWith('ed') && !_edNouns.contains(word));

  static bool _isIngWord(String word) =>
      word.length > 4 && word.endsWith('ing') && !_ingNouns.contains(word);

  /// The words of an embedded wh-question ([rest] from the wh-word up to
  /// [end]) that name what to search for, as `(start, end)`: after a
  /// pronoun subject and its verb ("where I put | my keys"), after the verb
  /// when the wh-word is the subject ("who took | my cup"), else the subject
  /// up to its verb ("where | the cat | went", "where | my glasses | are").
  static (int, int) _embeddedSpan(List<String> rest, int end) {
    var i = 1;
    if (i < end && _subjectPronouns.contains(rest[i])) {
      i++;
      while (i < end && _auxiliaries.contains(rest[i])) {
        i++;
      }
      return (math.min(i + 1, end), end); // past the verb
    }
    // An adverb or an auxiliary before the verb ("who just came in", "what
    // already happened") is not the subject.
    while (i < end && _auxiliaries.contains(rest[i])) {
      i++;
    }
    if (i < end && _endsSubject(rest[i])) return (i + 1, end);
    var content = i;
    while (content < end && _memoryLeading.contains(rest[content])) {
      content++;
    }
    // A pronoun subject ("who that | was", "what it | said") may be all of
    // it, and any past verb ends it; a determiner ("where my | left shoe
    // went") has its noun after it.
    final pronoun = content > i && _standalone.contains(rest[content - 1]);
    for (var j = pronoun ? content : content + 1; j < end; j++) {
      if (_endsSubject(rest[j]) || (pronoun && _isPastVerb(rest[j]))) {
        return (i, j);
      }
    }
    return (i, end);
  }

  /// A verb that ends an embedded question's subject: a clause verb, a
  /// known past form, or a present one.
  static bool _endsSubject(String word) =>
      _clauseVerbs.contains(word) ||
      _pastVerbs.contains(word) ||
      _isPresentVerb(word);

  /// The memory query in [rest], the words after a memory cue: up to the
  /// first clause word ("when", "because"…), without leading determiners and
  /// possessives ("the", "my") and without trailing filler ("anywhere",
  /// "today", "in the last minute", "here", "just now", "in the picture":
  /// a place marker reaches the phrase only when a past marker overrode
  /// it). "the sign about the wifi password" → "sign about the wifi
  /// password". An embedded wh-question names its thing
  /// ([_embeddedSpan]): "where I put my keys" → "keys", "where the cat
  /// went" → "cat". Empty when nothing is left, or only the speaker or the
  /// listener.
  @visibleForTesting
  static String memoryPhrase(List<String> rest) {
    var end = rest.indexWhere(_memoryClauseWords.contains);
    if (end < 0) end = rest.length;
    var start = 0;
    if (end > 0 && _whWords.contains(rest.first)) {
      (start, end) = _embeddedSpan(rest, end);
    }
    while (start < end && _memoryLeading.contains(rest[start])) {
      start++;
    }
    var trimmed = true;
    // The last word dropped was a time unit: an amount before it goes too
    // ("… 5 minutes ago"), never a number of its own ("gate 42").
    var afterUnit = false;
    while (trimmed && end > start) {
      trimmed = false;
      for (final tail in _memoryTrailingPhrases) {
        if (end - tail.length >= start &&
            _startsWith(rest, end - tail.length, tail)) {
          end -= tail.length;
          trimmed = true;
          afterUnit = false;
          break;
        }
      }
      if (trimmed) continue;
      final last = rest[end - 1];
      if (_memoryTrailing.contains(last) || (afterUnit && _isAmount(last))) {
        end--;
        trimmed = true;
        afterUnit = _timeUnits.contains(last);
      }
    }
    final words = rest.sublist(start, end);
    if (words.every(_selfOrListener.contains)) return '';
    return words.join(' ');
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

  static const _memoryCues = [
    _MemoryCue('where-was', ['where', 'was']),
    _MemoryCue('where-was', ['where', 'were']),
    _MemoryCue('where-is-my', ['where', 'is', 'my']),
    _MemoryCue('where-is-my', ['where', 'are', 'my']),
    _MemoryCue('when-seen', ['when', 'did', 'you', 'see']),
    _MemoryCue('when-seen', ['when', 'did', 'you', 'last', 'see']),
    _MemoryCue('when-seen', ['when', 'have', 'you', 'seen']),
    _MemoryCue('when-seen', [
      'when',
      'was',
      'the',
      'last',
      'time',
      'you',
      'saw',
    ]),
    _MemoryCue('seen', ['have', 'you', 'seen']),
    _MemoryCue('seen', ['did', 'you', 'see']),
    _MemoryCue('seen', ['have', 'you', 'noticed']),
    _MemoryCue('seen', ['did', 'you', 'notice']),
    _MemoryCue('seen', ['do', 'you', 'remember', 'seeing']),
    _MemoryCue('remember', ['do', 'you', 'remember']),
    _MemoryCue('seen', ['where', 'did', 'you', 'see']),
    _MemoryCue('where-did-i', ['where', 'did', 'i', 'leave']),
    _MemoryCue('where-did-i', ['where', 'did', 'i', 'put']),
    _MemoryCue('where-did-i', ['where', 'have', 'i', 'left']),
    _MemoryCue('where-did-i', ['where', 'have', 'i', 'put']),
    _MemoryCue('where-did-i', ['where', 'did', 'i', 'last', 'see']),
  ];

  /// Dropped at the start of a memory query (a pronoun alone leaves
  /// nothing to search with: the newest frame answers then), with the
  /// [_particles] ("where I put down | my keys").
  static const _memoryLeading = {
    ..._particles,
    'a',
    'an',
    'any',
    'some',
    'the',
    'it',
    'them',
    'they',
    'him',
    'my',
    'your',
    'our',
    'his',
    'her',
    'their',
    'this',
    'that',
    'these',
    'those',
  };

  /// Leading words that can be a whole subject or object ("who that was"),
  /// unlike a determiner ("my …", "the …").
  static const _standalone = {
    'it',
    'them',
    'they',
    'him',
    'this',
    'that',
    'these',
    'those',
  };

  /// The units of a time amount at the end of a memory query.
  static const _timeUnits = {
    'second',
    'seconds',
    'minute',
    'minutes',
    'hour',
    'hours',
  };

  /// An amount before a time unit: digits, or a number word.
  static bool _isAmount(String word) =>
      RegExp(r'^\d+$').hasMatch(word) || _numberWords.contains(word);

  static const _numberWords = {
    'one',
    'two',
    'three',
    'four',
    'five',
    'ten',
    'fifteen',
    'twenty',
    'thirty',
  };

  /// A verb's particle ("put down", "took away"): never what to search for.
  static const _particles = {'away', 'aside', 'back', 'down', 'off', 'up'};

  /// Dropped at the end of a memory query ("… anywhere", "… today", "… in
  /// the last minute"), with the [_particles] ("… what I just put down" has
  /// nothing left). Never "now": that makes it a question about the frame
  /// ([_presentTimeMarkers]); "just now" and the place markers are
  /// [_memoryTrailingPhrases].
  static const _memoryTrailing = {
    ..._particles,
    'a',
    'ago',
    'all',
    'again',
    'anywhere',
    'around',
    'at',
    'before',
    'by',
    'chance',
    'earlier',
    'few',
    'hour',
    'hours',
    'in',
    'last',
    'lately',
    'minute',
    'minutes',
    'night',
    'please',
    'recently',
    'second',
    'seconds',
    'the',
    'there',
    'time',
    'today',
    'week',
    'yesterday',
    'yet',
  };

  /// Dropped, as whole phrases, at the end of a memory query (with
  /// [_memoryTrailing]): "did you see a cat just now?" → "cat", "… a dog
  /// here earlier?" → "dog".
  static const _memoryTrailingPhrases = [
    ['just', 'now'],
    ['here'],
    ['in', 'the', 'picture'],
    ['in', 'the', 'frame'],
    ['in', 'the', 'image'],
    ['in', 'the', 'photo'],
    ['in', 'this', 'picture'],
    ['in', 'this', 'frame'],
    ['in', 'this', 'image'],
    ['in', 'this', 'photo'],
  ];

  /// After a memory cue, these mark the present: the question is about the
  /// frame ("where is my cup right now?"; "right now" is normalized to
  /// "now"; "just now" is the past).
  static const _presentTimeMarkers = ['now', 'currently', 'at the moment'];

  /// After a memory cue, these mark the frame too ("have you seen a dog
  /// here?"), unless a [_pastMarkers] word overrides them ("… here
  /// earlier?").
  static const _presentPlaceMarkers = [
    'here',
    'in the picture',
    'in the frame',
    'in the image',
    'in the photo',
    'in this picture',
    'in this frame',
    'in this image',
    'in this photo',
  ];

  /// Anywhere after a memory cue (or in it: "where was"), these put "here"
  /// and "in the picture" in the past: "did you see how many people were
  /// here?" asks about then.
  static const _pastMarkers = [
    'earlier',
    'ago',
    'before',
    'last',
    'just now',
    'was',
    'were',
  ];

  /// A wh-word right after a memory cue starts an embedded question: about
  /// the frame in the present ("did you see how many people are here?"), a
  /// memory in the past ("did you see where I put my keys?").
  static const _whWords = {'what', 'how', 'which', 'who', 'whose', 'where'};

  /// An embedded question with one of these is in the present.
  static const _presentVerbs = {'is', 'are', 'am'};

  /// Third-person present verbs: an embedded question or a relative clause
  /// with one is in the present ("what this sign says", "where the cat
  /// goes", "the man who sits by the door").
  static const _thirdPersonVerbs = {
    'belongs',
    'comes',
    'contains',
    'does',
    'faces',
    'goes',
    'hangs',
    'has',
    'holds',
    'lies',
    'looks',
    'means',
    'moves',
    'plays',
    'points',
    'reads',
    'runs',
    'says',
    'shows',
    'sits',
    'sleeps',
    'stands',
    'wears',
  };

  /// The subject of an embedded question that points at what is in view
  /// ([_presentEmbedded]).
  static const _demonstratives = {'this', 'that', 'these', 'those'};

  /// Start a relative clause after the phrase's noun ("the cat that is …").
  static const _relativePronouns = {'that', 'which', 'who'};

  /// Past forms that are not "-ed": with one, a demonstrative subject is
  /// the past ("who that was").
  static const _pastVerbs = {
    'was',
    'were',
    'did',
    'had',
    'went',
    'took',
    'left',
    'came',
    'ran',
    'sat',
    'stole',
    'got',
    'fell',
    'brought',
    'hid',
    'said',
    'saw',
    'stood',
    'wore',
    'held',
    'made',
  };

  /// "-ed" words that are not past verbs.
  static const _edNouns = {'bed', 'red', 'shed', 'sled', 'seed', 'need'};

  /// An embedded question's subject when it is a pronoun ("where I put …").
  static const _subjectPronouns = {'i', 'you', 'we', 'he', 'she', 'they'};

  /// Skipped between an embedded pronoun subject and its verb ("where I
  /// last saw …", "where I had put …"), and not counted when only the
  /// speaker or the listener is left ("did you see where I was?").
  static const _auxiliaries = {
    'already',
    'did',
    'had',
    'has',
    'have',
    'just',
    'last',
    'was',
    'were',
  };

  /// An embedded question's subject ends at the first of these ("where the
  /// cat went"); when the wh-word is the subject, its object follows one
  /// ("who took my cup").
  static const _clauseVerbs = {
    'am',
    'appeared',
    'are',
    'arrived',
    'brought',
    'came',
    'carried',
    'changed',
    'did',
    'disappeared',
    'does',
    'dropped',
    'fell',
    'goes',
    'got',
    'grabbed',
    'had',
    'happened',
    'has',
    'hid',
    'is',
    'left',
    'moved',
    'picked',
    'placed',
    'put',
    'ran',
    'sat',
    'stole',
    'took',
    'was',
    'went',
    'were',
  };

  /// Nouns ending in "-ing" that are not dropped before the body-part check
  /// ("hand painting" is a painting) nor read as the present progressive
  /// ("where is my painting?").
  static const _ingNouns = {
    'awning',
    'bedding',
    'building',
    'ceiling',
    'clothing',
    'drawing',
    'earring',
    'painting',
    'pudding',
    'railing',
    'sibling',
    'sling',
    'spring',
    'string',
    'swing',
    'thing',
    'wedding',
  };

  /// "Where was I?", "where were we?": the speaker or the listener alone.
  static const _selfOrListener = {'i', 'me', 'we', 'us', 'you'};

  /// A memory query's noun is its last word before the first of these:
  /// "hand" in "left hand on the table", "sign" in "sign about the wifi".
  static const _prepositions = {
    'about',
    'at',
    'by',
    'for',
    'from',
    'in',
    'near',
    'of',
    'on',
    'to',
    'under',
    'with',
  };

  /// Body parts are in view or nowhere, never a memory: "where is my hand?"
  /// stays a detail question about the frame.
  static const _bodyParts = {
    'arm',
    'arms',
    'ear',
    'ears',
    'eye',
    'eyes',
    'face',
    'finger',
    'fingers',
    'foot',
    'feet',
    'hand',
    'hands',
    'head',
    'leg',
    'legs',
    'nose',
    'thumb',
  };

  /// A memory query ends before these: "where was the mug when I left".
  static const _memoryClauseWords = {
    'when',
    'while',
    'because',
    'since',
    'after',
  };

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

final class _MemoryCue {
  const _MemoryCue(this.name, this.words);

  final String name;
  final List<String> words;
}

final class _Trigger {
  const _Trigger(this.kind, this.words);

  final _Kind kind;
  final List<String> words;
}
