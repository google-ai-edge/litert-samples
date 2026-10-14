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

import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/domain/models/route_decision.dart';
import 'package:litert_edge_demos/domain/use_cases/question_router.dart';
import 'package:litert_edge_demos/domain/vision/coco_vocabulary.dart';

void main() {
  const router = QuestionRouter();

  group('golden set (test_assets/router_golden.json)', () {
    final doc = jsonDecode(
      File('test_assets/router_golden.json').readAsStringSync(),
    ) as Map<String, Object?>;
    final items = (doc['items']! as List<Object?>).cast<Map<String, Object?>>();

    bool correct(Map<String, Object?> item, RouteDecision route) =>
        switch ((item['route'], route)) {
          ('fast', FastRoute(:final intent, :final cls)) =>
            intent.name == item['intent'] &&
                (item['class'] == null ||
                    cls == kCocoNames.indexOf(item['class']! as String)),
          ('detailed', DetailedRoute()) => true,
          ('memory', MemoryRoute()) => true,
          _ => false,
        };

    test('≥60 utterances, ≥95% overall, 100% on the must-be-detailed subset', () {
      expect(items.length, greaterThanOrEqualTo(60));
      final misses = <String>[];
      var mustTotal = 0;
      var mustRight = 0;
      final watch = Stopwatch()..start();
      for (final item in items) {
        final q = item['q']! as String;
        final route = router.classify(q);
        final ok = correct(item, route);
        if (!ok) {
          misses.add(
            '"$q" → $route (expected ${item['route']} '
            '${item['intent'] ?? ''} ${item['class'] ?? ''})',
          );
        }
        if (item['must'] == true) {
          mustTotal++;
          if (ok) mustRight++;
        }
      }
      watch.stop();
      final right = items.length - misses.length;
      final accuracy = right / items.length;
      // The accuracy report is what this golden run is for.
      // ignore: avoid_print
      print(
        'ROUTER golden=${items.length} correct=$right '
        'accuracy=${(accuracy * 100).toStringAsFixed(1)}% '
        'must_detailed=$mustRight/$mustTotal '
        'avg=${(watch.elapsedMicroseconds / items.length).toStringAsFixed(1)}µs'
        '${misses.isEmpty ? '' : '\n  ${misses.join('\n  ')}'}',
      );
      expect(mustRight, mustTotal, reason: misses.join('\n'));
      expect(accuracy, greaterThanOrEqualTo(0.95), reason: misses.join('\n'));
    });
  });

  test('rules report which rule decided', () {
    expect(router.classify(' How many cats do you see?').rule, 'count');
    expect(router.classify(' Is there a dog?').rule, 'presence');
    expect(router.classify(' What do you see?').rule, 'inventory');
    expect(router.classify(' What color is it?').rule, 'detail:color');
    expect(
      router.classify(' How many unicorns are there?').rule,
      'unknown-noun',
    );
    expect(router.classify(' How many cats are sleeping?').rule, 'qualifier');
    expect(
      router.classify(' Besides the cats, how many dogs do you see?').rule,
      'prefix',
    );
    expect(router.classify(' Good morning.').rule, 'default');
    expect(router.classify('').rule, 'default');
  });

  test('a number in the noun phrase never gets a fast presence answer '
      '(the router keeps digits, so resolveNoun must too)', () {
    for (final q in [
      ' Are there 2 cats?',
      ' Are there 3 people here?',
      ' Is there 1 dog?',
      ' Do you see 2 cups?',
      ' How many 2 cats are there?',
      ' Are there two cats?',
    ]) {
      expect(router.classify(q), isA<DetailedRoute>(), reason: q);
    }
  });

  test('a kind of person or animal goes to Gemma, not the whole class', () {
    for (final q in [
      ' How many kids are there?',
      ' How many women are there?',
      ' Is there a man here?',
      ' Do you see a boy?',
      ' Are there any girls?',
      ' How many adults do you see?',
      ' Is there a kitten?',
      ' How many puppies are there?',
      ' Do you see a guy?',
      ' Is there a child?',
    ]) {
      final route = router.classify(q);
      expect(route, isA<DetailedRoute>(), reason: q);
      expect(route.rule, 'unknown-noun', reason: q);
    }
    for (final q in [
      ' Is there anyone here?',
      ' Is there someone in front of me?',
      ' Is there a human?',
    ]) {
      expect(
        router.classify(q),
        isA<FastRoute>().having(
          (r) => r.cls,
          'cls',
          kCocoNames.indexOf('person'),
        ),
        reason: q,
      );
    }
  });

  test('"right now" is not a position cue; "right" alone is', () {
    expect(
      router.classify(' How many cats do you see right now?'),
      isA<FastRoute>(),
    );
    expect(router.classify(' What is on the right?'), isA<DetailedRoute>());
  });

  test('normalization expands contractions and drops punctuation', () {
    expect(normalizeQuestion(" What's in front of me?"), [
      'what',
      'is',
      'in',
      'front',
      'of',
      'me',
    ]);
    expect(normalizeQuestion(' Is there a TV… right now!'), [
      'is',
      'there',
      'a',
      'tv',
      'now',
    ]);
  });

  group('memory (rule 0, before the detail cues)', () {
    MemoryRoute memory(String q) {
      final route = router.classify(q);
      expect(route, isA<MemoryRoute>(), reason: q);
      return route as MemoryRoute;
    }

    test('the past tense and possessions go to the visual memory, with the '
        'noun phrase after the cue as the query', () {
      for (final (q, query, rule) in [
        (
          ' Where was the sign about the wifi password?',
          'sign about the wifi password',
          'memory:where-was',
        ),
        (' Where were my glasses?', 'glasses', 'memory:where-was'),
        (" Where's my phone?", 'phone', 'memory:where-is-my'),
        (' Where are my keys?', 'keys', 'memory:where-is-my'),
        (' When did you see the yellow mug?', 'yellow mug', 'memory:when-seen'),
        (' When did you last see the remote?', 'remote', 'memory:when-seen'),
        (
          ' When was the last time you saw my wallet?',
          'wallet',
          'memory:when-seen',
        ),
        (' Have you seen a giraffe?', 'giraffe', 'memory:seen'),
        (' Did you see a dog?', 'dog', 'memory:seen'),
        (' Have you seen my keys anywhere today?', 'keys', 'memory:seen'),
        (' Did you notice a red bag?', 'red bag', 'memory:seen'),
        (' Where did I leave my mug?', 'mug', 'memory:where-did-i'),
        (' Where did I put the charger?', 'charger', 'memory:where-did-i'),
        (' Hey, where was the cat?', 'cat', 'memory:where-was'),
        (' Where was the mug when I left?', 'mug', 'memory:where-was'),
        (
          ' Did you see a bicycle in the last minute?',
          'bicycle',
          'memory:seen',
        ),
      ]) {
        final route = memory(q);
        expect(route.query, query, reason: q);
        expect(route.rule, rule, reason: q);
      }
    });

    test('a cue with nothing to search for after it (nothing, a wh-word, a '
        'pronoun, filler or a particle) asks for the newest stored frame, not '
        'an embedding of the question', () {
      for (final q in [
        ' What did you see?',
        ' What have you seen?',
        ' What did you see earlier?',
        ' Have you seen?',
        ' Where was it?',
        ' Did you see what I just put down?',
        ' Did you see what?',
      ]) {
        final route = memory(q);
        expect(route.latest, isTrue, reason: q);
        expect(route.query, q.trim(), reason: '$q: the question, for the log');
      }
      expect(memory(' Did you see a dog?').latest, isFalse);
    });

    test('an adverb or an auxiliary before the verb, or a past or present '
        'verb after a pronoun, is never what to search for', () {
      for (final q in [
        ' Did you see who just came in?',
        ' Did you see what just fell?',
        ' Did you see what was just here?',
        ' Did you see what already happened?',
        ' Did you see what it said?',
        ' Did you see what this showed?',
        ' What did you see 5 minutes ago?',
      ]) {
        final route = memory(q);
        expect(route.latest, isTrue, reason: '$q → "${route.query}"');
      }
      for (final (q, query) in [
        (' Did you see who just took my cup?', 'cup'),
        (' Where was my cup 2 minutes ago?', 'cup'),
        (
          ' Have you seen my wallet that has my cards?',
          'wallet that has my cards',
        ),
        (' Did you see where the striped cat went?', 'striped cat'),
        (' Did you see gate 42?', 'gate 42'),
      ]) {
        expect(memory(q).query, query, reason: q);
      }
    });

    test('a particle after the verb is not what to search for', () {
      expect(memory(' Did you see where I put my keys down?').query, 'keys');
      expect(memory(' Did you see where I put down my keys?').query, 'keys');
    });

    test('the past of "be" is a past marker: it overrides "here" and the '
        'picture, and an embedded question with it is a memory', () {
      for (final q in [
        ' Did you see how many people were here?',
        ' Did you see what color the car was?',
        ' Where was the cat in the picture?',
      ]) {
        expect(router.classify(q), isA<MemoryRoute>(), reason: q);
      }
    });

    test(
      'the present also with a third-person verb, a relative clause in the '
      'present, a demonstrative subject, and an -ing after "where is my X"',
      () {
        for (final q in [
          ' Did you see what this sign says?',
          ' Did you see where the cat goes?',
          ' Did you see what this sign displays?',
          ' Where is my dog going?',
          ' Where is my phone charging?',
          ' Did you see the cat that is on the table?',
          ' Have you seen the man who sits by the door?',
        ]) {
          expect(router.classify(q), isA<DetailedRoute>(), reason: q);
        }
        for (final (q, query) in [
          (' Did you see where this cat went?', 'cat'),
          (' Did you see who that was?', 'Did you see who that was?'),
          (' Where is my charging cable?', 'charging cable'),
          (' Where is my painting?', 'painting'),
          (' Where was my dog going?', 'dog going'),
        ]) {
          expect(memory(q).query, query, reason: q);
        }
      },
    );

    test('the present stays about the frame: "where is the dog?" is a detail '
        'question, "do you see a dog?" a fast one', () {
      expect(router.classify(' Where is the dog?'), isA<DetailedRoute>());
      expect(router.classify(' Where is the dog?').rule, 'detail:where');
      expect(router.classify(' Where are the cats?'), isA<DetailedRoute>());
      expect(router.classify(' Do you see a dog?'), isA<FastRoute>());
      expect(router.classify(' Can you see a cup?'), isA<FastRoute>());
      expect(router.classify(' Was there a cat?'), isA<DetailedRoute>());
    });

    test('a present marker after the cue keeps the question on the frame '
        '(never trimmed as filler)', () {
      for (final (q, rule) in [
        (' Where is my cup right now?', 'detail:where'),
        (' Where are my keys now?', 'detail:where'),
        (" Where's my phone at the moment?", 'detail:where'),
        (' Where is my bag currently?', 'detail:where'),
        (' Have you seen a dog here?', 'default'),
        (' Did you see a cat in the picture?', 'default'),
        (' Did you see a cup in the frame?', 'default'),
      ]) {
        final route = router.classify(q);
        expect(route, isA<DetailedRoute>(), reason: q);
        expect(route.rule, rule, reason: q);
      }
    });

    test('a marker before the cue, or in a clause after the phrase, does '
        'not count: "now, where did I put…" is the idiom, not the present', () {
      expect(memory(' Now, where did I put my glasses?').query, 'glasses');
      expect(memory(' Where was the mug when I was here?').query, 'mug');
    });

    test('a body part as the noun stays a detail question', () {
      for (final q in [
        ' Where is my hand?',
        ' Where are my hands?',
        " Where's my left arm?",
        ' Where are my fingers?',
        ' Where was my face?',
        ' Have you seen my head?',
        ' Where is my hand on the table?',
        // A trailing -ing word is skipped: the noun is "finger".
        ' Where is my finger pointing?',
        ' Where is my finger pointing at?',
        ' Where are my hands going?',
      ]) {
        expect(router.classify(q), isA<DetailedRoute>(), reason: q);
      }
      // The noun decides, not any word: "hand cream" is a cream, and a
      // painting is not dropped as an -ing word.
      expect(memory(' Where did I leave my hand cream?').query, 'hand cream');
      expect(memory(' Where is my hand painting?').query, 'hand painting');
      expect(memory(' Where is my earring?').query, 'earring');
    });

    test('a wh-word right after the cue starts an embedded question: in the '
        'present (is/are/am) it is about the frame and the later rules '
        'decide', () {
      expect(
        router.classify(' Did you see how many people are here?').rule,
        'prefix',
      );
      expect(
        router.classify(' Did you see what color the car is?').rule,
        'detail:color',
      );
      expect(
        router.classify(' Did you see where the cat is?').rule,
        'detail:where',
      );
      expect(
        router.classify(' Have you seen which cup is mine?'),
        isA<DetailedRoute>(),
      );
      // "where" with "my" asks where a possession is, like "where are my
      // glasses?" — unless a present marker says now.
      expect(
        router.classify(' Have you seen where my glasses are now?'),
        isA<DetailedRoute>(),
      );
    });

    test('an embedded question in the past is a memory: the query is what '
        'it names', () {
      for (final (q, query, rule) in [
        (' Did you see where I put my keys?', 'keys', 'memory:seen'),
        (' Did you see where I left my phone?', 'phone', 'memory:seen'),
        (' Did you see where the cat went?', 'cat', 'memory:seen'),
        (' Did you see who took my cup?', 'cup', 'memory:seen'),
        (' Have you seen where my glasses are?', 'glasses', 'memory:seen'),
        (' Do you remember where I put my keys?', 'keys', 'memory:remember'),
        (
          ' Do you remember where I last saw the remote?',
          'remote',
          'memory:remember',
        ),
        (' Did you see where my left shoe went?', 'left shoe', 'memory:seen'),
        // Nothing named: the whole question is the query.
        (
          ' Did you see who came in?',
          'Did you see who came in?',
          'memory:seen',
        ),
        (
          ' Did you see what happened?',
          'Did you see what happened?',
          'memory:seen',
        ),
      ]) {
        final route = memory(q);
        expect(route.query, query, reason: q);
        expect(route.rule, rule, reason: q);
      }
      // "Do you remember seeing …" keeps its longer cue.
      expect(memory(' Do you remember seeing a red bag?').rule, 'memory:seen');
      // A body part named in the past is still in view or nowhere.
      expect(
        router.classify(' Did you see where my hand went?'),
        isA<DetailedRoute>(),
      );
    });

    test('a past marker puts "here" and "in the picture" in the past; '
        '"just now" is the past', () {
      for (final (q, query) in [
        (' Did you see a cat just now?', 'cat'),
        (' Did you see a dog here earlier?', 'dog'),
        (' Have you seen a cup in the picture before?', 'cup'),
        (' Where was my phone in the picture a minute ago?', 'phone'),
        (' Have you seen my keys here last time?', 'keys'),
        (' Where were my keys just now?', 'keys'),
        (' Did you see a dog in the picture last week?', 'dog'),
        // The cue's own "last" counts.
        (' When did you last see the remote here?', 'remote'),
      ]) {
        expect(memory(q).query, query, reason: q);
      }
      // Only the place markers: "now" and "at the moment" stay the present.
      expect(
        router.classify(' Where are my keys now, earlier they were here?'),
        isA<DetailedRoute>(),
      );
    });

    test('"in this picture/photo/frame/image" marks the frame', () {
      for (final q in [
        ' Did you see a cat in this picture?',
        ' Have you seen a cup in this photo?',
        ' Did you see a dog in this frame?',
        ' Have you seen a bag in this image?',
      ]) {
        expect(router.classify(q), isA<DetailedRoute>(), reason: q);
      }
    });

    test('the speaker or the listener alone is not a memory', () {
      for (final q in [
        ' Where was I?',
        ' Where were we?',
        ' Where were you?',
        ' Did you see where I was?',
        ' Have you seen where we were?',
      ]) {
        final route = router.classify(q);
        expect(route, isA<DetailedRoute>(), reason: q);
        expect(route.rule, 'detail:where', reason: q);
      }
      // A third-person pronoun still searches with the whole question.
      expect(memory(' Where was it?').query, 'Where was it?');
    });

    test('the phrase: determiners and possessives off the front, filler off '
        'the back, cut before a clause', () {
      expect(
        QuestionRouter.memoryPhrase(['the', 'red', 'mug', 'anywhere']),
        'red mug',
      );
      expect(
        QuestionRouter.memoryPhrase(['my', 'keys', 'when', 'i', 'left']),
        'keys',
      );
      expect(QuestionRouter.memoryPhrase(['the']), '');
      expect(QuestionRouter.memoryPhrase(const []), '');
      expect(
        QuestionRouter.memoryPhrase(['a', 'dog', 'here', 'a', 'minute', 'ago']),
        'dog',
      );
      expect(
        QuestionRouter.memoryPhrase(['where', 'i', 'put', 'my', 'keys']),
        'keys',
      );
      expect(QuestionRouter.memoryPhrase(['where', 'i', 'was']), '');
    });
  });
}
