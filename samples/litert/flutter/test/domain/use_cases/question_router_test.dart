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
}
