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
import 'package:litert_edge_demos/domain/use_cases/skill_question_router.dart';

/// Live device-fact and time questions skip retrieval
/// (deterministic), so Gemma sees only its skills for them; every
/// knowledge-base question still retrieves. Device questions clear the KB
/// gate (0.40–0.56, `test_assets/skill_trigger_similarity.json`), which is
/// why this router exists.
void main() {
  const router = SkillQuestionRouter();
  SkillTopic? topic(String q) => router.classify(q)?.topic;

  group('every measured skill trigger phrase routes to its skill', () {
    final golden = jsonDecode(
      File('test_assets/skill_trigger_similarity.json').readAsStringSync(),
    ) as Map<String, Object?>;
    // The app has no timer or camera-watch skills: their measured phrases
    // reach the model like any other chat (no route).
    const topics = {
      'device-info': SkillTopic.deviceFacts,
      'current-time': SkillTopic.time,
      'timer': null,
      'camera-watch': null,
    };
    for (final row in golden['phrases']! as List<Object?>) {
      final phrase = row! as Map<String, Object?>;
      final q = phrase['q']! as String;
      test(q, () {
        expect(topics.containsKey(phrase['skill']), isTrue, reason: q);
        expect(topic(q), topics[phrase['skill']], reason: q);
      });
    }
  });

  test('more live device-fact questions', () {
    for (final q in [
      'Which accelerator is running right now?',
      'Is Gemma running on the GPU?',
      'How much memory are you using?',
      'What processor does this device have?',
      'Are you using the NPU?',
      'Which backend is the detector using right now?',
      "What's running on the GPU at the moment?",
      'Which models are loaded?',
      'Are you running on the CPU or the GPU?',
      'What are you running on?',
    ]) {
      expect(topic(q), SkillTopic.deviceFacts, reason: q);
    }
  });

  test('more time questions; timers and watching are gone', () {
    const cases = {
      'What time is it now?': SkillTopic.time,
      'Can you tell me the time, please?': SkillTopic.time,
      "What's today's date?": SkillTopic.time,
      'What year is it?': SkillTopic.time,
      'Set a five minute timer.': null,
      'Cancel all timers': null,
      'Let me know when you see a dog.': null,
      'Stop looking for the cup.': null,
    };
    for (final MapEntry(key: q, value: expected) in cases.entries) {
      expect(topic(q), expected, reason: q);
    }
  });

  test('knowledge-base questions still retrieve: the whole on-topic golden '
      'set', () {
    final golden = jsonDecode(
      File('test_assets/kb_golden.json').readAsStringSync(),
    ) as Map<String, Object?>;
    for (final row in golden['on_topic']! as List<Object?>) {
      final q = (row! as Map<String, Object?>)['q']! as String;
      expect(router.classify(q), isNull, reason: q);
    }
  });

  test('knowledge-base questions about the same topics still retrieve', () {
    for (final q in [
      'What GPU does LiteRT use?',
      'Does LiteRT support the NPU on Android?',
      'How much memory does Gemma 4 E2B need?',
      'Which accelerator should I use for the detector?',
      'How do I run a model on the GPU with the compiled model API?',
      'What is the difference between the GPU and NPU accelerators?',
      'Why is my model running on the CPU instead of the GPU?',
      'How fast is YOLO26n on a Pixel 9 GPU?',
      'What does the device info skill do?',
      'What does the current_time intent do?',
      'How do I add a skill to the app?',
      'What should I watch for when quantizing a model?',
      'What is the time to first token on an iPhone?',
      'What time does the museum open?',
      'Is the GPU faster than the CPU?',
      'What is LiteRT running on?',
      'Why does the model give me the same answer every time?',
      'How long can an audio clip be for Gemma 4?',
      'Which entitlements does an iPhone app need to load a big model?',
    ]) {
      expect(router.classify(q), isNull, reason: q);
    }
  });

  test('off-topic chat is not a skill question either', () {
    for (final q in [
      'Hello!',
      'Tell me a joke.',
      'What is the capital of France?',
      'How long should I boil an egg?',
      "What's in this picture?",
      'How many cats are there?',
      '',
    ]) {
      expect(router.classify(q), isNull, reason: q);
    }
  });

  test('the route names the rule that fired (for the overlay)', () {
    expect(
      router.classify('Which accelerator is running right now?')?.rule,
      isNotEmpty,
    );
  });

  group('high-confidence actions run their intent directly', () {
    DirectAction? action(String q) => router.action(q);

    test('live device questions run device_info directly; its result is '
        'the reply', () {
      for (final q in [
        'What device am I on?',
        'Which accelerator is running?',
        'Which backends are you running on?',
        'What models are loaded and how much memory do you use?',
      ]) {
        final a = action(q);
        expect(a?.intent, 'device_info', reason: q);
        expect(a?.params, isEmpty, reason: q);
      }
      expect(action('What GPU does LiteRT use?'), isNull);
    });

    test('the time and date run current_time directly', () {
      for (final q in [
        'What time is it?',
        "What's the date today?",
        'What day of the week is it?',
        'Can you tell me the time, please?',
      ]) {
        expect(action(q)?.intent, 'current_time', reason: q);
        expect(action(q)?.params, isEmpty, reason: q);
      }
      expect(action('What time does the museum open?'), isNull);
    });

    test('negatives: no timers or watching any more', () {
      for (final q in [
        'Tell me when you see a dog',
        'Watch for a bottle',
        'Stop watching',
        'Set a timer for 10 seconds',
        'Tea timer for 3 minutes',
        'What is the input size of the YOLO 26 nano detector?',
      ]) {
        expect(action(q), isNull, reason: q);
      }
    });
  });
}
