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

import '../skills/app_intents.dart' as intents;
import 'question_router.dart' show normalizeQuestion;

/// Which live skill a Demo 1 question is about.
enum SkillTopic {
  /// The loaded models, their backends and accelerators, memory, hardware
  /// (`device-info`).
  deviceFacts,

  /// The time, date or day (`current-time`).
  time,
}

/// A request the app runs itself, without asking the model to call
/// the tool: [intent] with [params] (JSON-ready), [rule] for the logs.
final class const DirectAction(
  final String intent,
  final Map<String, Object?> params,
  final String rule,
);

/// A question routed to a skill; [rule] names what matched (overlay, logs).
final class const SkillRoute(final SkillTopic topic, final String rule) {
  @override
  String toString() => '${topic.name} ($rule)';
}

/// Recognizes the Demo 1 questions only a skill can answer, so
/// the turn skips retrieval and Gemma sees nothing but its skills.
///
/// Why: retrieval runs before every turn, and live device questions clear
/// the knowledge-base gate against the GPU and backend docs (0.40–0.56,
/// `test_assets/skill_trigger_similarity.json`); at temperature 0.6 Gemma
/// then sometimes answered "which accelerator is running" from the docs
/// instead of calling `device_info`. Rules, not similarity, so the outcome
/// is deterministic. Like Demo 3's `QuestionRouter`, it prefers a miss (the
/// question retrieves; the agent prompt still puts skills first) to a false
/// hit (a documentation question loses its excerpts).
///
/// Rules, in order:
/// 1. time: the whole question is a time/date question once polite filler
///    is dropped ("can you tell me what time it is please");
/// 2. device facts: a device term (accelerator, backend, GPU, memory, …)
///    plus a cue that it is about this app now ("are you", "right now",
///    "is running", "this device", "loaded"), and no documentation cue
///    ("how do I", "why", "should I", "my", "LiteRT", a phone model, …).
/// "What GPU does LiteRT use" retrieves; "which accelerator is running right
/// now" goes to the skill. About 0.1 ms.
final class SkillQuestionRouter {
  const SkillQuestionRouter();

  /// The requests the app runs itself: the time and live device facts.
  /// Null for everything else, which goes to the model as before.
  /// Deliberately narrow: a missed phrasing still works through the skill; a
  /// false hit would answer something nobody asked.
  DirectAction? action(String question) {
    final words = normalizeQuestion(question);
    if (words.isEmpty) return null;
    if (_timeCore(words) case final core?) {
      return DirectAction(
        intents.AppIntent.currentTime,
        const {},
        'time:$core',
      );
    }
    // Live device questions: device_info's text is the answer (the model
    // skipped the skill for "What device am I on?", and its phrasing of the
    // facts left out Gemma's GPU).
    if (classify(question) case SkillRoute(
      topic: SkillTopic.deviceFacts,
      :final rule,
    )) {
      return DirectAction(intents.AppIntent.deviceInfo, const {}, rule);
    }
    return null;
  }

  SkillRoute? classify(String question) {
    final words = normalizeQuestion(question);
    if (words.isEmpty) return null;
    final padded = ' ${words.join(' ')} ';
    bool has(String phrase) => padded.contains(' $phrase ');
    String? first(Iterable<String> phrases) {
      for (final phrase in phrases) {
        if (has(phrase)) return phrase;
      }
      return null;
    }

    if (_timeCore(words) case final core?) {
      return SkillRoute(SkillTopic.time, 'time:$core');
    }
    final term = first(_deviceTerms);
    final live = first(_liveCues);
    if (term != null && live != null && first(_deviceVetoes) == null) {
      return SkillRoute(SkillTopic.deviceFacts, 'device:$term+$live');
    }
    return null;
  }

  /// The question without leading filler and trailing neutral words, when
  /// what is left is exactly a time or date question.
  static String? _timeCore(List<String> words) {
    var start = 0;
    var end = words.length;
    while (start < end && _timePrefix.contains(words[start])) {
      start++;
    }
    while (end > start && _timeTail.contains(words[end - 1])) {
      end--;
    }
    final core = words.sublist(start, end).join(' ');
    return _timeQuestions.contains(core) ? core : null;
  }

  static const _timePrefix = {
    'hey',
    'hi',
    'ok',
    'okay',
    'so',
    'um',
    'uh',
    'well',
    'please',
    'can',
    'could',
    'would',
    'you',
    'tell',
    'me',
    'do',
    'know',
    'i',
    'want',
    'to',
    'just',
    'and',
    'quick',
    'question',
  };
  static const _timeTail = {
    'please',
    'now',
    'currently',
    'today',
    'exactly',
    'again',
    'here',
    'then',
    'for',
    'me',
  };
  static const _timeQuestions = {
    'time',
    'the time',
    'what time',
    'what time is it',
    'what time it is',
    'what is the time',
    'what is the current time',
    'what is the time and date',
    'what is the date and time',
    'the date',
    'the date and time',
    'what is the date',
    'what is the current date',
    'what date is it',
    'what date it is',
    'what is today s date',
    'what is todays date',
    'today s date',
    'todays date',
    'what day is it',
    'what day it is',
    'what day is today',
    'which day is it',
    'which day is today',
    'what is the day',
    'what day of the week is it',
    'what day of the week is today',
    'what day of the week it is',
    'which day of the week is it',
    'what year is it',
    'what month is it',
  };

  /// What device_info reports on. Multi-word entries are about the live
  /// app by themselves ("models are loaded").
  static const _deviceTerms = [
    'accelerator',
    'accelerators',
    'backend',
    'backends',
    'gpu',
    'cpu',
    'npu',
    'tpu',
    'hardware',
    'device',
    'memory',
    'ram',
    'processor',
    'processors',
    'cores',
    'chip',
    'chipset',
    'models are loaded',
    'model is loaded',
    'models loaded',
    'loaded models',
    'models are running',
    'models are you',
    'model are you running',
    'models do you',
    'running on',
  ];

  /// The question is about this app, now.
  static const _liveCues = [
    'now',
    'currently',
    'at the moment',
    'are you',
    'you are',
    'do you run',
    'do you use',
    'do you have',
    'your',
    'are we',
    'am i',
    'this app',
    'this device',
    'this phone',
    'this computer',
    'this mac',
    'this laptop',
    'this iphone',
    'loaded',
    'in use',
    'being used',
    'is running',
    'are running',
    'running',
    'using',
  ];

  /// How-to, why, recommendation, the user's own project, or a named
  /// product or phone: a documentation question.
  static const _deviceVetoes = [
    'how do',
    'how does',
    'how to',
    'how can',
    'how should',
    'how would',
    'why',
    'explain',
    'difference',
    'should',
    'can i',
    'could i',
    'my',
    'need',
    'needs',
    'require',
    'requires',
    'support',
    'supports',
    'supported',
    'recommend',
    'recommended',
    'skill',
    'skills',
    'litert',
    'tflite',
    'tensorflow',
    'flutter',
    'mediapipe',
    'pixel',
    'raspberry',
    'snapdragon',
    'samsung',
    'galaxy',
    'qualcomm',
    'mediatek',
    'jetson',
  ];
}
