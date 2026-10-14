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
import 'package:litert_edge_demos/selftest/self_test_runner.dart'
    show kSelfTestAudioQuestionAsset;
import 'package:litert_edge_demos/utils/wav.dart';

import '../../../integration_test/support/kb_golden_clips.dart';

void main() {
  final golden = parseKbGolden(
    jsonDecode(File('test_assets/kb_golden.json').readAsStringSync()),
  );

  test('the golden set: 19 on-topic, 12 off-topic questions and 12 lines of '
      'everyday chat', () {
    expect(golden.onTopic, hasLength(19));
    expect(golden.offTopic, hasLength(12));
    expect(golden.everydayChat, hasLength(12));
    expect(golden.onTopic.map((e) => e.doc), everyElement(endsWith('.md')));
    expect(
      {...golden.offTopic, ...golden.everydayChat},
      hasLength(24),
      reason: 'no line twice',
    );
  });

  test('the spoken golden set is not shipped (kb_retrieval_test speaks it '
      'into memory); the self-test clip is, as a 16 kHz mono PCM16 WAV '
      'within EmbeddingGemma 2\'s 30 s audio cap', () {
    final pubspec = File('pubspec.yaml').readAsStringSync();
    expect(pubspec, isNot(contains('kb_golden_audio')));
    expect(Directory('test_assets/kb_golden_audio').existsSync(), isFalse);
    expect(pubspec, contains('- assets/selftest/'));
    final wav = File(kSelfTestAudioQuestionAsset);
    expect(wav.existsSync(), isTrue);
    final duration = wavDuration(wav.readAsBytesSync());
    expect(duration, greaterThan(const Duration(seconds: 1)));
    expect(duration, lessThanOrEqualTo(const Duration(seconds: 30)));
  });

  test('a golden file of another shape is a FormatException', () {
    expect(
      () => parseKbGolden({'on_topic': <Object?>[]}),
      throwsFormatException,
    );
    expect(
      () => parseKbGolden({'on_topic': <Object?>[], 'off_topic': <Object?>[]}),
      throwsFormatException,
      reason: 'no everyday chat',
    );
    expect(
      () => parseKbGolden({
        'on_topic': [
          {'q': 'x'},
        ],
        'off_topic': <Object?>[],
        'everyday_chat': <Object?>[],
      }),
      throwsFormatException,
    );
    expect(
      () => parseKbGolden({
        'on_topic': <Object?>[],
        'off_topic': <Object?>[],
        'everyday_chat': [1],
      }),
      throwsFormatException,
    );
  });
}
