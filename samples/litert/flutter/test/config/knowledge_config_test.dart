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

import 'package:flutter_edge_ai/flutter_edge_ai.dart' show TaskType;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/knowledge_config.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/config/voice_config.dart';
import 'package:litert_edge_demos/domain/models/knowledge_query.dart';
import 'package:litert_edge_demos/selftest/self_test_runner.dart'
    show kSelfTestAudioQuestionAsset;

void main() {
  group('similarity gates per query kind', () {
    test('text 0.70; audio below it (audio scores lower than its words)', () {
      expect(kKbMinSimilarityText, 0.70);
      expect(kKbMinSimilarityAudio, lessThan(kKbMinSimilarityText));
      expect(kKbMinSimilarityAudio, greaterThan(0.6));
    });

    test('kbMinSimilarityFor picks each kind\'s gate', () {
      expect(kbMinSimilarityFor(KnowledgeQueryKind.text), kKbMinSimilarityText);
      expect(
        kbMinSimilarityFor(KnowledgeQueryKind.audio),
        kKbMinSimilarityAudio,
      );
    });
  });

  group('question caps', () {
    test('the text cap is derived from the embedder\'s token limit: '
        'prefix + question bytes + BOS + EOS + a dummy-prefix piece fit', () {
      final prefix = utf8.encode(TaskType.retrievalQuery.prefix).length;
      expect(prefix, 29);
      expect(kEmbedderConfig.maxInputTokens, kEmbedderMaxInputTokens);
      // At most one token per UTF-8 byte (byte fallback), plus the specials.
      expect(
        prefix + kKbMaxTextQueryUtf8Bytes + 3,
        lessThanOrEqualTo(kEmbedderMaxInputTokens),
      );
      // Not needlessly short: within the 32 spare bytes of the bound.
      expect(
        kEmbedderMaxInputTokens - (prefix + kKbMaxTextQueryUtf8Bytes + 3),
        lessThanOrEqualTo(32),
      );
    });

    test('every voice capture fits the audio cap, and the cap fits the '
        'embedder\'s token limit at the audio token rate', () {
      // A capture ends at kVoiceConfig.maxUtterance (Whisper's window), so a
      // spoken question is never refused as too long to search.
      expect(kVoiceConfig.maxUtterance, lessThanOrEqualTo(kKbMaxAudioQuery));
      expect(
        kKbMaxAudioQuery.inSeconds * kAudioTokensPerSecond,
        lessThan(kEmbedderMaxInputTokens),
      );
    });

    test('the embedder warms its audio encoder up with the self-test clip, '
        'an asset the app ships', () {
      expect(kKbAudioWarmUpAsset, 'assets/selftest/what_time_is_it.wav');
      expect(kSelfTestAudioQuestionAsset, kKbAudioWarmUpAsset);
    });
  });
}
