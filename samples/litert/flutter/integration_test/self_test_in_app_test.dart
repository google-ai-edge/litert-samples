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

// The in-app self-test (Models › Run self-test) on the real app's
// dependencies, macOS first:
//
//   flutter test integration_test/self_test_in_app_test.dart -d macos \
//     --dart-define=GEMMA_MODEL_PATH=<path/to/gemma-4-E2B-it.litertlm>
//
// Setup as the app does it (GEMMA_MODEL_PATH fills the chat model slot when
// no chat model is chosen; nothing is chosen or saved here), then the run:
// every step passes; steps 7a/7b use the app's own EmbeddingGemma 2 (one
// embedder per process: borrowed, never closed) and the bundled spoken
// "What time is it?"; afterwards the app has its chat model back and the
// knowledge base still searches, typed and spoken. Prints the report and
// `SELFTEST_IN_APP passed=… cosine=… report=…`.

import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/dependencies.dart';
import 'package:litert_edge_demos/config/env.dart';
import 'package:litert_edge_demos/domain/models/knowledge.dart';
import 'package:litert_edge_demos/domain/models/knowledge_query.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/domain/models/self_test.dart';
import 'package:litert_edge_demos/selftest/self_test_runner.dart'
    show kSelfTestAudioQuestionAsset;
import 'package:litert_edge_demos/utils/result.dart';

import 'support/app_window.dart';

void main() {
  initIntegrationTest();

  testWidgets('the in-app self-test passes, with the app\'s own embedder for '
      'audio ↔ text, and leaves the app\'s models working', (tester) async {
    if (kGemmaModelPath.isEmpty) fail('Pass --dart-define=GEMMA_MODEL_PATH=…');
    final deps = await AppDependencies.create();
    addTearDown(deps.dispose);
    await tester.pumpWidget(const MaterialApp(home: SizedBox.shrink()));

    final setup = await tester.runAsync(deps.prepareModels);
    expect(setup, isA<Ok<void>>(), reason: '$setup');
    final states = deps.models.states;
    expect(states.value[ModelId.embeddingGemma], isA<ModelReady>());
    expect(states.value[ModelId.chat], isA<ModelReady>());
    // The knowledge base installs its index once the embedder is ready.
    await tester.runAsync(() async {
      final watch = Stopwatch()..start();
      while (deps.knowledge.status.value is! KnowledgeReady) {
        if (watch.elapsed > const Duration(minutes: 2)) {
          fail('the knowledge base: ${deps.knowledge.status.value}');
        }
        await Future<void>.delayed(const Duration(milliseconds: 100));
      }
    });

    final lines = <String>[];
    final outcome = await tester.runAsync(
      () => deps.selfTest.run(progress: lines.add),
    );
    final SelfTestOutcome result;
    switch (outcome) {
      case Ok(:final value):
        result = value;
      case final other:
        fail('The self-test could not run: $other');
    }
    debugPrint(result.text);
    expect(result.text, contains('PASS  4a  chat model load + warm-up (gpu)'));
    expect(result.text, contains('PASS  7a  embedder load (audio + text)'));
    expect(
      result.text,
      contains(
        'embeddinggemma-2-740m · cpu · audio+text · the app\'s, loaded '
        'before the run',
      ),
    );
    expect(result.text, contains('PASS  7b  embedder audio ↔ text'));
    final cosine = RegExp(r'cosine (\d\.\d+) \(≥').firstMatch(result.text);
    expect(cosine, isNotNull);
    expect(result.passed, isTrue, reason: result.text);

    // The app's models after the run: its chat model loaded again, its
    // embedder never closed (the knowledge base searches typed and spoken).
    expect(states.value[ModelId.chat], isA<ModelReady>());
    expect(states.value[ModelId.embeddingGemma], isA<ModelReady>());
    final typed = await tester.runAsync(
      () =>
          deps.knowledge.retrieve(const KnowledgeQuery.text('What is LiteRT?')),
    );
    expect(
      (typed as Ok<Retrieval>?)?.value.outcome,
      RetrievalOutcome.used,
      reason: '$typed',
    );
    final clip = await tester.runAsync(
      () => rootBundle.load(kSelfTestAudioQuestionAsset),
    );
    final spoken = await tester.runAsync(
      () => deps.knowledge.retrieve(
        KnowledgeQuery.audio(
          clip!.buffer.asUint8List(clip.offsetInBytes, clip.lengthInBytes),
        ),
      ),
    );
    expect(spoken, isA<Ok<Retrieval>>(), reason: '$spoken');
    debugPrint(
      'SELFTEST_IN_APP passed=${result.passed} cosine=${cosine!.group(1)} '
      'spoken_top=${(spoken! as Ok<Retrieval>).value.topSimilarity} '
      'report=${result.reportPath}',
    );
  }, timeout: const Timeout(Duration(minutes: 10)));
}
