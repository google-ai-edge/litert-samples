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

// Gemma's audio path on a real target, without the UI (it pumps no frames,
// so it also runs while the screen is locked): the app's LlmService and
// conversation repository over Gemma 4 E2B with its audio encoder.
//
//   flutter test integration_test/gemma_audio_test.dart -d macos \
//     --dart-define=GEMMA_MODEL_PATH=<path/to/gemma-4-E2B-it.litertlm>
//
// 1. Resource measurements (printed for the spec's resource table, not a
//    gate: nothing is asserted on them; the spec sets no limit for the chat
//    model's audio encoder): the process's RSS before and after
//    LlmService.load with audio off, then with audio on, then off again
//    (`LLM_RSS audio=off|on|off(2) …`; the first load pays the GPU runtime's
//    one-time costs), and each warm-up's time.
// 2. The audio warm-up decision: the first and second audio turn's time to
//    first token on an engine warmed up without audio, then the first one on
//    an engine whose warm-up also sent a 1 s WAV (`AUDIO_TTFT …`), and what
//    that warm-up cost.
// 3. Gemma hears the question: test_assets/france_16k.pcm as a WAV with the
//    spoken question's prompt → a reply containing "paris".
// 4. A stopped audio turn on a fresh chat, stopped at its first words: on the
//    GPU flutter_edge_ai_litertlm rewinds it in place as the next turn starts
//    (the plugin's log line), so the recording stays in the conversation —
//    the follow-up can quote the spoken question, which no text holds — and
//    no rebuild from the text history runs.

import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_edge_ai/flutter_edge_ai.dart'
    show ModelType, PreferredBackend;
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:litert_edge_demos/config/bootstrap.dart';
import 'package:litert_edge_demos/config/env.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/repositories/conversation_repository_edge_ai.dart';
import 'package:litert_edge_demos/data/services/llm/llm_service.dart';
import 'package:litert_edge_demos/data/services/model_store/local_files.dart';
import 'package:litert_edge_demos/domain/models/assistant_event.dart';
import 'package:litert_edge_demos/domain/models/chat_model_config.dart';
import 'package:litert_edge_demos/domain/models/conversation_history.dart';
import 'package:litert_edge_demos/domain/use_cases/prompt_builder.dart';
import 'package:litert_edge_demos/utils/result.dart';
import 'package:litert_edge_demos/utils/wav.dart';

/// Gemma 4 E2B with both towers, or without the audio one to compare: built
/// by hand, this test drives the engine below the app's plan. The engine
/// places the towers (both on the CPU for this file).
ChatModelConfig _gemma({required bool audio}) => ChatModelConfig(
  name: 'Gemma 4 E2B',
  modelType: ModelType.gemma4,
  llm: LlmConfig(
    maxTokens: kLlmConfig.maxTokens,
    backend: PreferredBackend.gpu,
    maxNumImages: 1,
    supportImage: true,
    supportAudio: audio,
  ),
  tools: false,
);

String _mb(int? bytes) =>
    bytes == null ? '–' : '${(bytes / (1 << 20)).round()}';

/// One ask, collected: the reply text and its metrics.
Future<({String text, GenerationMetrics? metrics, Object? error})> _ask(
  EdgeAiConversationRepository repo,
  String prompt, {
  Uint8List? audioWav,
  int? turn,
  String? historyText,
}) async {
  final text = StringBuffer();
  GenerationMetrics? metrics;
  Object? error;
  await for (final event in repo.ask(
    prompt,
    audioWav: audioWav,
    turn: turn,
    historyText: historyText,
  )) {
    switch (event) {
      case AssistantTextDelta(:final text):
        textBuffer(text);
      case AssistantDone(metrics: final m):
        metrics = m;
      case AssistantFailed(error: final e):
        error = e;
      case AssistantContextReset():
        break;
    }
  }
  return (text: text.toString(), metrics: metrics, error: error);
}

/// Where [_ask] writes its deltas (one ask at a time).
late void Function(String text) textBuffer;

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('Gemma hears the question (RSS and first audio TTFT measured), '
      'and a stopped audio turn keeps its recording (rewound in place)', (
    tester,
  ) async {
    if (kGemmaModelPath.isEmpty) {
      fail('Pass --dart-define=GEMMA_MODEL_PATH=<path to the .litertlm>');
    }
    final logged = <String>[];
    final printer = debugPrint;
    debugPrint = (String? message, {int? wrapWidth}) {
      if (message != null) logged.add(message);
      printer(message, wrapWidth: wrapWidth);
    };
    addTearDown(() => debugPrint = printer);

    final pcm = (await rootBundle.load('test_assets/france_16k.pcm')).buffer
        .asUint8List();
    final france = wavFromPcm16(pcm, sampleRate: 16000);
    final path = await resolveLocalPath(kGemmaModelPath);
    expect(File(path).existsSync(), isTrue, reason: path);
    await initEdgeAi();
    final llm = LlmService();
    addTearDown(llm.close);
    expect(
      await llm.install(path: path, onProgress: (_) {}),
      isA<Ok<String>>(),
    );

    Future<LlmInfo> load({required bool audio}) async {
      final loaded = await llm.load(_gemma(audio: audio));
      return switch (loaded) {
        Ok(:final value) => value,
        Error(:final error) => fail('load (audio=$audio): $error'),
      };
    }

    Future<Duration> warmUp({required bool audio}) async {
      final warm = await llm.warmUp(
        kSampler,
        withImage: true,
        withAudio: audio,
      );
      return switch (warm) {
        Ok(:final value) => value,
        Error(:final error) => fail('warm-up (audio=$audio): $error'),
      };
    }

    // 1. Audio off.
    final off = await load(audio: false);
    final warmOff = await warmUp(audio: false);
    debugPrint(
      'LLM_RSS audio=off before=${_mb(off.rssBeforeBytes)}MB '
      'after=${_mb(off.rssAfterBytes)}MB '
      'delta=${_mb((off.rssAfterBytes ?? 0) - (off.rssBeforeBytes ?? 0))}MB '
      'load=${off.loadTime.inMilliseconds}ms warm=${warmOff.inMilliseconds}ms',
    );
    await llm.unload();

    // 2. Audio on, warmed up without audio: two audio turns.
    final on = await load(audio: true);
    final warmOn = await warmUp(audio: false);
    debugPrint(
      'LLM_RSS audio=on before=${_mb(on.rssBeforeBytes)}MB '
      'after=${_mb(on.rssAfterBytes)}MB '
      'delta=${_mb((on.rssAfterBytes ?? 0) - (on.rssBeforeBytes ?? 0))}MB '
      'load=${on.loadTime.inMilliseconds}ms warm=${warmOn.inMilliseconds}ms',
    );
    var repo = EdgeAiConversationRepository(llm: llm);
    expect(await repo.open(kVoiceChatProfile), isA<Ok<void>>());
    final ttfts = <int?>[];
    for (var i = 1; i <= 2; i++) {
      final reply = StringBuffer();
      textBuffer = reply.write;
      final watch = Stopwatch()..start();
      final (:text, :metrics, :error) = await _ask(
        repo,
        PromptBuilder.spokenQuestion,
        audioWav: france,
        turn: i,
        historyText: kSpokenQuestionPlaceholder,
      );
      expect(error, isNull);
      expect(reply.toString().toLowerCase(), contains('paris'));
      ttfts.add(metrics?.timeToFirstToken?.inMilliseconds);
      debugPrint(
        'AUDIO_TURN warmup=no turn=$i ttft=${metrics?.timeToFirstToken?.inMilliseconds}ms '
        'total=${watch.elapsedMilliseconds}ms prefill=${metrics?.prefillTokens} '
        'reply="${reply.toString().trim()}"',
      );
      expect(text, isEmpty, reason: 'the deltas went to the buffer');
    }

    // 4. A stopped audio turn on a fresh chat (no earlier turn names the
    //    city), stopped at its first words, then a follow-up only the
    //    recording can answer.
    expect(await repo.reset(ifCurrent: kVoiceChatProfile), isA<Ok<void>>());
    final story = StringBuffer();
    textBuffer = story.write;
    final stoppedTurn = _ask(
      repo,
      'Answer the question asked in this audio, then tell a 300-word story '
      'about that city.',
      audioWav: france,
      turn: 3,
      historyText: kSpokenQuestionPlaceholder,
    );
    final waitText = Stopwatch()..start();
    while (story.isEmpty && waitText.elapsed < const Duration(seconds: 30)) {
      await Future<void>.delayed(const Duration(milliseconds: 5));
    }
    final stopWatch = Stopwatch()..start();
    await repo.stop();
    final stoppedResult = await stoppedTurn;
    debugPrint(
      'AUDIO_STOP chars=${story.length} "$story" '
      'stop=${stopWatch.elapsedMilliseconds}ms '
      'stopped=${stoppedResult.metrics?.stopped}',
    );
    expect(stoppedResult.metrics?.stopped, isTrue);
    expect(
      story.toString().toLowerCase(),
      isNot(contains('france')),
      reason: 'the partial reply must not give the question away',
    );
    final follow = StringBuffer();
    textBuffer = follow.write;
    final followed = await _ask(
      repo,
      'Quote, word for word, the question I asked in my audio recording.',
      historyText: 'Quote the question I asked.',
    );
    expect(followed.error, isNull);
    expect(
      logged.where((l) => l.contains('rebuilt the chat from the text history')),
      isEmpty,
    );
    const rewound =
        '[FfiInferenceModel] rewound the conversation after a stopped turn';
    expect(logged.where((l) => l == rewound), hasLength(1));
    expect(followed.metrics?.contextReset, isFalse);
    debugPrint(
      'AUDIO_REWIND ttft=${followed.metrics?.timeToFirstToken?.inMilliseconds}ms '
      'ctx=${followed.metrics?.contextTokens} '
      'reply="${follow.toString().trim()}" history=${repo.history}',
    );
    expect(follow.toString().toLowerCase(), contains('france'));
    await repo.close();
    await llm.unload();

    // 2b. Audio on, warmed up with audio: the first audio turn.
    await load(audio: true);
    final warmAudio = await warmUp(audio: true);
    repo = EdgeAiConversationRepository(llm: llm);
    expect(await repo.open(kVoiceChatProfile), isA<Ok<void>>());
    final reply = StringBuffer();
    textBuffer = reply.write;
    final first = await _ask(
      repo,
      PromptBuilder.spokenQuestion,
      audioWav: france,
      turn: 1,
      historyText: kSpokenQuestionPlaceholder,
    );
    expect(first.error, isNull);
    expect(reply.toString().toLowerCase(), contains('paris'));
    debugPrint(
      'AUDIO_TTFT warmup=no first=${ttfts[0]}ms second=${ttfts[1]}ms '
      'warm_up=${warmOn.inMilliseconds}ms | warmup=audio '
      'first=${first.metrics?.timeToFirstToken?.inMilliseconds}ms '
      'warm_up=${warmAudio.inMilliseconds}ms',
    );
    await repo.close();
    await llm.unload();

    // 1b. Audio off again: the first load in the process pays one-time costs
    //     (the GPU runtime), so the audio on/off comparison is this one.
    final offAgain = await load(audio: false);
    final warmOffAgain = await warmUp(audio: false);
    debugPrint(
      'LLM_RSS audio=off(2) before=${_mb(offAgain.rssBeforeBytes)}MB '
      'after=${_mb(offAgain.rssAfterBytes)}MB '
      'delta=${_mb((offAgain.rssAfterBytes ?? 0) - (offAgain.rssBeforeBytes ?? 0))}MB '
      'load=${offAgain.loadTime.inMilliseconds}ms '
      'warm=${warmOffAgain.inMilliseconds}ms',
    );
    debugPrint = printer;
  }, timeout: const Timeout(Duration(minutes: 15)));
}
