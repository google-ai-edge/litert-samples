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

// Demo 1's voice loop on the real app (macOS): Gemma 4 E2B on the GPU,
// Whisper base int8 and Inflect-nano-v2 on the CPU, soloud playback, in both
// modes: the Whisper path (GEMMA_MODEL_PATH, no audio tower loaded: Voice →
// Whisper → Gemma) and audio mode (the same file as your own model, its audio
// tower loaded as its header has it: Voice → Gemma, the model hears the
// question).
//
//   flutter test integration_test/voice_loop_test.dart -d macos \
//     --dart-define=GEMMA_MODEL_PATH=<path/to/gemma-4-E2B-it.litertlm>
//
// The STT and TTS models are built into the app (tool/fetch_models.sh). The
// saved chat model choice is cleared for the run (GEMMA_MODEL_PATH loads with
// audio off) and put back at the end.
//
// 0. The barge-in question of the audio run ("Explain in detail how a
//    lighthouse works, in about 300 words.") spoken by the app's own TTS
//    into memory, before the app starts.
// 1. setup → home → Demo 1; the mode line says Voice → Whisper → Gemma.
// 2. Mic bypassed: test_assets/france_16k.pcm ("What is the capital of
//    France?") through the view model's PCM entry point — the one mic release
//    uses. Asserts "france" in the transcript, "paris" in the reply, and a
//    first audio chunk at 24 kHz. Prints `VOICE stt=… ttft=… ttfa=… tts1=…`.
// 3. Barge-in on a long typed reply while it is speaking and Gemma is still
//    generating, through the real mic button and a fixture mic that streams
//    the same audio in real time (no TCC prompt). Asserts the playback was
//    silenced ≤150 ms after the press, the mic opened and the generation was
//    stopped; the release runs a follow-up voice turn.
//    Prints `BARGE silenced=… stop_confirmed=… drain=… press_to_return=…`.
// 4. Gate: a silent and an all-zero capture make no LLM call.
// 5. Real mic probe, only if the app is already authorized (never prompts):
//    1.5 s of capture, `MIC real …` with its loudness and all-zero check.
// 6. Audio mode: back home, GEMMA_MODEL_PATH as a local file, its audio
//    tower on as its header sets it (the Models screen's flow:
//    useLocalFile, apply, reload), Demo 1 again; the mode line says
//    Voice → Gemma. Prints `LLM_RSS` for both loads (step 1's
//    GEMMA_MODEL_PATH load is the audio-off one: kLlmConfig has no audio
//    tower).
// 7. The France fixture as audio, twice: Gemma hears it (no STT), the reply
//    contains "paris"; the user entry is "🎤 2.1 s" until the transcript job
//    (after the playback drained) fills it with words containing "france",
//    and the conversation's text history gets them. Prints `AUDIO …` per
//    turn (first-turn TTFT vs the second's: the audio warm-up decision).
// 8. Barge-in in audio mode: the spoken lighthouse question, a press while it is
//    speaking and Gemma generating: silent ≤150 ms, generation stopped; the
//    follow-up audio turn (the fixture mic plays the France question) is
//    answered with "paris", on a chat rebuilt from the text history.
//    Prints `STORY_PARTIAL …` (the reply so far, before the press) and
//    `AUDIO_BARGE …`.
// (Audio is no longer a setting: a file with an audio tower always loads it,
// so there is no "audio off with the same file" step; step 1 measured the
// audio-off RSS.)

import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter/services.dart';
import 'package:flutter_edge_ai/flutter_edge_ai.dart'
    show ModelType, PreferredBackend;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/app.dart';
import 'package:litert_edge_demos/config/bootstrap.dart';
import 'package:litert_edge_demos/config/demos.dart';
import 'package:litert_edge_demos/config/dependencies.dart';
import 'package:litert_edge_demos/config/env.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/domain/models/chat_entry.dart';
import 'package:litert_edge_demos/domain/models/chat_model.dart';
import 'package:litert_edge_demos/domain/models/conversation_history.dart';
import 'package:litert_edge_demos/domain/models/knowledge_query.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/domain/models/voice.dart';
import 'package:litert_edge_demos/ui/features/home/views/home_screen.dart';
import 'package:litert_edge_demos/ui/features/voice_chat/view_models/voice_chat_view_model.dart';
import 'package:litert_edge_demos/ui/features/voice_chat/views/chat_keys.dart';
import 'package:litert_edge_demos/ui/features/voice_chat/views/voice_chat_screen.dart';
import 'package:litert_edge_demos/utils/pcm.dart';
import 'package:litert_edge_demos/utils/result.dart';
import 'package:provider/provider.dart';
import 'package:record/record.dart';

import 'support/app_window.dart';
import 'support/chat_model_settings.dart';
import 'support/fixtures.dart';
import 'support/pump.dart';
import 'support/question_audio.dart';

final _screenKey = GlobalKey();

Future<String> saveScreenshot(String name) async {
  final boundary =
      _screenKey.currentContext?.findRenderObject() as RenderRepaintBoundary?;
  if (boundary == null) fail('No RepaintBoundary to capture');
  final image = await boundary.toImage(pixelRatio: 2);
  final png = await image.toByteData(format: ui.ImageByteFormat.png);
  image.dispose();
  if (png == null) fail('PNG encoding failed');
  final file = File('${Directory.systemTemp.path}/$name.png');
  await file.writeAsBytes(png.buffer.asUint8List());
  return file.path;
}

VoiceChatViewModel chatViewModel(WidgetTester tester) =>
    Provider.of(tester.element(find.byType(VoiceChatScreen)), listen: false);

String describeChat(VoiceChatViewModel vm) =>
    'phase=${vm.phase.name} error=${vm.error} entries='
    '${vm.entries.map((e) => '${e.role.name}:"${e.text}"${e.interrupted ? '(i)' : ''}').join(' | ')}';

String ms(Duration? d) => d == null ? '–' : '${d.inMilliseconds}';

String mb(int? bytes) => bytes == null ? '–' : '${(bytes / (1 << 20)).round()}';

/// The chat model row's RSS figures: before and after the engine was built.
String rssOf(ModelState? state) => switch (state) {
  ModelReady(:final info) =>
    'before=${mb(info.chat?.rssBeforeBytes)}MB '
        'after=${mb(info.chat?.rssAfterBytes)}MB '
        'delta=${mb((info.chat?.rssAfterBytes ?? 0) - (info.chat?.rssBeforeBytes ?? 0))}MB '
        'load=${info.loadTime.inMilliseconds}ms '
        'warm=${info.warmUpTime.inMilliseconds}ms '
        'audio=${info.chat?.audio}',
  final other => '$other',
};

/// Taps Demo 1's tile and waits for its chat to open.
Future<VoiceChatViewModel> enterVoiceChat(WidgetTester tester) async {
  final tile = find.byKey(HomeKeys.tile(Demo.voiceChat));
  await pumpUntil(
    tester,
    () => tile.evaluate().isNotEmpty && tester.widget<ListTile>(tile).enabled,
    timeout: const Duration(seconds: 30),
    reason: 'the Demo 1 tile to be enabled',
  );
  await tester.tap(tile);
  await pumpUntil(
    tester,
    () => find.byType(VoiceChatScreen).evaluate().length == 1,
    timeout: const Duration(seconds: 5),
    reason: 'the chat screen to open',
  );
  final vm = chatViewModel(tester);
  await pumpUntil(
    tester,
    () => vm.isReady,
    timeout: const Duration(seconds: 30),
    reason: 'the chat to open',
    describe: () => describeChat(vm),
  );
  await tester.pump(const Duration(milliseconds: 400));
  return vm;
}

/// Back from Demo 1 to the home screen (the demo's view model is disposed).
Future<void> leaveVoiceChat(WidgetTester tester) async {
  Navigator.of(tester.element(find.byType(VoiceChatScreen))).pop();
  await pumpUntil(
    tester,
    () =>
        find.byType(VoiceChatScreen).evaluate().isEmpty &&
        find.byType(HomeScreen).evaluate().isNotEmpty,
    timeout: const Duration(seconds: 10),
    reason: 'the home screen',
  );
  await tester.pump(const Duration(milliseconds: 400));
}

/// Holds a press while [mic] plays [pcm] (in real time from the capture
/// start) until it has sent every byte, plus a moment for the capture to take
/// the last chunk, so the release hands the whole question over however the
/// timer was delayed.
Future<void> holdUntilPlayed(
  WidgetTester tester,
  FixtureMicService mic,
  Uint8List pcm,
) async {
  await pumpUntil(
    tester,
    () => mic.played,
    timeout: pcm16Duration(pcm.length, 16000) + const Duration(seconds: 3),
    reason: 'the fixture mic to play the whole question',
  );
  await pumpFor(tester, const Duration(milliseconds: 300));
}

/// Feeds [pcm] through the view model's PCM entry point and waits for the
/// turn's end.
Future<TurnResult> voiceTurn(
  WidgetTester tester,
  VoiceChatViewModel vm,
  Uint8List pcm,
) async {
  final turn = vm.submitUtterance(
    Utterance(pcm: pcm, held: pcm16Duration(pcm.length, 16000)),
  );
  TurnResult? result;
  unawaited(turn.then((r) => result = r));
  await pumpUntil(
    tester,
    () => result != null,
    timeout: const Duration(seconds: 90),
    reason: 'the voice turn to finish',
    describe: () => describeChat(vm),
  );
  return result!;
}

void main() {
  initIntegrationTest();

  testWidgets('voice loop: STT → Gemma → streamed TTS, barge-in, gate', (
    tester,
  ) async {
    if (kGemmaModelPath.isEmpty) {
      fail('Pass --dart-define=GEMMA_MODEL_PATH=<path to the .litertlm>');
    }
    final fixture = (await rootBundle.load('test_assets/france_16k.pcm')).buffer
        .asUint8List();
    expect(
      fixture.length,
      66604,
      reason: 'the clip as tool/make_question_audio.sh writes it',
    );
    // debugPrint is captured for the lines that prove the audio run's
    // rebuild (restored before the test ends, as flutter_test requires).
    final logged = <String>[];
    final printer = debugPrint;
    debugPrint = (String? message, {int? wrapWidth}) {
      if (message != null) logged.add(message);
      printer(message, wrapWidth: wrapWidth);
    };
    addTearDown(() => debugPrint = printer);
    final savedChatModel = await takeChatModelSettings();
    addTearDown(() => restoreChatModelSettings(savedChatModel));

    // 0. The audio run's barge-in question, spoken by the app's own TTS.
    await initEdgeAi();
    final tts = await loadAppTts();
    // A question, not a request for a story: asked aloud in audio mode,
    // Gemma 4 E2B tends to refuse "tell me a story" ("I don't have a story
    // about…"), and once refused it repeats the refusal from the history;
    // this step tests the barge-in, which needs a long reply.
    final story = await speakQuestion(
      tts,
      'Explain in detail how a lighthouse works, in about 300 words.',
      window: kWhisperSttConfig.window,
    );
    await tts.close();
    debugPrint(
      'STORY clip ${story.duration.inMilliseconds}ms '
      '(tts ${story.ttsTime.inMilliseconds}ms)',
    );

    final mic = FixtureMicService(fixture);
    final deps = await AppDependencies.create(mic: mic);
    await tester.pumpWidget(
      RepaintBoundary(
        key: _screenKey,
        child: App(dependencies: deps),
      ),
    );

    // 1. Setup (downloads STT/TTS on the first run), home, Demo 1.
    final setupWatch = Stopwatch()..start();
    await pumpUntil(
      tester,
      () => find.byType(HomeScreen).evaluate().isNotEmpty,
      timeout: const Duration(minutes: 12),
      reason: 'model setup',
      onPoll: () {
        for (final MapEntry(:key, :value) in deps.models.states.value.entries) {
          if (value case ModelFailed(:final message)) {
            fail('${key.spec.displayName} failed: $message');
          }
        }
      },
    );
    final states = deps.models.states.value;
    String row(ModelId id) => switch (states[id]) {
      ModelReady(:final info) =>
        '${info.modelId} ${info.detail ?? info.backend} '
            'load=${info.loadTime.inMilliseconds}ms '
            'warm=${info.warmUpTime.inMilliseconds}ms',
      final other => '$other',
    };
    debugPrint(
      'SETUP ${setupWatch.elapsedMilliseconds}ms | '
      'llm ${row(ModelId.chat)} | stt ${row(ModelId.whisperBase)} | '
      'tts ${row(ModelId.inflectNano)}',
    );
    expect(states[ModelId.chat], isA<ModelReady>());
    expect(states[ModelId.whisperBase], isA<ModelReady>());
    expect(states[ModelId.inflectNano], isA<ModelReady>());
    final rssAudioOffDefine = rssOf(states[ModelId.chat]);
    debugPrint('LLM_RSS load=GEMMA_MODEL_PATH $rssAudioOffDefine');
    expect(
      deps.conversation.capabilities.audio,
      isFalse,
      reason: 'GEMMA_MODEL_PATH loads with audio off',
    );
    await tester.pump(const Duration(milliseconds: 400));

    final vm = await enterVoiceChat(tester);
    expect(vm.speakReplies, isTrue);
    expect(vm.modeLine, 'Voice → Whisper → ${kDefineChatModel.name}');
    expect(find.text(vm.modeLine), findsOneWidget);

    // 2. The fixture through the PCM entry point (mic bypassed).
    final turn = vm.submitUtterance(
      Utterance(pcm: fixture, held: pcm16Duration(fixture.length, 16000)),
    );
    TurnResult? result;
    unawaited(turn.then((r) => result = r));
    await pumpUntil(
      tester,
      () => result != null,
      timeout: const Duration(seconds: 90),
      reason: 'the voice turn to finish',
      describe: () => describeChat(vm),
    );
    expect(result!.outcome, TurnOutcome.completed, reason: describeChat(vm));
    final user = vm.entries.lastWhere((e) => e.role == ChatRole.user);
    final reply = vm.entries.last;
    expect(user.spoken, isNull, reason: 'the Whisper path: words first');
    expect(user.text.toLowerCase(), contains('france'));
    expect(reply.role, ChatRole.assistant);
    expect(reply.text.toLowerCase(), contains('paris'));
    final voice = deps.diagnostics.latest.lastVoiceTurn!;
    final gen = deps.diagnostics.latest.lastGeneration!;
    expect(voice.firstAudio, isNotNull, reason: 'a first audio chunk arrived');
    expect(voice.sampleRate, 24000);
    expect(voice.stt, isNotNull);
    debugPrint(
      'VOICE stt=${ms(voice.stt)}ms ttft=${ms(gen.timeToFirstToken)}ms '
      'first_text=${ms(voice.firstText)}ms ttfa=${ms(voice.firstAudio)}ms '
      'tts1=${voice.ttsClauses.isEmpty ? '–' : voice.ttsClauses.first.inMilliseconds}ms '
      'tts=${voice.ttsClauses.map((d) => d.inMilliseconds).toList()} '
      'clauses=${voice.ttsClauses.length} rate=${voice.sampleRate} '
      'total=${ms(voice.total)}ms capture_close=${ms(voice.captureClose)} '
      'transcript="${user.text}" reply="${reply.text}"',
    );
    await tester.pump(const Duration(milliseconds: 400));
    debugPrint('SCREENSHOT ${await saveScreenshot('voice_loop')}');

    // 3. Barge-in while a long typed reply is speaking.
    await tester.enterText(
      find.byKey(ChatKeys.input),
      'Tell me a story about a lighthouse keeper in about 300 words.',
    );
    await tester.pump();
    await tester.tap(find.byKey(ChatKeys.send));
    await pumpUntil(
      tester,
      () => vm.phase == TurnPhase.speaking,
      timeout: const Duration(seconds: 60),
      reason: 'the long reply to start speaking',
      describe: () => describeChat(vm),
    );
    // Press while Gemma is still generating (~400 tokens at ~80 tok/s), so
    // the barge-in has to stop the generation, not just the audio.
    await pumpFor(tester, const Duration(milliseconds: 300));
    expect(vm.phase, TurnPhase.speaking, reason: 'still speaking');
    expect(
      deps.conversation.isGenerating.value,
      isTrue,
      reason: 'still generating',
    );
    final entriesBefore = vm.entries.length;

    final pressWatch = Stopwatch()..start();
    final gesture = await tester.startGesture(
      tester.getCenter(find.byKey(ChatKeys.mic)),
    );
    final pressToReturn = pressWatch.elapsed;
    final bargeIn = deps.diagnostics.latest.lastBargeIn;
    expect(bargeIn, isNotNull);
    expect(bargeIn!.wasPlaying, isTrue);
    expect(bargeIn.silenced, isNotNull);
    expect(
      bargeIn.silenced!,
      lessThanOrEqualTo(const Duration(milliseconds: 150)),
      reason: 'acceptance: silent ≤150 ms after the press',
    );
    await pumpUntil(
      tester,
      () => vm.isListening,
      timeout: const Duration(seconds: 3),
      reason: 'the barge-in to open the mic',
      describe: () => describeChat(vm),
    );
    expect(
      mic.starts,
      greaterThanOrEqualTo(1),
      reason: 'Listening is shown only once the capture runs',
    );
    final interrupted = vm.entries[entriesBefore];
    expect(interrupted.role, ChatRole.assistant);
    expect(interrupted.interrupted, isTrue);

    // Hold while the fixture mic plays the question (66 604 B, 2.08 s, in
    // real time from the capture start) until every byte was sent, then
    // release.
    await holdUntilPlayed(tester, mic, fixture);
    await gesture.up();
    await pumpUntil(
      tester,
      () => deps.diagnostics.latest.lastBargeIn?.interruptDone != null,
      timeout: const Duration(seconds: 12),
      reason: 'the interrupted turn to drain',
    );
    final drained = deps.diagnostics.latest.lastBargeIn!;
    // The detached turn's own end: stopped, with its stop latency.
    final stoppedGen = deps.diagnostics.latest.lastGeneration!;
    expect(stoppedGen.stopped, isTrue, reason: 'the barge-in stopped Gemma');
    await pumpUntil(
      tester,
      () => vm.phase == TurnPhase.idle || vm.phase == TurnPhase.error,
      timeout: const Duration(seconds: 90),
      reason: 'the follow-up voice turn',
      describe: () => describeChat(vm),
    );
    expect(vm.phase, TurnPhase.idle, reason: describeChat(vm));
    final followUser = vm.entries.lastWhere((e) => e.role == ChatRole.user);
    expect(followUser.text.toLowerCase(), contains('france'));
    expect(vm.entries.last.text.toLowerCase(), contains('paris'));
    final follow = deps.diagnostics.latest.lastVoiceTurn!;
    debugPrint(
      'BARGE silenced=${ms(bargeIn.silenced)}ms '
      'stop_confirmed=${ms(drained.stopConfirmed)}ms '
      'drain=${ms(drained.interruptDone)}ms '
      'press_to_return=${pressToReturn.inMicroseconds / 1000}ms '
      'interrupted_chars=${interrupted.text.length} '
      'gen_stopped=${stoppedGen.stopped} '
      'gen_stop_latency=${ms(stoppedGen.stopLatency)}ms '
      'gen_chunks=${stoppedGen.chunks} | '
      'FOLLOWUP capture_close=${ms(follow.captureClose)}ms '
      'stt=${ms(follow.stt)}ms ttfa=${ms(follow.firstAudio)}ms '
      'transcript="${followUser.text}" reply="${vm.entries.last.text}"',
    );

    // 4. The gate: no LLM call for silence or digital zeros.
    final generationBefore = deps.diagnostics.latest.lastGeneration;
    final silent = await vm.submitUtterance(
      Utterance(
        pcm: Uint8List.fromList(
          List.generate(32000, (i) => i.isEven ? (i ~/ 2) % 3 : 0),
        ),
        held: const Duration(seconds: 1),
      ),
    );
    expect(silent.outcome, TurnOutcome.notHeard);
    expect(vm.entries.last.role, ChatRole.notice);
    final zeros = await vm.submitUtterance(
      Utterance(pcm: Uint8List(32000), held: const Duration(seconds: 1)),
    );
    expect(zeros.outcome, TurnOutcome.micUnavailable);
    expect(vm.entries.last.role, ChatRole.error);
    expect(vm.entries.last.text, contains('Microphone access is off'));
    expect(
      identical(deps.diagnostics.latest.lastGeneration, generationBefore),
      isTrue,
      reason: 'no generation ran for either',
    );
    debugPrint(
      'GATE silent=${silent.outcome.name} zeros=${zeros.outcome.name} '
      'notice="${vm.entries[vm.entries.length - 2].text}"',
    );

    // 5. Real mic probe, only when already authorized.
    final probe = AudioRecorder();
    try {
      final authorized = await probe.hasPermission(request: false);
      if (!authorized) {
        debugPrint(
          'MIC real authorized=false (not granted to this test app; skipped, '
          'nothing was prompted)',
        );
      } else {
        final bytes = BytesBuilder();
        final stream = await probe.startStream(
          const RecordConfig(
            encoder: AudioEncoder.pcm16bits,
            sampleRate: 16000,
            numChannels: 1,
          ),
        );
        final sub = stream.listen(bytes.add);
        await pumpFor(tester, const Duration(milliseconds: 1500));
        await probe.stop();
        await sub.cancel();
        final pcm = bytes.takeBytes();
        final stats = analyzePcm16(pcm);
        debugPrint(
          'MIC real authorized=true bytes=${pcm.length} '
          'allZero=${stats.allZero} '
          'peak=${stats.peakFrameDbfs.toStringAsFixed(1)}dBFS',
        );
      }
    } finally {
      await probe.dispose();
    }

    // 6. Audio mode: the same file in place, its audio tower on (the Models
    //    screen's flow), then Demo 1 again.
    await leaveVoiceChat(tester);
    final local = await deps.chatModels.useLocalFile(kGemmaModelPath);
    final CustomChatModel adopted;
    switch (local) {
      case Ok(:final value):
        adopted = value;
      case Error(:final error):
        fail('GEMMA_MODEL_PATH as a local file: $error');
    }
    expect(
      adopted.towersOf(deps.chatModels.state.value.header!).supportAudio,
      isTrue,
      reason: 'its header has audio',
    );
    Future<void> applyAndReload(CustomChatModel model, String label) async {
      expect(await deps.chatModels.apply(model), isA<Ok<void>>());
      final reloadWatch = Stopwatch()..start();
      final reloaded = await deps.chatModelSwitcher.reload();
      expect(reloaded, isA<Ok<void>>(), reason: '$reloaded');
      await pumpUntil(
        tester,
        () => deps.models.states.value[ModelId.chat] is ModelReady,
        timeout: const Duration(minutes: 3),
        reason: 'the chat model to load ($label)',
      );
      debugPrint(
        'LLM_RSS load=$label reload=${reloadWatch.elapsedMilliseconds}ms '
        '${rssOf(deps.models.states.value[ModelId.chat])}',
      );
    }

    final gemma = adopted.copyWith(
      displayName: kDefineChatModel.name,
      modelType: ModelType.gemma4,
      backend: PreferredBackend.gpu,
      maxTokens: kLlmConfig.maxTokens,
      tools: true,
    );
    await applyAndReload(gemma, 'audio-on');
    expect(deps.conversation.capabilities.audio, isTrue);
    final audioVm = await enterVoiceChat(tester);
    expect(audioVm.modeLine, 'Voice → ${kDefineChatModel.name}');
    expect(audioVm.hasSkills, isTrue, reason: 'skills through the tools');

    // 7. The France question as audio, twice.
    final audioTurns = <String>[];
    for (var i = 1; i <= 2; i++) {
      final heard = await voiceTurn(tester, audioVm, fixture);
      expect(
        heard.outcome,
        TurnOutcome.completed,
        reason: describeChat(audioVm),
      );
      final asked = audioVm.entries.lastWhere((e) => e.role == ChatRole.user);
      final answer = audioVm.entries.last;
      expect(answer.role, ChatRole.assistant);
      expect(answer.text.toLowerCase(), contains('paris'));
      final spoken = asked.spoken;
      expect(spoken, isNotNull, reason: 'an audio turn: ${asked.text}');
      // The turn ended after its playback drained; the transcript job only
      // starts then, so the words are not there yet.
      expect(spoken!.transcript, isNull, reason: 'no words before playback');
      expect(asked.text, startsWith('🎤 '));
      final voiceTurnMetrics = deps.diagnostics.latest.lastVoiceTurn!;
      final generation = deps.diagnostics.latest.lastGeneration!;
      final retrieval = deps.diagnostics.latest.lastRetrieval;
      expect(voiceTurnMetrics.audio, isTrue);
      expect(voiceTurnMetrics.stt, isNull, reason: 'no STT before the reply');
      expect(retrieval?.queryKind, KnowledgeQueryKind.audio);
      final afterPlayback = Stopwatch()..start();
      await pumpUntil(
        tester,
        () =>
            audioVm.entries
                .firstWhere((e) => e.spoken?.turn == spoken.turn)
                .spoken
                ?.transcript !=
            null,
        timeout: const Duration(seconds: 30),
        reason: 'the transcript after the playback',
        describe: () => describeChat(audioVm),
      );
      final words = audioVm.entries
          .firstWhere((e) => e.spoken?.turn == spoken.turn)
          .text;
      expect(words.toLowerCase(), contains('france'));
      final history = deps.conversation.history;
      expect(
        history.where((t) => t.turn == spoken.turn).single.user,
        words,
        reason: 'the transcript reaches the text history',
      );
      final metrics = deps.diagnostics.latest.lastVoiceTurn!;
      final line =
          'AUDIO turn=$i ttft=${ms(generation.timeToFirstToken)}ms '
          'first_text=${ms(voiceTurnMetrics.firstText)}ms '
          'ttfa=${ms(voiceTurnMetrics.firstAudio)}ms '
          'total=${ms(voiceTurnMetrics.total)}ms '
          'kb_audio=${ms(retrieval?.latency)}ms '
          'kb_top=${retrieval?.topSimilarity?.toStringAsFixed(3)} '
          'kb=${retrieval?.outcome.name} '
          'transcript=${ms(metrics.transcript)}ms '
          'transcript_wait=${afterPlayback.elapsedMilliseconds}ms '
          'words="$words" reply="${answer.text}"';
      audioTurns.add(line);
      debugPrint(line);
    }
    debugPrint('SCREENSHOT ${await saveScreenshot('voice_loop_audio')}');

    // 8. Barge-in in audio mode, on the spoken lighthouse question. Should
    // the model still answer in one short line that has finished before the
    // press, ask again, at most three times, until the reply is still being
    // generated while it speaks.
    var storyGenerating = false;
    for (var attempt = 1; attempt <= 3 && !storyGenerating; attempt++) {
      final before = audioVm.entries.length;
      unawaited(
        audioVm.submitUtterance(
          Utterance(
            pcm: story.pcm,
            held: pcm16Duration(story.pcm.length, 16000),
          ),
        ),
      );
      await pumpUntil(
        tester,
        () =>
            audioVm.phase == TurnPhase.speaking ||
            (audioVm.phase == TurnPhase.idle &&
                audioVm.entries.length > before + 1),
        timeout: const Duration(seconds: 60),
        reason: 'the story to start speaking',
        describe: () => describeChat(audioVm),
      );
      if (audioVm.phase == TurnPhase.speaking) {
        await pumpFor(tester, const Duration(milliseconds: 300));
      }
      storyGenerating =
          audioVm.phase == TurnPhase.speaking &&
          deps.conversation.isGenerating.value;
      // The reply so far, on one line: shows what the model did when it
      // did not tell the story (a short refusal, a skill call).
      debugPrint(
        'STORY_PARTIAL attempt=$attempt generating=$storyGenerating '
        'reply="${audioVm.partialReply.value.replaceAll(RegExp(r'\s+'), ' ')}"',
      );
      if (!storyGenerating) {
        await pumpUntil(
          tester,
          () => audioVm.phase == TurnPhase.idle,
          timeout: const Duration(seconds: 60),
          reason: 'the short reply to finish',
          describe: () => describeChat(audioVm),
        );
      }
    }
    expect(audioVm.phase, TurnPhase.speaking, reason: 'still speaking');
    expect(
      storyGenerating,
      isTrue,
      reason: 'still generating (three spoken requests)',
    );
    final storyEntries = audioVm.entries.length;
    final storyQuestion = audioVm.entries.lastWhere(
      (e) => e.role == ChatRole.user,
    );
    expect(storyQuestion.spoken, isNotNull);
    // "rebuilt the chat from the text history" (the live chat cleared), or
    // "replayed the text history" (an interrupted skill call's fresh chat).
    bool rebuildLine(String l) =>
        l.contains('[Conversation] rebuilt the chat from the text history') ||
        l.contains('[Conversation] replayed the text history');
    final storyRebuildsBefore = logged.where(rebuildLine).length;
    final audioPressWatch = Stopwatch()..start();
    final audioGesture = await tester.startGesture(
      tester.getCenter(find.byKey(ChatKeys.mic)),
    );
    final audioPressToReturn = audioPressWatch.elapsed;
    final audioBargeIn = deps.diagnostics.latest.lastBargeIn!;
    expect(audioBargeIn.wasPlaying, isTrue);
    expect(
      audioBargeIn.silenced!,
      lessThanOrEqualTo(const Duration(milliseconds: 150)),
      reason: 'acceptance: silent ≤150 ms after the press (audio mode)',
    );
    await pumpUntil(
      tester,
      () => audioVm.isListening,
      timeout: const Duration(seconds: 3),
      reason: 'the barge-in to open the mic',
      describe: () => describeChat(audioVm),
    );
    final storyReply = audioVm.entries[storyEntries];
    expect(storyReply.role, ChatRole.assistant);
    expect(storyReply.interrupted, isTrue);
    await holdUntilPlayed(tester, mic, fixture);
    await audioGesture.up();
    await pumpUntil(
      tester,
      () => deps.diagnostics.latest.lastBargeIn?.interruptDone != null,
      timeout: const Duration(seconds: 12),
      reason: 'the interrupted audio turn to drain',
    );
    final audioDrained = deps.diagnostics.latest.lastBargeIn!;
    final storyGen = deps.diagnostics.latest.lastGeneration!;
    expect(storyGen.stopped, isTrue, reason: 'the barge-in stopped Gemma');
    await pumpUntil(
      tester,
      () => audioVm.phase == TurnPhase.idle || audioVm.phase == TurnPhase.error,
      timeout: const Duration(seconds: 90),
      reason: 'the follow-up audio turn',
      describe: () => describeChat(audioVm),
    );
    expect(audioVm.phase, TurnPhase.idle, reason: describeChat(audioVm));
    final followAsked = audioVm.entries.lastWhere(
      (e) => e.role == ChatRole.user,
    );
    expect(followAsked.spoken, isNotNull, reason: 'the follow-up is audio');
    expect(audioVm.entries.last.text.toLowerCase(), contains('paris'));
    final rebuilt = logged.where(rebuildLine).toList();
    expect(
      rebuilt.length,
      storyRebuildsBefore + 1,
      reason:
          'the stopped audio turn makes the next one rebuild the chat '
          'from the text history',
    );
    // The interrupted story stays in the history as its placeholder and the
    // partial reply; it gets no transcript (a barged-in question gets none:
    // its recording was dropped at the press).
    final storyHistory = deps.conversation.history.firstWhere(
      (t) => t.turn == storyQuestion.spoken!.turn,
    );
    expect(storyHistory.user, kSpokenQuestionPlaceholder);
    expect(storyHistory.assistant, isNotEmpty);
    expect(
      audioVm.entries
          .firstWhere((e) => e.spoken?.turn == storyQuestion.spoken!.turn)
          .spoken!
          .transcript,
      isNull,
    );
    final followVoice = deps.diagnostics.latest.lastVoiceTurn!;
    debugPrint(
      'AUDIO_BARGE silenced=${ms(audioBargeIn.silenced)}ms '
      'stop_confirmed=${ms(audioDrained.stopConfirmed)}ms '
      'drain=${ms(audioDrained.interruptDone)}ms '
      'press_to_return=${audioPressToReturn.inMicroseconds / 1000}ms '
      'interrupted_chars=${storyReply.text.length} '
      'gen_stopped=${storyGen.stopped} '
      'gen_stop_latency=${ms(storyGen.stopLatency)}ms | '
      'FOLLOWUP ttfa=${ms(followVoice.firstAudio)}ms '
      'rebuild="${rebuilt.last}" reply="${audioVm.entries.last.text}"',
    );
    // Its transcript too, after the playback.
    await pumpUntil(
      tester,
      () =>
          audioVm.entries.last.role == ChatRole.assistant &&
          audioVm.entries
                  .lastWhere((e) => e.role == ChatRole.user)
                  .spoken
                  ?.transcript !=
              null,
      timeout: const Duration(seconds: 30),
      reason: 'the follow-up transcript',
      describe: () => describeChat(audioVm),
    );

    for (final line in audioTurns) {
      debugPrint(line);
    }
    debugPrint = printer;
    await tester.pumpWidget(const SizedBox.shrink());
    await deps.dispose();
  }, timeout: const Timeout(Duration(minutes: 25)));
}
