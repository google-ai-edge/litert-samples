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
// Whisper base int8 and Inflect-nano-v2 on the CPU, soloud playback.
//
//   flutter test integration_test/voice_loop_test.dart -d macos \
//     --dart-define=GEMMA_MODEL_PATH=<path/to/gemma-4-E2B-it.litertlm>
//
// The STT and TTS models are built into the app (tool/fetch_models.sh).
//
// 1. setup → home → Demo 1.
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

import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/app.dart';
import 'package:litert_edge_demos/config/demos.dart';
import 'package:litert_edge_demos/config/dependencies.dart';
import 'package:litert_edge_demos/config/env.dart';
import 'package:litert_edge_demos/domain/models/chat_entry.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/domain/models/voice.dart';
import 'package:litert_edge_demos/ui/features/home/views/home_screen.dart';
import 'package:litert_edge_demos/ui/features/voice_chat/view_models/voice_chat_view_model.dart';
import 'package:litert_edge_demos/ui/features/voice_chat/views/chat_keys.dart';
import 'package:litert_edge_demos/ui/features/voice_chat/views/voice_chat_screen.dart';
import 'package:litert_edge_demos/utils/pcm.dart';
import 'package:provider/provider.dart';
import 'package:record/record.dart';

import 'support/app_window.dart';
import 'support/fixtures.dart';
import 'support/pump.dart';

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
    await tester.pump(const Duration(milliseconds: 400));

    final tile = find.byKey(HomeKeys.tile(Demo.voiceChat));
    await pumpUntil(
      tester,
      () => tester.widget<ListTile>(tile).enabled,
      timeout: const Duration(seconds: 10),
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
    expect(vm.speakReplies, isTrue);

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

    // Hold while the fixture mic plays the question (1.64 s), then release.
    await pumpFor(tester, const Duration(milliseconds: 2200));
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

    await tester.pumpWidget(const SizedBox.shrink());
    await deps.dispose();
  }, timeout: const Timeout(Duration(minutes: 20)));
}
