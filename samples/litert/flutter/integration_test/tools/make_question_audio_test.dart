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

// Speaks the questions the integration tests play into the microphone
// (test_assets/q_*.wav, test_assets/france_16k.pcm, test_assets/showcase/
// q_*.wav) with the app's own speech synthesizer, Inflect-nano-v2
// (Apache-2.0, built into the app), so the clips can ship with the sample.
// Run it through the wrapper, which copies the files out of the macOS
// sandbox into test_assets/:
//
//   tool/make_question_audio.sh
//
// 1. Inflect-nano-v2 exactly as setup loads it (ModelRepository's TTS step):
//    flutter_edge_ai initialized as the app does, the built-in bundle's
//    folder from BundledModelFiles, then the app's TtsService installs,
//    loads (24 kHz checked) and warms up.
// 2. Each question of [kQuestionClips]: synthesized (16-bit PCM, 24 kHz),
//    resampled to 16 kHz (support/resample.dart: polyphase windowed sinc),
//    150 ms of silence added before and after, checked against the app's
//    voice gate and moonshine's 5 s window, and written as 16 kHz mono PCM16:
//    a WAV (44-byte header) for `.wav`, raw little-endian samples for `.pcm`.
//    Inflect seeds its noise with a constant, so a text comes out the same on
//    a given machine.
// 3. Output: `<app documents>/question_audio/` (macOS:
//    ~/Library/Containers/com.google.ai.edge.examples.litertEdgeDemos/Data/
//    Documents/question_audio/), the showcase clips in `showcase/`.
//
// Prints `QUESTION_AUDIO <file> bytes=… duration=… voiced=… tts=… text="…"`
// per clip and `QUESTION_AUDIO_OUT=<dir>` at the end.

import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/bootstrap.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/config/voice_config.dart';
import 'package:litert_edge_demos/data/services/model_store/bundled_model_files.dart';
import 'package:litert_edge_demos/data/services/speech/tts_service.dart';
import 'package:litert_edge_demos/utils/pcm.dart';
import 'package:litert_edge_demos/utils/result.dart';
import 'package:path_provider/path_provider.dart';

import '../support/app_window.dart';
import '../support/resample.dart';
import '../support/wav.dart';
import '../support/wav_writer.dart';

/// The source of truth for the spoken test clips: each file's path under
/// test_assets/ and the words spoken in it. Change a wording here, then run
/// tool/make_question_audio.sh. What the tests need from each clip:
/// - france_16k.pcm: voice_loop_test and camera_assistant_test (step 7):
///   Whisper's transcript contains "france", Gemma's reply "paris";
///   test/utils/pcm_test.dart: more than 500 ms voiced.
/// - q_cats.wav: camera_assistant_test (moonshine's transcript contains
///   "cats", the fast route's count rule answers "I count two cats.") and
///   image_chat_test (the voice follow-up: Whisper's transcript contains
///   "cat").
/// - q_describe.wav: camera_assistant_test and network_camera_test, a
///   detailed question about the frame ("describe").
/// - q_sign.wav: camera_assistant_test, a detailed question ("say") whose
///   answer reads "42" off the sign.
/// - q_describe_detail.wav: camera_assistant_test, the long detailed
///   question interrupted by the barge-in.
/// - showcase/: showcase_test (the wording as its earlier runs logged it).
const kQuestionClips = <({String file, String text})>[
  (file: 'france_16k.pcm', text: 'What is the capital of France?'),
  (file: 'q_cats.wav', text: 'How many cats do you see?'),
  (file: 'q_describe.wav', text: 'Describe the scene.'),
  (file: 'q_sign.wav', text: 'What does the sign say?'),
  (
    file: 'q_describe_detail.wav',
    text: 'Describe this scene in detail, please.',
  ),
  (file: 'showcase/q_cats.wav', text: 'How many cats do you see?'),
  (file: 'showcase/q_see.wav', text: 'What do you see?'),
  (file: 'showcase/q_describe.wav', text: 'Describe the scene.'),
  (file: 'showcase/q_sign.wav', text: 'What does the sign say?'),
];

/// The clips' rate: what the app records and the recognizers hear.
const kClipSampleRate = 16000;

/// Silence before and after the speech: a person pressing the mic does not
/// start talking on the same millisecond, nor release on the last one.
const kClipSilence = Duration(milliseconds: 150);

/// [speech] (16-bit PCM at [kClipSampleRate]) with [kClipSilence] of digital
/// silence on each side.
Uint8List _withSilence(Uint8List speech) {
  final pad = kClipSilence.inMicroseconds * kClipSampleRate ~/ 1000000 * 2;
  return Uint8List(pad + speech.length + pad)
    ..setRange(pad, pad + speech.length, speech);
}

void main() {
  initIntegrationTest();

  testWidgets('speak the question clips with the app speech synthesizer', (
    tester,
  ) async {
    // 1. The app's TTS, loaded as setup loads it.
    await initEdgeAi();
    final directory = switch (await BundledModelFiles().directoryOf(
      kBundledInflectFiles,
    )) {
      Ok(:final value) => value,
      Error(:final error) => fail('The built-in Inflect bundle: $error'),
    };
    final tts = TtsService();
    addTearDown(tts.close);
    final installed = await tts.install(
      directory: directory,
      onProgress: (_) {},
    );
    expect(installed, isA<Ok<String>>(), reason: '$installed');
    final loaded = await tts.load();
    expect(loaded, isA<Ok<Duration>>(), reason: '$loaded');
    final warm = await tts.warmUp();
    expect(warm, isA<Ok<Duration>>(), reason: '$warm');
    final rate = tts.sampleRate;
    expect(rate, kTtsConfig.sampleRate);
    debugPrint(
      'QUESTION_AUDIO tts=${tts.modelId} rate=${rate}Hz '
      'backend=${kTtsConfig.backend.name} bundle=$directory',
    );

    // 2. A fresh output folder: nothing of an earlier run is left to copy.
    final documents = await getApplicationDocumentsDirectory();
    final out = Directory('${documents.path}/question_audio');
    if (out.existsSync()) out.deleteSync(recursive: true);
    out.createSync(recursive: true);

    // 3. Every clip.
    expect(
      kQuestionClips.map((c) => c.file).toSet(),
      hasLength(kQuestionClips.length),
      reason: 'one entry per file',
    );
    for (final (:file, :text) in kQuestionClips) {
      final watch = Stopwatch()..start();
      final spoken = await tts.synthesizer.synthesize(text);
      final ttsTime = watch.elapsed;
      expect(spoken, isNotEmpty, reason: 'no audio for "$text"');
      final speech = resamplePcm16(
        spoken,
        fromRate: rate,
        toRate: kClipSampleRate,
      );
      final pcm = _withSilence(speech);
      final duration = pcm16Duration(pcm.length, kClipSampleRate);
      // The app's own silence gate (VoiceAssistant drops a quieter capture).
      final voice = measureVoice(
        pcm,
        gateDbfs: kVoiceConfig.silenceGateDbfs,
        aboveFloorDb: kVoiceConfig.aboveFloorDb,
        floorCapDbfs: kVoiceConfig.floorCapDbfs,
        sampleRate: kClipSampleRate,
      );
      expect(
        voice.voiced,
        greaterThanOrEqualTo(kVoiceConfig.minVoiced),
        reason: '$file would be dropped as silence',
      );
      expect(
        duration,
        lessThanOrEqualTo(kMoonshineSttConfig.window),
        reason: '$file is longer than moonshine hears',
      );
      final Uint8List bytes;
      if (file.endsWith('.wav')) {
        bytes = wavFromPcm16(pcm, sampleRate: kClipSampleRate);
        // The tests' own reader gets the samples back.
        expect(pcm16FromWav(bytes), pcm);
      } else {
        expect(file, endsWith('.pcm'));
        bytes = pcm;
      }
      final target = File('${out.path}/$file');
      target.parent.createSync(recursive: true);
      target.writeAsBytesSync(bytes, flush: true);
      debugPrint(
        'QUESTION_AUDIO $file bytes=${bytes.length} '
        'duration=${duration.inMilliseconds}ms '
        'speech=${pcm16Duration(speech.length, kClipSampleRate).inMilliseconds}ms '
        'voiced=${voice.voiced.inMilliseconds}ms '
        'peak=${voice.peakFrameDbfs.toStringAsFixed(1)}dBFS '
        'tts=${ttsTime.inMilliseconds}ms text="$text"',
      );
    }
    debugPrint('QUESTION_AUDIO_OUT=${out.path}');
  }, timeout: const Timeout(Duration(minutes: 10)));
}
