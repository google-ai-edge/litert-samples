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
// q_*.wav) and the self-test's spoken question
// (assets/selftest/what_time_is_it.wav, an app asset) with the app's own
// speech synthesizer, Inflect-nano-v2 (Apache-2.0, built into the app), so
// the clips can ship with the sample. The knowledge base's spoken golden set
// is not shipped: kb_retrieval_test.dart speaks it into memory at start-up
// with the same code (support/question_audio.dart). Run it through the wrapper, which
// copies the files out of the macOS sandbox into test_assets/ (and assets/
// for the app asset):
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
//    voice gate and the window of the recognizer that hears it (moonshine's
//    5 s; Whisper's 30 s for the self-test clip, which is also
//    EmbeddingGemma 2's audio cap), and written as 16 kHz mono PCM16:
//    a WAV (44-byte header) for `.wav`, raw little-endian samples for `.pcm`
//    (support/question_audio.dart). Inflect seeds its noise with a constant,
//    so a text comes out the same on a given machine.
// 3. Output: `<app documents>/question_audio/` (macOS:
//    ~/Library/Containers/com.google.ai.edge.examples.litertEdgeDemos/Data/
//    Documents/question_audio/), the showcase clips in `showcase/`, the
//    self-test clip in `assets/selftest/`.
//
// Prints `QUESTION_AUDIO <file> bytes=… duration=… voiced=… tts=… text="…"`
// per clip and `QUESTION_AUDIO_OUT=<dir>` at the end.

import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/bootstrap.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:path_provider/path_provider.dart';

import '../support/app_window.dart';
import '../support/question_audio.dart';
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
/// - assets/selftest/what_time_is_it.wav: the self-test's step 7b and the
///   embedder's audio warm-up (an app asset, so the path starts at the
///   repository root).
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
  // The self-test's spoken question (step 7b), an app asset:
  // kSelfTestAudioQuestionAsset, its words kSelfTestAudioQuestionText.
  (file: 'assets/selftest/what_time_is_it.wav', text: 'What time is it?'),
];

/// The longest a clip may be: the window of the recognizer that hears it.
Duration _windowOf(String file) => file.startsWith('assets/')
    ? kWhisperSttConfig.window
    : kMoonshineSttConfig.window;

void main() {
  initIntegrationTest();

  testWidgets('speak the question clips with the app speech synthesizer', (
    tester,
  ) async {
    // 1. The app's TTS, loaded as setup loads it.
    await initEdgeAi();
    final tts = await loadAppTts();
    addTearDown(tts.close);
    debugPrint(
      'QUESTION_AUDIO tts=${tts.modelId} rate=${tts.sampleRate}Hz '
      'backend=${kTtsConfig.backend.name}',
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
      final clip = await speakQuestion(tts, text, window: _windowOf(file));
      final Uint8List bytes;
      if (file.endsWith('.wav')) {
        bytes = wavFromPcm16(clip.pcm, sampleRate: kClipSampleRate);
        // The tests' own reader gets the samples back.
        expect(pcm16FromWav(bytes), clip.pcm);
      } else {
        expect(file, endsWith('.pcm'));
        bytes = clip.pcm;
      }
      final target = File('${out.path}/$file');
      target.parent.createSync(recursive: true);
      target.writeAsBytesSync(bytes, flush: true);
      debugPrint(
        'QUESTION_AUDIO $file bytes=${bytes.length} '
        'duration=${clip.duration.inMilliseconds}ms '
        'speech=${clip.speech.inMilliseconds}ms '
        'voiced=${clip.voiced.inMilliseconds}ms '
        'peak=${clip.peakDbfs.toStringAsFixed(1)}dBFS '
        'tts=${clip.ttsTime.inMilliseconds}ms text="$text"',
      );
    }
    debugPrint('QUESTION_AUDIO_OUT=${out.path}');
  }, timeout: const Timeout(Duration(minutes: 10)));
}
