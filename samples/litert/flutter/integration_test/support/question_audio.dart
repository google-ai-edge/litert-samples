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

// Spoken test questions made with the app's own speech synthesizer,
// Inflect-nano-v2 (built into the app): tools/make_question_audio_test.dart
// writes the shipped clips with it, and kb_retrieval_test.dart speaks the
// knowledge base's golden questions into memory at start-up with it.

import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/config/voice_config.dart';
import 'package:litert_edge_demos/data/services/model_store/bundled_model_files.dart';
import 'package:litert_edge_demos/data/services/speech/tts_service.dart';
import 'package:litert_edge_demos/utils/pcm.dart';
import 'package:litert_edge_demos/utils/result.dart';

import 'resample.dart';

/// The clips' rate: what the app records and the recognizers hear.
const kClipSampleRate = 16000;

/// Silence before and after the speech: a person pressing the mic does not
/// start talking on the same millisecond, nor release on the last one.
const kClipSilence = Duration(milliseconds: 150);

/// The app's TTS, loaded exactly as setup loads it (ModelRepository's TTS
/// step): the built-in bundle's folder from BundledModelFiles, then the
/// app's TtsService installs, loads (24 kHz checked) and warms up. The caller
/// has initialized flutter_edge_ai and closes the service.
Future<TtsService> loadAppTts() async {
  final directory = switch (await BundledModelFiles().directoryOf(
    kBundledInflectFiles,
  )) {
    Ok(:final value) => value,
    Error(:final error) => fail('The built-in Inflect bundle: $error'),
  };
  final tts = TtsService();
  final installed = await tts.install(directory: directory, onProgress: (_) {});
  expect(installed, isA<Ok<String>>(), reason: '$installed');
  final loaded = await tts.load();
  expect(loaded, isA<Ok<Duration>>(), reason: '$loaded');
  final warm = await tts.warmUp();
  expect(warm, isA<Ok<Duration>>(), reason: '$warm');
  expect(tts.sampleRate, kTtsConfig.sampleRate);
  return tts;
}

/// One spoken question: 16 kHz mono PCM16 (no header) with [kClipSilence]
/// on each side, and its figures.
typedef SpokenQuestionClip = ({
  Uint8List pcm,
  Duration duration,
  Duration speech,
  Duration voiced,
  double peakDbfs,
  Duration ttsTime,
});

/// [text] spoken by [tts]: synthesized (16-bit PCM at the synthesizer's
/// rate), resampled to [kClipSampleRate] (support/resample.dart: polyphase
/// windowed sinc), padded with [kClipSilence], then checked against the
/// app's voice gate (VoiceAssistant drops a quieter capture) and [window]
/// (the recognizer's, which would cut a longer one). Inflect seeds its noise
/// with a constant, so a text comes out the same on a given machine.
Future<SpokenQuestionClip> speakQuestion(
  TtsService tts,
  String text, {
  required Duration window,
}) async {
  final watch = Stopwatch()..start();
  final spoken = await tts.synthesizer.synthesize(text);
  final ttsTime = watch.elapsed;
  expect(spoken, isNotEmpty, reason: 'no audio for "$text"');
  final speech = resamplePcm16(
    spoken,
    fromRate: tts.sampleRate,
    toRate: kClipSampleRate,
  );
  final pcm = _withSilence(speech);
  final duration = pcm16Duration(pcm.length, kClipSampleRate);
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
    reason: '"$text" would be dropped as silence',
  );
  expect(
    duration,
    lessThanOrEqualTo(window),
    reason: '"$text" is longer than its recognizer hears',
  );
  return (
    pcm: pcm,
    duration: duration,
    speech: pcm16Duration(speech.length, kClipSampleRate),
    voiced: voice.voiced,
    peakDbfs: voice.peakFrameDbfs,
    ttsTime: ttsTime,
  );
}

/// [speech] (16-bit PCM at [kClipSampleRate]) with [kClipSilence] of digital
/// silence on each side.
Uint8List _withSilence(Uint8List speech) {
  final pad = kClipSilence.inMicroseconds * kClipSampleRate ~/ 1000000 * 2;
  return Uint8List(pad + speech.length + pad)
    ..setRange(pad, pad + speech.length, speech);
}
