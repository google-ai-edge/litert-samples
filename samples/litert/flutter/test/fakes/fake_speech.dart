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

import 'dart:async';
import 'dart:typed_data';

import 'package:flutter_edge_ai/flutter_edge_ai.dart'
    show SpeechRecognizer, SpeechSynthesizer, SttModelType;
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/repositories/speech_repository.dart';
import 'package:litert_edge_demos/data/services/speech/stt_service.dart';
import 'package:litert_edge_demos/data/services/speech/tts_service.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';

/// A recognizer that returns [text]; can block on [gate] or throw [error].
class FakeRecognizer implements SpeechRecognizer {
  FakeRecognizer([this.text = 'What is the capital of France?']);

  String text;
  Error? error;
  Completer<void>? gate;
  final List<Uint8List> pcms = [];
  int closeCalls = 0;

  int get calls => pcms.length;

  @override
  String? language = 'en';

  @override
  Future<String> transcribe(Uint8List pcm16kMono, {String? language}) async {
    pcms.add(pcm16kMono);
    await gate?.future;
    if (error case final e?) throw e;
    return text;
  }

  @override
  void addCloseListener(void Function() listener) {}

  @override
  Future<void> close() async => closeCalls++;
}

/// Records every clause it synthesizes and returns 20 ms of non-zero PCM per
/// call. From call [gateFrom] (1-based) on it blocks on [gate]; a clause
/// containing [throwOn] throws; one containing [emptyOn] returns zero bytes
/// (Inflect does that for a non-speech clause).
class RecordingSynth implements SpeechSynthesizer {
  RecordingSynth({this.sampleRate = 24000});

  @override
  final int sampleRate;

  final List<String> synthesized = [];
  Completer<void>? gate;
  int gateFrom = 1;
  String? throwOn;
  String? emptyOn;
  int closeCalls = 0;

  @override
  Future<Uint8List> synthesize(String text) async {
    synthesized.add(text);
    if (gate != null && synthesized.length >= gateFrom) await gate!.future;
    if (throwOn case final marker? when text.contains(marker)) {
      throw StateError('synth boom');
    }
    if (emptyOn case final marker? when text.contains(marker)) {
      return Uint8List(0);
    }
    return Uint8List.fromList(List.filled(960, synthesized.length));
  }

  @override
  void addCloseListener(void Function() listener) {}

  @override
  Future<void> close() async => closeCalls++;
}

/// The installed model id the fake installer reports for [config].
String fakeSttModelId(SttConfig config) => switch (config.type) {
  SttModelType.moonshine => 'moonshine_tiny_5s_f32',
  _ => 'whisper_base_30s_i8',
};

/// A real [SttService] whose install and load are fakes. [recognizer] is
/// what every load returns unless [recognizerFor] picks one per config.
SttService fakeSttService({
  SpeechRecognizer? recognizer,
  SpeechRecognizer Function(SttConfig config)? recognizerFor,
  Future<String> Function()? install,
  Completer<void>? loadGate,
  List<String>? log,
  List<SttSource>? sources,
}) => SttService(
  install: (config, source, onProgress) async {
    sources?.add(source);
    onProgress(50);
    onProgress(100);
    log?.add('install ${fakeSttModelId(config)}');
    return install == null ? fakeSttModelId(config) : await install();
  },
  load: (config) async {
    log?.add('load ${fakeSttModelId(config)}');
    await loadGate?.future;
    return recognizerFor?.call(config) ?? recognizer ?? FakeRecognizer();
  },
);

/// Made-up store files for [id] (the fake installer never opens them).
SttSource fakeSttSource(ModelId id) => SttFromFiles(
  modelPath: '/store/${id.name}/model',
  tokenizerPath: '/store/${id.name}/tokenizer',
);

/// A real [TtsService] whose install and load are fakes.
TtsService fakeTtsService({
  SpeechSynthesizer? synthesizer,
  Future<String> Function()? install,
  List<String>? directories,
}) => TtsService(
  install: (config, directory, onProgress) async {
    directories?.add(directory);
    onProgress(100);
    return install == null ? 'inflect' : await install();
  },
  load: (config) async => synthesizer ?? RecordingSynth(),
);

/// A [SpeechRepository] over loaded fake models.
Future<SpeechRepository> loadedSpeech({
  required SpeechRecognizer recognizer,
  required SpeechSynthesizer synthesizer,
}) async {
  final stt = fakeSttService(recognizer: recognizer);
  final tts = fakeTtsService(synthesizer: synthesizer);
  await stt.install(
    ModelId.whisperBase,
    source: fakeSttSource(ModelId.whisperBase),
    onProgress: (_) {},
  );
  await stt.load(ModelId.whisperBase);
  await tts.install(directory: '/bundled', onProgress: (_) {});
  await tts.load();
  return SpeechRepository(stt: stt, tts: tts);
}
