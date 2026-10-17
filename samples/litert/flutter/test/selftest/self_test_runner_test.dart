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
    show ModelType, PreferredBackend;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/data/repositories/audio_repository.dart'
    show MicAccessException, PlaybackException;
import 'package:litert_edge_demos/domain/models/chat_model_config.dart';
import 'package:litert_edge_demos/domain/models/detection.dart';
import 'package:litert_edge_demos/domain/models/detector_spec.dart';
import 'package:litert_edge_demos/domain/models/hardware_profile.dart';
import 'package:litert_edge_demos/domain/models/knowledge_query.dart';
import 'package:litert_edge_demos/selftest/self_test_options.dart';
import 'package:litert_edge_demos/selftest/self_test_report.dart';
import 'package:litert_edge_demos/selftest/self_test_runner.dart';
import 'package:litert_edge_demos/utils/result.dart';
import 'package:litert_edge_demos/utils/wav.dart';

import '../fakes/fake_hardware.dart';
import '../fakes/fake_llm_service.dart' show engineCreateFailure;
import '../fakes/fake_self_test_adapters.dart';

/// [fakeSelfTestRunner] with the chat model as a file and a config (the
/// user's own settings unless it is the app's Gemma 4 E2B).
SelfTestRunner _runner({
  SelfTestOptions options = const SelfTestOptions(),
  required SelfTestDetector detector,
  required SelfTestGemma gemma,
  FakeNativeLogTap? tap,
  HardwareProfile profile = kFakeMacProfile,
  ModelFileChoice gemmaFile = kFakeSelfTestFile,
  ChatModelConfig chatConfig = kDefineChatModel,
  ModelFileChoice detectorFile = kFakeSelfTestFile,
  List<String> unhandled = const [],
  SelfTestAudio? audio,
  Duration audioTimeout = kSelfTestAudioTimeout,
}) => fakeSelfTestRunner(
  options: options,
  detector: detector,
  gemma: gemma,
  tap: tap,
  profile: profile,
  chatModel: SelfTestChatModel(
    file: gemmaFile,
    config: chatConfig,
    settingsSource: chatConfig == kDefineChatModel
        ? 'the app\'s Gemma 4 E2B settings'
        : 'your own .litertlm, saved settings',
  ),
  detectorFile: detectorFile,
  unhandled: unhandled,
  audio: audio,
  audioTimeout: audioTimeout,
);

Map<String, StepStatus> _statuses(SelfTestReport r) => {
  for (final s in r.steps) s.id: s.status,
};

/// The runner's wiring. The step lines of each scenario are pinned by
/// self_test_transcript_test.dart's goldens (a detector that fails on the
/// GPU with its native log lines, software and unnamed GPUs, the watchdog's
/// report, silent or quiet output), the rules behind them by
/// evidence_judge_test.dart, step_recorder_test.dart and
/// tapped_load_test.dart.
void main() {
  test('everything passes: exit 0, Metal confirmed by the log, memory '
      'sampled around every step, both models closed', () async {
    final tap = FakeNativeLogTap();
    final detector = FakeSelfTestDetector(
      tap: tap,
      logLines: const [
        'I0000 delegate_metal.mm:88] Created a Metal device.',
        'unrelated',
      ],
    );
    final gemma = FakeSelfTestGemma();
    final report = await _runner(
      detector: detector,
      gemma: gemma,
      tap: tap,
    ).run();

    expect(_statuses(report), {
      '1': StepStatus.pass,
      '2': StepStatus.pass,
      '3': StepStatus.pass,
      '4a': StepStatus.pass,
      '4b': StepStatus.pass,
      '5': StepStatus.pass,
      '7a': StepStatus.pass,
      '7b': StepStatus.pass,
      '7c': StepStatus.pass,
    });
    expect(report.passed, isTrue);
    expect(report.exitCode, 0);
    expect(detector.detects, 11, reason: '10 golden runs + 1 after Gemma');
    final step2 = report.steps[1];
    expect(
      step2.details,
      contains(
        'gpu → gpu (confirmed: API) · api Metal (confirmed: native log) · '
        'adapter Apple M4 Pro (inferred)',
      ),
    );
    expect(step2.details, contains(startsWith('log: I0000 delegate_metal')));
    expect(step2.details.where((d) => d.contains('unrelated')), isEmpty);
    expect(report.steps[2].details.last, contains('golden classes ✓'));
    expect(
      report.steps.every((s) => s.before != null && s.after != null),
      isTrue,
    );
    expect(detector.closed && gemma.closed, isTrue);

    final text = formatSelfTest(report, reportPath: '/out/selftest.txt');
    expect(
      text,
      startsWith('$kSelfTestBegin\nresult     PASS · 9 pass · exit 0\n'),
    );
    expect(text.trimRight(), endsWith(kSelfTestEnd));
    expect(text, contains('report     /out/selftest.txt'));
    expect(
      text,
      contains('cpu        Apple M4 Pro · 14 cores (10P+4E) · arm64'),
    );
    expect(text, contains('4a chat model    '));
    expect(text, contains('7a embedder load '), reason: 'its RSS row');
  });

  group('the embedder steps (7a load, 7b audio ↔ text)', () {
    test('the bundled clip and its words, each embedded as the knowledge base '
        'embeds a question; above the calibrated cosine passes; the '
        'embedder is closed after the run', () async {
      final embedder = FakeSelfTestEmbedder();
      final report = await fakeSelfTestRunner(
        detector: FakeSelfTestDetector(),
        gemma: FakeSelfTestGemma(),
        embedder: embedder,
      ).run();

      expect(_statuses(report)['7a'], StepStatus.pass);
      expect(_statuses(report)['7b'], StepStatus.pass);
      final (audio, text) = (embedder.queries[0], embedder.queries[1]);
      expect(audio, isA<KnowledgeAudioQuery>());
      expect(
        pcm16FromWav((audio as KnowledgeAudioQuery).wav),
        hasLength(38400),
      );
      expect((text as KnowledgeTextQuery).text, kSelfTestAudioQuestionText);
      final details = report.steps.firstWhere((s) => s.id == '7b').details;
      expect(details.first, startsWith('cosine 0.800 (≥ 0.73)'));
      expect(details.first, contains('(1.20 s)'));
      expect(embedder.closed, isTrue);
    });

    test('the gate sits between the measured pair (0.8525) and unrelated '
        'pairs (up to 0.606), with a margin on both sides', () {
      expect(kSelfTestAudioTextMinCosine, lessThan(0.8525 - 0.1));
      expect(kSelfTestAudioTextMinCosine, greaterThan(0.606 + 0.1));
      expect(
        kSelfTestAudioQuestionAsset,
        'assets/selftest/what_time_is_it.wav',
      );
    });

    test('below the gate fails 7b and says what it means', () async {
      final report = await fakeSelfTestRunner(
        detector: FakeSelfTestDetector(),
        gemma: FakeSelfTestGemma(),
        embedder: FakeSelfTestEmbedder(audio: const [0, 1, 0]),
      ).run();

      final step = report.steps.firstWhere((s) => s.id == '7b');
      expect(step.status, StepStatus.fail);
      expect(step.details.first, startsWith('cosine 0.000 (< 0.73)'));
      expect(step.details.last, contains('audio path'));
      expect(report.passed, isFalse);
    });

    test('an embedder that is not ready fails 7a with the reason and skips '
        '7b', () async {
      final embedder = FakeSelfTestEmbedder(
        prepareError: Exception('The app\'s EmbeddingGemma is loading'),
      );
      final report = await fakeSelfTestRunner(
        detector: FakeSelfTestDetector(),
        gemma: FakeSelfTestGemma(),
        embedder: embedder,
      ).run();

      expect(_statuses(report)['7a'], StepStatus.fail);
      expect(
        report.steps.firstWhere((s) => s.id == '7a').details.single,
        contains('is loading'),
      );
      expect(_statuses(report)['7b'], StepStatus.skip);
      expect(_statuses(report)['7c'], StepStatus.skip);
      expect(embedder.queries, isEmpty);
      expect(embedder.images, isEmpty);
      expect(embedder.closed, isTrue);
    });

    test('7c: the image encoder loaded as the camera demo loads it, then the '
        'bundled photo made into a memory frame against its caption, each '
        'embedded as the visual memory embeds a frame and a question; above '
        'the gate passes', () async {
      final embedder = FakeSelfTestEmbedder();
      final framed = <Uint8List>[];
      final report = await fakeSelfTestRunner(
        detector: FakeSelfTestDetector(),
        gemma: FakeSelfTestGemma(),
        embedder: embedder,
        toMemoryFrame: (photo) async {
          framed.add(photo);
          return fakeMemoryFrame(photo);
        },
      ).run();

      final step = report.steps.firstWhere((s) => s.id == '7c');
      expect(step.status, StepStatus.pass);
      expect(embedder.imageLoads, 1);
      expect(framed.single, hasLength(6), reason: 'the bundled photo');
      expect(
        embedder.images.single,
        hasLength(40 << 10),
        reason: 'the memory frame, not the photo as it is',
      );
      expect(embedder.memoryTexts, [kSelfTestImageText]);
      expect(
        step.details.first,
        startsWith(
          'cosine 0.700 (≥ ${kSelfTestImageTextMinCosine.toStringAsFixed(2)})',
        ),
      );
      expect(step.details.first, contains('as a memory frame (40 KB PNG)'));
      expect(step.details[1], startsWith('image encoder loaded in 2.10 s'));
    });

    test('7c: a failed image encoder load, or a photo that does not make a '
        'memory frame, fails with the reason; one loaded by the camera demo '
        'before the run is said so', () async {
      final noEncoder = await fakeSelfTestRunner(
        detector: FakeSelfTestDetector(),
        gemma: FakeSelfTestGemma(),
        embedder: FakeSelfTestEmbedder(
          imagesLoadError: Exception('no vision tower'),
        ),
      ).run();
      final failed = noEncoder.steps.firstWhere((s) => s.id == '7c');
      expect(failed.status, StepStatus.fail);
      expect(failed.details.single, contains('loading the image encoder'));
      expect(failed.details.single, contains('no vision tower'));

      final undecodable = await fakeSelfTestRunner(
        detector: FakeSelfTestDetector(),
        gemma: FakeSelfTestGemma(),
        embedder: FakeSelfTestEmbedder(),
        toMemoryFrame: (_) async =>
            Result.error(Exception('the photo does not decode')),
      ).run();
      final notFramed = undecodable.steps.firstWhere((s) => s.id == '7c');
      expect(notFramed.status, StepStatus.fail);
      expect(notFramed.details.single, contains('as a memory frame'));

      final before = await fakeSelfTestRunner(
        detector: FakeSelfTestDetector(),
        gemma: FakeSelfTestGemma(),
        embedder: FakeSelfTestEmbedder(imagesAlreadyLoaded: true),
      ).run();
      expect(
        before.steps.firstWhere((s) => s.id == '7c').details[1],
        contains('loaded before the run'),
      );
    });

    test('7c below the gate, or a failed image embedding, fails with what it '
        'means', () async {
      final below = await fakeSelfTestRunner(
        detector: FakeSelfTestDetector(),
        gemma: FakeSelfTestGemma(),
        embedder: FakeSelfTestEmbedder(image: const [0, 0, 1]),
      ).run();
      final step = below.steps.firstWhere((s) => s.id == '7c');
      expect(step.status, StepStatus.fail);
      expect(step.details.last, contains('visual memory'));
      expect(below.passed, isFalse);

      final failed = await fakeSelfTestRunner(
        detector: FakeSelfTestDetector(),
        gemma: FakeSelfTestGemma(),
        embedder: FakeSelfTestEmbedder(imageError: Exception('stb')),
      ).run();
      final failedStep = failed.steps.firstWhere((s) => s.id == '7c');
      expect(failedStep.status, StepStatus.fail);
      expect(failedStep.details.single, contains('stb'));
    });

    test('a failed audio embedding, a clip that is not a WAV, or a missing '
        'clip fail 7b with the reason', () async {
      Future<SelfTestStep> step7b({
        FakeSelfTestEmbedder? embedder,
        Future<Result<Uint8List>> Function()? clip,
      }) async {
        final report = await fakeSelfTestRunner(
          detector: FakeSelfTestDetector(),
          gemma: FakeSelfTestGemma(),
          embedder: embedder,
          loadAudioQuestion: clip,
        ).run();
        return report.steps.firstWhere((s) => s.id == '7b');
      }

      final engine = await step7b(
        embedder: FakeSelfTestEmbedder(audioError: Exception('miniaudio')),
      );
      final notWav = await step7b(
        clip: () async => Result.ok(Uint8List.fromList([1, 2, 3])),
      );
      final missing = await step7b(
        clip: () async => Result.error(Exception('not in this build')),
      );

      expect(engine.details.single, contains('embedding the clip failed'));
      expect(notWav.details.single, contains('the bundled clip'));
      expect(missing.details.single, contains('not in this build'));
      for (final s in [engine, notWav, missing]) {
        expect(s.status, StepStatus.fail);
      }
    });

    test(
      'the app\'s own embedder (in-app) says so instead of a load time',
      () async {
        final report = await fakeSelfTestRunner(
          detector: FakeSelfTestDetector(),
          gemma: FakeSelfTestGemma(),
          embedder: FakeSelfTestEmbedder(borrowed: true),
        ).run();

        expect(
          report.steps.firstWhere((s) => s.id == '7a').details.single,
          'embeddinggemma-2-740m · cpu · audio+text · the app\'s, loaded '
          'before the run',
        );
      },
    );
  });

  test('--detector-cpu-retry: an explicit, labelled CPU attempt; its steps '
      'are information only and the result stays FAIL', () async {
    final detector = FakeSelfTestDetector(failOn: {DetectorBackend.gpu});
    final report = await _runner(
      options: const SelfTestOptions(detectorCpuRetry: true),
      detector: detector,
      gemma: FakeSelfTestGemma(),
    ).run();
    expect(detector.loads, [DetectorBackend.gpu, DetectorBackend.cpu]);
    expect(_statuses(report), {
      '1': StepStatus.pass,
      '2': StepStatus.fail,
      '2b': StepStatus.info,
      '3': StepStatus.info,
      '4a': StepStatus.pass,
      '4b': StepStatus.pass,
      '5': StepStatus.info,
      '7a': StepStatus.pass,
      '7b': StepStatus.pass,
      '7c': StepStatus.pass,
    });
    expect(report.steps[2].title, contains('information only'));
    expect(report.steps[2].details.first, contains('still a failure'));
    expect(report.steps[3].title, contains('on the CPU attempt'));
    expect(report.passed, isFalse);
  });

  test(
    'Gemma on a GPU the strict engine cannot create: a failure with the '
    "plugin's reason, never a CPU retry; generation skipped; exit 1",
    () async {
      final gemma = FakeSelfTestGemma(loadError: engineCreateFailure());
      final report = await _runner(
        detector: FakeSelfTestDetector(),
        gemma: gemma,
      ).run();

      expect(gemma.loads, [PreferredBackend.gpu]);
      expect(gemma.generations, 0);
      expect(_statuses(report), {
        '1': StepStatus.pass,
        '2': StepStatus.pass,
        '3': StepStatus.pass,
        '4a': StepStatus.fail,
        '4b': StepStatus.skip,
        '5': StepStatus.pass,
        '7a': StepStatus.pass,
        '7b': StepStatus.pass,
        '7c': StepStatus.pass,
      });
      final why = report.steps[3].details.first;
      expect(why, contains('load failed (no fallback)'));
      expect(why, contains('did not load on gpu'));
      expect(why, contains('Can not find OpenCL library on this device'));
      expect(report.exitCode, 1);
    },
  );

  test('a missing model file is a failure with the reason', () async {
    final gemma = FakeSelfTestGemma();
    final report = await _runner(
      detector: FakeSelfTestDetector(),
      gemma: gemma,
      gemmaFile: const ModelFileChoice(
        source: 'model store',
        problem: 'not in the model store: pass --gemma=PATH',
      ),
    ).run();
    expect(report.steps[3].status, StepStatus.fail);
    expect(report.steps[3].details.single, contains('--gemma=PATH'));
    expect(gemma.loads, isEmpty);
    expect(
      formatSelfTest(report),
      contains(
        'chat model Gemma 4 E2B: none (model store): not in the model store',
      ),
    );
  });

  test('your own chat model: loaded with its own type, backend, context, '
      'images and tools; the report says which and where from', () async {
    const own = ChatModelConfig(
      name: 'Gemma 3 1B NPU',
      modelType: ModelType.gemmaIt,
      llm: LlmConfig(
        maxTokens: 1280,
        backend: PreferredBackend.npu,
        maxNumImages: 1,
      ),
      tools: false,
    );
    final gemma = FakeSelfTestGemma();
    final report = await _runner(
      detector: FakeSelfTestDetector(),
      gemma: gemma,
      chatConfig: own,
      gemmaFile: const ModelFileChoice(
        path: '/store/custom/g3_ekv1280.litertlm',
        source: 'model store, custom/',
      ),
    ).run();

    expect(gemma.configs.single, same(own));
    expect(report.steps[3].title, 'chat model load + warm-up (npu)');
    expect(report.steps[3].status, StepStatus.pass);
    expect(report.steps[4].title, startsWith('chat model generate'));
    expect(
      report.steps[3].details,
      isNot(contains(startsWith('vision encoder'))),
      reason: 'no vision tower',
    );
    final text = formatSelfTest(report);
    expect(
      text,
      contains(
        'chat model Gemma 3 1B NPU (your own .litertlm, saved settings) · '
        '/store/custom/g3_ekv1280.litertlm (model store, custom/) · npu · '
        'ctx 1280 · towers none · tools off · type gemmaIt',
      ),
    );
  });

  test('without --detector the built-in detector is used and the report '
      'says so', () async {
    final detector = FakeSelfTestDetector();
    final runner = _runner(
      detector: detector,
      gemma: FakeSelfTestGemma(),
      detectorFile: ModelFileChoice.bundledDetector,
    );

    final report = await runner.run();

    expect(report.passed, isTrue);
    expect(detector.models.first.bundled, isTrue);
    expect(detector.models.first.path, kDetModelAsset);
    expect(
      formatSelfTest(report),
      contains(
        'detector (bundled) assets/models/yolo26n_fp16_rawhead.tflite · '
        'backend gpu',
      ),
    );
  });

  test('--gemma-backend overrides the chat model\'s own backend', () async {
    final gemma = FakeSelfTestGemma();
    final report = await _runner(
      options: const SelfTestOptions(gemmaBackend: PreferredBackend.npu),
      detector: FakeSelfTestDetector(),
      gemma: gemma,
    ).run();

    expect(gemma.loads.single, PreferredBackend.npu);
    expect(gemma.configs.single.llm.maxTokens, kLlmConfig.maxTokens);
    expect(report.steps[3].title, 'chat model load + warm-up (npu)');
    expect(
      formatSelfTest(report),
      contains('Gemma 4 E2B settings, backend from --gemma-backend'),
    );
  });

  test('stop: the step in flight ends on its own, every later step is '
      'skipped (the chat model is never loaded), the models are '
      'closed', () async {
    final detector = FakeSelfTestDetector()..loadGate = Completer<void>();
    final gemma = FakeSelfTestGemma();
    final runner = _runner(detector: detector, gemma: gemma);

    final running = runner.run();
    await pumpEventQueue();
    expect(detector.loads, hasLength(1), reason: 'step 2 hangs');
    runner.stop();
    detector.loadGate!.complete();
    final report = await running;

    expect(gemma.loads, isEmpty);
    expect(detector.detects, 0);
    expect(detector.closed, isTrue);
    expect(gemma.closed, isTrue);
    expect(_statuses(report)['2'], StepStatus.pass);
    expect(_statuses(report)['4a'], StepStatus.skip);
    expect(
      report.steps.firstWhere((s) => s.id == '4a').details.single,
      contains('stopped after its time limit'),
    );
  });

  group('chatEngineMayBeLoaded (one chat engine per process)', () {
    test('false until step 4a starts, true from then until the chat model '
        'is closed, false after', () async {
      final atStart = <String, bool>{};
      late final SelfTestRunner runner;
      runner = fakeSelfTestRunner(
        detector: FakeSelfTestDetector(),
        gemma: FakeSelfTestGemma(),
        audio: FakeSelfTestAudio(),
        progress: (line) {
          if (line.endsWith('…')) {
            atStart[line.split(' ')[1]] = runner.chatEngineMayBeLoaded;
          }
        },
      );
      expect(runner.chatEngineMayBeLoaded, isFalse);

      await runner.run();

      expect(atStart, {
        '1': false,
        '2': false,
        '3': false,
        '4a': false, // announced before its body runs
        '4b': true,
        '5': true,
        '6a': true,
        '6b': true,
        '7a': true,
        '7b': true,
        '7c': true,
      });
      expect(runner.chatEngineMayBeLoaded, isFalse);
    });

    test('a run hung in the chat model load holds it; one stopped while the '
        'detector loads never does', () async {
      final hungGemma = FakeSelfTestGemma()..loadGate = Completer<void>();
      final inChat = fakeSelfTestRunner(
        detector: FakeSelfTestDetector(),
        gemma: hungGemma,
      );
      final chatRun = inChat.run();
      await pumpEventQueue();
      expect(hungGemma.loads, hasLength(1), reason: 'step 4a hangs');
      inChat.stop();
      expect(inChat.chatEngineMayBeLoaded, isTrue);
      hungGemma.loadGate!.complete();
      await chatRun;
      expect(inChat.chatEngineMayBeLoaded, isFalse, reason: 'closed by now');

      final hungDetector = FakeSelfTestDetector()..loadGate = Completer<void>();
      final gemma = FakeSelfTestGemma();
      final inDetector = fakeSelfTestRunner(
        detector: hungDetector,
        gemma: gemma,
      );
      final detectorRun = inDetector.run();
      await pumpEventQueue();
      inDetector.stop();
      expect(inDetector.chatEngineMayBeLoaded, isFalse);
      hungDetector.loadGate!.complete();
      await detectorRun;
      expect(inDetector.chatEngineMayBeLoaded, isFalse);
      expect(gemma.loads, isEmpty);
    });
  });

  test(
    'an unhandled Dart error during the run fails it and is listed',
    () async {
      final unhandled = <String>[];
      final runner = _runner(
        detector: FakeSelfTestDetector(),
        gemma: FakeSelfTestGemma(),
        unhandled: unhandled,
      );
      unhandled.add('Bad state: lost frame');
      final report = await runner.run();
      expect(report.steps.every((s) => s.status == StepStatus.pass), isTrue);
      expect(report.passed, isFalse);
      expect(report.exitCode, 1);
      final text = formatSelfTest(report);
      expect(text, contains('FAIL · 1 unhandled error(s) · 9 pass · exit 1'));
      expect(text, contains('unhandled  Bad state: lost frame'));
    },
  );

  group('step 6: audio', () {
    Future<SelfTestReport> run(
      SelfTestAudio audio, {
      Duration timeout = kSelfTestAudioTimeout,
    }) => _runner(
      detector: FakeSelfTestDetector(),
      gemma: FakeSelfTestGemma(),
      audio: audio,
      audioTimeout: timeout,
    ).run();

    test('pass: the tone reaches the sink monitor, the mic hears the room; '
        'both reported in dBFS; the audio is closed', () async {
      final audio = FakeSelfTestAudio();
      final report = await run(audio);
      expect(_statuses(report)['6a'], StepStatus.pass);
      expect(_statuses(report)['6b'], StepStatus.pass);
      expect(report.passed, isTrue);
      expect(audio.monitorStarted, isTrue);
      expect(audio.played, [48000], reason: '1 s at 24 kHz, PCM16');
      expect(audio.closed, isTrue);
      final output = report.steps.singleWhere((s) => s.id == '6a');
      expect(output.title, 'audio output (tone → sink monitor)');
      expect(
        output.details,
        contains(
          'played 1.00 s of 440 Hz at -9.1 dBFS on Built-in Audio Analog '
          'Stereo · PulseAudio (on PipeWire 1.0.5)',
        ),
      );
      expect(
        output.details,
        contains(startsWith('sink monitor: loudest 100 ms -9.0 dBFS')),
      );
      final mic = report.steps.singleWhere((s) => s.id == '6b');
      expect(mic.details.first, startsWith('Built-in Audio Analog Stereo'));
      expect(
        mic.details[1],
        '1.00 s · RMS -35.0 dBFS · loudest 100 ms -35.0 dBFS',
      );
      final text = formatSelfTest(report);
      expect(text, contains('PASS  6a  audio output'));
      expect(text, contains('PASS  6b  microphone (1 s, default input)'));
    });

    test('a flagged output (the auto_null dummy sink) is a WARN even when the '
        'monitor hears the tone', () async {
      final report = await run(FakeSelfTestAudio(outputCaution: true));
      final output = report.steps.singleWhere((s) => s.id == '6a');
      expect(output.status, StepStatus.warn);
      expect(
        output.details.last,
        'the output check flagged this device: Dummy Output · pulseaudio '
        'dummy sink (auto_null): nothing is audible',
      );
      expect(report.passed, isTrue, reason: 'a WARN does not fail the run');
    });

    test('no output device: the output check\'s message', () async {
      final report = await run(
        FakeSelfTestAudio(
          outputError: const PlaybackException(
            'No audio output (no PulseAudio/PipeWire/ALSA device): …',
          ),
        ),
      );
      expect(_statuses(report)['6a'], StepStatus.fail);
      expect(
        report.steps.singleWhere((s) => s.id == '6a').details.single,
        'Playback failed: No audio output (no PulseAudio/PipeWire/ALSA '
        'device): …',
      );
      expect(_statuses(report)['6b'], StepStatus.pass, reason: 'independent');
    });

    test('no mic: FAIL with what to do', () async {
      final report = await run(
        FakeSelfTestAudio(
          micError: const MicAccessException(
            'Microphone unavailable: install pulseaudio-utils (parecord)',
          ),
        ),
      );
      expect(_statuses(report)['6b'], StepStatus.fail);
      expect(
        report.steps.singleWhere((s) => s.id == '6b').details.single,
        'no microphone: Microphone unavailable: install pulseaudio-utils '
        '(parecord)',
      );
      expect(report.passed, isFalse);
    });

    test('a silent mic is a WARN, not a FAIL: the run still passes', () async {
      final quiet = await run(FakeSelfTestAudio(micPcm: fakePcmAt(-75)));
      expect(_statuses(quiet)['6b'], StepStatus.warn);
      expect(
        quiet.steps.singleWhere((s) => s.id == '6b').details.last,
        startsWith('silent (below -60.0 dBFS): the room may be quiet'),
      );
      expect(quiet.passed, isTrue);
      expect(formatSelfTest(quiet), contains('1 warn · exit 0'));

      final zeros = await run(FakeSelfTestAudio(micPcm: Uint8List(32000)));
      expect(_statuses(zeros)['6b'], StepStatus.warn);
      expect(
        zeros.steps.singleWhere((s) => s.id == '6b').details.last,
        startsWith('all digital zeros'),
      );
    });

    test('timeout: a hung sound server fails the step instead of hanging the '
        'run', () async {
      final audio = FakeSelfTestAudio(hangMic: true);
      final report = await run(
        audio,
        timeout: const Duration(milliseconds: 50),
      );
      expect(_statuses(report)['6a'], StepStatus.pass);
      expect(_statuses(report)['6b'], StepStatus.fail);
      expect(
        report.steps.singleWhere((s) => s.id == '6b').details.single,
        'the microphone did not answer within 0.05 s: the sound server or the '
        'device hung (no audio stack?)',
      );
      expect(audio.closed, isTrue);
    });

    test('a close that hangs does not hold the report back', () async {
      final report = await run(
        FakeSelfTestAudio(hangMic: true, hangClose: true),
        timeout: const Duration(milliseconds: 50),
      );
      expect(_statuses(report)['6b'], StepStatus.fail);
    });

    test('macOS: no monitor source; the tone is played, labelled as not '
        'measured', () async {
      final audio = FakeSelfTestAudio(
        monitorUnavailable: 'macos has no monitor source to record the output',
      );
      final report = await run(audio);
      final output = report.steps.singleWhere((s) => s.id == '6a');
      expect(output.status, StepStatus.pass);
      expect(output.title, 'audio output (played, not measured)');
      expect(
        output.details.last,
        'not measured: macos has no monitor source to record the output',
      );
      expect(audio.monitorStarted, isFalse);
    });

    test('--skip-audio: no step 6, and the options line says so', () async {
      final report = await _runner(
        options: const SelfTestOptions(skipAudio: true),
        detector: FakeSelfTestDetector(),
        gemma: FakeSelfTestGemma(),
      ).run();
      expect(report.steps.where((s) => s.id.startsWith('6')), isEmpty);
      expect(
        formatSelfTest(report),
        contains('options    --skip-audio (no step 6)'),
      );
    });
  });
}
