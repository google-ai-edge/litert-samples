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
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_edge_ai/flutter_edge_ai.dart'
    show ModelType, PreferredBackend;
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/model_catalog.dart';
import 'package:litert_edge_demos/domain/models/chat_model_config.dart';
import 'package:litert_edge_demos/domain/models/detection.dart';
import 'package:litert_edge_demos/selftest/self_test_options.dart';
import 'package:litert_edge_demos/selftest/self_test_report.dart';
import 'package:litert_edge_demos/selftest/self_test_runner.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../fakes/fake_hardware.dart';
import '../fakes/fake_self_test_adapters.dart';

/// Characterization of the self-test's observable output, against
/// `test/goldens/selftest/<scenario>.txt`:
/// - two whole transcripts (`all_pass_macos`, `all_pass_linux_t4`
///   [_matchReport]): every progress line (the `[SELFTEST]` lines), every
///   report line (`formatSelfTest`) and the exit code — the header, the
///   device and the memory blocks are pinned there only;
/// - every other scenario ([_matchSteps]): what it decides, the result line
///   and the steps block. Header lines a scenario is about (its options, its
///   chat model, its detector file) are asserted in the test itself.
///
/// Step timings, the total time and stack frames are masked; everything else
/// must match byte for byte. The report is parsed by tools and read by
/// users: a diff here is a format change. `fvm flutter test
/// --update-goldens test/selftest/self_test_transcript_test.dart` rewrites
/// the files; review the diff before committing it.
void main() {
  group('transcripts', () {
    test('everything passes on macOS: Metal from the log, the tone played '
        'but not measured, the chat model\'s sha256', () async {
      final tap = FakeNativeLogTap();
      final audio = FakeSelfTestAudio(
        monitorUnavailable: 'macOS has no monitor source to record the output',
      );
      final t = await _transcript(
        (progress) => fakeSelfTestRunner(
          detector: FakeSelfTestDetector(
            tap: tap,
            logLines: const [
              'I0000 delegate_metal.mm:88] Created a Metal device.',
              'unrelated',
            ],
          ),
          gemma: FakeSelfTestGemma(),
          tap: tap,
          chatModel: const SelfTestChatModel(
            file: kFakeSelfTestFile,
            config: kDefineChatModel,
            settingsSource: 'the app\'s Gemma 4 E2B settings',
            sha256: 'abc123',
          ),
          audio: audio,
          progress: progress,
        ),
      );
      expect(t.statuses, {
        '1': StepStatus.pass,
        '2': StepStatus.pass,
        '3': StepStatus.pass,
        '4a': StepStatus.pass,
        '4b': StepStatus.pass,
        '5': StepStatus.pass,
        '6a': StepStatus.pass,
        '6b': StepStatus.pass,
      });
      expect(audio.monitorStarted, isFalse);
      expect(audio.closed, isTrue);
      _matchReport('all_pass_macos', t);
    });

    test('everything passes on a Linux T4: the adapter from the log, the tone '
        'measured on the sink monitor', () async {
      final tap = FakeNativeLogTap();
      final t = await _transcript(
        (progress) => fakeSelfTestRunner(
          detector: FakeSelfTestDetector(
            tap: tap,
            logLines: const [
              'I0000 webgpu.cc:1] Selected adapter: Tesla T4, arch=turing, '
                  'vendor=nvidia, backend=Vulkan, adapterType=Discrete GPU',
            ],
          ),
          gemma: FakeSelfTestGemma(),
          tap: tap,
          profile: kFakeLinuxT4Profile,
          audio: FakeSelfTestAudio(),
          progress: progress,
        ),
      );
      expect(t.report.passed, isTrue);
      _matchReport('all_pass_linux_t4', t);
    });

    group('detector', () {
      test('on the CPU (--detector-backend=cpu): passes, no adapter '
          'needed', () async {
        final detector = FakeSelfTestDetector(
          frame: (call) =>
              fakeGoldenCatsFrame(call, backend: DetectorBackend.cpu),
        );
        final t = await _transcript(
          (progress) => fakeSelfTestRunner(
            options: const SelfTestOptions(
              detectorBackend: DetectorBackend.cpu,
              gemmaBackend: PreferredBackend.cpu,
            ),
            detector: detector,
            gemma: FakeSelfTestGemma(),
            profile: kFakeUnnamedLinuxProfile,
            progress: progress,
          ),
        );
        expect(detector.loads, [DetectorBackend.cpu]);
        expect(t.report.passed, isTrue);
        expect(t.step('2').title, 'detector load (cpu)');
        expect(
          t.header('chat model'),
          contains(
            'settings, backend from --gemma-backend) · /models/file '
            '(model store) · cpu ·',
          ),
        );
        expect(
          t.header('detector'),
          'detector   /models/file (model store) · backend cpu',
        );
        _matchSteps('detector_cpu', t);
      });

      test('the GPU load fails: the native errors, no fallback, steps 3 and '
          '5 skipped', () async {
        final tap = FakeNativeLogTap();
        final detector = FakeSelfTestDetector(
          failOn: {DetectorBackend.gpu},
          tap: tap,
          logLines: const [
            'INFO: attempting the GPU',
            'E0000 00:00:1 webgpu.cc:12] Failed to create a Vulkan instance',
          ],
        );
        final t = await _transcript(
          (progress) => fakeSelfTestRunner(
            detector: detector,
            gemma: FakeSelfTestGemma(),
            tap: tap,
            progress: progress,
          ),
        );
        expect(detector.loads, [DetectorBackend.gpu]);
        expect(t.statuses['3'], StepStatus.skip);
        expect(t.statuses['5'], StepStatus.skip);
        expect(t.report.exitCode, 1);
        _matchSteps('detector_gpu_fails', t);
      });

      test('--detector-cpu-retry, the CPU loads: 2b, 3 and 5 are information '
          'only; still FAIL', () async {
        final t = await _transcript(
          (progress) => fakeSelfTestRunner(
            options: const SelfTestOptions(detectorCpuRetry: true),
            detector: FakeSelfTestDetector(failOn: {DetectorBackend.gpu}),
            gemma: FakeSelfTestGemma(),
            progress: progress,
          ),
        );
        expect(t.statuses['2b'], StepStatus.info);
        expect(t.statuses['3'], StepStatus.info);
        expect(t.statuses['5'], StepStatus.info);
        expect(t.report.passed, isFalse);
        expect(t.header('options'), 'options    --detector-cpu-retry');
        _matchSteps('detector_cpu_retry_loads', t);
      });

      test('--detector-cpu-retry, the CPU fails too: 2b information, 3 and 5 '
          'skipped', () async {
        final detector = FakeSelfTestDetector(
          failOn: {DetectorBackend.gpu, DetectorBackend.cpu},
        );
        final t = await _transcript(
          (progress) => fakeSelfTestRunner(
            options: const SelfTestOptions(detectorCpuRetry: true),
            detector: detector,
            gemma: FakeSelfTestGemma(),
            progress: progress,
          ),
        );
        expect(detector.loads, [DetectorBackend.gpu, DetectorBackend.cpu]);
        expect(t.statuses['2b'], StepStatus.info);
        expect(t.step('2b').details.first, 'the CPU failed too');
        expect(t.statuses['3'], StepStatus.skip);
        _matchSteps('detector_cpu_retry_fails_too', t);
      });

      test('no detector file: step 2 fails with the reason, no CPU attempt '
          'even when asked', () async {
        final detector = FakeSelfTestDetector();
        final t = await _transcript(
          (progress) => fakeSelfTestRunner(
            options: const SelfTestOptions(detectorCpuRetry: true),
            detector: detector,
            gemma: FakeSelfTestGemma(),
            detectorFile: const ModelFileChoice(
              source: '--detector',
              problem: 'not readable: /nope.tflite (No such file).',
            ),
            progress: progress,
          ),
        );
        expect(detector.loads, isEmpty);
        expect(t.statuses.containsKey('2b'), isFalse);
        expect(
          t.header('detector'),
          'detector   none (--detector): not readable: /nope.tflite '
          '(No such file).',
        );
        _matchSteps('detector_file_missing', t);
      });

      test('the runtime reports the CPU for a GPU load: a confirmed mismatch '
          'fails the step', () async {
        final t = await _transcript(
          (progress) => fakeSelfTestRunner(
            detector: FakeSelfTestDetector(reportAs: DetectorBackend.cpu),
            gemma: FakeSelfTestGemma(),
            progress: progress,
          ),
        );
        expect(t.statuses['2'], StepStatus.fail);
        expect(t.step('2').details, contains('requested gpu but runs on cpu'));
        _matchSteps('detector_mismatch', t);
      });
    });

    group('cats golden and detector again', () {
      test('wrong classes, runs not identical, step 5 differs', () async {
        final t = await _transcript(
          (progress) => fakeSelfTestRunner(
            detector: FakeSelfTestDetector(frame: fakeWrongFrame),
            gemma: FakeSelfTestGemma(),
            progress: progress,
          ),
        );
        expect(t.statuses['3'], StepStatus.fail);
        expect(t.statuses['5'], StepStatus.fail);
        _matchSteps('golden_wrong_not_identical', t);
      });

      test('a custom image: no golden, consistency only', () async {
        final t = await _transcript(
          (progress) => fakeSelfTestRunner(
            detector: FakeSelfTestDetector(),
            gemma: FakeSelfTestGemma(),
            loadImage: () async => Result.ok(
              SelfTestImage(
                frame: FakeTinyFrame(),
                label: '/tmp/street.jpg',
                golden: false,
              ),
            ),
            imageLabel: '/tmp/street.jpg',
            progress: progress,
          ),
        );
        expect(t.report.passed, isTrue);
        expect(t.header('image'), 'image      /tmp/street.jpg');
        _matchSteps('custom_image', t);
      });

      test('the image does not load: step 3 fails, step 5 skipped', () async {
        final t = await _transcript(
          (progress) => fakeSelfTestRunner(
            detector: FakeSelfTestDetector(),
            gemma: FakeSelfTestGemma(),
            loadImage: () async =>
                Result.error(Exception('cannot decode /tmp/x.jpg')),
            progress: progress,
          ),
        );
        expect(t.statuses['3'], StepStatus.fail);
        expect(t.statuses['5'], StepStatus.skip);
        _matchSteps('image_fails', t);
      });

      test('a detection fails during the golden runs: no reference, step 5 '
          'skipped', () async {
        final detector = FakeSelfTestDetector(failDetects: {3});
        final t = await _transcript(
          (progress) => fakeSelfTestRunner(
            detector: detector,
            gemma: FakeSelfTestGemma(),
            progress: progress,
          ),
        );
        expect(detector.detects, 3);
        expect(t.statuses['5'], StepStatus.skip);
        _matchSteps('detect_fails_golden', t);
      });

      test('the detection after the chat model fails', () async {
        final t = await _transcript(
          (progress) => fakeSelfTestRunner(
            detector: FakeSelfTestDetector(failDetects: {11}),
            gemma: FakeSelfTestGemma(),
            progress: progress,
          ),
        );
        expect(t.statuses['5'], StepStatus.fail);
        _matchSteps('detect_fails_again', t);
      });
    });

    group('chat model', () {
      test('the load fails: the native errors, generation skipped', () async {
        final tap = FakeNativeLogTap();
        final gemma = FakeSelfTestGemma(
          loadError: Exception('LiteRT-LM: failed to open the model'),
          tap: tap,
          logLines: const [
            'I0000 engine.cc:1] Created a WebGPU environment',
            'E0000 engine.cc:2] Failed to allocate 2.4 GB',
          ],
        );
        final t = await _transcript(
          (progress) => fakeSelfTestRunner(
            detector: FakeSelfTestDetector(),
            gemma: gemma,
            tap: tap,
            progress: progress,
          ),
        );
        expect(gemma.generations, 0);
        expect(t.statuses['4b'], StepStatus.skip);
        _matchSteps('chat_load_fails', t);
      });

      test('the engine comes up on the CPU: a failure, never a CPU '
          'retry', () async {
        final t = await _transcript(
          (progress) => fakeSelfTestRunner(
            detector: FakeSelfTestDetector(),
            gemma: FakeSelfTestGemma(active: PreferredBackend.cpu),
            progress: progress,
          ),
        );
        expect(t.statuses['4a'], StepStatus.fail);
        _matchSteps('chat_backend_mismatch', t);
      });

      test('generation fails', () async {
        final t = await _transcript(
          (progress) => fakeSelfTestRunner(
            detector: FakeSelfTestDetector(),
            gemma: FakeSelfTestGemma(generateError: Exception('session lost')),
            progress: progress,
          ),
        );
        expect(t.statuses['4b'], StepStatus.fail);
        _matchSteps('chat_generate_fails', t);
      });

      test('too few chunks, no engine metrics, a long reply cut at 100 '
          'characters', () async {
        final t = await _transcript(
          (progress) => fakeSelfTestRunner(
            detector: FakeSelfTestDetector(),
            gemma: FakeSelfTestGemma(
              chunks: 3,
              engineMetrics: false,
              text:
                  'Once upon a time,\n\na lighthouse keeper named Elias '
                  'watched the sea every night and counted the ships that '
                  'passed his rock.',
            ),
            progress: progress,
          ),
        );
        expect(t.statuses['4b'], StepStatus.fail);
        _matchSteps('chat_few_chunks', t);
      });

      test('your own model on the NPU, images and tools off', () async {
        const own = ChatModelConfig(
          name: 'Gemma 3 1B NPU',
          modelType: ModelType.gemmaIt,
          llm: LlmConfig(
            maxTokens: 1280,
            backend: PreferredBackend.npu,
            supportImage: false,
            maxNumImages: 1,
          ),
          tools: false,
        );
        final t = await _transcript(
          (progress) => fakeSelfTestRunner(
            detector: FakeSelfTestDetector(),
            gemma: FakeSelfTestGemma(),
            chatModel: const SelfTestChatModel(
              file: ModelFileChoice(
                path: '/store/custom/g3_ekv1280.litertlm',
                source: 'model store, custom/',
              ),
              config: own,
              settingsSource: 'your own .litertlm, saved settings',
              sha256: 'feedface',
            ),
            progress: progress,
          ),
        );
        expect(t.report.passed, isTrue);
        expect(
          t.step('4a').details.first,
          startsWith(
            'Gemma 3 1B NPU (gemma-4-E2B-it) · maxTokens 1280 · image off · '
            'tools off · type gemmaIt ·',
          ),
          reason: 'the load line says what was loaded, as the header does',
        );
        expect(
          t.header('chat model'),
          'chat model Gemma 3 1B NPU (your own .litertlm, saved settings) · '
          '/store/custom/g3_ekv1280.litertlm (model store, custom/) · npu · '
          'ctx 1280 · images off · tools off · type gemmaIt',
        );
        expect(t.header('sha256'), 'sha256     feedface');
        _matchSteps('chat_own_npu', t);
      });

      test('no chat model: 4a fails with the reason, nothing loaded', () async {
        final gemma = FakeSelfTestGemma();
        final t = await _transcript(
          (progress) => fakeSelfTestRunner(
            detector: FakeSelfTestDetector(),
            gemma: gemma,
            chatModel: const SelfTestChatModel(
              file: ModelFileChoice(
                source: 'no chat model',
                problem:
                    'No chat model yet: choose a .litertlm in the Chat model '
                    'card, or pass --gemma.',
              ),
              config: kDefineChatModel,
              settingsSource: 'no chat model',
            ),
            progress: progress,
          ),
        );
        expect(gemma.loads, isEmpty);
        expect(
          t.header('chat model'),
          'chat model Gemma 4 E2B: none (no chat model): No chat model yet: '
          'choose a .litertlm in the Chat model card, or pass --gemma.',
        );
        _matchSteps('no_chat_model', t);
      });
    });

    test('the hardware probe throws: a failed step, the run goes on without '
        'a profile', () async {
      final t = await _transcript(
        (progress) => fakeSelfTestRunner(
          detector: FakeSelfTestDetector(),
          gemma: FakeSelfTestGemma(),
          hardware: ThrowingHardwareInfoService(StateError('no sysctl')),
          progress: progress,
        ),
      );
      expect(t.statuses['1'], StepStatus.fail);
      expect(t.step('1').details.first, 'threw: Bad state: no sysctl');
      expect(t.report.hardware, isNull);
      expect(t.text, contains('--- device ---\nprobe failed\n'));
      _matchSteps('probe_throws', t);
    });

    group('software and unnamed GPUs', () {
      for (final allowed in [false, true]) {
        final suffix = allowed ? 'allowed' : 'strict';
        test('llvmpipe only, $suffix', () async {
          final t = await _transcript(
            (progress) => fakeSelfTestRunner(
              options: SelfTestOptions(allowSoftwareGpu: allowed),
              detector: FakeSelfTestDetector(),
              gemma: FakeSelfTestGemma(),
              profile: kFakeLlvmpipeProfile,
              progress: progress,
            ),
          );
          expect(t.report.passed, allowed);
          expect(t.report.softwareGpuAllowed, allowed);
          if (allowed) {
            expect(t.header('options'), 'options    --allow-software-gpu');
          }
          _matchSteps('software_gpu_$suffix', t);
        });

        test('an adapter nothing can name on Linux, $suffix', () async {
          final t = await _transcript(
            (progress) => fakeSelfTestRunner(
              options: SelfTestOptions(allowSoftwareGpu: allowed),
              detector: FakeSelfTestDetector(),
              gemma: FakeSelfTestGemma(),
              profile: kFakeUnnamedLinuxProfile,
              progress: progress,
            ),
          );
          expect(t.report.passed, allowed);
          _matchSteps('unnamed_linux_$suffix', t);
        });
      }

      // Each GPU step alone must still label the result line: only the chat
      // model, or only the detector, on llvmpipe.
      test('llvmpipe for the chat model only (detector on the CPU), '
          'allowed', () async {
        final t = await _transcript(
          (progress) => fakeSelfTestRunner(
            options: const SelfTestOptions(
              allowSoftwareGpu: true,
              detectorBackend: DetectorBackend.cpu,
            ),
            detector: FakeSelfTestDetector(),
            gemma: FakeSelfTestGemma(),
            profile: kFakeLlvmpipeProfile,
            progress: progress,
          ),
        );
        expect(t.report.passed, isTrue);
        expect(t.report.softwareGpuAllowed, isTrue);
        _matchSteps('software_gpu_chat_only_allowed', t);
      });

      test('llvmpipe for the detector only (chat model on the CPU), '
          'allowed', () async {
        final t = await _transcript(
          (progress) => fakeSelfTestRunner(
            options: const SelfTestOptions(
              allowSoftwareGpu: true,
              gemmaBackend: PreferredBackend.cpu,
            ),
            detector: FakeSelfTestDetector(),
            gemma: FakeSelfTestGemma(),
            profile: kFakeLlvmpipeProfile,
            progress: progress,
          ),
        );
        expect(t.report.passed, isTrue);
        expect(t.report.softwareGpuAllowed, isTrue);
        _matchSteps('software_gpu_detector_only_allowed', t);
      });

      test('a CPU attempt never sets the label', () async {
        final t = await _transcript(
          (progress) => fakeSelfTestRunner(
            options: const SelfTestOptions(
              allowSoftwareGpu: true,
              detectorCpuRetry: true,
              gemmaBackend: PreferredBackend.cpu,
            ),
            detector: FakeSelfTestDetector(failOn: {DetectorBackend.gpu}),
            gemma: FakeSelfTestGemma(),
            profile: kFakeLlvmpipeProfile,
            progress: progress,
          ),
        );
        expect(t.statuses['2b'], StepStatus.info);
        expect(t.report.softwareGpuAllowed, isFalse);
        expect(
          t.header('options'),
          'options    --detector-cpu-retry --allow-software-gpu',
        );
        _matchSteps('software_gpu_cpu_retry_allowed', t);
      });
    });

    group('closing', () {
      test('the detector, then the chat model, then the audio', () async {
        final closes = <String>[];
        await fakeSelfTestRunner(
          detector: FakeSelfTestDetector(closeLog: closes),
          gemma: FakeSelfTestGemma(closeLog: closes),
          audio: FakeSelfTestAudio(closeLog: closes),
        ).run();
        expect(closes, ['detector', 'gemma', 'audio']);
      });

      test('a run that throws out of a step still closes everything, in '
          'order', () async {
        final closes = <String>[];
        final runner = fakeSelfTestRunner(
          detector: FakeSelfTestDetector(closeLog: closes),
          gemma: FakeSelfTestGemma(closeLog: closes),
          audio: FakeSelfTestAudio(closeLog: closes),
          progress: (line) {
            if (line.startsWith('step 4a ')) throw StateError('sink broke');
          },
        );
        await expectLater(runner.run(), throwsStateError);
        expect(closes, ['detector', 'gemma', 'audio']);
      });

      test('a stopped run closes everything too', () async {
        final closes = <String>[];
        await _transcript(
          (progress) => fakeSelfTestRunner(
            detector: FakeSelfTestDetector(closeLog: closes),
            gemma: FakeSelfTestGemma(closeLog: closes),
            audio: FakeSelfTestAudio(closeLog: closes),
            progress: progress,
          ),
          stopAt: 'step 1 ',
        );
        expect(closes, ['detector', 'gemma', 'audio']);
      });
    });

    group('stop after the time limit', () {
      test('during the detector load: it ends on its own, every later step '
          'is skipped, everything is closed', () async {
        final detector = FakeSelfTestDetector();
        final gemma = FakeSelfTestGemma();
        final audio = FakeSelfTestAudio();
        final t = await _transcript(
          (progress) => fakeSelfTestRunner(
            detector: detector,
            gemma: gemma,
            audio: audio,
            progress: progress,
          ),
          stopAt: 'step 2 ',
        );
        expect(t.statuses, {
          '1': StepStatus.pass,
          '2': StepStatus.pass,
          '3': StepStatus.skip,
          '4a': StepStatus.skip,
          '4b': StepStatus.skip,
          '5': StepStatus.skip,
          '6a': StepStatus.skip,
          '6b': StepStatus.skip,
        });
        expect(gemma.loads, isEmpty);
        expect(detector.detects, 0);
        expect(audio.played, isEmpty);
        expect(detector.closed && gemma.closed && audio.closed, isTrue);
        _matchSteps('stop_during_detector_load', t);
      });

      test('during the chat model load: generation, step 5 and audio '
          'skipped', () async {
        final gemma = FakeSelfTestGemma();
        final audio = FakeSelfTestAudio();
        final t = await _transcript(
          (progress) => fakeSelfTestRunner(
            detector: FakeSelfTestDetector(),
            gemma: gemma,
            audio: audio,
            progress: progress,
          ),
          stopAt: 'step 4a ',
        );
        expect(t.statuses['4a'], StepStatus.pass);
        expect(t.statuses['4b'], StepStatus.skip);
        expect(t.statuses['5'], StepStatus.skip);
        expect(gemma.generations, 0);
        _matchSteps('stop_during_chat_load', t);
      });

      test('during a failed GPU load with --detector-cpu-retry: no CPU '
          'attempt', () async {
        final detector = FakeSelfTestDetector(failOn: {DetectorBackend.gpu});
        final t = await _transcript(
          (progress) => fakeSelfTestRunner(
            options: const SelfTestOptions(detectorCpuRetry: true),
            detector: detector,
            gemma: FakeSelfTestGemma(),
            progress: progress,
          ),
          stopAt: 'step 2 ',
        );
        expect(detector.loads, [DetectorBackend.gpu]);
        expect(t.statuses['2b'], StepStatus.skip);
        _matchSteps('stop_during_detector_cpu_retry', t);
      });

      test('during the audio output: it times out, the microphone is '
          'skipped', () async {
        final audio = FakeSelfTestAudio(hangOutput: true);
        final t = await _transcript(
          (progress) => fakeSelfTestRunner(
            detector: FakeSelfTestDetector(),
            gemma: FakeSelfTestGemma(),
            audio: audio,
            audioTimeout: const Duration(milliseconds: 50),
            progress: progress,
          ),
          stopAt: 'step 6a ',
        );
        expect(t.statuses['6a'], StepStatus.fail);
        expect(t.statuses['6b'], StepStatus.skip);
        expect(audio.micRecorded, isFalse);
        _matchSteps('stop_during_audio_output', t);
      });
    });

    test('the watchdog\'s report while the chat model load hangs: the steps '
        'so far plus a timeout step naming it', () async {
      final gemma = FakeSelfTestGemma()..loadGate = Completer<void>();
      final progress = <String>[];
      final runner = fakeSelfTestRunner(
        detector: FakeSelfTestDetector(),
        gemma: gemma,
        progress: progress.add,
      );
      final running = runner.run();
      await pumpEventQueue();
      expect(gemma.loads, hasLength(1), reason: 'step 4a hangs');
      final partial = runner.report(timedOut: const Duration(seconds: 90));
      final t = _Transcript(partial, List.of(progress));
      gemma.loadGate!.complete();
      await running;
      expect(partial.steps.last.id, 'T');
      expect(
        partial.steps.last.details.single,
        'step 4a chat model load + warm-up (gpu) did not finish within 90 s '
        '(--timeout)',
      );
      _matchSteps('timeout_during_chat_load', t);
    });

    group('audio', () {
      Future<_Transcript> audioRun(
        FakeSelfTestAudio audio, {
        Duration timeout = kSelfTestAudioTimeout,
      }) => _transcript(
        (progress) => fakeSelfTestRunner(
          detector: FakeSelfTestDetector(),
          gemma: FakeSelfTestGemma(),
          profile: kFakeLinuxT4Profile,
          audio: audio,
          audioTimeout: timeout,
          progress: progress,
        ),
      );

      test('no output device; no microphone', () async {
        final t = await audioRun(
          FakeSelfTestAudio(outputError: kFakeNoOutput, micError: kFakeNoMic),
        );
        expect(t.statuses['6a'], StepStatus.fail);
        expect(t.statuses['6b'], StepStatus.fail);
        _matchSteps('audio_no_output_no_mic', t);
      });

      test('the monitor does not start; the microphone delivers '
          'nothing', () async {
        final t = await audioRun(
          FakeSelfTestAudio(
            monitorStartError: Exception('parecord exited with 1'),
            micPcm: Uint8List(0),
          ),
        );
        expect(t.statuses['6a'], StepStatus.fail);
        expect(t.statuses['6b'], StepStatus.fail);
        _matchSteps('audio_monitor_start_fails_mic_empty', t);
      });

      test('the monitor capture fails; the microphone hands digital '
          'zeros', () async {
        final t = await audioRun(
          FakeSelfTestAudio(
            monitorStopError: Exception('parecord died'),
            micPcm: Uint8List(32000),
          ),
        );
        expect(t.statuses['6a'], StepStatus.fail);
        expect(t.statuses['6b'], StepStatus.warn);
        _matchSteps('audio_monitor_stop_fails_mic_zeros', t);
      });

      test('playback fails while the monitor records; a silent '
          'microphone', () async {
        final t = await audioRun(
          FakeSelfTestAudio(
            playError: Exception('soloud: no device'),
            micPcm: fakePcmAt(-75),
          ),
        );
        expect(t.statuses['6a'], StepStatus.fail);
        expect(t.statuses['6b'], StepStatus.warn);
        expect(t.report.passed, isFalse);
        _matchSteps('audio_play_fails_mic_silent', t);
      });

      test('no monitor here and playback fails', () async {
        final t = await audioRun(
          FakeSelfTestAudio(
            monitorUnavailable: 'macOS has no monitor source',
            playError: Exception('soloud: no device'),
          ),
        );
        expect(t.statuses['6a'], StepStatus.fail);
        _matchSteps('audio_unmeasured_play_fails', t);
      });

      test('nothing reached the sound server; the microphone hangs', () async {
        final t = await audioRun(
          FakeSelfTestAudio(monitorPcm: Uint8List(32000), hangMic: true),
          timeout: const Duration(milliseconds: 50),
        );
        expect(t.statuses['6a'], StepStatus.fail);
        expect(t.statuses['6b'], StepStatus.fail);
        _matchSteps('audio_nothing_mic_hangs', t);
      });

      test('too quiet; the close hangs and is not waited for', () async {
        final audio = FakeSelfTestAudio(
          monitorPcm: fakePcmAt(-55),
          hangClose: true,
        );
        final t = await audioRun(
          audio,
          timeout: const Duration(milliseconds: 50),
        );
        expect(t.statuses['6a'], StepStatus.fail);
        expect(audio.closed, isTrue);
        expect(
          t.progress,
          contains(
            'the audio did not close within 0.05 s; exiting releases it',
          ),
        );
        _matchSteps('audio_too_quiet_close_hangs', t);
      });

      test('a flagged output device is a WARN', () async {
        final t = await audioRun(FakeSelfTestAudio(outputCaution: true));
        expect(t.statuses['6a'], StepStatus.warn);
        expect(t.report.passed, isTrue);
        _matchSteps('audio_caution', t);
      });

      test('a flagged device without a monitor is a WARN too', () async {
        final t = await audioRun(
          FakeSelfTestAudio(
            outputCaution: true,
            monitorUnavailable: 'macOS has no monitor source',
          ),
        );
        expect(t.statuses['6a'], StepStatus.warn);
        _matchSteps('audio_caution_unmeasured', t);
      });
    });

    test('--skip-audio: no step 6, the options line says so', () async {
      final t = await _transcript(
        (progress) => fakeSelfTestRunner(
          options: const SelfTestOptions(skipAudio: true),
          detector: FakeSelfTestDetector(),
          gemma: FakeSelfTestGemma(),
          progress: progress,
        ),
      );
      expect(t.statuses.keys, ['1', '2', '3', '4a', '4b', '5']);
      expect(t.header('options'), 'options    --skip-audio (no step 6)');
      _matchSteps('skip_audio', t);
    });

    test('an unhandled error fails a run whose steps all passed', () async {
      final t = await _transcript(
        (progress) => fakeSelfTestRunner(
          detector: FakeSelfTestDetector(),
          gemma: FakeSelfTestGemma(),
          unhandled: ['Bad state: lost frame'],
          progress: progress,
        ),
      );
      expect(t.report.passed, isFalse);
      expect(t.header('unhandled'), 'unhandled  Bad state: lost frame');
      _matchSteps('unhandled_error', t);
    });
  });
}

final class _Transcript {
  _Transcript(this.report, this.progress);

  final SelfTestReport report;

  /// The progress lines, in order.
  final List<String> progress;

  /// The progress, the report and the exit code.
  String get text => _render(progress, report);

  /// The report's lines.
  List<String> get _report => formatSelfTest(
    report,
    reportPath: '/out/selftest.txt',
  ).trimRight().split('\n');

  /// The report's result line and its steps block.
  String get decided {
    final lines = _report;
    final from = lines.indexOf('--- steps ---');
    final to = lines.indexWhere((l) => l.startsWith('--- memory'), from);
    return [
      lines.firstWhere((l) => l.startsWith('result ')),
      ...lines.sublist(from, to),
      '',
    ].join('\n');
  }

  /// The report's header line [label] (`options    --skip-audio …`).
  String header(String label) =>
      _report.singleWhere((l) => l.startsWith(label.padRight(11)));

  Map<String, StepStatus> get statuses => {
    for (final s in report.steps) s.id: s.status,
  };

  SelfTestStep step(String id) => report.steps.singleWhere((s) => s.id == id);
}

/// Builds the runner with a progress sink, runs it and renders the
/// transcript. With [stopAt], [SelfTestRunner.stop] is called when a
/// progress line starting with it appears (the step it names is in flight).
Future<_Transcript> _transcript(
  SelfTestRunner Function(void Function(String line) progress) build, {
  String? stopAt,
}) async {
  final progress = <String>[];
  late final SelfTestRunner runner;
  runner = build((line) {
    progress.add(line);
    if (stopAt != null && line.startsWith(stopAt) && line.endsWith('…')) {
      runner.stop();
    }
  });
  final report = await runner.run();
  return _Transcript(report, progress);
}

String _render(List<String> progress, SelfTestReport report) => [
  '--- progress ---',
  ...progress,
  '--- report ---',
  formatSelfTest(report, reportPath: '/out/selftest.txt').trimRight(),
  '--- exit ${report.exitCode} ---',
  '',
].join('\n');

/// A step that ran: its time varies. Skipped steps and the watchdog's `T`
/// step take no time and keep their literal `0.00 s`.
final _stepHeader = RegExp(
  r'^((?:PASS|FAIL|INFO|WARN)  (?!T  ).*) \d+\.\d\d s$',
);
final _took = RegExp(r' · took \d+\.\d\d s$');
final _stackFrame = RegExp(r'^ {11}(?:#\d+ .*|<asynchronous suspension>)$');

/// Masks what changes from run to run (the timings of steps that ran, the
/// total time) and with the code's layout (stack frames).
String _mask(String text) => [
  for (final line in text.split('\n'))
    if (_stepHeader.firstMatch(line) case final m?)
      '${m.group(1)} <time>'
    else if (_took.hasMatch(line))
      line.replaceFirst(_took, ' · took <time>')
    else if (_stackFrame.hasMatch(line))
      '           <stack frame>'
    else
      line,
].join('\n');

/// The whole transcript: kept for two runs only, so a change to the header,
/// the device or the memory block rewrites two files.
void _matchReport(String name, _Transcript t) => _matchGolden(name, t.text);

/// What scenario [name] decides: its result line and its steps.
void _matchSteps(String name, _Transcript t) => _matchGolden(name, t.decided);

void _matchGolden(String name, String transcript) {
  final file = File('test/goldens/selftest/$name.txt');
  final masked = _mask(transcript);
  if (autoUpdateGoldenFiles) {
    file.parent.createSync(recursive: true);
    file.writeAsStringSync(masked);
    return;
  }
  expect(
    file.existsSync(),
    isTrue,
    reason: '${file.path} is missing: run with --update-goldens',
  );
  expect(masked, file.readAsStringSync(), reason: file.path);
}
