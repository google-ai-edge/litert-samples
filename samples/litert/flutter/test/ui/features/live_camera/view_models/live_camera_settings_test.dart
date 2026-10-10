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

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/repositories/diagnostics_repository.dart';
import 'package:litert_edge_demos/data/repositories/live_camera_settings_repository.dart';
import 'package:litert_edge_demos/data/repositories/live_detection_repository.dart';
import 'package:litert_edge_demos/data/repositories/speech_repository.dart';
import 'package:litert_edge_demos/data/services/settings/typed_settings.dart';
import 'package:litert_edge_demos/domain/models/camera_side_event.dart';
import 'package:litert_edge_demos/domain/models/camera_source.dart';
import 'package:litert_edge_demos/domain/models/detection.dart';
import 'package:litert_edge_demos/domain/models/frame_source_spec.dart';
import 'package:litert_edge_demos/domain/models/live_state.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/domain/use_cases/camera_turn_responder.dart';
import 'package:litert_edge_demos/domain/use_cases/voice_assistant.dart';
import 'package:litert_edge_demos/ui/core/debug_overlay.dart';
import 'package:litert_edge_demos/ui/features/live_camera/view_models/live_camera_view_model.dart';
import 'package:litert_edge_demos/ui/features/live_camera/views/camera_settings_sheet.dart';
import 'package:litert_edge_demos/ui/features/live_camera/views/live_camera_screen.dart';
import 'package:litert_edge_demos/utils/result.dart';
import 'package:provider/provider.dart';

import '../../../../fakes/fake_audio_repository.dart';
import '../../../../fakes/fake_conversation_repository.dart';
import '../../../../fakes/fake_detector.dart';
import '../../../../fakes/fake_frame_source.dart';
import '../../../../fakes/fake_settings_store.dart';
import '../../../../fakes/fake_speech.dart';

const _phone = 'http://192.168.1.23:8080/video';

const _cpuInfo = DetectorInfo(
  backend: DetectorBackend.cpu,
  fullyAccelerated: false,
  verifyAbsolute: 0,
  verifyRelative: 0,
  verifyReference: VerifyReference.interpreter,
  createTime: Duration(milliseconds: 20),
  verifyTime: Duration(milliseconds: 100),
  firstRunTime: Duration(milliseconds: 30),
);

/// Demo 3 with its settings: the camera source and the detector backend.
void main() {
  late FakeConversationRepository conversation;
  late InMemorySettingsStore store;
  late FakeDetector detector;
  late List<FrameSourceSpec> specs;
  late List<FakeFrameSource> sources;
  late LiveDetectionRepository live;
  late ValueNotifier<Map<ModelId, ModelState>> models;
  late DiagnosticsRepository diagnostics;
  late List<String> log;
  late Result<void> Function() reloadResult;
  late SpeechRepository speech;

  /// A device camera that cannot open (Linux board without one).
  bool deviceCameraFails = false;

  /// Network sources wait on this in their start (a camera connecting).
  Completer<void>? holdNetworkStart;

  setUp(() async {
    conversation = FakeConversationRepository();
    store = InMemorySettingsStore();
    detector = FakeDetector(autoComplete: true);
    specs = [];
    sources = [];
    log = [];
    deviceCameraFails = false;
    holdNetworkStart = null;
    live = LiveDetectionRepository(
      detector: detector,
      createSource: (spec) {
        specs.add(spec);
        log.add('start ${spec.runtimeType}');
        if (spec is CameraSourceSpec && deviceCameraFails) {
          return const Result.error(
            FrameSourceUnavailableException('No camera found (camera_desktop)'),
          );
        }
        final source = FakeFrameSource(
          label: switch (spec) {
            NetworkSourceSpec(:final url) => networkCameraLabel(url),
            _ => 'fake',
          },
        )..startGate = spec is NetworkSourceSpec ? holdNetworkStart : null;
        sources.add(source);
        return Result.ok(source);
      },
    );
    models = ValueNotifier({
      ModelId.yolo26n: const ModelReady(
        LoadedModelInfo(
          modelId: 'yolo26n',
          backend: 'gpu',
          loadTime: Duration.zero,
          warmUpTime: Duration.zero,
          detail: 'GPU fp32 full',
        ),
      ),
    });
    diagnostics = DiagnosticsRepository(
      models: models,
      minInterval: Duration.zero,
      sampleRss: false,
    );
    reloadResult = () => const Result.ok(null);
    speech = await loadedSpeech(
      recognizer: FakeRecognizer(),
      synthesizer: RecordingSynth(),
    );
  });

  tearDown(() async {
    await live.close();
    await conversation.close();
    diagnostics.dispose();
    models.dispose();
  });

  LiveCameraSettingsRepository settingsWith({
    Result<FrameSourceSpec> environment = const Result.ok(CameraSourceSpec()),
    String backendDefine = '',
  }) => LiveCameraSettingsRepository(
    settings: TypedSettings(store: store),
    environment: environment,
    backendDefine: backendDefine,
  );

  Future<Result<void>> reload() async {
    log.add('reload');
    final result = reloadResult();
    if (result is Ok<void>) {
      detector.info = _cpuInfo;
      models.value = {
        ModelId.yolo26n: const ModelReady(
          LoadedModelInfo(
            modelId: 'yolo26n',
            backend: 'cpu',
            loadTime: Duration.zero,
            warmUpTime: Duration.zero,
            detail: 'CPU (chosen)',
            explicitCpu: true,
          ),
        ),
      };
    } else {
      models.value = {
        ModelId.yolo26n: const ModelFailed('CPU refused', backend: 'cpu'),
      };
    }
    return result;
  }

  LiveCameraViewModel create({
    LiveCameraSettingsRepository? settings,
    TargetPlatform platform = TargetPlatform.macOS,
  }) => LiveCameraViewModel(
    conversation: conversation,
    live: live,
    assistant: VoiceAssistant<CameraSideEvent>(
      speech: speech,
      audio: FakeAudioRepository()..autoDrain = true,
      responders: CameraTurnResponder(
        capture: live.capture,
        conversation: conversation,
        encode: live.encodeForLlm,
      ),
      diagnostics: diagnostics,
    ),
    diagnostics: diagnostics,
    activateStt: (_) async => const Result.ok(null),
    settings: settings ?? settingsWith(),
    models: models,
    reloadDetector: reload,
    platform: platform,
  );

  /// Pumps the event queue until [condition] holds, boundedly.
  Future<void> until(bool Function() condition, {String? reason}) async {
    for (var i = 0; i < 50 && !condition(); i++) {
      await pumpEventQueue();
    }
    expect(condition(), isTrue, reason: reason);
  }

  /// Demo 3 entered: its first start ended, then one more pump for the
  /// settings' load (it reads from memory, microtasks only).
  Future<void> entered(LiveCameraViewModel vm) async {
    await until(() => vm.startLive.result != null, reason: 'started');
    await pumpEventQueue();
  }

  /// The settings' queue ran dry.
  Future<void> applied(LiveCameraViewModel vm) =>
      until(() => !vm.applying, reason: 'the queue ran dry');

  /// Demo 3 entered on the saved network camera, which still connects; the
  /// saved choice is loaded.
  Future<void> connectingToSaved(LiveCameraViewModel vm) => until(
    () =>
        live.state.value is LiveStarting &&
        vm.sourceChoice.kind == CameraSourceKind.network,
    reason: 'connecting to the saved network camera',
  );

  group('view model', () {
    test('a saved network camera is what Demo 3 opens; its Retry is '
        'Reconnect', () async {
      await store.setString('camera.source', 'network');
      await store.setString('camera.networkUrl', _phone);
      final vm = create();
      await until(
        () =>
            live.state.value is LiveRunning &&
            vm.sourceChoice.kind == CameraSourceKind.network,
      );

      expect((specs.single as NetworkSourceSpec).url, Uri.parse(_phone));
      expect(vm.sourceIsNetwork, isTrue);
      expect(vm.retryLabel, 'Reconnect');
      expect(vm.sourceChoice.kind, CameraSourceKind.network);
      expect(live.state.value, isA<LiveRunning>());
      expect(
        (live.state.value as LiveRunning).source,
        'Network camera · 192.168.1.23:8080',
      );
      vm.dispose();
    });

    test('applying a network camera saves it and restarts live detection on '
        'it; back to the device camera likewise', () async {
      final vm = create();
      await entered(vm);
      expect(specs.single, isA<CameraSourceSpec>());

      expect(
        vm.applySettings(
          kind: CameraSourceKind.network,
          networkUrl: _phone,
          backend: DetectorBackend.gpu,
        ),
        isNull,
      );
      await applied(vm);
      expect(store.values['camera.source'], 'network');
      expect(specs.last, isA<NetworkSourceSpec>());
      expect(sources.first.stopped, isTrue, reason: 'the device camera');
      expect(log.where((l) => l == 'reload'), isEmpty);

      vm.applySettings(
        kind: CameraSourceKind.device,
        networkUrl: _phone,
        backend: DetectorBackend.gpu,
      );
      await applied(vm);
      expect(specs.last, isA<CameraSourceSpec>());
      expect(store.values['camera.networkUrl'], _phone, reason: 'kept');
      vm.dispose();
    });

    test('the placeholder URL is refused with what to fix; nothing is '
        'applied', () async {
      final vm = create();
      await entered(vm);
      final error = vm.applySettings(
        kind: CameraSourceKind.network,
        networkUrl: kNetworkCameraUrlExample,
        backend: DetectorBackend.gpu,
      );
      await pumpEventQueue();
      expect(error, contains('Replace 192.168.x.x'));
      expect(specs.length, 1);
      expect(store.values, isEmpty);
      vm.dispose();
    });

    test('a source fixed by the build is not changed by the sheet', () async {
      final vm = create(
        settings: settingsWith(
          environment: const Result.ok(FixtureSourceSpec(['/fixtures'])),
        ),
      );
      await entered(vm);
      expect(vm.sourceLock, 'FRAME_SOURCE=fixture');
      vm.applySettings(
        kind: CameraSourceKind.network,
        networkUrl: _phone,
        backend: DetectorBackend.gpu,
      );
      await applied(vm);
      expect(specs, [isA<FixtureSourceSpec>()]);
      vm.dispose();
    });

    test('the device camera failing offers the network camera (first on '
        'Linux)', () async {
      deviceCameraFails = true;
      final vm = create(platform: TargetPlatform.linux);
      await entered(vm);
      expect(vm.startError, contains('No camera found'));
      expect(vm.offerNetworkCamera, isTrue);
      expect(vm.preferNetworkCamera, isTrue);
      vm.dispose();
    });

    test(
      'a GPU failure is shown with the CPU as the way out: switching '
      'stops the source, reloads the detector, restarts — in that order',
      () async {
        models.value = {
          ModelId.yolo26n: const ModelFailed(
            'YOLO26n is only partly on the GPU',
            backend: 'gpu',
          ),
        };
        detector.info = null;
        final vm = create();
        await entered(vm);
        expect(vm.detectorFailure?.backend, 'gpu');
        expect(vm.detectorFailure?.message, contains('only partly on the GPU'));
        expect(vm.canChooseBackend, isTrue);
        expect(vm.startError, contains('detector is not loaded'));

        vm.runDetectorOn(DetectorBackend.cpu);
        await until(() => !vm.applying);

        expect(store.values['detector.backend'], 'cpu');
        // No source opened before: the first start failed on the detector.
        expect(log, ['reload', 'start CameraSourceSpec']);
        expect(vm.detectorFailure, isNull);
        expect(vm.backend, DetectorBackend.cpu);
        expect(vm.detectorLabel, 'CPU (chosen)');
        expect(vm.detectorOnCpu, isTrue);
        expect(live.state.value, isA<LiveRunning>());
        vm.dispose();
      },
    );

    test('switching while running stops the running source before the '
        'reload; a failed reload does not restart', () async {
      final vm = create();
      await entered(vm);
      expect(live.state.value, isA<LiveRunning>());
      reloadResult = () =>
          const Result.error(FrameSourceUnavailableException('CPU refused'));

      vm.runDetectorOn(DetectorBackend.cpu);
      await until(() => !vm.applying);

      expect(sources.single.stopped, isTrue);
      expect(log, ['start CameraSourceSpec', 'reload']);
      expect(vm.setDetectorBackend.error, isTrue);
      expect(vm.detectorFailure?.backend, 'cpu');
      expect(live.state.value, isA<LiveStopped>());
      vm.dispose();
    });

    test('DETECTOR_BACKEND locks the backend: no switch', () async {
      final vm = create(settings: settingsWith(backendDefine: 'gpu'));
      await entered(vm);
      expect(vm.canChooseBackend, isFalse);
      expect(vm.backendLock, 'DETECTOR_BACKEND=gpu');
      vm.runDetectorOn(DetectorBackend.cpu);
      expect(vm.applying, isFalse);
      await pumpEventQueue();
      expect(log.where((l) => l == 'reload'), isEmpty);
      vm.dispose();
    });

    test(
      'a second Apply while the first still connects is not dropped: the '
      'connecting start is stopped and the newest choice runs next',
      () async {
        final vm = create();
        await entered(vm);
        holdNetworkStart = Completer<void>();
        const urlA = 'http://192.168.1.99:8080/video'; // nobody there
        vm.applySettings(
          kind: CameraSourceKind.network,
          networkUrl: urlA,
          backend: DetectorBackend.gpu,
        );
        await until(() => live.state.value is LiveStarting);
        expect(vm.applying, isTrue);
        final connectingA = sources.last;
        expect((specs.last as NetworkSourceSpec).url.host, '192.168.1.99');

        vm.applySettings(
          kind: CameraSourceKind.network,
          networkUrl: _phone,
          backend: DetectorBackend.gpu,
        );
        await until(() => connectingA.stopped);
        expect(
          connectingA.stopped,
          isTrue,
          reason: 'A stopped while connecting',
        );
        holdNetworkStart!.complete(); // A's (fake) start returns; B starts
        await until(() => !vm.applying);

        expect((specs.last as NetworkSourceSpec).url, Uri.parse(_phone));
        expect(store.values['camera.networkUrl'], _phone);
        expect(vm.sourceChoice.networkUrl, _phone);
        expect(live.state.value, isA<LiveRunning>());
        expect(
          (live.state.value as LiveRunning).source,
          'Network camera · 192.168.1.23:8080',
        );
        expect(vm.applying, isFalse);
        expect(vm.startError, isNull);
        vm.dispose();
      },
    );

    test('a reload that ends unavailable (no backend to offer) still shows '
        'why', () async {
      final vm = create();
      await entered(vm);
      reloadResult = () =>
          const Result.error(FrameSourceUnavailableException('bad file'));
      vm.runDetectorOn(DetectorBackend.cpu);
      await applied(vm);
      models.value = {
        ModelId.yolo26n: const ModelUnavailable('DETECTOR_BACKEND must be …'),
      };
      expect(vm.detectorFailure?.message, contains('DETECTOR_BACKEND'));
      expect(vm.detectorFailure?.backend, isNull);
      vm.dispose();
    });

    test('both changed: the source first, then the detector switch — never '
        'both at once', () async {
      final vm = create();
      await entered(vm);
      vm.applySettings(
        kind: CameraSourceKind.network,
        networkUrl: _phone,
        backend: DetectorBackend.cpu,
      );
      await until(() => !vm.applying);
      expect(log, [
        'start CameraSourceSpec',
        'start NetworkSourceSpec',
        'reload',
        'start NetworkSourceSpec',
      ]);
      vm.dispose();
    });
  });

  // The apply queue's waits, merges and failures, pinned before the queue
  // moved out of the view model (LiveSettingsApplier).
  group('view model: applies, waits and failures', () {
    const urlA = 'http://192.168.1.99:8080/video'; // nobody there

    Future<void> saveNetworkCamera() async {
      await store.setString('camera.source', 'network');
      await store.setString('camera.networkUrl', _phone);
    }

    test('saved settings that cannot be read: both reasons, joined; the '
        'defaults stay', () async {
      await store.setString('camera.source', 'bogus');
      await store.setString('detector.backend', 'tpu');
      final vm = create();
      await until(() => vm.settingsError != null);
      expect(vm.settingsError, contains('"bogus" is not device or network'));
      expect(vm.settingsError, contains(' · '));
      expect(vm.settingsError, contains('tpu'));
      expect(vm.sourceChoice, CameraSourceChoice.standard);
      expect(vm.backend, DetectorBackend.gpu);
      vm.dispose();
    });

    test('a camera source that cannot be saved: the action error says so; '
        'nothing restarts', () async {
      final vm = create();
      await entered(vm);
      store.failWritesOf.add('camera.source');
      vm.applySettings(
        kind: CameraSourceKind.network,
        networkUrl: _phone,
        backend: DetectorBackend.gpu,
      );
      await applied(vm);
      expect(
        vm.settingsActionError,
        startsWith('Could not apply the camera source: '),
      );
      expect(specs, hasLength(1));
      expect(vm.sourceChoice, CameraSourceChoice.standard);
      expect(live.state.value, isA<LiveRunning>());
      expect(vm.applying, isFalse);
      vm.dispose();
    });

    test('a detector backend that cannot be saved: the action error says so; '
        'no reload, the source keeps running', () async {
      final vm = create();
      await entered(vm);
      store.failWritesOf.add('detector.backend');
      vm.runDetectorOn(DetectorBackend.cpu);
      expect(vm.applying, isTrue);
      await applied(vm);
      expect(
        vm.settingsActionError,
        startsWith('Could not switch the detector: '),
      );
      expect(log, ['start CameraSourceSpec']);
      expect(vm.backend, DetectorBackend.gpu);
      expect(live.state.value, isA<LiveRunning>());
      expect(vm.applying, isFalse);
      vm.dispose();
    });

    test("a failed reload shows the detector's own failure, not the action "
        'error', () async {
      final vm = create();
      await entered(vm);
      reloadResult = () =>
          const Result.error(FrameSourceUnavailableException('CPU refused'));
      vm.runDetectorOn(DetectorBackend.cpu);
      await applied(vm);
      expect(vm.setDetectorBackend.error, isTrue);
      expect(vm.detectorFailure?.message, 'CPU refused');
      expect(vm.settingsActionError, isNull);
      expect(vm.backend, DetectorBackend.cpu, reason: 'saved before the load');
      vm.dispose();
    });

    test('switching the detector while the saved network camera still '
        'connects: that start is stopped and waited for, then the reload '
        'and the restart', () async {
      await saveNetworkCamera();
      holdNetworkStart = Completer<void>();
      final vm = create();
      await connectingToSaved(vm);
      expect(vm.startLive.running, isTrue);
      final connecting = sources.single;

      vm.runDetectorOn(DetectorBackend.cpu);
      await until(() => connecting.stopped);
      expect(connecting.stopped, isTrue, reason: 'stopped while connecting');
      expect(vm.detectorReloading, isTrue);
      expect(log, ['start NetworkSourceSpec'], reason: 'waits for the start');

      holdNetworkStart!.complete();
      await until(() => !vm.applying);
      expect(log, [
        'start NetworkSourceSpec',
        'reload',
        'start NetworkSourceSpec',
      ]);
      expect(vm.detectorReloading, isFalse);
      expect(live.state.value, isA<LiveRunning>());
      vm.dispose();
    });

    test('an Apply while the first start still connects (no Apply running): '
        'that start is stopped and waited for, then the new source '
        'starts', () async {
      await saveNetworkCamera();
      holdNetworkStart = Completer<void>();
      final vm = create();
      await connectingToSaved(vm);
      expect(vm.startLive.running, isTrue);
      final connecting = sources.single;

      vm.applySettings(
        kind: CameraSourceKind.device,
        networkUrl: _phone,
        backend: DetectorBackend.gpu,
      );
      await until(() => connecting.stopped);
      expect(connecting.stopped, isTrue);
      expect(store.values['camera.source'], 'device');
      expect(log, ['start NetworkSourceSpec'], reason: 'waits for the start');
      expect(vm.applying, isTrue);

      holdNetworkStart!.complete();
      await until(() => !vm.applying);
      expect(log, ['start NetworkSourceSpec', 'start CameraSourceSpec']);
      expect(live.state.value, isA<LiveRunning>());
      expect(vm.sourceIsNetwork, isFalse);
      expect(vm.applying, isFalse);
      vm.dispose();
    });

    test('applying the same source again: nothing restarts while it runs; '
        'after a failed start it reconnects', () async {
      deviceCameraFails = true;
      final vm = create();
      await entered(vm);
      expect(vm.startError, contains('No camera found'));

      deviceCameraFails = false;
      vm.applySettings(
        kind: CameraSourceKind.device,
        networkUrl: kNetworkCameraUrlExample,
        backend: DetectorBackend.gpu,
      );
      await applied(vm);
      expect(specs, hasLength(2), reason: 'reconnected');
      expect(live.state.value, isA<LiveRunning>());
      expect(vm.startError, isNull);
      expect(store.values['camera.source'], 'device');

      vm.applySettings(
        kind: CameraSourceKind.device,
        networkUrl: kNetworkCameraUrlExample,
        backend: DetectorBackend.gpu,
      );
      await applied(vm);
      expect(specs, hasLength(2), reason: 'running: nothing to apply');
      expect(sources.last.stopped, isFalse);
      expect(vm.applying, isFalse);
      vm.dispose();
    });

    test('applying the same source again after a runtime failure '
        'reconnects', () async {
      final vm = create();
      await entered(vm);
      sources.single.failAtRuntime(Exception('unplugged'));
      await until(() => live.state.value is LiveFailed);
      expect(live.state.value, isA<LiveFailed>());

      vm.applySettings(
        kind: CameraSourceKind.device,
        networkUrl: kNetworkCameraUrlExample,
        backend: DetectorBackend.gpu,
      );
      await applied(vm);
      expect(specs, hasLength(2));
      expect(live.state.value, isA<LiveRunning>());
      vm.dispose();
    });

    test('the same network camera applied twice while it connects: the '
        'stopped start (not a live failure) reconnects it', () async {
      final vm = create();
      await entered(vm);
      holdNetworkStart = Completer<void>();
      vm.applySettings(
        kind: CameraSourceKind.network,
        networkUrl: _phone,
        backend: DetectorBackend.gpu,
      );
      await until(() => live.state.value is LiveStarting);
      final connecting = sources.last;
      final states = <LiveState>[];
      void record() => states.add(live.state.value);
      live.state.addListener(record);
      addTearDown(() => live.state.removeListener(record));

      vm.applySettings(
        kind: CameraSourceKind.network,
        networkUrl: _phone,
        backend: DetectorBackend.gpu,
      );
      await until(() => connecting.stopped);
      expect(connecting.stopped, isTrue);
      holdNetworkStart!.complete();
      await until(() => !vm.applying);
      expect(log, [
        'start CameraSourceSpec',
        'start NetworkSourceSpec',
        'start NetworkSourceSpec',
      ]);
      expect(states.whereType<LiveFailed>(), isEmpty, reason: 'only stopped');
      expect(live.state.value, isA<LiveRunning>());
      expect(vm.startError, isNull);
      expect(vm.applying, isFalse);
      vm.dispose();
    });

    test('pending choices merge field by field: the source of one Apply and '
        'the backend of a later action both run, the source first', () async {
      final vm = create();
      await entered(vm);
      holdNetworkStart = Completer<void>();
      vm.applySettings(
        kind: CameraSourceKind.network,
        networkUrl: urlA,
        backend: DetectorBackend.gpu,
      );
      await until(() => live.state.value is LiveStarting);
      vm
        ..applySettings(
          kind: CameraSourceKind.network,
          networkUrl: _phone,
          backend: DetectorBackend.gpu,
        )
        ..runDetectorOn(DetectorBackend.cpu);
      await pumpEventQueue();
      holdNetworkStart!.complete();
      await until(() => !vm.applying);
      expect(log, [
        'start CameraSourceSpec',
        'start NetworkSourceSpec',
        'start NetworkSourceSpec',
        'reload',
        'start NetworkSourceSpec',
      ]);
      expect((specs.last as NetworkSourceSpec).url, Uri.parse(_phone));
      expect(vm.backend, DetectorBackend.cpu);
      expect(vm.applying, isFalse);
      vm.dispose();
    });

    test('the newest pending backend wins: a later Apply on the GPU cancels a '
        'queued "Run detector on CPU"', () async {
      final vm = create();
      await entered(vm);
      holdNetworkStart = Completer<void>();
      vm.applySettings(
        kind: CameraSourceKind.network,
        networkUrl: urlA,
        backend: DetectorBackend.gpu,
      );
      await until(() => live.state.value is LiveStarting);
      vm
        ..runDetectorOn(DetectorBackend.cpu)
        ..applySettings(
          kind: CameraSourceKind.network,
          networkUrl: _phone,
          backend: DetectorBackend.gpu,
        );
      await pumpEventQueue();
      holdNetworkStart!.complete();
      await until(() => !vm.applying);
      expect(log.where((l) => l == 'reload'), isEmpty);
      expect((specs.last as NetworkSourceSpec).url, Uri.parse(_phone));
      expect(vm.backend, DetectorBackend.gpu);
      vm.dispose();
    });

    test('leaving ends the queue: no step runs after dispose', () async {
      final vm = create();
      await entered(vm);
      holdNetworkStart = Completer<void>();
      vm.applySettings(
        kind: CameraSourceKind.network,
        networkUrl: _phone,
        backend: DetectorBackend.cpu,
      );
      await until(() => live.state.value is LiveStarting);
      vm.dispose();
      holdNetworkStart!.complete();
      await until(() => !vm.applying);
      expect(log, ['start CameraSourceSpec', 'start NetworkSourceSpec']);
      expect(store.values['detector.backend'], isNull);
    });

    test('leaving while a switch waits for a connecting start ends the '
        'queue, though the start command no longer notifies', () async {
      await saveNetworkCamera();
      holdNetworkStart = Completer<void>();
      final vm = create();
      await connectingToSaved(vm);
      vm.runDetectorOn(DetectorBackend.cpu);
      await until(() => sources.single.stopped);
      expect(vm.applying, isTrue, reason: 'waits for the start');

      vm.dispose();
      holdNetworkStart!.complete(); // the aborted start returns after
      await until(() => !vm.applying, reason: 'the wait ended with dispose');
      expect(log, ['start NetworkSourceSpec'], reason: 'no reload, no start');
    });

    test('Retry while a start is in progress joins it: no second start, and '
        'it completes when that start ended', () async {
      await saveNetworkCamera();
      holdNetworkStart = Completer<void>();
      final vm = create();
      await connectingToSaved(vm);
      expect(vm.startLive.running, isTrue);

      var retried = false;
      unawaited(vm.retryStart().then((_) => retried = true));
      await pumpEventQueue();
      expect(retried, isFalse, reason: 'the start still connects');
      expect(specs, hasLength(1));

      holdNetworkStart!.complete();
      await until(() => retried);
      expect(specs, hasLength(1));
      expect(live.state.value, isA<LiveRunning>());
      vm.dispose();
    });

    test('a locked backend queues nothing for "Run detector on CPU"', () async {
      final vm = create(settings: settingsWith(backendDefine: 'gpu'));
      await entered(vm);
      vm.runDetectorOn(DetectorBackend.cpu);
      expect(vm.applying, isFalse);
      vm.dispose();
    });

    test('settings without a detector reload: the backend cannot be chosen; '
        'its switch refuses', () async {
      final vm = LiveCameraViewModel(
        conversation: conversation,
        live: live,
        assistant: VoiceAssistant<CameraSideEvent>(
          speech: speech,
          audio: FakeAudioRepository()..autoDrain = true,
          responders: CameraTurnResponder(
            capture: live.capture,
            conversation: conversation,
            encode: live.encodeForLlm,
          ),
          diagnostics: diagnostics,
        ),
        diagnostics: diagnostics,
        activateStt: (_) async => const Result.ok(null),
        settings: settingsWith(),
        models: models,
      );
      await entered(vm);
      expect(vm.settingsAvailable, isTrue);
      expect(vm.canChooseBackend, isFalse);
      vm.runDetectorOn(DetectorBackend.cpu);
      expect(vm.applying, isFalse);
      await pumpEventQueue();
      expect(vm.settingsActionError, isNull);
      expect(store.values['detector.backend'], isNull);
      vm.dispose();
    });

    test('without the settings repository: no settings, both switches '
        'refuse, an Apply changes nothing', () async {
      final vm = LiveCameraViewModel(
        conversation: conversation,
        live: live,
        assistant: VoiceAssistant<CameraSideEvent>(
          speech: speech,
          audio: FakeAudioRepository()..autoDrain = true,
          responders: CameraTurnResponder(
            capture: live.capture,
            conversation: conversation,
            encode: live.encodeForLlm,
          ),
          diagnostics: diagnostics,
        ),
        diagnostics: diagnostics,
        activateStt: (_) async => const Result.ok(null),
        source: const Result.ok(CameraSourceSpec()),
        models: models,
        reloadDetector: reload,
      );
      await entered(vm);
      expect(vm.settingsAvailable, isFalse);
      expect(vm.sourceLock, isNull);
      expect(vm.backendLock, isNull);
      expect(vm.canChooseBackend, isFalse);
      expect(vm.offerNetworkCamera, isFalse);
      expect(vm.settingsError, isNull);

      expect(
        vm.applySettings(
          kind: CameraSourceKind.network,
          networkUrl: 'not a url',
          backend: DetectorBackend.cpu,
        ),
        isNull,
        reason: 'nothing editable: nothing to check',
      );
      expect(vm.applying, isTrue, reason: 'an empty Apply still queues');
      await applied(vm);
      expect(vm.applying, isFalse);
      expect(specs, hasLength(1));
      expect(log.where((l) => l == 'reload'), isEmpty);
      expect(vm.applySource.result, isNull, reason: 'never ran');
      expect(vm.setDetectorBackend.result, isNull, reason: 'never ran');
      vm.dispose();
    });
  });

  group('screen', () {
    /// Real time for the repository's work and frames for the view model's
    /// (it runs in the test's fake zone), in turns, until [condition] holds;
    /// bounded.
    Future<void> pumpUntil(
      WidgetTester tester,
      bool Function() condition, {
      String? reason,
    }) async {
      for (var i = 0; i < 50 && !condition(); i++) {
        await tester.runAsync(pumpEventQueue);
        await tester.pump();
      }
      expect(condition(), isTrue, reason: reason);
    }

    Future<LiveCameraViewModel> pump(
      WidgetTester tester, {
      TargetPlatform platform = TargetPlatform.macOS,
    }) async {
      late LiveCameraViewModel vm;
      await tester.pumpWidget(
        MaterialApp(
          home: ChangeNotifierProvider(
            create: (_) => vm = create(platform: platform),
            child: const LiveCameraScreen(),
          ),
        ),
      );
      await pumpUntil(
        tester,
        () => vm.startLive.result != null,
        reason: 'started',
      );
      return vm;
    }

    /// The view model's stop runs in the test's fake zone: close the
    /// repository in real time here, not in tearDown.
    Future<void> tearDownScreen(WidgetTester tester) async {
      await tester.pumpWidget(const SizedBox.shrink());
      await tester.pump(Duration.zero);
      await tester.runAsync(live.close);
    }

    testWidgets('the detector failure card: reason and "Run detector on '
        'CPU"; tapping it reloads and the chip says CPU (chosen)', (
      tester,
    ) async {
      models.value = {
        ModelId.yolo26n: const ModelFailed(
          'GPU rejected YOLO26n: Vulkan device lost',
          backend: 'gpu',
        ),
      };
      detector.info = null;
      final vm = await pump(tester);

      expect(find.byKey(LiveCameraKeys.detectorFailure), findsOneWidget);
      expect(
        find.textContaining('The detector failed on the GPU: GPU rejected'),
        findsOneWidget,
      );
      await tester.tap(find.byKey(LiveCameraKeys.otherBackend));
      await pumpUntil(tester, () => !vm.applying, reason: 'switched');

      expect(find.byKey(LiveCameraKeys.detectorFailure), findsNothing);
      expect(find.text('fake · CPU (chosen)'), findsOneWidget);
      await tearDownScreen(tester);
    });

    testWidgets('the settings sheet applies a network camera URL', (
      tester,
    ) async {
      final vm = await pump(tester);
      await tester.tap(find.byKey(LiveCameraKeys.settings));
      await tester.pumpAndSettle();
      expect(find.byKey(CameraSettingsKeys.sheet), findsOneWidget);

      await tester.tap(find.byKey(CameraSettingsKeys.networkCamera));
      await tester.pump();
      // The placeholder is refused in place.
      await tester.tap(find.byKey(CameraSettingsKeys.apply));
      await tester.pump();
      expect(find.textContaining('Replace 192.168.x.x'), findsOneWidget);

      await tester.enterText(find.byKey(CameraSettingsKeys.url), _phone);
      await tester.tap(find.byKey(CameraSettingsKeys.apply));
      await pumpUntil(tester, () => !vm.applying, reason: 'applied');
      await tester.pumpAndSettle();

      expect(find.byKey(CameraSettingsKeys.sheet), findsNothing);
      expect(specs.last, isA<NetworkSourceSpec>());
      expect(store.values['camera.networkUrl'], _phone);
      await tearDownScreen(tester);
    });

    testWidgets('on Linux a missing device camera offers the network camera '
        'first, as a labelled button', (tester) async {
      deviceCameraFails = true;
      await pump(tester, platform: TargetPlatform.linux);

      expect(find.textContaining('No camera found'), findsOneWidget);
      expect(find.text('Use a network camera'), findsOneWidget);
      expect(find.text('Camera'), findsOneWidget, reason: 'app bar button');
      await tester.tap(find.byKey(LiveCameraKeys.useNetworkCamera));
      await tester.pumpAndSettle();
      expect(find.byKey(CameraSettingsKeys.sheet), findsOneWidget);
      await tearDownScreen(tester);
    });
  });

  test('the overlay labels the chosen CPU', () async {
    models.value = {
      ModelId.yolo26n: const ModelReady(
        LoadedModelInfo(
          modelId: 'yolo26n',
          backend: 'cpu',
          loadTime: Duration.zero,
          warmUpTime: Duration.zero,
          detail: 'CPU (chosen)',
          explicitCpu: true,
        ),
      ),
    };
    final liveState = ValueNotifier<LiveState>(const LiveRunning('fake'));
    final repo = DiagnosticsRepository(
      models: models,
      liveState: liveState,
      minInterval: Duration.zero,
      sampleRss: false,
    );
    addTearDown(() {
      repo.dispose();
      liveState.dispose();
    });
    // minInterval zero: the coalesced flush is a zero-length timer.
    await pumpEventQueue();
    expect(
      debugOverlayLines(repo.snapshot.value),
      contains(startsWith('det CPU (chosen) ·')),
    );
  });
}
