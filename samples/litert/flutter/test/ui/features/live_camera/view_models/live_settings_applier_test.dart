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

import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/repositories/live_camera_settings_repository.dart';
import 'package:litert_edge_demos/data/repositories/live_detection_repository.dart';
import 'package:litert_edge_demos/data/services/settings/settings_store.dart';
import 'package:litert_edge_demos/data/services/settings/typed_settings.dart';
import 'package:litert_edge_demos/domain/models/camera_source.dart';
import 'package:litert_edge_demos/domain/models/detection.dart';
import 'package:litert_edge_demos/domain/models/frame_source_info.dart';
import 'package:litert_edge_demos/domain/models/frame_source_spec.dart';
import 'package:litert_edge_demos/domain/models/live_state.dart';
import 'package:litert_edge_demos/ui/features/live_camera/view_models/live_settings_applier.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../../../fakes/fake_detector.dart';
import '../../../../fakes/fake_frame_source.dart';
import '../../../../fakes/fake_settings_store.dart';

const _phone = 'http://192.168.1.23:8080/video';
const _nobody = 'http://192.168.1.99:8080/video';

/// The view model's side of the applier, played: a start opens the
/// settings' source on the live repository for [owner] (as the view model's
/// `startLive` does) and is kept while it runs, so a second start joins it;
/// a stop stops [owner]'s source.
final class _Screen {
  _Screen({required this.live, required this.settings, required this.owner});

  final LiveDetectionRepository live;
  final LiveCameraSettingsRepository settings;
  final Object owner;

  /// The start in progress; null when none runs.
  Future<void>? running;
  Result<FrameSourceInfo>? lastStart;

  bool get starting => running != null;
  bool get startFailed => lastStart is Error<FrameSourceInfo>;
  bool needsReconnect() => startFailed || live.state.value is LiveFailed;

  Future<void> start() =>
      running ??= _start().whenComplete(() => running = null);

  Future<void> _start() async {
    lastStart = switch (await settings.resolveSource()) {
      Ok(:final value) => await live.start(value, owner: owner),
      Error(:final error) => Result.error(error),
    };
  }

  Future<void> stop() => live.stop(owner: owner);
}

/// [SettingsStore] whose next read of one key can be held: it takes the
/// value when it is made and delivers it only once released — a read that
/// began before a save and ends after it.
final class _HeldReads implements SettingsStore {
  _HeldReads(this._store);

  final SettingsStore _store;
  String? _key;
  Completer<void>? _release;

  /// A held read has its value and waits for its release.
  bool waiting = false;

  /// Holds the next read of [key]; completing the result delivers it.
  Completer<void> hold(String key) {
    _key = key;
    return _release = Completer<void>();
  }

  @override
  Future<String?> getString(String key) async {
    final value = await _store.getString(key);
    final release = _release;
    if (key == _key && release != null) {
      _key = null;
      _release = null;
      waiting = true;
      await release.future;
      waiting = false;
    }
    return value;
  }

  @override
  Future<bool?> getBool(String key) => _store.getBool(key);

  @override
  Future<void> setString(String key, String value) =>
      _store.setString(key, value);

  @override
  Future<void> setBool(String key, {required bool value}) =>
      _store.setBool(key, value: value);

  @override
  Future<void> remove(String key) => _store.remove(key);
}

/// The settings repository, with every save of a choice logged: the step
/// that saves it has begun.
final class _LoggedSettings extends LiveCameraSettingsRepository {
  _LoggedSettings({
    required super.settings,
    required super.environment,
    required super.backendDefine,
    required this.log,
  });

  final List<String> log;

  @override
  Future<Result<void>> saveSource(CameraSourceChoice choice) {
    log.add('source ${choice.kind.name}');
    return super.saveSource(choice);
  }

  @override
  Future<Result<void>> saveBackend(DetectorBackend backend) {
    log.add('backend ${backend.name}');
    return super.saveBackend(backend);
  }
}

void main() {
  late InMemorySettingsStore store;
  late _HeldReads reads;
  late List<String> log;
  late List<FakeFrameSource> sources;
  late LiveDetectionRepository live;
  late Object owner;
  late int changes;
  late Result<void> Function() reloadResult;
  bool deviceCameraFails = false;
  Completer<void>? holdNetworkStart;
  Completer<void>? holdReload;

  setUp(() {
    store = InMemorySettingsStore();
    reads = _HeldReads(store);
    log = [];
    sources = [];
    owner = Object();
    changes = 0;
    reloadResult = () => const Result.ok(null);
    deviceCameraFails = false;
    holdNetworkStart = null;
    holdReload = null;
    live = LiveDetectionRepository(
      detector: FakeDetector(autoComplete: true),
      createSource: (spec) {
        log.add('start ${spec.runtimeType}');
        if (spec is CameraSourceSpec && deviceCameraFails) {
          return const Result.error(
            FrameSourceUnavailableException('No camera found'),
          );
        }
        final source = FakeFrameSource()
          ..startGate = spec is NetworkSourceSpec ? holdNetworkStart : null;
        sources.add(source);
        return Result.ok(source);
      },
    );
  });

  tearDown(() => live.close());

  LiveCameraSettingsRepository settingsWith({
    Result<FrameSourceSpec> environment = const Result.ok(CameraSourceSpec()),
    String backendDefine = '',
  }) => _LoggedSettings(
    settings: TypedSettings(store: reads),
    environment: environment,
    backendDefine: backendDefine,
    log: log,
  );

  Future<Result<void>> reload() async {
    log.add('reload');
    await holdReload?.future;
    return reloadResult();
  }

  /// The applier over [settings] (none: `withoutSettings`), with the
  /// screen it starts and stops live detection through; nothing started
  /// yet.
  (LiveSettingsApplier, _Screen) create({
    LiveCameraSettingsRepository? settings,
    bool withoutSettings = false,
    bool canReload = true,
  }) {
    final repository = settings ?? settingsWith();
    final screen = _Screen(live: live, settings: repository, owner: owner);
    final applier = LiveSettingsApplier(
      start: screen.start,
      runningStart: () => screen.running,
      stop: screen.stop,
      needsReconnect: screen.needsReconnect,
      onSwitch: () => log.add('switch'),
      onChanged: () => changes++,
      settings: withoutSettings ? null : repository,
      reloadDetector: canReload ? reload : null,
    );
    addTearDown(applier.close);
    return (applier, screen);
  }

  /// The applier with its saved choices loaded and live detection running.
  Future<(LiveSettingsApplier, _Screen)> running({
    LiveCameraSettingsRepository? settings,
  }) async {
    final (applier, screen) = create(settings: settings);
    await applier.load();
    await screen.start();
    log.clear();
    changes = 0;
    return (applier, screen);
  }

  /// Runs the camera-source step as the queue does; its result.
  Future<Result<void>?> sourceStep(
    LiveSettingsApplier applier,
    CameraSourceChoice choice,
  ) async {
    await applier.applySource.execute(choice);
    return applier.applySource.result;
  }

  /// Runs the detector-backend step as the queue does; its result.
  Future<Result<void>?> backendStep(
    LiveSettingsApplier applier,
    DetectorBackend backend,
  ) async {
    await applier.setDetectorBackend.execute(backend);
    return applier.setDetectorBackend.result;
  }

  /// Pumps the event queue until [condition] holds, boundedly.
  Future<void> until(bool Function() condition, {String? reason}) async {
    for (var i = 0; i < 50 && !condition(); i++) {
      await pumpEventQueue();
    }
    expect(condition(), isTrue, reason: reason);
  }

  /// The queue ran dry.
  Future<void> applied(LiveSettingsApplier applier) =>
      until(() => !applier.applying, reason: 'the queue ran dry');

  group('loading and what can be chosen', () {
    test('the saved choices load; the change is announced', () async {
      await store.setString('camera.source', 'network');
      await store.setString('camera.networkUrl', _phone);
      await store.setString('detector.backend', 'cpu');
      final (applier, _) = create();
      expect(applier.sourceChoice, CameraSourceChoice.standard);
      expect(applier.backend, DetectorBackend.gpu);

      await applier.load();
      expect(applier.sourceChoice.kind, CameraSourceKind.network);
      expect(applier.sourceChoice.networkUrl, _phone);
      expect(applier.backend, DetectorBackend.cpu);
      expect(applier.loadError, isNull);
      expect(changes, 1);
    });

    test(
      'unreadable choices keep their defaults and say why, joined',
      () async {
        await store.setString('camera.source', 'bogus');
        await store.setString('detector.backend', 'tpu');
        final (applier, _) = create();
        await applier.load();
        expect(applier.loadError, contains('"bogus" is not device or network'));
        expect(applier.loadError, contains(' · '));
        expect(applier.sourceChoice, CameraSourceChoice.standard);
        expect(applier.backend, DetectorBackend.gpu);
      },
    );

    test('a load that read the camera source before an Apply saved it '
        'keeps the applied source', () async {
      final (applier, screen) = create();
      await screen.start();
      final release = reads.hold('camera.source');
      final loading = applier.load(); // reads the device camera, held
      await until(() => reads.waiting);

      applier.apply(
        kind: CameraSourceKind.network,
        networkUrl: _phone,
        backend: DetectorBackend.gpu,
      );
      await until(() => !applier.applying);
      expect(applier.sourceChoice.kind, CameraSourceKind.network);

      release.complete();
      await loading;
      expect(
        applier.sourceChoice.kind,
        CameraSourceKind.network,
        reason: 'not the device camera the load read before the save',
      );
      expect(applier.sourceChoice.networkUrl, _phone);
      expect(store.values['camera.source'], 'network');
    });

    test('a load that read the detector backend before a switch saved it '
        'keeps the switched backend', () async {
      final (applier, screen) = create();
      await screen.start();
      final release = reads.hold('detector.backend');
      final loading = applier.load(); // reads the GPU, held
      await until(() => reads.waiting);

      applier.runDetectorOn(DetectorBackend.cpu);
      await until(() => !applier.applying);
      expect(applier.backend, DetectorBackend.cpu);

      release.complete();
      await loading;
      expect(
        applier.backend,
        DetectorBackend.cpu,
        reason: 'not the GPU the load read before the save',
      );
      expect(store.values['detector.backend'], 'cpu');
      expect(applier.loadError, isNull);
    });

    test('an Apply of the default choices before the load has read the '
        'saved ones is applied, not taken for "unchanged"', () async {
      await store.setString('camera.source', 'network');
      await store.setString('camera.networkUrl', _phone);
      await store.setString('detector.backend', 'cpu');
      final (applier, screen) = create();
      final release = reads.hold('camera.source');
      final loading = applier.load(); // reads the network camera, held
      await until(() => reads.waiting);

      applier.apply(
        kind: CameraSourceKind.device,
        networkUrl: kNetworkCameraUrlExample, // the defaults, exactly
        backend: DetectorBackend.gpu,
      );
      await applied(applier);
      expect(log, contains('source device'));
      expect(log, contains('backend gpu'));
      expect(store.values['camera.source'], 'device');
      expect(store.values['detector.backend'], 'gpu');

      release.complete();
      await loading;
      expect(applier.sourceChoice, CameraSourceChoice.standard);
      expect(applier.backend, DetectorBackend.gpu);
      expect(screen.lastStart, isA<Ok<FrameSourceInfo>>());
    });

    test('"Run detector on GPU" before the load has read the saved CPU is '
        'applied', () async {
      await store.setString('detector.backend', 'cpu');
      final (applier, _) = create();
      final release = reads.hold('detector.backend');
      final loading = applier.load(); // reads the CPU, held
      await until(() => reads.waiting);

      applier.runDetectorOn(DetectorBackend.gpu);
      await applied(applier);
      expect(log, containsAllInOrder(['backend gpu', 'reload']));
      expect(store.values['detector.backend'], 'gpu');

      release.complete();
      await loading;
      expect(applier.backend, DetectorBackend.gpu);
    });

    test('a load that ends after close changes nothing', () async {
      await store.setString('detector.backend', 'cpu');
      final (applier, _) = create();
      final loading = applier.load();
      applier.close();
      await loading;
      expect(applier.backend, DetectorBackend.gpu);
      expect(changes, 0);
    });

    test('what can be chosen: everything with settings and a reload; the '
        "build's defines lock their setting; nothing without settings", () {
      final (all, _) = create();
      expect(all.available, isTrue);
      expect(all.sourceEditable, isTrue);
      expect(all.canChooseBackend, isTrue);
      expect(all.sourceLock, isNull);
      expect(all.backendLock, isNull);

      final (fixture, _) = create(
        settings: settingsWith(
          environment: const Result.ok(FixtureSourceSpec(['/fixtures'])),
        ),
      );
      expect(fixture.sourceLock, 'FRAME_SOURCE=fixture');
      expect(fixture.sourceEditable, isFalse);
      expect(fixture.canChooseBackend, isTrue);

      final (locked, _) = create(settings: settingsWith(backendDefine: 'gpu'));
      expect(locked.backendLock, 'DETECTOR_BACKEND=gpu');
      expect(locked.canChooseBackend, isFalse);
      expect(locked.sourceEditable, isTrue);

      final (noReload, _) = create(canReload: false);
      expect(noReload.canChooseBackend, isFalse);

      final (none, _) = create(withoutSettings: true);
      expect(none.available, isFalse);
      expect(none.sourceEditable, isFalse);
      expect(none.canChooseBackend, isFalse);
      expect(none.sourceLock, isNull);
      expect(none.backendLock, isNull);
    });

    test('a network camera URL is checked', () {
      final (applier, _) = create();
      expect(applier.validateNetworkUrl(_phone), isNull);
      expect(
        applier.validateNetworkUrl(kNetworkCameraUrlExample),
        contains('Replace 192.168.x.x'),
      );
    });
  });

  group('apply', () {
    test('a URL to fix is returned; nothing is queued or saved', () async {
      final (applier, _) = await running();
      expect(
        applier.apply(
          kind: CameraSourceKind.network,
          networkUrl: kNetworkCameraUrlExample,
          backend: DetectorBackend.gpu,
        ),
        contains('Replace 192.168.x.x'),
      );
      expect(applier.applying, isFalse);
      expect(changes, 0);
      await pumpEventQueue();
      expect(log, isEmpty);
      expect(store.values['camera.source'], isNull);
    });

    test(
      'a new source: queued at once, saved, the live view interrupted, '
      'the source restarted on it; the same backend reloads nothing',
      () async {
        final (applier, screen) = await running();
        expect(
          applier.apply(
            kind: CameraSourceKind.network,
            networkUrl: ' $_phone ',
            backend: DetectorBackend.gpu,
          ),
          isNull,
        );
        expect(applier.applying, isTrue, reason: 'before any await');
        expect(changes, 1);
        await applied(applier);

        expect(log, ['source network', 'switch', 'start NetworkSourceSpec']);
        expect(store.values['camera.networkUrl'], _phone, reason: 'trimmed');
        expect(applier.sourceChoice.networkUrl, _phone);
        expect(sources.first.stopped, isTrue, reason: 'the device camera');
        expect(applier.applySource.result, isA<Ok<void>>());
        expect(live.state.value, isA<LiveRunning>());
        expect(applier.applying, isFalse);
        expect(changes, 2, reason: 'queued, then done');
      },
    );

    test('both changed: the source first, then the detector switch (stop, '
        'reload, restart) — never both at once', () async {
      final (applier, screen) = await running();
      applier.apply(
        kind: CameraSourceKind.network,
        networkUrl: _phone,
        backend: DetectorBackend.cpu,
      );
      await applied(applier);
      expect(log, [
        'source network',
        'switch',
        'start NetworkSourceSpec',
        'backend cpu',
        'switch',
        'reload',
        'start NetworkSourceSpec',
      ]);
      expect(sources[1].stopped, isTrue, reason: 'stopped for the reload');
      expect(store.values['detector.backend'], 'cpu');
      expect(applier.backend, DetectorBackend.cpu);
      expect(applier.setDetectorBackend.result, isA<Ok<void>>());
      expect(live.state.value, isA<LiveRunning>());
    });

    test('the same source while it runs: nothing; after a failed start: it '
        'reconnects', () async {
      deviceCameraFails = true;
      final (applier, screen) = await running();
      expect(screen.startFailed, isTrue);

      deviceCameraFails = false;
      applier.apply(
        kind: CameraSourceKind.device,
        networkUrl: kNetworkCameraUrlExample,
        backend: DetectorBackend.gpu,
      );
      await applied(applier);
      expect(log, ['source device', 'switch', 'start CameraSourceSpec']);
      expect(live.state.value, isA<LiveRunning>());

      log.clear();
      applier.apply(
        kind: CameraSourceKind.device,
        networkUrl: kNetworkCameraUrlExample,
        backend: DetectorBackend.gpu,
      );
      expect(applier.applying, isTrue);
      await applied(applier);
      expect(log, isEmpty);
    });

    test('the same source after a runtime failure reconnects', () async {
      final (applier, _) = await running();
      sources.single.failAtRuntime(Exception('unplugged'));
      await until(() => live.state.value is LiveFailed);

      applier.apply(
        kind: CameraSourceKind.device,
        networkUrl: kNetworkCameraUrlExample,
        backend: DetectorBackend.gpu,
      );
      await applied(applier);
      expect(log, ['source device', 'switch', 'start CameraSourceSpec']);
      expect(live.state.value, isA<LiveRunning>());
    });

    test('a locked source and a locked backend are left out', () async {
      final (applier, _) = await running(
        settings: settingsWith(
          environment: const Result.ok(FixtureSourceSpec(['/fixtures'])),
          backendDefine: 'gpu',
        ),
      );
      expect(
        applier.apply(
          kind: CameraSourceKind.network,
          networkUrl: 'not a url',
          backend: DetectorBackend.cpu,
        ),
        isNull,
        reason: 'the source is not editable: its URL is not checked',
      );
      expect(applier.applying, isTrue, reason: 'an empty Apply still queues');
      await applied(applier);
      expect(log, isEmpty);
    });

    test('"Run detector on CPU" is queued only when the backend can be '
        'chosen', () async {
      final (locked, _) = create(settings: settingsWith(backendDefine: 'gpu'));
      locked.runDetectorOn(DetectorBackend.cpu);
      expect(locked.applying, isFalse);
      expect(changes, 0);

      final (applier, _) = await running();
      applier.runDetectorOn(DetectorBackend.cpu);
      expect(applier.applying, isTrue);
      await applied(applier);
      expect(log, [
        'backend cpu',
        'switch',
        'reload',
        'start CameraSourceSpec',
      ]);
    });
  });

  group('serialized, newest wins', () {
    /// An Apply of a network camera nobody answers: it waits in its start.
    Future<(LiveSettingsApplier, _Screen)> connecting() async {
      final (applier, screen) = await running();
      holdNetworkStart = Completer<void>();
      applier.apply(
        kind: CameraSourceKind.network,
        networkUrl: _nobody,
        backend: DetectorBackend.gpu,
      );
      await until(() => live.state.value is LiveStarting);
      expect(screen.starting, isTrue);
      expect(applier.applying, isTrue);
      log.clear();
      return (applier, screen);
    }

    test('a choice that arrives while an Apply waits on a connecting start '
        'stops that start; the newest choice runs next', () async {
      final (applier, _) = await connecting();
      final connectingSource = sources.last;

      applier.apply(
        kind: CameraSourceKind.network,
        networkUrl: _phone,
        backend: DetectorBackend.gpu,
      );
      await until(() => connectingSource.stopped);
      await pumpEventQueue();
      expect(log, isEmpty, reason: 'waits for the stopped start to end');

      holdNetworkStart!.complete();
      await applied(applier);
      expect(log, ['source network', 'switch', 'start NetworkSourceSpec']);
      expect(applier.sourceChoice.networkUrl, _phone);
      expect(live.state.value, isA<LiveRunning>());
      expect(applier.applying, isFalse);
    });

    test(
      'the same choice again while it connects: its start is stopped '
      '(stopped, not failed) and the failed start makes it reconnect',
      () async {
        final (applier, screen) = await connecting();
        final states = <LiveState>[];
        void record() => states.add(live.state.value);
        live.state.addListener(record);
        addTearDown(() => live.state.removeListener(record));
        final connectingSource = sources.last;

        applier.apply(
          kind: CameraSourceKind.network,
          networkUrl: _nobody,
          backend: DetectorBackend.gpu,
        );
        await until(() => connectingSource.stopped);

        holdNetworkStart!.complete();
        await applied(applier);
        expect(log, ['source network', 'switch', 'start NetworkSourceSpec']);
        expect(states.whereType<LiveFailed>(), isEmpty, reason: 'only stopped');
        expect(states, contains(isA<LiveStopped>()));
        expect(screen.startFailed, isFalse, reason: 'the reconnect started');
        expect(live.state.value, isA<LiveRunning>());
        expect(applier.applying, isFalse);
      },
    );

    test('pending choices merge field by field: the source of one, the '
        'backend of a later one', () async {
      final (applier, _) = await connecting();
      applier
        ..apply(
          kind: CameraSourceKind.network,
          networkUrl: _phone,
          backend: DetectorBackend.gpu,
        )
        ..runDetectorOn(DetectorBackend.cpu);
      holdNetworkStart!.complete();
      await applied(applier);
      expect(log, [
        'source network',
        'switch',
        'start NetworkSourceSpec',
        'backend cpu',
        'switch',
        'reload',
        'start NetworkSourceSpec',
      ]);
    });

    test('the newest pending backend wins', () async {
      final (applier, _) = await connecting();
      applier
        ..runDetectorOn(DetectorBackend.cpu)
        ..apply(
          kind: CameraSourceKind.network,
          networkUrl: _phone,
          backend: DetectorBackend.gpu,
        );
      holdNetworkStart!.complete();
      await applied(applier);
      expect(log, ['source network', 'switch', 'start NetworkSourceSpec']);
      expect(applier.backend, DetectorBackend.gpu);
    });

    test('a step waits for a start still in progress, after stopping it '
        '(not only inside the queue)', () async {
      await store.setString('camera.source', 'network');
      await store.setString('camera.networkUrl', _phone);
      holdNetworkStart = Completer<void>();
      final (applier, screen) = create();
      await applier.load();
      unawaited(screen.start());
      await until(() => live.state.value is LiveStarting);
      expect(screen.starting, isTrue);
      final first = sources.single;

      final switching = backendStep(applier, DetectorBackend.cpu);
      await until(() => first.stopped);
      expect(log, ['start NetworkSourceSpec', 'backend cpu', 'switch']);

      holdNetworkStart!.complete();
      expect(await switching, isA<Ok<void>>());
      expect(log, [
        'start NetworkSourceSpec',
        'backend cpu',
        'switch',
        'reload',
        'start NetworkSourceSpec',
      ]);
    });

    test('close ends a wait on a start that never ends: the queue finishes '
        'and nothing after the wait runs', () async {
      await store.setString('camera.source', 'network');
      await store.setString('camera.networkUrl', _nobody);
      holdNetworkStart = Completer<void>(); // not before the close
      final (applier, screen) = create();
      await applier.load();
      unawaited(screen.start());
      await until(() => live.state.value is LiveStarting);
      expect(screen.starting, isTrue);

      applier.runDetectorOn(DetectorBackend.cpu);
      await until(() => sources.single.stopped);
      await pumpEventQueue();
      expect(applier.applying, isTrue, reason: 'waits for the start to end');

      applier.close();
      await applied(applier);
      expect(log, ['start NetworkSourceSpec', 'backend cpu', 'switch']);
      // Only so that tearDown's close of the repository is not queued
      // behind it.
      holdNetworkStart!.complete();
    });

    test('close ends the queue: a queued step never runs, a running one '
        'stops at its next step', () async {
      final (applier, _) = await running();
      holdReload = Completer<void>();
      applier.apply(
        kind: CameraSourceKind.network,
        networkUrl: _phone,
        backend: DetectorBackend.cpu,
      );
      await until(() => log.lastOrNull == 'reload');

      applier
        ..runDetectorOn(DetectorBackend.gpu)
        ..close();
      holdReload!.complete();
      await applied(applier);
      expect(log, [
        'source network',
        'switch',
        'start NetworkSourceSpec',
        'backend cpu',
        'switch',
        'reload',
      ], reason: 'no restart after the reload, no queued switch back');
    });
  });

  group('the steps', () {
    test('without settings both switches refuse', () async {
      final (applier, _) = create(withoutSettings: true);
      final source = await sourceStep(applier, CameraSourceChoice.standard);
      expect(
        (source as Error<void>).error.toString(),
        'The camera source cannot be chosen here',
      );
      final backend = await backendStep(applier, DetectorBackend.cpu);
      expect(
        (backend as Error<void>).error.toString(),
        'The detector backend cannot be chosen',
      );
      expect(log, isEmpty);
    });

    test('without a reload the backend switch refuses', () async {
      final (applier, _) = create(canReload: false);
      expect(
        await backendStep(applier, DetectorBackend.cpu),
        isA<Error<void>>(),
      );
      expect(store.values['detector.backend'], isNull);
    });

    test('a save that fails: the error, nothing else', () async {
      final (applier, _) = await running();
      store.failWritesOf.addAll(['camera.source', 'detector.backend']);
      expect(
        await sourceStep(
          applier,
          const CameraSourceChoice(
            kind: CameraSourceKind.network,
            networkUrl: _phone,
          ),
        ),
        isA<Error<void>>(),
      );
      expect(
        await backendStep(applier, DetectorBackend.cpu),
        isA<Error<void>>(),
      );
      expect(log, ['source network', 'backend cpu'], reason: 'tried only');
      expect(applier.sourceChoice, CameraSourceChoice.standard);
      expect(applier.backend, DetectorBackend.gpu);
      expect(live.state.value, isA<LiveRunning>());
    });

    test('a locked backend refuses to save', () async {
      final (applier, _) = create(settings: settingsWith(backendDefine: 'gpu'));
      final result = await backendStep(applier, DetectorBackend.cpu);
      expect(
        (result as Error<void>).error.toString(),
        'The detector backend is fixed by DETECTOR_BACKEND=gpu',
      );
    });

    test('a failed reload: its error; the backend stays chosen; the source '
        'is not restarted', () async {
      final (applier, _) = await running();
      reloadResult = () =>
          const Result.error(FrameSourceUnavailableException('CPU refused'));
      final result = await backendStep(applier, DetectorBackend.cpu);
      expect((result as Error<void>).error.toString(), 'CPU refused');
      expect(log, ['backend cpu', 'switch', 'reload']);
      expect(applier.backend, DetectorBackend.cpu);
      expect(store.values['detector.backend'], 'cpu');
      expect(live.state.value, isA<LiveStopped>());
    });

    test('a step that ends after close stops there: saved, but no switch or '
        'restart', () async {
      final (applier, _) = await running();
      final applying = sourceStep(
        applier,
        const CameraSourceChoice(
          kind: CameraSourceKind.network,
          networkUrl: _phone,
        ),
      );
      applier.close();
      expect(await applying, isA<Ok<void>>());
      expect(log, ['source network'], reason: 'saved, nothing after it');
      expect(store.values['camera.source'], 'network');
      expect(applier.sourceChoice.kind, CameraSourceKind.network);
    });
  });
}
