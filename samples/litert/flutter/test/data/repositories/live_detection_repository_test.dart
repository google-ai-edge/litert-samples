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

import 'package:fake_async/fake_async.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/repositories/live_detection_repository.dart';
import 'package:litert_edge_demos/data/services/frames/frame_source.dart';
import 'package:litert_edge_demos/domain/models/frame_source_info.dart';
import 'package:litert_edge_demos/domain/models/frame_source_spec.dart';
import 'package:litert_edge_demos/domain/models/live_state.dart';
import 'package:litert_edge_demos/utils/frame_rate_gate.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../fakes/fake_detector.dart';
import '../../fakes/fake_frame_source.dart';
import '../../support/frames.dart';

/// Lets queued microtasks and zero-length timers run.
Future<void> settle() => Future<void>.delayed(Duration.zero);

const _spec = FixtureSourceSpec(['/fixtures']);
const _period = 66667; // µs at 15 fps

void main() {
  late FakeDetector detector;
  late List<FakeFrameSource> sources;
  late FakeFrameSource source; // the newest one
  late int clock; // µs
  late LiveDetectionRepository repo;
  final owner = Object();

  LiveDetectionRepository build({Duration statsInterval = Duration.zero}) =>
      LiveDetectionRepository(
        detector: detector,
        createSource: (spec) {
          final s = source = FakeFrameSource();
          sources.add(s);
          return Result.ok(s);
        },
        statsInterval: statsInterval,
        clockMicros: () => clock,
      );

  setUp(() {
    detector = FakeDetector();
    sources = [];
    clock = 0;
    repo = build();
  });

  tearDown(() => repo.close());

  Future<void> start() async {
    final started = await repo.start(_spec, owner: owner);
    expect(started, isA<Ok<FrameSourceInfo>>());
  }

  /// One frame [dtMicros] after the previous one.
  void emitAfter(int dtMicros, [FrameView? frame]) {
    clock += dtMicros;
    source.emit(frame);
  }

  group('FrameRateGate', () {
    test('a 30 fps source is cut to exactly 15 fps', () {
      final gate = FrameRateGate(
        fps: 15,
        slack: const Duration(milliseconds: 12),
      );
      var passed = 0;
      for (var i = 0; i < 300; i++) {
        if (gate.tryPass(i * 33333)) passed++;
      }
      expect(passed, 150);
    });

    test('a jittery 15 fps source passes every frame', () {
      final gate = FrameRateGate(
        fps: 15,
        slack: const Duration(milliseconds: 12),
      );
      const jitter = [0, -8000, 5000, -11000, 9000, -3000];
      for (var i = 0; i < 150; i++) {
        expect(
          gate.tryPass(i * _period + jitter[i % jitter.length]),
          isTrue,
          reason: 'frame $i',
        );
      }
    });

    test('a 60 fps source is cut to 15; after a gap there is no burst', () {
      final gate = FrameRateGate(
        fps: 15,
        slack: const Duration(milliseconds: 12),
      );
      var passed = 0;
      for (var i = 0; i < 240; i++) {
        if (gate.tryPass(i * 16667)) passed++;
      }
      expect(passed, 60);
      // 2 s pause, then a 60 fps burst: one frame, then the cadence again.
      const t0 = 6000000;
      expect(gate.tryPass(t0), isTrue);
      expect(gate.tryPass(t0 + 16667), isFalse);
      expect(gate.tryPass(t0 + 33333), isFalse);
      expect(gate.tryPass(t0 + 66667), isTrue);
    });
  });

  test('start opens the source, publishes its preview and Running', () async {
    expect(repo.state.value, isA<LiveStopped>());

    await start();

    expect(source.startCalls, 1);
    expect(repo.state.value, const TypeMatcher<LiveRunning>());
    expect((repo.state.value as LiveRunning).source, 'fake');
    expect(repo.preview.value, same(source.preview));
    expect(repo.sourceInfo?.label, 'fake');
  });

  test('latest frame wins: while a frame is in flight the rest are dropped, '
      'and the next frame after the result goes through', () async {
    await start();

    emitAfter(0, TestFrame.rgba(width: 640));
    emitAfter(_period, TestFrame.rgba(width: 320));
    emitAfter(_period, TestFrame.rgba(width: 160));
    expect(detector.calls, hasLength(1), reason: 'one slot');
    expect(detector.calls.single.width, 640);

    detector.complete();
    await settle();
    expect(repo.frames.value?.frameId, detector.calls.single.frameId);
    expect(repo.frames.value?.width, 640);

    emitAfter(_period, TestFrame.rgba(width: 480));
    expect(detector.calls, hasLength(2));
    expect(
      detector.calls.last.width,
      480,
      reason: 'the newest frame, not a queued one',
    );
    detector.complete();
    await settle();
    expect(repo.stats.value.droppedBusy, 2);
  });

  test('rate: a 30 fps source with an instant detector sends 15 fps', () async {
    detector.autoComplete = true;
    await start();

    for (var i = 0; i < 60; i++) {
      emitAfter(33333);
      await settle();
    }

    expect(detector.calls, hasLength(30));
    expect(repo.stats.value.droppedRate, 30);
    expect(repo.stats.value.processed, 30);
    expect(repo.stats.value.fps, closeTo(15, 0.1));
  });

  test('pause closes the gate (the frame in flight still lands), shows the '
      'reason, and resume lets the next frame through at once', () async {
    await start();
    emitAfter(0);
    expect(detector.inFlight, 1);

    repo.setDuty(DetectorDuty.paused, reason: 'Gemma');
    final paused = repo.state.value;
    expect(paused, isA<LivePaused>());
    expect((paused as LivePaused).reason, 'Gemma');

    detector.complete(); // the in-flight frame finishes
    await settle();
    expect(repo.frames.value, isNotNull);
    for (var i = 0; i < 10; i++) {
      emitAfter(_period);
    }
    expect(detector.calls, hasLength(1), reason: 'nothing sent while paused');
    expect(repo.stats.value.droppedPaused, 10);

    repo.setDuty(DetectorDuty.live);
    expect(repo.state.value, isA<LiveRunning>());
    emitAfter(1000); // right after resume, well inside the old cadence
    expect(detector.calls, hasLength(2));
    detector.complete();
    await settle();
  });

  test('a pause requested before start applies when it starts', () async {
    repo.setDuty(DetectorDuty.paused, reason: 'Gemma');

    await start();

    expect(repo.state.value, isA<LivePaused>());
    emitAfter(0);
    expect(detector.calls, isEmpty);
  });

  test('stop with a frame in flight waits for it, does not publish it, and '
      'ends Stopped with the source stopped', () async {
    await start();
    emitAfter(0);
    expect(detector.inFlight, 1);

    var stopped = false;
    final stopping = repo.stop(owner: owner).then((_) => stopped = true);
    await settle();
    expect(source.stopped, isTrue, reason: 'no new frames');
    expect(stopped, isFalse, reason: 'waits for the frame in flight');

    detector.complete();
    await stopping;

    expect(
      repo.frames.value,
      isNull,
      reason: 'a stopped session publishes nothing',
    );
    expect(repo.state.value, isA<LiveStopped>());
    expect(repo.preview.value, isNull);
  });

  test('owner token: a stale owner cannot stop the next owner; a new owner '
      'takes over and the old source is stopped first', () async {
    final demo1 = Object();
    final demo3 = Object();
    expect(await repo.start(_spec, owner: demo1), isA<Ok<FrameSourceInfo>>());
    final first = source;

    expect(await repo.start(_spec, owner: demo3), isA<Ok<FrameSourceInfo>>());
    expect(first.stopped, isTrue);
    final second = source;

    await repo.stop(owner: demo1); // a late dispose from the old screen
    expect(second.stopped, isFalse);
    expect(repo.state.value, isA<LiveRunning>());

    await repo.stop(owner: demo3);
    expect(second.stopped, isTrue);
    expect(repo.state.value, isA<LiveStopped>());
  });

  group('a slow start (a network camera connecting) is reached at once', () {
    late Completer<void> gate;
    late LiveDetectionRepository slow;

    setUp(() {
      gate = Completer<void>();
      slow = LiveDetectionRepository(
        detector: FakeDetector(autoComplete: true),
        createSource: (_) {
          final s = source = FakeFrameSource()..startGate = gate;
          sources.add(s);
          return Result.ok(s);
        },
      );
      addTearDown(slow.close);
    });

    test("the owner's stop stops the starting source now; the start ends "
        'as Stopped, not Failed', () async {
      final starting = slow.start(_spec, owner: owner);
      await settle();
      expect(slow.state.value, isA<LiveStarting>());
      final stopping = slow.stop(owner: owner);
      await settle();
      expect(source.stopped, isTrue, reason: 'not queued behind the start');

      gate.complete(); // the fake source finishes its start anyway
      final result = await starting;
      await stopping;
      expect(result, isA<Error<FrameSourceInfo>>());
      expect(slow.state.value, isA<LiveStopped>());
    });

    test("another owner's stop does not touch it; a take-over does", () async {
      final starting = slow.start(_spec, owner: owner);
      await settle();
      unawaited(slow.stop(owner: Object()));
      await settle();
      expect(source.stopped, isFalse);

      final first = source;
      final takeOver = slow.start(_spec, owner: Object());
      await settle();
      expect(first.stopped, isTrue);
      gate.complete();
      expect(await starting, isA<Error<FrameSourceInfo>>());
      expect(await takeOver, isA<Ok<FrameSourceInfo>>());
      expect(slow.state.value, isA<LiveRunning>());
    });

    test('close stops it too', () async {
      unawaited(slow.start(_spec, owner: owner));
      await settle();
      final closing = slow.close();
      await settle();
      expect(source.stopped, isTrue);
      gate.complete();
      await closing;
    });
  });

  test(
    'a detector error fails the pipeline visibly and stops the source',
    () async {
      await start();
      emitAfter(0);

      detector.fail(Exception('worker died'));
      await settle();
      await settle();

      final state = repo.state.value;
      expect(state, isA<LiveFailed>());
      expect((state as LiveFailed).message, contains('worker died'));
      expect(source.stopped, isTrue);
      emitAfter(_period);
      expect(detector.calls, hasLength(1));
    },
  );

  test('start fails visibly when the detector is not loaded', () async {
    detector.info = null;

    final started = await repo.start(_spec, owner: owner);

    expect(started, isA<Error<FrameSourceInfo>>());
    expect(repo.state.value, isA<LiveFailed>());
    expect(sources, isEmpty);
  });

  test('a source that fails to start leaves Failed with its message', () async {
    final failing = LiveDetectionRepository(
      detector: detector,
      createSource: (_) => Result.ok(
        FakeFrameSource()
          ..startResult = const Result.error(
            FrameSourceUnavailableException('no images'),
          ),
      ),
    );
    addTearDown(failing.close);

    final started = await failing.start(_spec, owner: owner);

    expect(started, isA<Error<FrameSourceInfo>>());
    expect((failing.state.value as LiveFailed).message, contains('no images'));
    expect(failing.preview.value, isNull);
  });

  test(
    'a source whose start throws a non-camera exception leaves Failed '
    '(not Starting) with a Retry that works, and the source is stopped',
    () async {
      final throwing = FakeFrameSource()
        ..startThrows = StateError('MissingPluginException: availableCameras');
      // The first start gets the throwing source, the Retry a working one.
      final queue = [throwing, FakeFrameSource()];
      final repoUnderTest = LiveDetectionRepository(
        detector: detector,
        createSource: (_) => Result.ok(queue.removeAt(0)),
      );
      addTearDown(repoUnderTest.close);

      final started = await repoUnderTest.start(_spec, owner: owner);

      expect(started, isA<Error<FrameSourceInfo>>());
      final state = repoUnderTest.state.value;
      expect(state, isA<LiveFailed>());
      expect((state as LiveFailed).message, contains('availableCameras'));
      expect(throwing.stopped, isTrue);
      expect(repoUnderTest.preview.value, isNull);

      expect(
        await repoUnderTest.start(_spec, owner: owner),
        isA<Ok<FrameSourceInfo>>(),
        reason: 'Retry',
      );
      expect(repoUnderTest.state.value, isA<LiveRunning>());
    },
  );

  test('a source whose stop throws still ends Stopped and frees the owner '
      'slot', () async {
    await start();
    final first = source..stopThrows = StateError('dispose failed');

    await repo.stop(owner: owner);

    expect(first.stopped, isTrue);
    expect(repo.state.value, isA<LiveStopped>());
    expect(repo.preview.value, isNull);

    // The slot is free: a new owner starts, and the old owner's late stop is
    // a no-op.
    final next = Object();
    expect(await repo.start(_spec, owner: next), isA<Ok<FrameSourceInfo>>());
    await repo.stop(owner: owner);
    expect(repo.state.value, isA<LiveRunning>());
  });

  test('a source error while it is still starting leaves Failed (not '
      'Running) and stops it', () async {
    final failing = LiveDetectionRepository(
      detector: detector,
      createSource: (_) {
        final s = source = FakeFrameSource()
          ..failDuringStart = const FrameSourceUnavailableException(
            'camera unplugged',
          );
        return Result.ok(s);
      },
    );
    addTearDown(failing.close);

    final started = await failing.start(_spec, owner: owner);
    await settle();
    await settle();

    expect(started, isA<Error<FrameSourceInfo>>());
    expect((failing.state.value as LiveFailed).message, contains('unplugged'));
    expect(source.stopped, isTrue);
    expect(failing.preview.value, isNull);
  });

  test(
    'a frame detected while the source is still starting fails: the state '
    'stays Failed (not Running), start reports it, the source is stopped',
    () async {
      final failingDetector = FakeDetector(
        failWith: const FrameSourceUnavailableException('GPU lost'),
      );
      final repoUnderTest = LiveDetectionRepository(
        detector: failingDetector,
        createSource: (_) {
          final s = source = FakeFrameSource()..emitDuringStart = true;
          return Result.ok(s);
        },
      );
      addTearDown(repoUnderTest.close);

      final started = await repoUnderTest.start(_spec, owner: owner);
      await settle();
      await settle();

      expect(failingDetector.calls, hasLength(1));
      expect(started, isA<Error<FrameSourceInfo>>());
      final state = repoUnderTest.state.value;
      expect(state, isA<LiveFailed>());
      expect((state as LiveFailed).message, contains('GPU lost'));
      expect(source.stopped, isTrue);
    },
  );

  test(
    'a runtime source error fails the pipeline and stops the source',
    () async {
      await start();

      source.failAtRuntime(
        const FrameSourceUnavailableException('Camera error: gone'),
      );
      await settle();
      await settle();

      expect((repo.state.value as LiveFailed).message, 'Camera error: gone');
      expect(source.stopped, isTrue);
    },
  );

  // The watchdog tests run in fake time (fake_async): the repository's
  // timers are fake, and its clock is the fake clock plus [suspended], so a
  // test can play the app being suspended (the clock jumps, no timer runs).
  // Each window is asserted on both sides of its edge.

  /// Runs [body] in fake time with the repository clock it should use.
  void inFakeTime(
    void Function(FakeAsync async, int Function() clockMicros) body,
  ) {
    fakeAsync((async) {
      body(async, () => async.elapsed.inMicroseconds);
    });
  }

  /// Closes [r] inside fake time (a stop waits up to its stop timeout).
  void closeIn(FakeAsync async, LiveDetectionRepository r) {
    unawaited(r.close());
    async.elapse(const Duration(seconds: 1));
  }

  test('a frame the detector never answers fails visibly (no silent 0 fps) '
      'at the stall timeout, not before; a restart is refused until the slot '
      'frees, then works', () {
    inFakeTime((async, clockMicros) {
      const stallTimeout = Duration(milliseconds: 50);
      final stalled = LiveDetectionRepository(
        detector: detector,
        createSource: (_) => Result.ok(source = FakeFrameSource()),
        stallTimeout: stallTimeout,
        stopTimeout: const Duration(milliseconds: 50),
        clockMicros: clockMicros,
      );
      unawaited(stalled.start(_spec, owner: owner));
      async.flushMicrotasks();
      source.emit();
      expect(detector.inFlight, 1);

      async.elapse(stallTimeout - const Duration(milliseconds: 1));
      expect(stalled.state.value, isA<LiveRunning>());
      async.elapse(const Duration(milliseconds: 1));

      final state = stalled.state.value;
      expect(state, isA<LiveFailed>());
      expect((state as LiveFailed).message, contains('did not answer'));
      // The source stops once the frame in flight is given up on (bounded
      // by the stop timeout).
      async.elapse(const Duration(milliseconds: 50));
      expect(source.stopped, isTrue);

      Result<FrameSourceInfo>? retry;
      unawaited(stalled.start(_spec, owner: owner).then((r) => retry = r));
      // The retry first gives the stuck frame its stop timeout again.
      async.elapse(const Duration(milliseconds: 49));
      expect(retry, isNull);
      async.elapse(const Duration(milliseconds: 1));
      expect(retry, isA<Error<FrameSourceInfo>>(), reason: 'slot still taken');
      expect(
        (stalled.state.value as LiveFailed).message,
        contains('still busy'),
      );

      detector.complete(); // the stuck frame finally returns
      async.flushMicrotasks();
      Result<FrameSourceInfo>? next;
      unawaited(stalled.start(_spec, owner: owner).then((r) => next = r));
      async.flushMicrotasks();
      expect(next, isA<Ok<FrameSourceInfo>>());
      expect(stalled.state.value, isA<LiveRunning>());
      closeIn(async, stalled);
    });
  });

  group('source watchdog (camera_desktop never reports a lost camera)', () {
    const sourceTimeout = Duration(milliseconds: 150);
    const watchdogInterval = Duration(milliseconds: 25);
    const justBefore = Duration(milliseconds: 1);

    LiveDetectionRepository watched(
      int Function() clockMicros, {
      Duration? ownStallTimeout,
    }) => LiveDetectionRepository(
      detector: FakeDetector(autoComplete: true),
      createSource: (_) => Result.ok(
        source = FakeFrameSource()
          ..startResult = ownStallTimeout == null
              ? null
              : Result.ok(
                  FrameSourceInfo(
                    label: 'Network camera · 127.0.0.1:8080',
                    width: 640,
                    height: 480,
                    format: FramePixelFormat.rgba8888,
                    mirrored: false,
                    stallTimeout: ownStallTimeout,
                  ),
                ),
      ),
      sourceTimeout: sourceTimeout,
      watchdogInterval: watchdogInterval,
      clockMicros: clockMicros,
    );

    /// A frame every 20 ms for [duration], the last one at its end.
    void emitFor(FakeAsync async, Duration duration) {
      final end = async.elapsed + duration;
      while (async.elapsed < end) {
        source.emit();
        async.elapse(const Duration(milliseconds: 20));
      }
      source.emit();
    }

    void startIn(FakeAsync async, LiveDetectionRepository r) {
      unawaited(r.start(_spec, owner: owner));
      async.flushMicrotasks();
      expect(r.state.value, isA<LiveRunning>());
    }

    test('a source that stops emitting while Running fails the pipeline '
        '(not Running at 0 fps) within one check of the timeout, and Retry '
        'starts again', () {
      inFakeTime((async, clockMicros) {
        final r = watched(clockMicros);
        startIn(async, r);

        emitFor(async, const Duration(milliseconds: 300)); // healthy
        expect(r.state.value, isA<LiveRunning>());

        // Unplugged / interrupted / taken by another app: frames just stop.
        async.elapse(sourceTimeout - justBefore);
        expect(r.state.value, isA<LiveRunning>());
        async.elapse(watchdogInterval + justBefore);

        final state = r.state.value;
        expect(state, isA<LiveFailed>());
        expect(
          (state as LiveFailed).message,
          contains('The camera stopped delivering frames'),
        );
        expect(source.stopped, isTrue);

        startIn(async, r);
        emitFor(async, const Duration(milliseconds: 100));
        expect(r.state.value, isA<LiveRunning>());
        closeIn(async, r);
      });
    });

    test('a source that never delivers a frame fails too', () {
      inFakeTime((async, clockMicros) {
        final r = watched(clockMicros);
        startIn(async, r);

        async.elapse(sourceTimeout - justBefore);
        expect(r.state.value, isA<LiveRunning>());
        async.elapse(watchdogInterval + justBefore);

        expect(
          (r.state.value as LiveFailed).message,
          contains('The camera stopped delivering frames'),
        );
        closeIn(async, r);
      });
    });

    test("a source's own stall timeout (a network camera's backstop) "
        'replaces the default', () {
      inFakeTime((async, clockMicros) {
        const own = Duration(milliseconds: 600);
        final r = watched(clockMicros, ownStallTimeout: own);
        startIn(async, r);
        emitFor(async, const Duration(milliseconds: 100));

        // Silent for far longer than the default 150 ms, just short of 600.
        async.elapse(own - justBefore);
        expect(r.state.value, isA<LiveRunning>());

        async.elapse(watchdogInterval + justBefore);
        expect(r.state.value, isA<LiveFailed>());
        closeIn(async, r);
      });
    });

    test('stats report the source rate and size (before the gate)', () {
      inFakeTime((async, clockMicros) {
        final r = watched(clockMicros);
        startIn(async, r);
        emitFor(async, const Duration(milliseconds: 400));

        final stats = r.stats.value;
        expect((stats.sourceWidth, stats.sourceHeight), (640, 480));
        expect(stats.sourceFps, closeTo(50, 1e-6), reason: 'one per 20 ms');
        closeIn(async, r);
      });
    });
  });

  group('watchdogs run only while the detector is live', () {
    const stallTimeout = Duration(milliseconds: 50);
    const sourceTimeout = Duration(milliseconds: 100);
    const watchdogInterval = Duration(milliseconds: 20);
    const justBefore = Duration(milliseconds: 1);

    LiveDetectionRepository guarded(
      FakeDetector d,
      int Function() clockMicros, {
      FakeFrameSource Function()? newSource,
    }) => LiveDetectionRepository(
      detector: d,
      createSource: (_) =>
          Result.ok(source = newSource?.call() ?? FakeFrameSource()),
      stallTimeout: stallTimeout,
      stopTimeout: const Duration(milliseconds: 50),
      sourceTimeout: sourceTimeout,
      watchdogInterval: watchdogInterval,
      clockMicros: clockMicros,
    );

    /// A frame every 20 ms for [duration].
    void emitFor(FakeAsync async, Duration duration) {
      final end = async.elapsed + duration;
      while (async.elapsed < end) {
        source.emit();
        async.elapse(const Duration(milliseconds: 20));
      }
    }

    void startIn(FakeAsync async, LiveDetectionRepository r) {
      unawaited(r.start(_spec, owner: owner));
      async.flushMicrotasks();
    }

    test('a pause with a frame in flight, then time advancing past both '
        'timeouts, then a resume does not fail', () {
      inFakeTime((async, clockMicros) {
        final r = guarded(detector, clockMicros);
        startIn(async, r);
        source.emit();
        expect(detector.inFlight, 1);

        r.setDuty(DetectorDuty.paused, reason: 'Gemma');
        // Neither frames nor the detector's reply for 6x the stall timeout.
        async.elapse(const Duration(milliseconds: 300));
        expect(r.state.value, isA<LivePaused>());

        r.setDuty(DetectorDuty.live);
        async.elapse(stallTimeout - justBefore);
        expect(r.state.value, isA<LiveRunning>(), reason: 'a fresh window');

        detector.complete();
        async.flushMicrotasks();
        expect(r.frames.value, isNotNull, reason: 'the frame landed');
        detector.autoComplete = true;
        emitFor(async, const Duration(milliseconds: 200));
        expect(r.state.value, isA<LiveRunning>());
        closeIn(async, r);
      });
    });

    test('after a resume, a frame that is still stuck fails after a fresh '
        'stall timeout (re-armed, not disabled)', () {
      inFakeTime((async, clockMicros) {
        final r = guarded(detector, clockMicros);
        startIn(async, r);
        source.emit();
        r.setDuty(DetectorDuty.paused, reason: 'Gemma');
        async.elapse(const Duration(milliseconds: 150));

        r.setDuty(DetectorDuty.live);
        emitFor(async, const Duration(milliseconds: 40)); // the source is fine
        async.elapse(
          stallTimeout - const Duration(milliseconds: 40) - justBefore,
        );
        expect(r.state.value, isA<LiveRunning>());
        async.elapse(justBefore);

        final state = r.state.value;
        expect(state, isA<LiveFailed>());
        expect((state as LiveFailed).message, contains('did not answer'));
        closeIn(async, r);
      });
    });

    test('a frame sent while starting, then a pause and a resume before '
        'Running: if it never returns it still fails (its stall watchdog is '
        're-armed on entering Running)', () {
      inFakeTime((async, clockMicros) {
        final resolve = Completer<void>();
        final r = guarded(
          detector,
          clockMicros,
          newSource: () => FakeFrameSource()
            ..emitDuringStart = true
            ..resolveGate = resolve,
        );
        Result<FrameSourceInfo>? started;
        unawaited(r.start(_spec, owner: owner).then((s) => started = s));
        async.flushMicrotasks();
        expect(detector.inFlight, 1, reason: 'sent while starting');
        r
          ..setDuty(DetectorDuty.paused, reason: 'Gemma')
          ..setDuty(DetectorDuty.live);
        async.elapse(Duration.zero);
        resolve.complete();
        async.flushMicrotasks();
        expect(started, isA<Ok<FrameSourceInfo>>());
        expect(r.state.value, isA<LiveRunning>());

        emitFor(async, const Duration(milliseconds: 40)); // the source is fine
        async.elapse(
          stallTimeout - const Duration(milliseconds: 40) - justBefore,
        );
        expect(r.state.value, isA<LiveRunning>());
        async.elapse(justBefore);

        final state = r.state.value;
        expect(state, isA<LiveFailed>());
        expect((state as LiveFailed).message, contains('did not answer'));
        closeIn(async, r);
      });
    });

    test('a pause and a resume while stop waits for the frame in flight do '
        'not re-arm the watchdogs: the stopped pipeline stays Stopped', () {
      inFakeTime((async, clockMicros) {
        final r = guarded(detector, clockMicros);
        startIn(async, r);
        emitFor(async, const Duration(milliseconds: 40));
        source.emit();
        expect(detector.inFlight, 1);

        var stopped = false;
        unawaited(r.stop(owner: owner).then((_) => stopped = true));
        async.flushMicrotasks();
        expect(stopped, isFalse, reason: 'waits for the frame in flight');
        // The GPU arbiter hands the GPU to Gemma and back during that wait.
        r
          ..setDuty(DetectorDuty.paused, reason: 'Gemma')
          ..setDuty(DetectorDuty.live);
        async.elapse(const Duration(milliseconds: 10));
        detector.complete();
        async.flushMicrotasks();
        expect(stopped, isTrue);
        expect(r.state.value, isA<LiveStopped>());

        // Far past both timeouts: nothing watches a stopped pipeline.
        async.elapse(sourceTimeout * 5);
        expect(r.state.value, isA<LiveStopped>());
        closeIn(async, r);
      });
    });

    test('a pause and a resume while stop waits for a frame that never '
        'returns do not fail the stopped pipeline either', () {
      inFakeTime((async, clockMicros) {
        final r = guarded(detector, clockMicros);
        startIn(async, r);
        source.emit();
        expect(detector.inFlight, 1);

        var stopped = false;
        unawaited(r.stop(owner: owner).then((_) => stopped = true));
        async.flushMicrotasks();
        r
          ..setDuty(DetectorDuty.paused, reason: 'Gemma')
          ..setDuty(DetectorDuty.live);
        // The stop gives the stuck frame its stop timeout (50 ms), then ends.
        async.elapse(const Duration(milliseconds: 60));
        expect(stopped, isTrue);
        expect(r.state.value, isA<LiveStopped>());

        async.elapse(sourceTimeout * 5);
        expect(r.state.value, isA<LiveStopped>());
        detector.complete();
        closeIn(async, r);
      });
    });

    test('a suspended app (debugger pause, sleep) with a frame in flight '
        'does not fail: the overdue stall timer must not beat the reply '
        'queued behind it', () {
      fakeAsync((async) {
        var suspended = 0;
        final r = guarded(
          detector,
          () => async.elapsed.inMicroseconds + suspended,
        );
        startIn(async, r);
        source.emit();
        expect(detector.inFlight, 1);
        // The worker's reply is the next event after the stall timer...
        Timer(
          stallTimeout + const Duration(milliseconds: 10),
          detector.complete,
        );

        // ...and the main isolate was suspended past the stall deadline:
        // the clock jumped, no timer ran.
        suspended += const Duration(milliseconds: 300).inMicroseconds;
        async.elapse(stallTimeout); // the overdue stall timer runs first
        expect(
          r.state.value,
          isA<LiveRunning>(),
          reason: 'fired late: re-armed',
        );
        async.elapse(const Duration(milliseconds: 10));

        expect(r.state.value, isA<LiveRunning>());
        expect(r.frames.value, isNotNull, reason: 'the reply was taken');
        closeIn(async, r);
      });
    });

    test('a suspended app does not count as a silent camera: frames that '
        'arrive right after the suspension keep it Running', () {
      fakeAsync((async) {
        var suspended = 0;
        final r = guarded(
          FakeDetector(autoComplete: true),
          () => async.elapsed.inMicroseconds + suspended,
        );
        startIn(async, r);
        emitFor(async, const Duration(milliseconds: 100));

        // Suspended for 3x the source timeout; the next frame is queued
        // behind the overdue watchdog check.
        suspended += const Duration(milliseconds: 300).inMicroseconds;
        Timer(watchdogInterval + justBefore, source.emit);
        async.elapse(watchdogInterval);
        expect(r.state.value, isA<LiveRunning>(), reason: 'a fresh window');
        emitFor(async, const Duration(milliseconds: 100));

        expect(r.state.value, isA<LiveRunning>());
        closeIn(async, r);
      });
    });
  });

  test('black frames: the warning is shown while the pipeline keeps running, '
      'cleared by bright frames and reset by a stop (when it turns on and '
      'which frames are sampled: black_frame_detector_test.dart)', () async {
    detector.autoComplete = true;
    await start();
    final dark = TestFrame.rgba(fill: 1); // luma 1: covered or zeroed
    final bright = TestFrame.rgba(fill: 128);

    for (var i = 0; i < 53; i++) {
      emitAfter(_period, dark);
      await settle();
    }
    expect(repo.blackFrames.value, isTrue, reason: 'dark for over 2 s');
    expect(repo.luma, closeTo(1, 1e-9));
    expect(repo.state.value, isA<LiveRunning>(), reason: 'a warning only');
    expect(detector.calls, hasLength(53), reason: 'every frame detected');

    for (var i = 0; i < 16; i++) {
      emitAfter(_period, bright);
      await settle();
    }
    expect(repo.blackFrames.value, isFalse, reason: 'luma recovered');
    expect(repo.luma, closeTo(128, 1e-9));

    for (var i = 0; i < 50; i++) {
      emitAfter(_period, dark);
      await settle();
    }
    expect(repo.blackFrames.value, isTrue);
    await repo.stop(owner: owner);
    expect(repo.blackFrames.value, isFalse, reason: 'stopped: no warning');
    expect(repo.luma, isNull);
  });

  test('keeps the last 5 frames and publishes p50 timings', () async {
    detector.autoComplete = true;
    await start();

    for (var i = 0; i < 8; i++) {
      emitAfter(_period);
      await settle();
    }

    expect(repo.recent, hasLength(5));
    expect(repo.recent.last.frameId, repo.frames.value!.frameId);
    final stats = repo.stats.value;
    expect(stats.processed, 8);
    expect(stats.preMs, 1.0);
    expect(stats.runMs, 4.0);
    expect(stats.postMs, 1.0);
    expect(stats.fps, closeTo(15, 0.1));
  });

  test('stats are coalesced to the interval', () async {
    final coalesced = build(statsInterval: const Duration(milliseconds: 250));
    addTearDown(coalesced.close);
    detector.autoComplete = true;
    await coalesced.start(_spec, owner: owner);
    var updates = 0;
    coalesced.stats.addListener(() => updates++);

    for (var i = 0; i < 15; i++) {
      clock += _period;
      source.emit();
      await settle();
    }

    expect(updates, inInclusiveRange(3, 5), reason: '1 s of frames at ≤4 Hz');
  });

  test('after close, start fails and stop/setDuty are no-ops', () async {
    await start();
    await repo.close();

    expect(source.stopped, isTrue);
    expect(
      await repo.start(_spec, owner: owner),
      isA<Error<FrameSourceInfo>>(),
    );
    await repo.stop(owner: owner);
    repo.setDuty(DetectorDuty.paused);
  });
}
