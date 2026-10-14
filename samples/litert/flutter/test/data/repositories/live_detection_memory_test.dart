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

import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/repositories/live_detection_repository.dart';
import 'package:litert_edge_demos/domain/models/frame_source_info.dart';
import 'package:litert_edge_demos/domain/models/frame_source_spec.dart';
import 'package:litert_edge_demos/domain/models/live_state.dart';
import 'package:litert_edge_demos/domain/models/scene_snapshot.dart';
import 'package:litert_edge_demos/domain/models/visual_memory.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../fakes/fake_detector.dart';
import '../../fakes/fake_frame_source.dart';

Future<void> settle() => pumpEventQueue();

const _spec = FixtureSourceSpec(['/fixtures']);
const _period = 66667; // µs at 15 fps

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  late FakeDetector detector;
  late FakeFrameSource source;
  late int clock;
  late LiveDetectionRepository repo;
  late bool mirrored;
  final owner = Object();

  setUp(() {
    detector = FakeDetector();
    clock = 0;
    mirrored = false;
    repo = LiveDetectionRepository(
      detector: detector,
      createSource: (spec) => Result.ok(
        source = FakeFrameSource()
          ..startResult = Result.ok(
            FrameSourceInfo(
              label: 'fake',
              width: 640,
              height: 480,
              format: FramePixelFormat.rgba8888,
              mirrored: mirrored,
            ),
          ),
      ),
      statsInterval: Duration.zero,
      memoryCaptureTimeout: const Duration(milliseconds: 300),
      clockMicros: () => clock,
    );
  });

  tearDown(() => repo.close());

  Future<void> start() async {
    expect(await repo.start(_spec, owner: owner), isA<Ok<FrameSourceInfo>>());
  }

  void emitAfter(int dtMicros) {
    clock += dtMicros;
    source.emit();
  }

  CaptureUnavailableException failureOf(Result<MemoryShot> result) =>
      (result as Error<MemoryShot>).error as CaptureUnavailableException;

  test('without a running source it fails at once', () async {
    final result = await repo.captureMemory();
    expect(failureOf(result).kind, CaptureFailure.notRunning);
  });

  test('the next frame that passes the gate carries it: the frame is '
      'published as usual and its memory frame comes back', () async {
    await start();
    final capturing = repo.captureMemory();
    expect(repo.capturePending, isTrue);
    emitAfter(_period);
    expect(detector.memoryCalls.single, isNotNull);
    final spec = detector.memoryCalls.single!;
    expect(spec.maxSide, 320);
    expect(spec.greySide, 32);
    expect(spec.unmirror, isFalse);
    detector.complete();
    final shot = (await capturing as Ok<MemoryShot>).value;

    expect(shot.frameId, detector.calls.single.frameId);
    expect(shot.png, isNotNull);
    expect((shot.width, shot.height), (320, 240));
    expect(repo.frames.value?.frameId, shot.frameId, reason: 'published');
    expect(repo.capturePending, isFalse);
  });

  test('it never bypasses the rate gate', () async {
    await start();
    emitAfter(_period);
    detector.complete();
    await settle();
    final capturing = repo.captureMemory();
    emitAfter(1000); // the gate drops it
    expect(detector.calls, hasLength(1));
    emitAfter(_period);
    expect(detector.memoryCalls.last, isNotNull);
    detector.complete();
    expect(await capturing, isA<Ok<MemoryShot>>());
  });

  test('a question\'s capture takes the next frame first; the memory takes '
      'the one after', () async {
    await start();
    final memory = repo.captureMemory();
    final question = repo.capture();
    emitAfter(_period);
    expect(detector.snapshotCalls.single, isTrue);
    expect(detector.memoryCalls.single, isNull);
    detector.complete();
    expect(await question, isA<Ok<SceneSnapshot>>());
    emitAfter(_period);
    expect(detector.memoryCalls.last, isNotNull);
    detector.complete();
    expect(await memory, isA<Ok<MemoryShot>>());
  });

  test('a pause holds it (no bypass) until it times out', () async {
    await start();
    repo.setDuty(DetectorDuty.paused, reason: 'Gemma');
    final capturing = repo.captureMemory();
    emitAfter(_period);
    emitAfter(_period);
    expect(detector.calls, isEmpty);
    expect(failureOf(await capturing).kind, CaptureFailure.timedOut);
  });

  test(
    'one at a time: a second request while one is pending fails at once',
    () async {
      await start();
      final first = repo.captureMemory();
      final second = await repo.captureMemory();
      expect(failureOf(second).message, contains('already pending'));
      emitAfter(_period);
      detector.complete();
      expect(await first, isA<Ok<MemoryShot>>());
    },
  );

  test(
    'a memory frame the worker could not make fails only the memory '
    'capture: the frame is published and the pipeline keeps running',
    () async {
      await start();
      detector.memoryShotError = 'The memory frame could not be made: PNG';
      final capturing = repo.captureMemory();
      emitAfter(_period);
      final frameId = detector.calls.single.frameId;
      detector.complete();
      final result = await capturing;
      expect(
        (result as Error<MemoryShot>).error,
        isA<MemoryFrameException>().having(
          (e) => e.message,
          'message',
          contains('PNG'),
        ),
      );
      expect(repo.frames.value?.frameId, frameId, reason: 'published');
      expect(repo.state.value, isA<LiveRunning>());
      expect(repo.capturePending, isFalse);

      // The next frame is detected as usual, and the next memory capture works.
      detector.memoryShotError = null;
      emitAfter(_period);
      detector.complete();
      await settle();
      expect(repo.frames.value?.frameId, detector.calls.last.frameId);
      final again = repo.captureMemory();
      emitAfter(_period);
      detector.complete();
      expect(await again, isA<Ok<MemoryShot>>());
      expect(repo.state.value, isA<LiveRunning>());
    },
  );

  test('a stop fails it as "not running"', () async {
    await start();
    final capturing = repo.captureMemory();
    await repo.stop(owner: owner);
    expect(failureOf(await capturing).kind, CaptureFailure.notRunning);
  });

  test('a mirroring source asks for the frame mirrored back', () async {
    mirrored = true;
    await start();
    final capturing = repo.captureMemory();
    emitAfter(_period);
    expect(detector.memoryCalls.single!.unmirror, isTrue);
    detector.complete();
    expect((await capturing as Ok<MemoryShot>).value.unmirrored, isTrue);
  });

  test('a start is a new detection session (the memory forgets on it) '
      'unless it resumes one: the restart after a background release or a '
      'detector switch keeps it', () async {
    expect(repo.sourceStarts.value, 0);
    await start();
    expect(repo.sourceStarts.value, 1);
    await start(); // a take-over (a source change, Retry)
    expect(repo.sourceStarts.value, 2);
    await repo.stop(owner: owner);
    expect(repo.sourceStarts.value, 2, reason: 'a stop is no new session');
    expect(
      await repo.start(_spec, owner: owner, newSession: false),
      isA<Ok<FrameSourceInfo>>(),
    );
    expect(repo.sourceStarts.value, 2, reason: 'a resumed session');
    expect(repo.state.value, isA<LiveRunning>());
  });
}
