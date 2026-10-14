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

import 'package:flutter/foundation.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/live_camera_config.dart';
import 'package:litert_edge_demos/data/repositories/visual_memory/memory_store.dart';
import 'package:litert_edge_demos/data/repositories/visual_memory_repository.dart';
import 'package:litert_edge_demos/data/services/detector/frame_message.dart';
import 'package:litert_edge_demos/data/services/detector/memory_shot.dart';
import 'package:litert_edge_demos/domain/models/live_state.dart';
import 'package:litert_edge_demos/domain/models/model_id.dart';
import 'package:litert_edge_demos/domain/models/model_state.dart';
import 'package:litert_edge_demos/domain/models/visual_memory.dart';
import 'package:litert_edge_demos/domain/vision/coco_vocabulary.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../fakes/fake_detector.dart';
import '../../fakes/fake_visual_memory.dart';
import '../../support/desk_scene.dart';
import '../../support/frames.dart';

const _s = 1000000;

void main() {
  late int now;
  late bool idle;
  late bool capturePending;
  late FakeMemoryCapture capture;
  late FakeMemoryEmbedder embedder;
  late ValueNotifier<int> starts;
  late ValueNotifier<bool> black;
  late ValueNotifier<LiveState> live;
  late ValueNotifier<Map<ModelId, ModelState>>? models;
  late VisualMemoryRepository memory;
  final owner = Object();

  VisualMemoryRepository build({
    MemoryStore? store,
    CaptureMemoryShot? captureShot,
  }) => VisualMemoryRepository(
    capture: captureShot ?? capture.call,
    embedder: embedder,
    enableImages: embedder.enableImages,
    sourceStarts: starts,
    blackFrames: black,
    liveState: live,
    capturePending: () => capturePending,
    models: models,
    tickInterval: null,
    store: store,
    clockMicros: () => now,
  );

  setUp(() {
    now = 100 * _s;
    idle = true;
    capturePending = false;
    capture = FakeMemoryCapture();
    embedder = FakeMemoryEmbedder(
      imageVectors: {
        1: [1, 0, 0],
        2: [0, 1, 0],
        3: [0, 0, 1],
        // Scene 1 again, a little moved: image cosine 0.995 to it.
        4: [0.995, 0.0998, 0],
      },
      textVectors: {
        'mug': [0.9, 0.1, 0.1], // scene 1
        'sign': [0.7, 0.7, 0.1], // scenes 1 and 2 alike
        'giraffe': [0.5, 0.5, 0.5], // no scene well
        'faint': [0.63, 0, 0.7766], // scene 1 at 0.630: over the floor only
      },
    );
    starts = ValueNotifier(0);
    black = ValueNotifier(false);
    live = ValueNotifier(const LiveRunning('fake'));
    models = null;
    memory = build();
    memory.start(owner: owner, idle: () => idle);
  });

  tearDown(() {
    memory.close();
    starts.dispose();
    black.dispose();
    live.dispose();
  });

  /// One sampler step at [atSeconds] (the clock moves there first).
  Future<void> tickAt(int atSeconds) async {
    now = atSeconds * _s;
    await memory.tick();
  }

  group('sampler', () {
    test('the first frame is asked for at once (no reference), '
        'embedded and stored with its detections', () async {
      capture.results.add(Result.ok(fakeShot(frameId: 7, cats: 2)));
      await tickAt(100);

      expect(capture.calls.single.reference, isNull);
      expect(embedder.images, hasLength(1));
      final entry = memory.entries.single;
      expect(entry.frameId, 7);
      expect(entry.atMicros, 100 * _s);
      expect(entry.counts, {kCocoNames.indexOf('cat'): 2});
      expect(entry.vector, [1, 0, 0]);
      final status = memory.status.value;
      expect(status.active, isTrue);
      expect(status.frames, 1);
      expect(status.bytes, 100);
      expect(status.unavailable, isNull);
    });

    test('the next one waits 3 s; it carries the last frame\'s thumbnail; an '
        'unchanged scene (no PNG) is not embedded', () async {
      capture.results.add(Result.ok(fakeShot()));
      await tickAt(100);
      await tickAt(102);
      expect(capture.calls, hasLength(1), reason: 'not due yet');

      capture.results.add(Result.ok(fakeShot(frameId: 2, png: false)));
      await tickAt(103);
      expect(capture.calls, hasLength(2));
      expect(capture.calls.last.reference, memory.entries.single.grey);
      expect(embedder.images, hasLength(1), reason: 'nothing to embed');
      expect(memory.entries, hasLength(1));
    });

    test('an unchanged scene once 10 s have passed refreshes the newest '
        'frame\'s time in place: no duplicate, no embedding', () async {
      capture.results.add(Result.ok(fakeShot(frameId: 1)));
      await tickAt(100);
      final bytes = memory.status.value.bytes;
      capture.results.add(Result.ok(fakeShot(frameId: 2, png: false)));
      await tickAt(104);
      expect(memory.entries.single.atMicros, 100 * _s, reason: 'not due');

      capture.results.add(Result.ok(fakeShot(frameId: 3, png: false)));
      await tickAt(110);
      final entry = memory.entries.single;
      expect(entry.atMicros, 110 * _s);
      expect(entry.frameId, 1, reason: 'the frame that was embedded');
      expect(embedder.images, hasLength(1));
      expect(memory.status.value.bytes, bytes);

      // The next refresh is due 10 s after this one.
      capture.results.add(Result.ok(fakeShot(frameId: 4, png: false)));
      await tickAt(115);
      expect(memory.entries.single.atMicros, 110 * _s);
      capture.results.add(Result.ok(fakeShot(frameId: 5, png: false)));
      await tickAt(120);
      expect(memory.entries.single.atMicros, 120 * _s);
      expect(memory.entries, hasLength(1));
    });

    test('a refresh moves the last-seen time only: the first sighting stays, '
        'and the chip and a miss measure the span from it', () async {
      capture.results.add(Result.ok(fakeShot(frameId: 1)));
      await tickAt(100);
      expect(memory.entries.single.firstSeenMicros, 100 * _s);
      for (final (i, at) in [110, 120].indexed) {
        capture.results.add(Result.ok(fakeShot(frameId: 2 + i, png: false)));
        await tickAt(at);
      }
      final entry = memory.entries.single;
      expect(entry.atMicros, 120 * _s, reason: 'last seen');
      expect(entry.firstSeenMicros, 100 * _s, reason: 'first seen');
      expect(memory.status.value.span, const Duration(seconds: 20));

      now = 125 * _s;
      final miss =
          (await memory.recall('giraffe') as Ok<MemoryRecall>).value
              as MemoryMiss;
      expect(miss.span, const Duration(seconds: 25));
      expect(memory.status.value.span, const Duration(seconds: 25));
    });

    test('a changed scene is stored as a new frame', () async {
      capture.results.add(Result.ok(fakeShot()));
      await tickAt(100);
      capture.results.add(Result.ok(fakeShot(scene: 2, frameId: 3)));
      await tickAt(110);
      expect(memory.entries, hasLength(2));
      expect(embedder.images, hasLength(2));
    });

    test('a busy demo drops the sample: no capture, nothing queued; the gap '
        'restarts at the first idle tick', () async {
      idle = false;
      await tickAt(100);
      await tickAt(101);
      expect(capture.calls, isEmpty);

      idle = true;
      capture.results.add(Result.ok(fakeShot()));
      await tickAt(102);
      expect(capture.calls, isEmpty, reason: 'idle only just now');
      await tickAt(104);
      expect(capture.calls, isEmpty);
      await tickAt(105);
      expect(capture.calls, hasLength(1));
      expect(memory.entries, hasLength(1));
    });

    test('busy from 100 s to 110 s, idle at 110.5 s: no capture before '
        '113.5 s', () async {
      Future<void> tickAtMs(int ms) async {
        now = ms * 1000;
        await memory.tick();
      }

      // Every 500 ms tick of a 10 s turn is busy.
      idle = false;
      for (var ms = 100000; ms <= 110000; ms += 500) {
        await tickAtMs(ms);
      }
      idle = true;
      for (var ms = 110500; ms < 113500; ms += 500) {
        await tickAtMs(ms);
        expect(capture.calls, isEmpty, reason: 'at $ms ms');
      }
      capture.results.add(Result.ok(fakeShot()));
      await tickAtMs(113500);
      expect(capture.calls, hasLength(1));
    });

    test('a question capture pending, a paused or stopped detector, or black '
        'frames: no capture', () async {
      capturePending = true;
      await tickAt(100);
      capturePending = false;
      live.value = LivePaused(
        source: 'fake',
        reason: 'Gemma',
        since: DateTime(2026),
      );
      await tickAt(101);
      live.value = const LiveStopped();
      await tickAt(102);
      live.value = const LiveRunning('fake');
      black.value = true;
      await tickAt(103);
      expect(capture.calls, isEmpty);
    });

    test('a question that starts while the frame comes back wins: the frame '
        'is dropped before its embedding', () async {
      capture.gate = Completer();
      capture.results.add(Result.ok(fakeShot()));
      final step = tickAt(100);
      idle = false; // the mic went down
      capture.gate!.complete();
      await step;
      expect(embedder.images, isEmpty);
      expect(memory.entries, isEmpty);
    });

    test('one attempt at a time: a tick while one runs does nothing', () async {
      embedder.gate = Completer();
      capture.results.add(Result.ok(fakeShot()));
      final first = tickAt(100);
      await pumpEventQueue();
      await tickAt(200);
      expect(capture.calls, hasLength(1));
      embedder.gate!.complete();
      await first;
      expect(memory.entries, hasLength(1));
    });

    test('the store\'s caps apply (oldest out)', () async {
      memory.close();
      memory = build(store: MemoryStore(maxEntries: 2, maxBytes: 1 << 20));
      memory.start(owner: owner, idle: () => idle);
      for (var i = 0; i < 3; i++) {
        capture.results.add(Result.ok(fakeShot(frameId: i, scene: i + 1)));
        await tickAt(100 + i * 10);
      }
      expect([for (final e in memory.entries) e.frameId], [1, 2]);
      expect(memory.status.value.frames, 2);
    });

    test('a capture failure is not an embedding failure', () async {
      await tickAt(100); // nothing queued: the capture fails
      expect(memory.status.value.failures, 0);
      expect(memory.status.value.unavailable, isNull);
    });
  });

  group('scene change: the worker\'s test on synthetic desk frames', () {
    /// The frames the next captures see, in order.
    final frames = <TestFrame>[];

    /// The live pipeline's capture with the worker's real [memoryShot] (the
    /// shipped thresholds) over the next frame, against the reference the
    /// memory passes (its newest frame's thumbnail).
    Future<Result<MemoryShot>> deskCapture({Uint8List? reference}) async {
      final message = FrameMessage.copyOf(frames.removeAt(0), frameId: 1);
      return Result.ok(
        memoryShot(
          message,
          message.materialize(),
          FakeDetector.catsFor(message, cats: 0),
          MemoryShotSpec(
            maxSide: kMemoryThumbMaxSide,
            greySide: kMemoryGreySide,
            cellPixels: kMemoryCellPixels,
            maxShift: kMemoryMaxShift,
            unmirror: false,
            reference: reference,
            changeThreshold: kMemorySceneChange,
            cellChange: kMemoryCellChange,
            changedCells: kMemoryChangedCells,
          ),
        ),
      );
    }

    test(
      'an object put down or taken away is stored as a new frame; noise '
      'and a 10 % brightness change refresh the newest frame in place',
      () async {
        memory.close();
        memory = build(captureShot: deskCapture);
        memory.start(owner: owner, idle: () => idle);
        var at = 100;
        for (final object in DeskObject.values) {
          memory.clear();
          final on = {object};
          frames.addAll([
            deskScene(seed: 1),
            deskScene(objects: on, seed: 2),
            deskScene(objects: on, noise: 6, seed: 3),
            deskScene(objects: on, seed: 4, brightness: 0.9),
            deskScene(objects: on, seed: 5, brightness: 1.1),
            deskScene(seed: 6),
          ]);
          final what = object.name;
          final t0 = at;
          await tickAt(at);
          expect(memory.entries, hasLength(1), reason: '$what: the empty desk');
          await tickAt(at += 10);
          expect(
            memory.entries,
            hasLength(2),
            reason: '$what put down: stored',
          );
          for (final change in ['noise', '10 % darker', '10 % brighter']) {
            await tickAt(at += 10);
            expect(memory.entries, hasLength(2), reason: '$what, $change');
            expect(
              memory.entries.last.atMicros,
              at * _s,
              reason: '$what, $change: refreshed in place',
            );
            expect(memory.entries.last.firstSeenMicros, (t0 + 10) * _s);
          }
          await tickAt(at += 10);
          expect(memory.entries, hasLength(3), reason: '$what taken away');
          expect(frames, isEmpty);
          at += 10;
        }
        expect(embedder.images, hasLength(3 * DeskObject.values.length));
      },
    );
  });

  group('failures', () {
    test('each failed embedding counts; three in a row turn the memory off '
        'with the reason; clear starts afresh', () async {
      embedder.imageError = Exception('vision tower');
      for (var i = 0; i < 3; i++) {
        capture.results.add(Result.ok(fakeShot(frameId: i)));
        await tickAt(100 + i * 10);
        expect(memory.status.value.failures, i + 1);
      }
      final status = memory.status.value;
      expect(status.failuresInARow, 3);
      expect(status.unavailable, contains('3 frames in a row failed'));
      expect(status.unavailable, contains('vision tower'));

      // Off: no more sampling; a question gets the reason.
      await tickAt(200);
      expect(capture.calls, hasLength(3));
      final recall = await memory.recall('mug');
      expect(
        (recall as Ok<MemoryRecall>).value,
        isA<MemoryOff>().having(
          (o) => o.reason,
          'reason',
          contains('3 frames in a row'),
        ),
      );

      embedder.imageError = null;
      memory.clear();
      expect(memory.status.value.failures, 0);
      expect(memory.status.value.unavailable, isNull);
      capture.results.add(Result.ok(fakeShot()));
      await tickAt(300);
      expect(memory.entries, hasLength(1));
    });

    test('a success resets the run of failures, not the count', () async {
      embedder.imageError = Exception('once');
      capture.results.add(Result.ok(fakeShot()));
      await tickAt(100);
      embedder.imageError = null;
      capture.results.add(Result.ok(fakeShot(scene: 2)));
      await tickAt(110);
      expect(memory.status.value.failures, 1);
      expect(memory.status.value.failuresInARow, 0);
      expect(memory.entries, hasLength(1));
    });

    test('a memory frame the worker could not make counts like a failed '
        'embedding', () async {
      for (var i = 0; i < 3; i++) {
        capture.results.add(
          const Result.error(MemoryFrameException('PNG failed')),
        );
        await tickAt(100 + i * 10);
      }
      final status = memory.status.value;
      expect(status.failures, 3);
      expect(status.unavailable, contains('a memory frame could not be made'));
      expect(status.unavailable, contains('PNG failed'));
      expect(embedder.images, isEmpty);
    });

    test('the model row\'s state comes first: loading or failed is off with '
        'that reason, and no image encoder load is asked for', () async {
      memory.close();
      models = ValueNotifier({ModelId.embeddingGemma: const ModelLoading()});
      embedder.unavailable = 'EmbeddingGemma was loaded without images';
      memory = build();
      memory.start(owner: owner, idle: () => idle);
      expect(
        memory.status.value.unavailable,
        'EmbeddingGemma is still loading',
      );
      expect(memory.status.value.preparing, isFalse);
      models!.value = {ModelId.embeddingGemma: const ModelFailed('no file')};
      expect(memory.status.value.unavailable, contains('no file'));
      final recall = await memory.recall('mug');
      expect((recall as Ok<MemoryRecall>).value, isA<MemoryOff>());
      expect(embedder.enables, 0);
    });
  });

  group('image encoder (loaded on the first start)', () {
    test('an embedder loaded without images is prepared, not off: the image '
        'encoder is asked for once, nothing is sampled until it is loaded, '
        'then sampling starts', () async {
      memory.close();
      embedder.unavailable = 'EmbeddingGemma was loaded without images';
      embedder.enableGate = Completer();
      memory = build();
      memory.start(owner: owner, idle: () => idle);
      expect(embedder.enables, 1);
      expect(memory.status.value.preparing, isTrue);
      expect(memory.status.value.unavailable, isNull);

      capture.results.add(Result.ok(fakeShot()));
      await tickAt(100);
      await tickAt(101);
      expect(capture.calls, isEmpty);
      expect(embedder.enables, 1, reason: 'one load at a time');
      final asked = await memory.recall('mug');
      expect(
        (asked as Ok<MemoryRecall>).value,
        isA<MemoryEmpty>(),
        reason: 'nothing remembered while preparing',
      );

      embedder.enableGate!.complete();
      await pumpEventQueue();
      expect(memory.status.value.preparing, isFalse);
      await tickAt(102);
      expect(capture.calls, hasLength(1));
      expect(memory.entries, hasLength(1));
    });

    test(
      'the row turning ready while the camera is open starts the load',
      () async {
        memory.close();
        models = ValueNotifier({ModelId.embeddingGemma: const ModelLoading()});
        embedder.unavailable = 'EmbeddingGemma was loaded without images';
        memory = build();
        memory.start(owner: owner, idle: () => idle);
        expect(embedder.enables, 0);
        models!.value = {
          ModelId.embeddingGemma: const ModelReady(
            LoadedModelInfo(
              modelId: 'embeddinggemma-2-740m',
              backend: 'cpu',
              loadTime: Duration(seconds: 1),
              warmUpTime: Duration(milliseconds: 100),
            ),
          ),
        };
        expect(embedder.enables, 1);
        await pumpEventQueue();
        expect(memory.status.value.preparing, isFalse);
        expect(memory.status.value.unavailable, isNull);
      },
    );

    test('a failed load turns the memory off with the reason (no sampling); '
        '"Clear memory" tries again', () async {
      memory.close();
      embedder.unavailable = 'EmbeddingGemma was loaded without images';
      embedder.enableError = Exception('vision tower missing');
      memory = build();
      memory.start(owner: owner, idle: () => idle);
      await pumpEventQueue();
      final status = memory.status.value;
      expect(status.preparing, isFalse);
      expect(status.unavailable, contains('image encoder failed'));
      expect(status.unavailable, contains('vision tower missing'));
      await tickAt(100);
      expect(capture.calls, isEmpty);
      final recall = await memory.recall('mug');
      expect((recall as Ok<MemoryRecall>).value, isA<MemoryOff>());
      expect(embedder.enables, 1, reason: 'no retry storm');

      // A new detection session does not retry a 2 s reload; Clear does.
      starts.value++;
      await tickAt(110);
      expect(embedder.enables, 1);
      embedder.enableError = null;
      memory.clear();
      await pumpEventQueue();
      expect(embedder.enables, 2);
      expect(memory.status.value.unavailable, isNull);
    });
  });

  group('switch (camera settings)', () {
    test('off at the start: no image encoder load, no sampling, a question '
        'gets MemoryDisabled', () async {
      memory.close();
      embedder.unavailable = 'EmbeddingGemma was loaded without images';
      memory = build();
      memory.start(owner: owner, idle: () => idle, enabled: false);
      expect(embedder.enables, 0);
      final status = memory.status.value;
      expect(status.enabled, isFalse);
      expect(status.preparing, isFalse);
      capture.results.add(Result.ok(fakeShot()));
      await tickAt(100);
      expect(capture.calls, isEmpty);
      final recall = await memory.recall('mug');
      expect((recall as Ok<MemoryRecall>).value, isA<MemoryDisabled>());
      expect(embedder.texts, isEmpty);
    });

    test(
      'switched off while running: every frame is forgotten and sampling '
      'stops; on again: it resumes (loading images first if needed)',
      () async {
        capture.results.add(Result.ok(fakeShot()));
        await tickAt(100);
        expect(memory.entries, hasLength(1));

        memory.setEnabled(enabled: false);
        expect(memory.entries, isEmpty);
        expect(memory.status.value.enabled, isFalse);
        await tickAt(110);
        expect(capture.calls, hasLength(1));

        memory.setEnabled(enabled: true);
        expect(memory.status.value.enabled, isTrue);
        capture.results.add(Result.ok(fakeShot()));
        await tickAt(111);
        expect(capture.calls, hasLength(2));
        expect(memory.entries, hasLength(1));
      },
    );

    test('switched on with images missing: the load starts then', () async {
      memory.close();
      embedder.unavailable = 'EmbeddingGemma was loaded without images';
      memory = build();
      memory.start(owner: owner, idle: () => idle, enabled: false);
      memory.setEnabled(enabled: true);
      expect(embedder.enables, 1);
    });
  });

  group('clearing', () {
    Future<void> remember() async {
      capture.results.add(Result.ok(fakeShot()));
      await tickAt(100);
      expect(memory.entries, hasLength(1));
    }

    test('a new detection session (a source change, Retry after a failure) '
        'forgets every frame', () async {
      await remember();
      starts.value++;
      expect(memory.entries, isEmpty);
      expect(memory.status.value.frames, 0);
    });

    test('leaving the demo (its owner\'s stop) forgets every frame and stops '
        'sampling; a stop from anyone else does nothing', () async {
      await remember();
      memory.stop(owner: Object());
      expect(memory.entries, hasLength(1));

      memory.stop(owner: owner);
      expect(memory.entries, isEmpty);
      expect(memory.status.value.active, isFalse);
      capture.results.add(Result.ok(fakeShot()));
      await tickAt(200);
      expect(capture.calls, hasLength(1), reason: 'no owner, no sampling');
    });

    test('a new owner takes over afresh', () async {
      await remember();
      memory.start(owner: Object(), idle: () => true);
      expect(memory.entries, isEmpty);
    });

    test('"Clear memory" during an embedding drops its result', () async {
      embedder.gate = Completer();
      capture.results.add(Result.ok(fakeShot()));
      final step = tickAt(100);
      await pumpEventQueue();
      memory.clear();
      embedder.gate!.complete();
      await step;
      expect(memory.entries, isEmpty);
    });
  });

  group('recall', () {
    Future<void> rememberScenes() async {
      // Scene 1 at 100 s, 2 at 110 s, 3 at 120 s.
      for (var i = 0; i < 3; i++) {
        capture.results.add(Result.ok(fakeShot(frameId: i, scene: i + 1)));
        await tickAt(100 + i * 10);
      }
    }

    test('nothing remembered yet: MemoryEmpty, no embedding', () async {
      final recall = await memory.recall('mug');
      expect((recall as Ok<MemoryRecall>).value, isA<MemoryEmpty>());
      expect(embedder.texts, isEmpty);
    });

    test('a phrase that matches one frame clearly: a hit with its age, and '
        'the best cosine in the status', () async {
      await rememberScenes();
      now = 130 * _s;
      final recall = (await memory.recall('mug') as Ok<MemoryRecall>).value;
      final hit = recall as MemoryHit;
      expect(hit.entry.frameId, 0);
      expect(hit.age, const Duration(seconds: 30));
      expect(hit.cosine, closeTo(0.9878, 1e-3));
      expect(hit.rival, closeTo(0.1098, 1e-3));
      expect(memory.status.value.lastBestCosine, hit.cosine);
      expect(embedder.texts, ['mug']);
    });

    test(
      'a phrase alike to two different scenes: an ambiguous hit with the '
      'newest of the competing frames (maybe in more than one place)',
      () async {
        await rememberScenes();
        now = 130 * _s;
        final recall = (await memory.recall('sign') as Ok<MemoryRecall>).value;
        final hit = recall as MemoryHit;
        expect(hit.ambiguous, isTrue);
        expect(hit.entry.frameId, 1, reason: 'scene 2, at 110 s, the newer');
        expect(hit.age, const Duration(seconds: 20));
        expect(hit.rival, closeTo(0.7035, 1e-3));
      },
    );

    test('one scene seen twice: the newest near-identical frame is reported '
        '(its latest sighting), not the one that won by noise', () async {
      // Scene 1 a little moved at 100 s (it scores best), scene 2 at 110 s,
      // scene 1 at 120 s (0.006 lower).
      for (final (i, scene) in [4, 2, 1].indexed) {
        capture.results.add(Result.ok(fakeShot(frameId: i, scene: scene)));
        await tickAt(100 + i * 10);
      }
      now = 130 * _s;
      final hit =
          (await memory.recall('mug') as Ok<MemoryRecall>).value as MemoryHit;
      expect(hit.ambiguous, isFalse);
      expect(hit.entry.frameId, 2);
      expect(hit.age, const Duration(seconds: 10));
      expect(hit.cosine, closeTo(0.9879, 1e-3));
      expect(memory.status.value.lastBestCosine, closeTo(0.9940, 1e-3));
    });

    test('a near-identical frame within the tie whose thumbnail shows a '
        'change (the keys put down in the best frame, taken away again) is '
        'not its latest sighting: the best frame is reported', () async {
      // 128×128 thumbnails, as the worker makes them: an empty desk, and
      // the same with keys on it.
      final desk = Uint8List(128 * 128)..fillRange(0, 128 * 128, 100);
      final keys = Uint8List.fromList(desk);
      for (var y = 32; y < 64; y++) {
        keys.fillRange(y * 128 + 32, y * 128 + 64, 200);
      }
      for (final (i, (scene, grey)) in [
        (4, keys),
        (2, desk),
        (1, desk),
      ].indexed) {
        capture.results.add(
          Result.ok(fakeShot(frameId: i, scene: scene, grey: grey)),
        );
        await tickAt(100 + i * 10);
      }
      now = 130 * _s;
      final hit =
          (await memory.recall('mug') as Ok<MemoryRecall>).value as MemoryHit;
      expect(hit.ambiguous, isFalse);
      expect(hit.entry.frameId, 0, reason: 'the frame with the keys');
      expect(hit.age, const Duration(seconds: 30));
    });

    test('the newest frame, for a question with nothing to search for: a hit '
        'without a cosine, nothing embedded; empty, switched off and off as '
        'for a recall', () async {
      expect(
        (await memory.latest() as Ok<MemoryRecall>).value,
        isA<MemoryEmpty>(),
      );
      for (final (i, scene) in [1, 2].indexed) {
        capture.results.add(Result.ok(fakeShot(frameId: i, scene: scene)));
        await tickAt(100 + i * 10);
      }
      now = 125 * _s;
      final texts = embedder.texts.length;
      final hit =
          (await memory.latest() as Ok<MemoryRecall>).value as MemoryHit;
      expect(hit.entry.frameId, 1);
      expect(hit.cosine, isNull);
      expect(hit.latest, isTrue);
      expect(hit.age, const Duration(seconds: 15));
      expect(embedder.texts, hasLength(texts));

      memory.setEnabled(enabled: false);
      expect(
        (await memory.latest() as Ok<MemoryRecall>).value,
        isA<MemoryDisabled>(),
      );
    });

    test(
      'one scene only (no rival): the floor plus the margin decides',
      () async {
        capture.results.add(Result.ok(fakeShot()));
        await tickAt(100);
        final faint = (await memory.recall('faint') as Ok<MemoryRecall>).value;
        expect(faint, isA<MemoryMiss>());
        expect((faint as MemoryMiss).reason, startsWith('no rival'));
        expect(faint.cosine, closeTo(0.630, 1e-3));
        final mug = (await memory.recall('mug') as Ok<MemoryRecall>).value;
        expect(mug, isA<MemoryHit>());
        expect((mug as MemoryHit).rival, isNull);
      },
    );

    test('a phrase no frame matches well: a miss below the floor', () async {
      await rememberScenes();
      final recall = (await memory.recall('giraffe') as Ok<MemoryRecall>).value;
      expect((recall as MemoryMiss).reason, 'below the floor 0.62');
      expect(recall.cosine, closeTo(0.577, 1e-3));
    });

    test('a phrase that cannot be embedded is an Error', () async {
      await rememberScenes();
      embedder.textError = Exception('too long');
      final recall = await memory.recall('mug');
      expect(recall, isA<Error<MemoryRecall>>());
    });
  });
}
