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

import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/live_camera_config.dart';
import 'package:litert_edge_demos/data/services/detector/frame_message.dart';
import 'package:litert_edge_demos/data/services/detector/memory_shot.dart';
import 'package:litert_edge_demos/data/services/frames/frame_source.dart';
import 'package:litert_edge_demos/domain/models/frame_source_info.dart';
import 'package:litert_edge_demos/domain/models/scene_snapshot.dart';
import 'package:litert_edge_demos/domain/models/visual_memory.dart';

import '../../../fakes/fake_detector.dart';
import '../../../support/desk_scene.dart';
import '../../../support/frames.dart';

/// [w]×[h] RGBA, the left half red, the right half blue.
TestFrame halves({int w = 1280, int h = 720}) {
  final bytes = Uint8List(w * h * 4);
  for (var y = 0; y < h; y++) {
    for (var x = 0; x < w; x++) {
      final o = (y * w + x) * 4;
      final left = x < w ~/ 2;
      bytes[o] = left ? 255 : 0;
      bytes[o + 2] = left ? 0 : 255;
      bytes[o + 3] = 255;
    }
  }
  return TestFrame(w, h, FramePixelFormat.rgba8888, 0, [
    FramePlane(bytes: bytes, bytesPerRow: w * 4, bytesPerPixel: 4),
  ]);
}

MemoryShotSpec spec({Uint8List? reference, bool unmirror = false}) =>
    MemoryShotSpec(
      maxSide: 320,
      greySide: kMemoryGreySide,
      cellPixels: kMemoryCellPixels,
      maxShift: kMemoryMaxShift,
      unmirror: unmirror,
      reference: reference,
      changeThreshold: kMemorySceneChange,
      cellChange: kMemoryCellChange,
      changedCells: kMemoryChangedCells,
    );

MemoryShot shotOf(TestFrame view, MemoryShotSpec spec, {int frameId = 9}) {
  final message = FrameMessage.copyOf(view, frameId: frameId);
  return memoryShot(
    message,
    message.materialize(),
    FakeDetector.catsFor(message, cats: 2),
    spec,
  );
}

Future<(int, int, Uint8List)> decode(Uint8List png) async {
  final codec = await ui.instantiateImageCodec(png);
  final image = (await codec.getNextFrame()).image;
  final data = (await image.toByteData())!;
  final out = (image.width, image.height, data.buffer.asUint8List());
  image.dispose();
  codec.dispose();
  return out;
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('downscale and thumbnail', () {
    test('area averaging: each output pixel is the mean of its block', () {
      // 4×2 RGBA: two 2×2 blocks, the first (0,40,80,120)-ish, the second 200.
      final bytes = Uint8List.fromList([
        0, 0, 0, 255, 40, 40, 40, 255, 200, 0, 0, 255, 200, 0, 0, 255, //
        80, 80, 80, 255, 120, 120, 120, 255, 200, 0, 0, 255, 200, 0, 0, 255,
      ]);
      final rgb = downscaleRgb(
        RgbaPixels(width: 4, height: 2, bytes: bytes),
        2,
        1,
      );
      expect(rgb, [60, 60, 60, 200, 0, 0]);
      final mirrored = downscaleRgb(
        RgbaPixels(width: 4, height: 2, bytes: bytes),
        2,
        1,
        mirror: true,
      );
      expect(mirrored, [200, 0, 0, 60, 60, 60]);
    });

    test('a target larger than the source is refused', () {
      expect(
        () => downscaleRgb(
          RgbaPixels(width: 2, height: 2, bytes: Uint8List(16)),
          3,
          2,
        ),
        throwsArgumentError,
      );
    });

    test('the long side is fitted, never scaled up', () {
      expect(fitLongSide(1280, 720, 320), (320, 180));
      expect(fitLongSide(720, 1280, 320), (180, 320));
      expect(fitLongSide(200, 100, 320), (200, 100));
    });

    test('the grey thumbnail is BT.601 luma per cell; the difference is the '
        'mean absolute difference on 0–255', () {
      final white = Uint8List(64 * 32 * 3)..fillRange(0, 64 * 32 * 3, 255);
      final red = Uint8List(64 * 32 * 3);
      for (var i = 0; i < red.length; i += 3) {
        red[i] = 255;
      }
      final a = greyThumbnail(white, 64, 32, 32);
      final b = greyThumbnail(red, 64, 32, 32);
      expect(a, everyElement(255));
      expect(b, everyElement(76)); // 0.299 × 255
      expect(greyDifference(a, a), 0);
      expect(greyDifference(a, b), 179);
      // Smaller than the thumbnail: pixels repeat.
      expect(greyThumbnail(Uint8List(3 * 4), 2, 2, 32), hasLength(1024));
      expect(() => greyDifference(a, Uint8List(10)), throwsArgumentError);
    });
  });

  group('memoryShot', () {
    test('no reference: the PNG is made, at most 320 px, with the frame\'s '
        'detections and thumbnail', () async {
      final shot = shotOf(halves(), spec());
      expect(shot.frameId, 9);
      expect(shot.detections.count, 2);
      expect((shot.width, shot.height), (320, 180));
      expect(shot.grey, hasLength(128 * 128));
      expect(shot.difference, isNull);
      final (w, h, rgba) = await decode(shot.png!);
      expect((w, h), (320, 180));
      expect([rgba[0], rgba[2]], [255, 0], reason: 'left stays red');
      final right = (90 * 320 + 319) * 4;
      expect([rgba[right], rgba[right + 2]], [0, 255]);
    });

    test('the same scene again: no PNG (nothing to embed; the memory '
        'refreshes its newest frame instead)', () {
      final first = shotOf(halves(), spec());
      final same = shotOf(halves(), spec(reference: first.grey));
      expect(same.difference, 0);
      expect(same.png, isNull);
      expect((same.width, same.height), (320, 180));
    });

    test('a changed scene makes the PNG', () {
      final first = shotOf(halves(), spec());
      final other = shotOf(
        TestFrame.rgba(width: 1280, height: 720, fill: 255),
        spec(reference: first.grey),
      );
      expect(other.difference, greaterThan(6));
      expect(other.png, isNotNull);
    });

    test('a scene change reports its mean and changed cells', () {
      final first = shotOf(halves(), spec());
      expect(first.changedCells, isNull, reason: 'no reference');
      final other = shotOf(
        TestFrame.rgba(width: 1280, height: 720, fill: 255),
        spec(reference: first.grey),
      );
      expect(other.changedCells, greaterThanOrEqualTo(512));
    });

    test('a mirroring source: the PNG is mirrored back', () async {
      final shot = shotOf(halves(), spec(unmirror: true));
      expect(shot.unmirrored, isTrue);
      final (_, _, rgba) = await decode(shot.png!);
      expect([rgba[0], rgba[2]], [0, 255], reason: 'left is blue now');
    });
  });
  group('scene change on a synthetic desk (the shipped thresholds)', () {
    /// The worker's verdict on [current] against [reference]'s thumbnail,
    /// printed with the numbers behind it.
    MemoryShot judge(String what, TestFrame reference, TestFrame current) {
      final ref = shotOf(reference, spec()).grey;
      final shot = shotOf(current, spec(reference: ref));
      final change = sceneChange(
        ref,
        shot.grey,
        cellThreshold: kMemoryCellChange,
        cellPixels: kMemoryCellPixels,
        maxShift: kMemoryMaxShift,
      );
      // The measured values are what these tests are for.
      // ignore: avoid_print
      print(
        'SCENE_CHANGE $what: mean ${change.mean.toStringAsFixed(2)} max '
        '${change.max.toStringAsFixed(1)} cells≥${kMemoryCellChange.round()} '
        '${change.cells} gain ${change.gain.toStringAsFixed(3)} shift '
        '(${change.shiftX}, ${change.shiftY}) (raw mean '
        '${greyDifference(ref, shot.grey).toStringAsFixed(2)}) → '
        '${shot.png == null ? 'refreshed in place' : 'stored'}',
      );
      expect(shot.difference, change.mean);
      expect(shot.changedCells, change.cells);
      return shot;
    }

    test('keys, a phone, a mug, a book or a laptop put on the desk or taken '
        'off it is a scene change: the frame is stored', () {
      final empty = deskScene(seed: 1);
      for (final object in DeskObject.values) {
        final added = judge(
          '${object.name} added',
          empty,
          deskScene(objects: {object}, seed: 2),
        );
        expect(added.png, isNotNull, reason: '${object.name} added');
        final removed = judge(
          '${object.name} removed',
          deskScene(objects: {object}, seed: 3),
          deskScene(seed: 4),
        );
        expect(removed.png, isNotNull, reason: '${object.name} removed');
      }
    });

    test('sensor noise and a 10 % brightness change are the same scene: no '
        'PNG (a forced tick refreshes the frame in place)', () {
      final still = deskScene(objects: {DeskObject.mug}, seed: 1);
      for (final (what, frame) in [
        ('noise ±3', deskScene(objects: {DeskObject.mug}, seed: 2)),
        ('noise ±8', deskScene(objects: {DeskObject.mug}, noise: 8, seed: 3)),
        (
          '10 % darker',
          deskScene(objects: {DeskObject.mug}, seed: 4, brightness: 0.9),
        ),
        (
          '10 % brighter',
          deskScene(objects: {DeskObject.mug}, seed: 5, brightness: 1.1),
        ),
      ]) {
        expect(judge(what, still, frame).png, isNull, reason: what);
      }
    });

    test('a hand-held camera that moves 3, 4 or 8 px (sensor noise ±3) is '
        'the same scene: no PNG, the frame is refreshed in place', () {
      for (final objects in [
        <DeskObject>{},
        {DeskObject.mug},
        {DeskObject.book, DeskObject.mug},
        {DeskObject.laptop},
      ]) {
        final still = deskScene(objects: objects, seed: 1);
        final scene = objects.isEmpty
            ? 'desk'
            : objects.map((o) => o.name).join('+');
        for (final (dx, dy) in const [
          (3, 0),
          (0, 3),
          (-3, 5),
          (4, 4),
          (-4, -4),
          (8, 0),
          (0, 8),
          (8, 8),
          (-8, -8),
        ]) {
          final what = '$scene, camera moved ($dx, $dy) px';
          final moved = deskScene(
            objects: objects,
            seed: 2,
            shiftX: dx,
            shiftY: dy,
          );
          expect(judge(what, still, moved).png, isNull, reason: what);
        }
      }
    });

    test('the highest-contrast edge (a black phone on white paper) with the '
        'camera moved by half a thumbnail pixel or more is the same scene; '
        'the phone taken away is a change', () {
      final still = phoneOnPaper(seed: 1);
      for (final (dx, dy) in const [
        (5, 0),
        (5, 3),
        (-5, -3),
        (15, 0),
        (0, 8),
        (8, 8),
      ]) {
        final what = 'phone on paper, camera moved ($dx, $dy) px';
        final moved = phoneOnPaper(seed: 2, shiftX: dx, shiftY: dy);
        expect(judge(what, still, moved).png, isNull, reason: what);
      }
      expect(
        judge(
          'phone taken off the paper',
          still,
          phoneOnPaper(phone: false),
        ).png,
        isNotNull,
      );
    });

    test('an object put down or taken away while the camera moves 4 px is '
        'still a scene change', () {
      for (final object in DeskObject.values) {
        final added = judge(
          '${object.name} added, camera moved (4, 4) px',
          deskScene(seed: 1),
          deskScene(objects: {object}, seed: 2, shiftX: 4, shiftY: 4),
        );
        expect(added.png, isNotNull, reason: '${object.name} added');
        final removed = judge(
          '${object.name} removed, camera moved (-4, 4) px',
          deskScene(objects: {object}, seed: 3),
          deskScene(seed: 4, shiftX: -4, shiftY: 4),
        );
        expect(removed.png, isNotNull, reason: '${object.name} removed');
      }
    });

    test('thumbnail noise of ±4 grey levels per cell is not a change', () {
      final ref = shotOf(deskScene(seed: 1), spec()).grey;
      final random = math.Random(7);
      const side = kMemoryGreySide * kMemoryCellPixels;
      // One offset per cell, on all of its pixels.
      final offsets = [
        for (var i = 0; i < kMemoryGreySide * kMemoryGreySide; i++)
          random.nextInt(9) - 4,
      ];
      final noisy = Uint8List.fromList([
        for (var i = 0; i < ref.length; i++)
          (ref[i] +
                  offsets[(i ~/ side ~/ kMemoryCellPixels) * kMemoryGreySide +
                      i % side ~/ kMemoryCellPixels])
              .clamp(0, 255),
      ]);
      final change = sceneChange(
        ref,
        noisy,
        cellThreshold: kMemoryCellChange,
        cellPixels: kMemoryCellPixels,
        maxShift: kMemoryMaxShift,
      );
      // The measured values are what this test is for.
      // ignore: avoid_print
      print(
        'SCENE_CHANGE thumbnail noise ±4: mean '
        '${change.mean.toStringAsFixed(2)} max ${change.max.toStringAsFixed(1)} '
        'cells ${change.cells}',
      );
      expect(change.mean, lessThan(kMemorySceneChange));
      expect(change.cells, lessThan(kMemoryChangedCells));
    });

    test('the brightness match: the median ratio, dark cells out, at most 2× '
        'either way', () {
      final a = Uint8List(1024)..fillRange(0, 1024, 100);
      final half = Uint8List(1024)..fillRange(0, 1024, 50);
      expect(sceneChange(a, half, cellThreshold: 24).gain, 2);
      expect(sceneChange(a, half, cellThreshold: 24).mean, 0);
      final third = Uint8List(1024)..fillRange(0, 1024, 33);
      final clamped = sceneChange(a, third, cellThreshold: 24);
      expect(clamped.gain, 2, reason: 'clamped');
      expect(clamped.mean, 34);
      expect(clamped.cells, 1024);
      // An object over a third of the view does not move the median.
      final object = Uint8List.fromList(a)..fillRange(0, 340, 200);
      final withObject = sceneChange(a, object, cellThreshold: 24);
      expect(withObject.gain, 1);
      expect(withObject.cells, 340);
      // Black on both sides: nothing votes, the gain is 1.
      final black = Uint8List(1024);
      expect(sceneChange(black, black, cellThreshold: 24).gain, 1);
      expect(
        () => sceneChange(a, Uint8List(10), cellThreshold: 24),
        throwsArgumentError,
      );
    });
  });
}
