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
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/services/frames/fixture_frame_source.dart';
import 'package:litert_edge_demos/data/services/frames/frame_source.dart';
import 'package:litert_edge_demos/domain/models/frame_source_info.dart';
import 'package:litert_edge_demos/domain/models/preview_source.dart';
import 'package:litert_edge_demos/utils/result.dart';

/// Writes a [w]×[h] PNG filled with [color] to [path].
Future<void> writePng(String path, int w, int h, Color color) async {
  final recorder = ui.PictureRecorder();
  Canvas(recorder).drawRect(
    Rect.fromLTWH(0, 0, w.toDouble(), h.toDouble()),
    Paint()..color = color,
  );
  final image = await recorder.endRecording().toImage(w, h);
  final png = await image.toByteData(format: ui.ImageByteFormat.png);
  image.dispose();
  await File(path).writeAsBytes(png!.buffer.asUint8List());
}

/// The source's schedule, driven by the test: [timer] and [clockMicros]
/// are the source's seams, [runUntil] fires the due callbacks in order,
/// moving the clock to each one's time.
final class _Schedule {
  int now = 0; // µs
  final List<_ScheduledTimer> _timers = [];

  int Function() get clockMicros =>
      () => now;

  Timer timer(Duration delay, void Function() callback) {
    final t = _ScheduledTimer(now + delay.inMicroseconds, callback);
    _timers.add(t);
    return t;
  }

  /// The next pending callback's due time; null when none is pending.
  int? get nextDue {
    _timers.removeWhere((t) => !t.isActive);
    if (_timers.isEmpty) return null;
    return _timers.map((t) => t.due).reduce((a, b) => a < b ? a : b);
  }

  /// Fires every callback due before [end] µs, the clock at its due time
  /// plus [late] µs (a stalled app), then moves the clock to [end] (never
  /// back).
  void runUntil(int end, {int late = 0}) {
    while (true) {
      final due = nextDue;
      if (due == null || due >= end) break;
      final t = _timers.firstWhere((t) => t.isActive && t.due == due);
      now = due + late;
      t.fire();
    }
    if (end > now) now = end;
  }
}

final class _ScheduledTimer implements Timer {
  _ScheduledTimer(this.due, this._callback);

  final int due;
  final void Function() _callback;
  bool _active = true;

  void fire() {
    _active = false;
    _callback();
  }

  @override
  void cancel() => _active = false;

  @override
  bool get isActive => _active;

  @override
  int get tick => _active ? 0 : 1;
}

void main() {
  late Directory dir;

  setUp(() => dir = Directory.systemTemp.createTempSync('fixture_source'));
  tearDown(() => dir.deleteSync(recursive: true));

  testWidgets('decodes each image once to RGBA, emits on schedule, holds each '
      'image, and shows it as the preview', (tester) async {
    await tester.runAsync(() async {
      await writePng('${dir.path}/a_red.png', 8, 6, const Color(0xFFFF0000));
      await writePng('${dir.path}/b_blue.png', 6, 8, const Color(0xFF0000FF));
      File('${dir.path}/notes.txt').writeAsStringSync('ignored');
      // The schedule runs on the test's clock: exact, whatever the load.
      final schedule = _Schedule();
      final source = FixtureFrameSource(
        paths: [dir.path],
        fps: 50,
        hold: const Duration(milliseconds: 120),
        clockMicros: schedule.clockMicros,
        timer: schedule.timer,
      );
      final frames = <String>[]; // '<w>x<h> <first RGBA pixel>'
      final previews = <ui.Image?>[];
      final image = (source.preview as ImagePreviewSource).image;
      image.addListener(() => previews.add(image.value));

      final started = await source.start(
        (f) => frames.add(
          '${f.width}x${f.height} ${f.planes.single.bytes.take(4).join(',')}',
        ),
      );
      expect(frames, isEmpty, reason: 'the first frame follows start()');
      schedule.runUntil(400000); // 400 ms at 50 fps: frames at 0, 20 … 380
      await source.stop();
      final count = frames.length;
      schedule.runUntil(500000);

      final info = (started as Ok<FrameSourceInfo>).value;
      expect(info.label, 'fixture (2 images)');
      expect(
        (info.width, info.height, info.format),
        (8, 6, FramePixelFormat.rgba8888),
      );
      expect(source.imagePaths, isEmpty, reason: 'released on stop');
      expect(frames.length, count, reason: 'nothing after stop');
      expect(schedule.nextDue, isNull, reason: 'no tick left scheduled');
      expect(count, 20);
      const red = '8x6 255,0,0,255';
      const blue = '6x8 0,0,255,255';
      // Held 120 ms: six frames per image, alternating; the last two are
      // the start of the fourth hold.
      expect(frames, [
        ...List.filled(6, red),
        ...List.filled(6, blue),
        ...List.filled(6, red),
        ...List.filled(2, blue),
      ]);
      expect(previews.whereType<ui.Image>(), hasLength(4));
      expect(previews.last, isNull, reason: 'preview cleared on stop');
    });
  });

  testWidgets('mirrored: frames and preview are flipped like camera_desktop '
      'and the source says so (the un-mirroring fixture)', (tester) async {
    await tester.runAsync(() async {
      // 4×2: left half red, right half blue.
      final recorder = ui.PictureRecorder();
      Canvas(recorder)
        ..drawRect(
          const Rect.fromLTWH(0, 0, 2, 2),
          Paint()..color = const Color(0xFFFF0000),
        )
        ..drawRect(
          const Rect.fromLTWH(2, 0, 2, 2),
          Paint()..color = const Color(0xFF0000FF),
        );
      final img = await recorder.endRecording().toImage(4, 2);
      final png = await img.toByteData(format: ui.ImageByteFormat.png);
      img.dispose();
      await File('${dir.path}/halves.png')
          .writeAsBytes(png!.buffer.asUint8List());

      final source = FixtureFrameSource(paths: [dir.path], mirrored: true);
      final frames = <List<int>>[];
      final first = Completer<void>();
      final started = await source.start((f) {
        frames.add(List.of(f.planes.single.bytes.take(16)));
        if (!first.isCompleted) first.complete();
      });
      await first.future.timeout(const Duration(seconds: 15));
      final preview = (source.preview as ImagePreviewSource).image.value!;
      final previewBytes = (await preview.toByteData())!.buffer.asUint8List();
      await source.stop();

      final info = (started as Ok<FrameSourceInfo>).value;
      expect(info.mirrored, isTrue);
      expect(info.previewMirrored, isTrue);
      expect(info.overlayMirrored, isFalse, reason: 'both mirrored');
      expect(info.label, contains('mirrored'));
      expect(frames.first.sublist(0, 4), [0, 0, 255, 255], reason: 'blue left');
      expect(frames.first.sublist(12, 16), [255, 0, 0, 255]);
      expect(previewBytes.sublist(0, 4), [0, 0, 255, 255]);
    });
  });

  testWidgets('downscales images over maxSide once at decode', (tester) async {
    await tester.runAsync(() async {
      await writePng('${dir.path}/wide.png', 40, 20, const Color(0xFF00FF00));
      final source = FixtureFrameSource(
        paths: ['${dir.path}/wide.png'],
        maxSide: 10,
      );
      final frames = <FrameView>[];
      final first = Completer<void>();

      await source.start((f) {
        frames.add(f);
        if (!first.isCompleted) first.complete();
      });
      await first.future.timeout(const Duration(seconds: 15));
      final (w, h, bpr) = (
        frames.first.width,
        frames.first.height,
        frames.first.planes.single.bytesPerRow,
      );
      await source.stop();

      expect((w, h, bpr), (10, 5, 40));
    });
  });

  testWidgets('drift-free: frame k is due at k × period; after a stall the '
      'missed frames are skipped, not burst', (tester) async {
    await tester.runAsync(() async {
      await writePng('${dir.path}/a.png', 4, 4, const Color(0xFFFF0000));
      final schedule = _Schedule();
      final source = FixtureFrameSource(
        paths: [dir.path],
        fps: 50, // a 20 ms period
        clockMicros: schedule.clockMicros,
        timer: schedule.timer,
      );
      var frames = 0;
      await source.start((_) => frames++);

      // Each tick runs 5 ms late: the next one is still due on the grid.
      schedule.runUntil(100000, late: 5000);
      expect(frames, 5, reason: 'at 0, 20, 40, 60, 80 ms');
      expect(schedule.nextDue, 100000);

      // The app stalls for 75 ms: the tick due at 100 ms runs at 175 ms.
      schedule.runUntil(100001, late: 75000);
      expect(frames, 6);
      expect(
        schedule.nextDue,
        180000,
        reason: 'the next grid slot, not the four missed ones at once',
      );
      schedule.runUntil(260000);
      expect(frames, 10, reason: 'back on the grid: 180, 200, 220, 240');
      await source.stop();
    });
  });

  testWidgets(
    'a missing path, an empty directory or a second start are errors',
    (tester) async {
      await tester.runAsync(() async {
        final missing = await FixtureFrameSource(paths: ['${dir.path}/nope'])
            .start((_) {});
        expect((missing as Error).error.toString(), contains('not found'));

        final empty = await FixtureFrameSource(paths: [dir.path]).start((_) {});
        expect((empty as Error).error.toString(), contains('No images'));

        await writePng('${dir.path}/one.png', 4, 4, const Color(0xFFFFFFFF));
        final once = FixtureFrameSource(paths: [dir.path]);
        expect(await once.start((_) {}), isA<Ok<FrameSourceInfo>>());
        expect(await once.start((_) {}), isA<Error<FrameSourceInfo>>());
        await once.stop();
      });
    },
  );
}
