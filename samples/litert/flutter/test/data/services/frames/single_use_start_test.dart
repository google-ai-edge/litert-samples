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
import 'dart:ui' as ui;

import 'package:flutter/foundation.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/services/frames/frame_source.dart';
import 'package:litert_edge_demos/domain/models/frame_source_info.dart';
import 'package:litert_edge_demos/domain/models/frame_source_spec.dart';
import 'package:litert_edge_demos/domain/models/preview_source.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../../support/frames.dart';

const _info = FrameSourceInfo(
  label: 'test',
  width: 640,
  height: 480,
  format: FramePixelFormat.rgba8888,
  mirrored: false,
);

/// The smallest source on [SingleUseStart]: its acquisition can be held
/// open with [gate], and [checksStopped] false plays a source that forgets
/// the stop check (the mixin must still deliver nothing after stop).
final class _Source with SingleUseStart {
  _Source({this.checksStopped = true});

  final bool checksStopped;
  int acquires = 0;
  int releases = 0;
  Completer<void>? gate;
  final ValueNotifier<ui.Image?> _image = ValueNotifier(null);

  @override
  PreviewSource get preview => ImagePreviewSource(_image);

  @override
  String get sourceKind => 'test';

  @override
  FrameSourceUnavailableException get stoppedWhileStarting =>
      const FrameSourceUnavailableException(
        'Stopped while the test source started',
      );

  @override
  Future<Result<FrameSourceInfo>> acquire(
    void Function(FrameView) onFrame, {
    void Function(Exception error)? onError,
  }) async {
    acquires++;
    await gate?.future;
    if (checksStopped && stopped) return Result.error(stoppedWhileStarting);
    attach(onFrame, onError: onError);
    return const Result.ok(_info);
  }

  @override
  Future<void> release() async {
    releases++;
    _image.dispose();
  }

  /// A frame from the device.
  void emit(FrameView frame) => frameCallback?.call(frame);

  /// An error from the device; whether it was reported.
  bool fail(Exception error) {
    if (!markFailed()) return false;
    errorCallback?.call(error);
    return true;
  }

  bool get hasFailed => failed;
}

String _errorOf(Result<FrameSourceInfo> result) => switch (result) {
  Ok() => fail('expected an error, got $result'),
  Error(:final error) => error.toString(),
};

void main() {
  final frame = TestFrame.rgba(width: 4, height: 4);
  late List<FrameView> frames;
  late List<Exception> errors;

  setUp(() {
    frames = [];
    errors = [];
  });

  test('start once: a second start is refused without acquiring again; '
      'frames and the one error reach the callbacks start was given', () async {
    final source = _Source();

    expect(
      await source.start(frames.add, onError: errors.add),
      isA<Ok<FrameSourceInfo>>(),
    );
    expect(
      _errorOf(await source.start(frames.add, onError: errors.add)),
      'A test source is single-use',
    );
    expect(source.acquires, 1);

    source.emit(frame);
    expect(frames, [same(frame)]);
    expect(
      source.fail(const FrameSourceUnavailableException('unplugged')),
      isTrue,
    );
    expect(
      source.fail(const FrameSourceUnavailableException('again')),
      isFalse,
    );
    expect(errors.map((e) => '$e'), ['unplugged'], reason: 'once');
    expect(source.hasFailed, isTrue);
    await source.stop();
  });

  test('stop releases once, however often it is called; after it nothing '
      'goes out and a start is refused', () async {
    final source = _Source();
    await source.start(frames.add, onError: errors.add);

    await Future.wait([source.stop(), source.stop()]);
    await source.stop();

    expect(source.releases, 1);
    source.emit(frame);
    expect(frames, isEmpty);
    expect(source.fail(const FrameSourceUnavailableException('late')), isFalse);
    expect(errors, isEmpty);
    expect(source.hasFailed, isFalse, reason: 'no error after stop');
    expect(
      _errorOf(await source.start(frames.add)),
      'A test source is single-use',
    );
  });

  test('stop while starting: start fails with the source\'s stopped error, '
      'release ran once, and nothing is attached', () async {
    final source = _Source()..gate = Completer<void>();

    final starting = source.start(frames.add, onError: errors.add);
    await source.stop();
    source.gate!.complete();

    expect(_errorOf(await starting), 'Stopped while the test source started');
    expect(source.releases, 1);
    source.emit(frame);
    expect(frames, isEmpty);
  });

  test(
    'a source that attaches after stop anyway still delivers nothing',
    () async {
      final source = _Source(checksStopped: false)..gate = Completer<void>();

      final starting = source.start(frames.add, onError: errors.add);
      await source.stop();
      source.gate!.complete();
      await starting;

      source.emit(frame);
      expect(frames, isEmpty);
      expect(
        source.fail(const FrameSourceUnavailableException('late')),
        isFalse,
      );
      expect(errors, isEmpty);
    },
  );

  test('a start after a stop that came first reaches the source\'s stop '
      'check', () async {
    final source = _Source();

    await source.stop();

    expect(
      _errorOf(await source.start(frames.add)),
      'Stopped while the test source started',
    );
    expect(source.releases, 1);
  });

  test('an error before attach is claimed but goes nowhere', () async {
    final source = _Source()..gate = Completer<void>();
    final starting = source.start(frames.add, onError: errors.add);

    expect(source.fail(const FrameSourceUnavailableException('early')), isTrue);
    source.gate!.complete();
    await starting;

    expect(errors, isEmpty, reason: 'start had not attached onError yet');
    expect(
      source.fail(const FrameSourceUnavailableException('later')),
      isFalse,
    );
    await source.stop();
  });
}
