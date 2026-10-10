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
import 'package:litert_edge_demos/data/repositories/audio_repository.dart';
import 'package:litert_edge_demos/domain/use_cases/push_to_talk_capture.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../fakes/fake_audio_repository.dart';

/// Records the STT window each start asked for.
class _RecordingAudio extends FakeAudioRepository {
  final List<Duration> maxLengths = [];

  @override
  Future<Result<CaptureHandle>> startCapture({
    required Duration maxLength,
    required void Function() onLimit,
  }) {
    maxLengths.add(maxLength);
    return super.startCapture(maxLength: maxLength, onLimit: onLimit);
  }
}

Future<void> settle() => pumpEventQueue();

void main() {
  late _RecordingAudio audio;
  late PushToTalkCapture mic;
  var limits = 0;

  void onLimit() => limits++;

  setUp(() {
    audio = _RecordingAudio();
    mic = PushToTalkCapture(
      audio: audio,
      maxLength: const Duration(seconds: 30),
    );
    limits = 0;
  });

  tearDown(() => audio.close());

  test('a press opens the mic with the STT window', () async {
    expect(mic.pressed, isFalse);

    final started = await mic.open(onLimit: onLimit);

    expect(
      started,
      isA<CaptureOpened>().having((s) => s.released, 'released', false),
    );
    expect(mic.pressed, isTrue);
    expect(audio.maxLengths, [const Duration(seconds: 30)]);
    expect(audio.captures.single.isOpen, isTrue);
  });

  test('pressAgain: false without a press; a no-op while the mic '
      'captures', () async {
    expect(mic.pressAgain(), isFalse);
    await mic.open(onLimit: onLimit);

    expect(mic.pressAgain(), isTrue);

    expect(audio.log.where((e) => e == 'startCapture'), hasLength(1));
    expect(audio.captures.single.isOpen, isTrue);
  });

  test('a release without a press finds nothing', () {
    expect(mic.release(), isNull);
  });

  group('a release while the mic captures', () {
    test('hands the capture over, unstopped, for the turn to close', () async {
      await mic.open(onLimit: onLimit);

      final release = mic.release();

      expect(release, isA<ReleasedListening>());
      final handle = await (release! as ReleasedListening).handOver;
      expect(handle, same(audio.captures.single));
      expect(audio.captures.single.isOpen, isTrue);
      expect(mic.pressed, isFalse);
      expect(mic.release(), isNull, reason: 'the press has ended');
    });

    test('an action right after it takes the press over: nothing is '
        'handed over, the capture is cancelled', () async {
      await mic.open(onLimit: onLimit);
      final release = mic.release()! as ReleasedListening;

      await mic.drop();

      expect(await release.handOver, isNull);
      expect(audio.captures.single.cancelled, isTrue);
    });

    test('a new press right after it is not the released one: nothing is '
        'handed over', () async {
      await mic.open(onLimit: onLimit);
      final release = mic.release()! as ReleasedListening;
      unawaited(mic.drop());
      final next = mic.open(onLimit: onLimit);

      expect(await release.handOver, isNull);
      expect(await next, isA<CaptureOpened>());
      expect(mic.pressed, isTrue);
    });
  });

  group('a slow start (the audio warm-up still running)', () {
    setUp(() => audio.startGate = Completer<void>());

    test('a release while the mic opens: the capture is cancelled once it '
        'starts, nothing recorded', () async {
      final opening = mic.open(onLimit: onLimit);
      await settle();
      final release = mic.release();
      expect(release, isA<ReleasedWhileOpening>());
      expect(mic.pressed, isTrue, reason: 'until the start has ended');

      audio.startGate!.complete();

      expect(
        await opening,
        isA<CaptureOpened>().having((s) => s.released, 'released', true),
      );
      expect(await (release! as ReleasedWhileOpening).cancelled, isTrue);
      final capture = audio.captures.single;
      expect(capture.cancelled, isTrue);
      expect(capture.stopped, isFalse);
      expect(mic.pressed, isFalse);
    });

    test('the caller of open sees the result before the waiting release '
        'resumes', () async {
      final order = <String>[];
      unawaited(mic.open(onLimit: onLimit).then((_) => order.add('open')));
      await settle();
      final release = mic.release()! as ReleasedWhileOpening;
      unawaited(release.cancelled.then((_) => order.add('release')));

      audio.startGate!.complete();
      await settle();

      expect(order, ['open', 'release']);
    });

    test('pressed again before the start: the press keeps the capture and '
        'the release gets nothing', () async {
      final opening = mic.open(onLimit: onLimit);
      await settle();
      final release = mic.release()! as ReleasedWhileOpening;
      expect(mic.pressAgain(), isTrue);

      audio.startGate!.complete();

      expect(
        await opening,
        isA<CaptureOpened>().having((s) => s.released, 'released', false),
      );
      expect(await release.cancelled, isFalse);
      expect(audio.captures.single.isOpen, isTrue);
      expect(mic.pressed, isTrue);
    });

    test('two slips: only the last release cancels the capture', () async {
      final opening = mic.open(onLimit: onLimit);
      await settle();
      final first = mic.release()! as ReleasedWhileOpening;
      mic.pressAgain();
      final second = mic.release()! as ReleasedWhileOpening;

      audio.startGate!.complete();
      await opening;

      expect(await first.cancelled, isFalse);
      expect(await second.cancelled, isTrue);
      expect(audio.captures.single.cancelled, isTrue);
    });

    test('a slip, then the start fails: the press reports it, the release '
        'gets nothing', () async {
      audio.startError = const MicAccessException('off');
      final opening = mic.open(onLimit: onLimit);
      await settle();
      final release = mic.release()! as ReleasedWhileOpening;

      audio.startGate!.complete();

      expect(
        await opening,
        isA<CaptureFailed>().having(
          (s) => s.error,
          'error',
          isA<MicAccessException>(),
        ),
      );
      expect(await release.cancelled, isFalse);
      expect(mic.pressed, isFalse);
    });

    test('dropped while the mic opens: the capture it gets is cancelled, '
        'and a waiting release gets nothing', () async {
      final opening = mic.open(onLimit: onLimit);
      await settle();
      final release = mic.release()! as ReleasedWhileOpening;

      await mic.drop();
      expect(mic.pressed, isFalse);
      audio.startGate!.complete();

      expect(await opening, isA<CaptureDropped>());
      expect(await release.cancelled, isFalse);
      expect(audio.captures.single.cancelled, isTrue);
    });

    test('a slip, then dropped as soon as the start ends: the drop cancels '
        'the capture, the release gets nothing', () async {
      final opening = mic.open(onLimit: onLimit);
      await settle();
      final release = mic.release()! as ReleasedWhileOpening;
      audio.startGate!.complete();
      await opening;

      await mic.drop();

      expect(await release.cancelled, isFalse);
      expect(audio.log.where((e) => e == 'capture.cancel'), hasLength(1));
    });

    test('a drop after the release cancelled the capture has nothing left '
        'to cancel', () async {
      final opening = mic.open(onLimit: onLimit);
      await settle();
      final release = mic.release()! as ReleasedWhileOpening;
      audio.startGate!.complete();
      await opening;
      expect(await release.cancelled, isTrue);

      await mic.drop();

      expect(audio.log.where((e) => e == 'capture.cancel'), hasLength(1));
      expect(mic.pressed, isFalse);
    });
  });

  test('a start that fails ends the press; the next press opens the mic '
      'again', () async {
    audio.startError = Exception('busy');

    final failed = await mic.open(onLimit: onLimit);

    expect(failed, isA<CaptureFailed>());
    expect('${(failed as CaptureFailed).error}', contains('busy'));
    expect(mic.pressed, isFalse);
    expect(mic.release(), isNull);

    audio.startError = null;
    expect(await mic.open(onLimit: onLimit), isA<CaptureOpened>());
  });

  group('drop', () {
    test('while capturing: cancels the capture, completes after the '
        'cancel', () async {
      await mic.open(onLimit: onLimit);

      await mic.drop();

      expect(audio.captures.single.cancelled, isTrue);
      expect(mic.pressed, isFalse);
      expect(mic.release(), isNull);
    });

    test('without a press: completes, nothing to cancel', () async {
      await mic.drop();
      expect(audio.log, isEmpty);
    });
  });

  group('the STT window', () {
    test('reaching it while the press is current calls onLimit', () async {
      await mic.open(onLimit: onLimit);

      audio.lastOnLimit!();

      expect(limits, 1);
    });

    test('a press that was handed over or dropped no longer reaches '
        'onLimit', () async {
      await mic.open(onLimit: onLimit);
      final handedOver = audio.lastOnLimit!;
      await (mic.release()! as ReleasedListening).handOver;
      await mic.open(onLimit: onLimit);
      final dropped = audio.lastOnLimit!;
      await mic.drop();

      handedOver();
      dropped();

      expect(limits, 0);
    });
  });
}
