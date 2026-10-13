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
import 'dart:io' show ProcessException;
import 'dart:typed_data';

import 'package:fake_async/fake_async.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/voice_config.dart';
import 'package:litert_edge_demos/data/repositories/audio_repository.dart';
import 'package:litert_edge_demos/data/repositories/audio_repository_device.dart';
import 'package:litert_edge_demos/domain/audio/audio_device_checks.dart';
import 'package:litert_edge_demos/domain/models/audio_devices.dart';
import 'package:litert_edge_demos/domain/models/hardware_profile.dart';
import 'package:litert_edge_demos/domain/models/voice.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../fakes/fake_audio_devices.dart';
import '../../support/pcm.dart';

void main() {
  late FakeAudioSession session;
  late FakeMicService mic;
  late FakePcmPlayer player;
  late FakeAudioDeviceService devices;
  late DeviceAudioRepository audio;

  DeviceAudioRepository build({
    Duration streamEndTimeout = const Duration(seconds: 1),
    HostPlatform platform = HostPlatform.macos,
    Duration emptyCaptureCheckAfter = const Duration(milliseconds: 500),
  }) => DeviceAudioRepository(
    session: session,
    mic: mic,
    player: player,
    deviceService: devices,
    platform: platform,
    streamEndTimeout: streamEndTimeout,
    emptyCaptureCheckAfter: emptyCaptureCheckAfter,
  );

  setUp(() {
    session = FakeAudioSession();
    mic = FakeMicService();
    player = FakePcmPlayer();
    devices = FakeAudioDeviceService();
    audio = build();
  });

  tearDown(() => audio.close());

  CaptureHandle started(Result<CaptureHandle> result) => switch (result) {
    Ok(:final value) => value,
    Error(:final error) => throw StateError('capture failed: $error'),
  };

  Utterance utterance(Result<Utterance> result) => switch (result) {
    Ok(:final value) => value,
    Error(:final error) => throw StateError('stop failed: $error'),
  };

  PlaybackHandle playback(Result<PlaybackHandle> result) => switch (result) {
    Ok(:final value) => value,
    Error(:final error) => throw StateError('playback failed: $error'),
  };

  group('capture', () {
    test('prepares once, accumulates the chunks, meters each one, and stop '
        'waits for the stream to close', () async {
      final capture = started(
        await audio.startCapture(
          maxLength: const Duration(seconds: 30),
          onLimit: () => fail('no limit'),
        ),
      );
      expect(session.configureCalls, 1);
      expect(player.inits, 1);

      mic
        ..deliver(tone(const Duration(milliseconds: 100)))
        ..deliver(tone(const Duration(milliseconds: 100)));
      await pumpEventQueue();
      expect(audio.inputLevel.value, greaterThan(0.5));

      final result = utterance(await capture.stop());
      expect(result.pcm.length, 2 * 3200);
      expect(result.held, greaterThan(Duration.zero));
      expect(mic.stops, 1);
      expect(audio.inputLevel.value, 0);

      // A second capture reuses the prepared session.
      started(
        await audio.startCapture(
          maxLength: const Duration(seconds: 30),
          onLimit: () {},
        ),
      );
      expect(session.configureCalls, 1);
    });

    test('the chunk the mic flushes after stop is kept (the '
        'last word)', () async {
      mic.tailOnStop = tone(const Duration(milliseconds: 100));
      final capture = started(
        await audio.startCapture(
          maxLength: const Duration(seconds: 30),
          onLimit: () {},
        ),
      );
      mic.deliver(tone(const Duration(milliseconds: 100)));
      await pumpEventQueue();

      final result = utterance(await capture.stop());
      expect(result.pcm.length, 2 * 3200);
    });

    test('the hold counts from the mic start, not from the '
        'permission dialog', () async {
      mic.permissionGate = Completer<void>();
      final starting = audio.startCapture(
        maxLength: const Duration(seconds: 30),
        onLimit: () {},
      );
      await Future<void>.delayed(const Duration(milliseconds: 150));
      mic.permissionGate!.complete();
      final capture = started(await starting);

      final result = utterance(await capture.stop());
      expect(result.held, lessThan(const Duration(milliseconds: 100)));
    });

    test('the hold counts from the mic start, not from a slow audio '
        'warm-up', () async {
      player.initGate = Completer<void>();
      final starting = audio.startCapture(
        maxLength: const Duration(seconds: 30),
        onLimit: () {},
      );
      await Future<void>.delayed(const Duration(milliseconds: 150));
      expect(mic.starts, 0, reason: 'the capture waits for the warm-up');
      player.initGate!.complete();
      final capture = started(await starting);

      final result = utterance(await capture.stop());
      expect(result.held, lessThan(const Duration(milliseconds: 100)));
    });

    test('the STT window counts from the mic start, not from a slow audio '
        'warm-up', () {
      fakeAsync((async) {
        // Its own mic: the fake runs its calls in order, and a call queued in
        // this zone never ends outside it.
        mic = FakeMicService();
        player.initGate = Completer<void>();
        final local = build();
        var limits = 0;
        Result<CaptureHandle>? result;
        unawaited(
          local
              .startCapture(
                maxLength: const Duration(seconds: 5),
                onLimit: () => limits++,
              )
              .then((r) => result = r),
        );
        async.elapse(const Duration(milliseconds: 2600)); // a cold start
        expect(result, isNull, reason: 'still warming up');
        player.initGate!.complete();
        async.flushMicrotasks();
        final capture = started(result!);

        // 7.599 s after the request, 4.999 s after the mic started.
        async.elapse(const Duration(milliseconds: 4999));
        expect(limits, 0);
        async.elapse(const Duration(milliseconds: 1));
        expect(limits, 1);
        unawaited(capture.cancel());
        unawaited(local.close());
        async.flushMicrotasks();
      });
    });

    test('keeps at most maxLength and calls onLimit once', () async {
      var limits = 0;
      final capture = started(
        await audio.startCapture(
          maxLength: const Duration(milliseconds: 100), // 3200 bytes
          onLimit: () => limits++,
        ),
      );
      mic
        ..deliver(Uint8List(2000))
        ..deliver(Uint8List(2000))
        ..deliver(Uint8List(2000));
      await pumpEventQueue();

      expect(limits, 1);
      expect(utterance(await capture.stop()).pcm.length, 3200);
    });

    test(
      'without permission: a mic-access error, the mic never starts',
      () async {
        mic.permission = false;
        final result = await audio.startCapture(
          maxLength: const Duration(seconds: 30),
          onLimit: () {},
        );

        expect(result, isA<Error<CaptureHandle>>());
        expect(
          (result as Error<CaptureHandle>).error,
          isA<MicAccessException>().having(
            (e) => e.message,
            'message',
            contains('Microphone access is off'),
          ),
        );
        expect(mic.starts, 0);
      },
    );

    test('a format change mid-capture fails the stop (the STT would get '
        'wrong audio)', () async {
      final capture = started(
        await audio.startCapture(
          maxLength: const Duration(seconds: 30),
          onLimit: () {},
        ),
      );
      mic.onFormatChanged!('pcm16bits 48000 Hz 1 ch');

      final result = await capture.stop();
      expect((result as Error<Utterance>).error, isA<MicFormatException>());
    });

    test(
      'a stream that never closes after stop does not hang the stop',
      () async {
        await audio.close();
        audio = build(streamEndTimeout: const Duration(milliseconds: 50));
        mic.closeOnStop = false;
        final capture = started(
          await audio.startCapture(
            maxLength: const Duration(seconds: 30),
            onLimit: () {},
          ),
        );
        mic.deliver(tone(const Duration(milliseconds: 100)));
        await pumpEventQueue();

        final result = utterance(await capture.stop());
        expect(result.pcm.length, 3200);
      },
    );

    test('a new capture cancels the old one; the old handle\'s late calls '
        'never touch the new capture', () async {
      final old = started(
        await audio.startCapture(
          maxLength: const Duration(seconds: 30),
          onLimit: () {},
        ),
      );
      final current = started(
        await audio.startCapture(
          maxLength: const Duration(seconds: 30),
          onLimit: () {},
        ),
      );
      expect(mic.stops, 1, reason: 'the old capture was closed');

      await old.cancel();
      expect(mic.stops, 1, reason: 'a late cancel is a no-op');
      expect(await old.stop(), isA<Error<Utterance>>());

      mic.deliver(tone(const Duration(milliseconds: 100)));
      await pumpEventQueue();
      expect(utterance(await current.stop()).pcm.length, 3200);
    });
  });

  group('playback', () {
    setUp(() async {
      expect(await audio.prepare(), isA<Ok<void>>());
    });

    test('drained completes when the engine reports the end', () async {
      final handle = playback(audio.beginPlayback(24000));
      handle.enqueue(Uint8List(4800));
      handle.end();
      var drained = false;
      unawaited(handle.drained.then((_) => drained = true));
      await pumpEventQueue();
      expect(drained, isFalse);
      expect(player.outputs.single.ended, isTrue);

      player.outputs.single.finishPlaying();
      await pumpEventQueue();
      expect(drained, isTrue);
    });

    test('stop silences the output and completes drained', () async {
      final handle = playback(audio.beginPlayback(24000));
      handle.enqueue(Uint8List(4800));

      final stopping = handle.stop();
      expect(
        player.outputs.single.stopped,
        isTrue,
        reason: 'stopped synchronously, before the future',
      );
      await stopping;
      await handle.drained;
      handle.enqueue(Uint8List(4800));
      expect(player.outputs.single.chunks, hasLength(1));
    });

    test('a new playback stops the previous one; the stale handle is '
        'ignored', () async {
      final old = playback(audio.beginPlayback(24000));
      old.enqueue(Uint8List(4800));
      final current = playback(audio.beginPlayback(24000));

      expect(player.outputs.first.stopped, isTrue);
      old.enqueue(Uint8List(4800));
      expect(player.outputs.first.chunks, hasLength(1));
      current.enqueue(Uint8List(4800));
      expect(player.outputs.last.chunks, hasLength(1));
    });

    test('end with nothing queued is drained at once', () async {
      final handle = playback(audio.beginPlayback(24000));
      handle.end();
      await handle.drained;
    });

    test('an engine that never reports the end is treated as drained after '
        'the queued audio plus the slack', () {
      fakeAsync((async) {
        // Its own mic: the fake runs its calls in order, and a call queued in
        // this zone never ends outside it.
        mic = FakeMicService();
        final local = build();
        unawaited(local.prepare());
        async.flushMicrotasks();
        final handle = playback(local.beginPlayback(24000));
        handle.enqueue(Uint8List(48000)); // 1 s at 24 kHz
        handle.end();
        var drained = false;
        unawaited(handle.drained.then((_) => drained = true));

        async.elapse(const Duration(milliseconds: 2900));
        expect(drained, isFalse);
        async.elapse(const Duration(milliseconds: 200)); // 1 s + 2 s slack
        expect(drained, isTrue);
        unawaited(local.close());
        async.flushMicrotasks();
      });
    });
  });

  test('a stall between chunks does not end the drain early, '
      'and a drain that does time out stops the output', () {
    fakeAsync((async) {
      final local = DeviceAudioRepository(
        session: session,
        mic: FakeMicService(), // as above: its calls stay in this zone
        player: player,
        deviceService: devices,
        clockMicros: () => async.elapsed.inMicroseconds,
      );
      unawaited(local.prepare());
      async.flushMicrotasks();
      final handle = playback(local.beginPlayback(24000));
      handle.enqueue(Uint8List(48000)); // t=0: 1 s of audio
      async.elapse(const Duration(seconds: 5)); // generation stalls
      handle.enqueue(Uint8List(144000)); // t=5: 3 s more, plays until ~8 s
      handle.end();
      var drained = false;
      unawaited(handle.drained.then((_) => drained = true));

      async.elapse(const Duration(milliseconds: 2500)); // t=7.5
      expect(drained, isFalse, reason: 'audio still plays until ~8 s');
      expect(player.outputs.single.stopped, isFalse);

      async.elapse(const Duration(milliseconds: 2600)); // t=10.1 (8 s + 2 s)
      expect(drained, isTrue);
      expect(
        player.outputs.single.stopped,
        isTrue,
        reason: 'a voice nobody can stop any more must not keep playing',
      );
      unawaited(local.close());
      async.flushMicrotasks();
    });
  });

  test('beginPlayback before prepare fails instead of playing nothing', () {
    expect(audio.beginPlayback(24000), isA<Error<PlaybackHandle>>());
  });

  test('a failed prepare is an error, and prepare again retries', () async {
    player.initError = StateError('no output device');
    final failed = await audio.prepare();
    expect(
      (failed as Error<void>).error.toString(),
      contains('no output device'),
    );

    player.initError = null;
    expect(await audio.prepare(), isA<Ok<void>>());
    expect(player.inits, 2);
  });

  group('requestMicAccess', () {
    test(
      'configures the session first, then asks the mic; granted is Ok',
      () async {
        expect(await audio.requestMicAccess(), isA<Ok<void>>());
        expect(session.configureCalls, 1);
      },
    );

    test(
      'denied: MicAccessException with the platform\'s settings path',
      () async {
        mic.permission = false;
        final result = await audio.requestMicAccess();
        final error = (result as Error<void>).error;
        expect(error, isA<MicAccessException>());
        expect('$error', kMicAccessMessage);
      },
    );
  });

  group('input devices and mic errors', () {
    const noServer = DeviceUnavailable(
      'No sound server: PulseAudio/PipeWire is not running (pactl info: '
      'Connection failure: Connection refused). Start it: …',
    );

    test('requestMicAccess publishes the input; an unusable one is a '
        'MicAccessException with what to do', () async {
      expect(await audio.requestMicAccess(), isA<Ok<void>>());
      expect(
        audio.devices.value.input,
        isA<DeviceReady>().having((d) => d.name, 'name', 'Fake Mic'),
      );

      devices.input = noServer;
      final result = await audio.requestMicAccess();
      final error = (result as Error<void>).error;
      expect(error, isA<MicAccessException>());
      expect('$error', noServer.message);
      expect(audio.devices.value.input, same(noServer));
    });

    test('a missing parecord at start: the install hint, published as the '
        'input state', () async {
      final linux = build(platform: HostPlatform.linux);
      addTearDown(linux.close);
      devices.system = const AudioSystem(
        server: SoundServer(name: 'PulseAudio (on PipeWire 1.0.5)'),
      );
      mic.startError = const ProcessException(
        'parecord',
        ['--raw'],
        'No such file or directory',
        2,
      );
      final result = await linux.startCapture(
        maxLength: const Duration(seconds: 5),
        onLimit: () {},
      );
      final error = (result as Error<CaptureHandle>).error;
      expect(error, isA<MicAccessException>());
      expect('$error', kParecordMissing);
      expect(
        linux.devices.value.input,
        isA<DeviceUnavailable>().having(
          (d) => d.message,
          'message',
          kParecordMissing,
        ),
      );
    });

    test(
      'after a failed check a press checks once more: still broken → '
      'the message, and the mic is never started; fixed → it starts',
      () async {
        devices.input = noServer;
        await audio.requestMicAccess();
        expect(devices.inputChecks, 1);

        final failed = await audio.startCapture(
          maxLength: const Duration(seconds: 5),
          onLimit: () {},
        );
        expect('${(failed as Error<CaptureHandle>).error}', noServer.message);
        expect(devices.inputChecks, 2, reason: 'one check per press');
        expect(mic.starts, 0);

        devices.input = const DeviceReady('USB Mic');
        final started = await audio.startCapture(
          maxLength: const Duration(seconds: 5),
          onLimit: () {},
        );
        expect(started, isA<Ok<CaptureHandle>>());
        expect(devices.inputChecks, 3);
        expect(mic.starts, 1);
        await (started as Ok<CaptureHandle>).value.cancel();

        final again = await audio.startCapture(
          maxLength: const Duration(seconds: 5),
          onLimit: () {},
        );
        expect(again, isA<Ok<CaptureHandle>>());
        expect(devices.inputChecks, 3, reason: 'no check while it works');
        await (again as Ok<CaptureHandle>).value.cancel();
      },
    );

    test('a stream that ends before stop with no audio (parecord exited): '
        'stop says why, from a fresh input check', () async {
      final capture = started(
        await audio.startCapture(
          maxLength: const Duration(seconds: 5),
          onLimit: () {},
        ),
      );
      devices.input = noServer;
      await mic.endStream();
      await pumpEventQueue();
      final result = await capture.stop();
      final error = (result as Error<Utterance>).error;
      expect(error, isA<MicAccessException>());
      expect('$error', noServer.message);
    });

    test(
      '… and when the device still looks fine, it says no audio arrived',
      () async {
        final capture = started(
          await audio.startCapture(
            maxLength: const Duration(seconds: 5),
            onLimit: () {},
          ),
        );
        await mic.endStream();
        await pumpEventQueue();
        final error = ((await capture.stop()) as Error<Utterance>).error;
        expect(
          '$error',
          allOf(
            startsWith('No audio arrived from the microphone in '),
            endsWith(' ms (Fake Mic · test); press again'),
          ),
        );
      },
    );

    test('Linux, as record 7.1.1 behaves: parecord exited but the app\'s '
        'stream stays open (no onDone is forwarded); a press that delivered '
        'nothing is checked, and the check says why', () async {
      final linux = build(
        platform: HostPlatform.linux,
        emptyCaptureCheckAfter: Duration.zero,
      );
      addTearDown(linux.close);
      devices.system = const AudioSystem(
        server: SoundServer(name: 'PulseAudio (on PipeWire 1.0.5)'),
      );
      final capture = started(
        await linux.startCapture(
          maxLength: const Duration(seconds: 5),
          onLimit: () {},
        ),
      );
      devices.input = noServer;
      final error = ((await capture.stop()) as Error<Utterance>).error;
      expect(error, isA<MicAccessException>());
      expect('$error', noServer.message);
      expect(linux.devices.value.input, same(noServer));
    });

    test('… below the threshold, or off Linux, an empty press stays an empty '
        'utterance (the assistant\'s gate reports it)', () async {
      final capture = started(
        await audio.startCapture(
          maxLength: const Duration(seconds: 5),
          onLimit: () {},
        ),
      );
      final result = await capture.stop();
      expect(result, isA<Ok<Utterance>>());
      expect((result as Ok<Utterance>).value.pcm, isEmpty);
      expect(devices.inputChecks, 0);
    });
  });

  group('output device check', () {
    test('prepare names the default output and publishes it', () async {
      expect(await audio.prepare(), isA<Ok<void>>());
      expect(
        audio.devices.value.output,
        isA<DeviceReady>().having((d) => d.name, 'name', 'Fake Speakers'),
      );
    });

    test('the Null device fails prepare (never plays in silence), shuts the '
        'engine down, and the next prepare starts it again', () async {
      final linux = build(platform: HostPlatform.linux);
      addTearDown(linux.close);
      devices.system = const AudioSystem(
        server: SoundServer(name: 'PulseAudio (on PipeWire 1.0.5)'),
      );
      player.playbackDevices = const [
        AudioDevice(id: '0', name: 'NULL Playback Device', isDefault: true),
      ];
      final failed = await linux.prepare();
      final error = (failed as Error<void>).error;
      expect(error, isA<PlaybackException>());
      expect(
        '$error',
        startsWith(
          'Playback failed: No audio output (no PulseAudio/PipeWire/ALSA '
          'device)',
        ),
      );
      expect(player.disposes, 1);
      expect(linux.devices.value.output, isA<DeviceUnavailable>());
      expect(
        linux.beginPlayback(24000),
        isA<Error<PlaybackHandle>>(),
        reason: 'no playback on a failed prepare',
      );

      player.playbackDevices = const [
        AudioDevice(id: '0', name: 'Built-in Audio', isDefault: true),
      ];
      expect(await linux.prepare(), isA<Ok<void>>());
      expect(player.inits, 2);
      expect(
        linux.devices.value.output,
        isA<DeviceReady>().having(
          (d) => d.detail,
          'detail',
          'PulseAudio (on PipeWire 1.0.5)',
        ),
      );
    });

    test(
      'iOS: the devices are never listed (it would reset the session)',
      () async {
        final ios = build(platform: HostPlatform.ios);
        addTearDown(ios.close);
        player.listError = StateError('must not be called on iOS');
        expect(await ios.prepare(), isA<Ok<void>>());
        expect(
          ios.devices.value.output,
          isA<DeviceReady>().having(
            (d) => d.name,
            'name',
            'system default output',
          ),
        );
      },
    );

    test('an engine that cannot list its devices is not trusted', () async {
      player.listError = StateError('no context');
      final failed = await audio.prepare();
      expect(
        '${(failed as Error<void>).error}',
        contains('The output device could not be checked'),
      );
    });
  });

  group('close racing a prepare or a capture start', () {
    /// The repository's closed error.
    Matcher isClosedError<T>() => isA<Error<T>>().having(
      (e) => '${e.error}',
      'error',
      contains('AudioRepository closed'),
    );

    /// Starts [future]'s completion tracking.
    ({bool done}) Function() tracked(Future<void> future) {
      var done = false;
      unawaited(future.then((_) => done = true));
      return () => (done: done);
    }

    test('during prepare: the engine is not disposed while it starts, the '
        'prepare fails as closed, nothing is ready afterwards', () async {
      player.initGate = Completer<void>();
      final preparing = audio.prepare();
      await pumpEventQueue();
      expect(player.calls, ['init']);

      final closed = tracked(audio.close());
      await pumpEventQueue();
      expect(closed().done, isFalse, reason: 'close waits for the start');
      expect(player.disposes, 0);

      player.initGate!.complete();
      expect(await preparing, isClosedError<void>());
      await pumpEventQueue();
      expect(closed().done, isTrue);
      expect(player.calls, ['init', 'init done', 'dispose']);
      expect(audio.beginPlayback(24000), isClosedError<PlaybackHandle>());
    });

    test('during a capture start\'s permission check: the closed error, the '
        'mic never starts', () async {
      mic.permissionGate = Completer<void>();
      final starting = audio.startCapture(
        maxLength: const Duration(seconds: 30),
        onLimit: () {},
      );
      await pumpEventQueue();

      final closing = audio.close();
      mic.permissionGate!.complete();

      expect(await starting, isClosedError<CaptureHandle>());
      await closing;
      expect(mic.starts, 0);
      expect(mic.usedAfterDispose, isEmpty);
    });

    test('during a capture start\'s prepare: the closed error, the mic never '
        'starts', () async {
      player.initGate = Completer<void>();
      final starting = audio.startCapture(
        maxLength: const Duration(seconds: 30),
        onLimit: () {},
      );
      await pumpEventQueue();

      final closing = audio.close();
      player.initGate!.complete();

      expect(await starting, isClosedError<CaptureHandle>());
      await closing;
      expect(mic.starts, 0);
      expect(player.calls, ['init', 'init done', 'dispose']);
    });

    test('while the mic starts: close waits, the start stops the mic it '
        'opened and fails as closed, then the mic is disposed', () async {
      mic.startGate = Completer<void>();
      final starting = audio.startCapture(
        maxLength: const Duration(seconds: 30),
        onLimit: () {},
      );
      await pumpEventQueue();
      expect(mic.starts, 1);

      final closed = tracked(audio.close());
      await pumpEventQueue();
      expect(closed().done, isFalse, reason: 'close waits for the start');
      expect(mic.disposes, 0);

      mic.startGate!.complete();
      expect(await starting, isClosedError<CaptureHandle>());
      await pumpEventQueue();
      expect(closed().done, isTrue);
      expect(mic.stops, 1, reason: 'the mic it opened is stopped');
      expect(mic.disposes, 1);
      expect(mic.usedAfterDispose, isEmpty);
    });

    /// [audio] replaced by one with a fresh mic and a 50 ms close bound.
    void boundedClose() {
      unawaited(audio.close());
      mic = FakeMicService();
      player = FakePcmPlayer();
      audio = DeviceAudioRepository(
        session: session,
        mic: mic,
        player: player,
        deviceService: devices,
        platform: HostPlatform.macos,
        closeWait: const Duration(milliseconds: 50),
      );
    }

    test('a start that never finishes does not hold close forever: the mic '
        'dispose queued behind the dialog is left to run', () async {
      boundedClose();
      mic.permissionGate = Completer<void>(); // a dialog left open
      final starting = audio.startCapture(
        maxLength: const Duration(seconds: 30),
        onLimit: () {},
      );
      await pumpEventQueue();

      await audio.close().timeout(const Duration(seconds: 2));
      expect(player.disposes, 1);
      expect(mic.disposes, 0, reason: 'record runs it after the dialog');

      mic.permissionGate!.complete();
      expect(await starting, isClosedError<CaptureHandle>());
      await pumpEventQueue();
      expect(mic.starts, 0);
      expect(mic.disposes, 1);
      expect(mic.usedAfterDispose, isEmpty);
    });

    test('requestMicAccess waiting on the permission dialog does not hold '
        'close forever either, and fails as closed once answered', () async {
      boundedClose();
      expect(await audio.prepare(), isA<Ok<void>>());
      mic.permissionGate = Completer<void>(); // a dialog left open
      final asking = audio.requestMicAccess();
      await pumpEventQueue();

      final watch = Stopwatch()..start();
      await audio.close().timeout(const Duration(seconds: 2));
      expect(watch.elapsed, lessThan(const Duration(seconds: 1)));
      expect(player.disposes, 1);

      mic.permissionGate!.complete();
      expect(await asking, isClosedError<void>());
      await pumpEventQueue();
      expect(mic.disposes, 1);
      expect(devices.inputChecks, 0, reason: 'nothing is checked after close');
    });

    test('a second close waits for the first to finish', () async {
      player.initGate = Completer<void>();
      final preparing = audio.prepare();
      await pumpEventQueue();

      final first = audio.close();
      var secondDone = false;
      final second = audio.close().then((_) => secondDone = true);
      await pumpEventQueue();
      expect(secondDone, isFalse, reason: 'the first still waits for prepare');

      player.initGate!.complete();
      await preparing;
      await second;
      expect(player.disposes, 1, reason: 'disposed before the second returns');
      expect(mic.disposes, 1);
      await first;
    });

    test('twice, also concurrently: one shutdown', () async {
      started(
        await audio.startCapture(
          maxLength: const Duration(seconds: 30),
          onLimit: () {},
        ),
      );

      await Future.wait([audio.close(), audio.close()]);
      await audio.close();

      expect(mic.disposes, 1);
      expect(player.disposes, 1);
      expect(mic.stops, 1, reason: 'the open capture was closed once');
    });

    test('afterwards every call fails as closed and nothing touches the mic, '
        'the engine or a disposed notifier', () async {
      final capture = started(
        await audio.startCapture(
          maxLength: const Duration(seconds: 30),
          onLimit: () {},
        ),
      );
      final reply = playback(audio.beginPlayback(24000));
      final playCalls = player.calls.length;

      await audio.close();
      expect(mic.stops, 1);
      expect(player.outputs.single.stopped, isTrue);

      expect(await audio.prepare(), isClosedError<void>());
      expect(await audio.requestMicAccess(), isClosedError<void>());
      expect(
        await audio.startCapture(
          maxLength: const Duration(seconds: 30),
          onLimit: () {},
        ),
        isClosedError<CaptureHandle>(),
      );
      expect(audio.beginPlayback(24000), isClosedError<PlaybackHandle>());
      // The old handles: no mic or engine call, no notifier touched.
      expect(await capture.stop(), isA<Error<Utterance>>());
      await capture.cancel();
      reply.enqueue(tone(const Duration(milliseconds: 100)));
      reply.end();
      await reply.stop();
      await pumpEventQueue();

      expect(mic.usedAfterDispose, isEmpty);
      expect(mic.stops, 1);
      expect(player.calls.length, playCalls + 1, reason: 'only the dispose');
      expect(player.outputs.single.chunks, isEmpty);
    });

    test('requestMicAccess with a close during its prepare: the closed '
        'error, no permission request on a disposed mic', () async {
      player.initGate = Completer<void>();
      final asking = audio.requestMicAccess();
      await pumpEventQueue();

      final closing = audio.close();
      player.initGate!.complete();

      expect(await asking, isClosedError<void>());
      await closing;
      expect(mic.usedAfterDispose, isEmpty);
    });
  });
}
