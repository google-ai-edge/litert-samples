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

// DeviceAudioRepository and the platform session: Android takes the audio
// focus per turn and gives it back, iOS keeps the session active between
// turns, and an interruption cancels the capture, stops the playback and
// makes the next turn prepare again.
import 'dart:async';

import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/repositories/audio_repository.dart';
import 'package:litert_edge_demos/data/repositories/audio_repository_device.dart';
import 'package:litert_edge_demos/domain/models/audio_interruption.dart';
import 'package:litert_edge_demos/domain/models/hardware_profile.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../../fakes/fake_audio_devices.dart';
import '../../support/pcm.dart';

void main() {
  late FakeAudioSession session;
  late FakeMicService mic;
  late FakePcmPlayer player;
  late DeviceAudioRepository audio;
  late List<AudioInterruption> heard;

  void build(HostPlatform platform) {
    audio = DeviceAudioRepository(
      session: session,
      mic: mic,
      player: player,
      deviceService: FakeAudioDeviceService(),
      platform: platform,
    );
    heard = [];
    audio.interruptions.listen(heard.add);
  }

  setUp(() {
    session = FakeAudioSession();
    mic = FakeMicService();
    player = FakePcmPlayer();
  });

  tearDown(() => audio.close());

  Future<CaptureHandle> capture() async => switch (await audio.startCapture(
    maxLength: const Duration(seconds: 30),
    onLimit: () {},
  )) {
    Ok(:final value) => value,
    Error(:final error) => throw StateError('capture failed: $error'),
  };

  PlaybackHandle play() => switch (audio.beginPlayback(24000)) {
    Ok(:final value) => value,
    Error(:final error) => throw StateError('playback failed: $error'),
  };

  group('Android: the audio focus per turn', () {
    setUp(() => build(HostPlatform.android));

    test('the warm-up takes no focus (the user\'s music plays on while a '
        'demo is open); a press takes it, a second press keeps it, the end '
        'of the turn gives it back', () async {
      expect(await audio.prepare(), isA<Ok<void>>());
      expect(session.calls, ['configure']);

      final first = await capture();
      expect(session.calls, ['configure', 'activate']);
      await first.stop();
      await capture();
      expect(session.activations, 1, reason: 'held until the turn ends');

      await audio.deactivate();
      expect(session.calls.last, 'deactivate');
      expect(mic.starts, 2);
    });

    test('a spoken reply takes the focus for its turn and gives it back '
        'after it drained; the next turn takes it again', () async {
      expect(await audio.activate(), isA<Ok<void>>());
      final playback = play()..enqueue(tone(const Duration(milliseconds: 50)));
      playback.end();
      player.outputs.single.finishPlaying();
      await playback.drained;
      expect(session.deactivations, 0, reason: 'the turn decides, not drain');

      await audio.deactivate();
      expect(session.calls, ['configure', 'activate', 'deactivate']);

      expect(await audio.activate(), isA<Ok<void>>());
      expect(session.calls.last, 'activate');
    });

    test('a deactivate called after an activate that still prepares wins: '
        'no focus is left behind', () async {
      player.initGate = Completer<void>();
      final activating = audio.activate();
      final deactivating = audio.deactivate();
      player.initGate!.complete();
      expect(await activating, isA<Ok<void>>());
      await deactivating;
      expect(session.activations, 0);
    });

    test(
      'an activate right after a deactivate ends with the focus held',
      () async {
        await audio.activate();
        final deactivating = audio.deactivate();
        final activating = audio.activate();
        await deactivating;
        expect(await activating, isA<Ok<void>>());
        expect(session.calls.where((c) => c != 'configure').last, 'activate');
      },
    );

    test('a refused focus (a call) fails the press with the focus message, '
        'and the mic never starts', () async {
      session.activateResult = false;
      final started = await audio.startCapture(
        maxLength: const Duration(seconds: 30),
        onLimit: () {},
      );
      expect(
        started,
        isA<Error<CaptureHandle>>().having(
          (e) => e.error,
          'error',
          isA<AudioFocusException>().having(
            (e) => e.message,
            'message',
            kAudioFocusMessage,
          ),
        ),
      );
      expect(mic.starts, 0);

      // The call ended: the next press asks again.
      session.activateResult = true;
      await capture();
      expect(session.activations, 2);
    });

    test(
      'an interruption (a call) cancels the capture, gives the focus back '
      'at once, then tells the listeners; the next press asks again',
      () async {
        final held = await capture();
        mic.deliver(tone(const Duration(milliseconds: 100)));
        await pumpEventQueue();

        session.interrupt(AudioInterruption.taken);
        expect(heard, [AudioInterruption.taken]);
        await pumpEventQueue();

        expect(mic.stops, 1, reason: 'the capture was cancelled');
        expect(
          await held.stop(),
          isA<Error<Object?>>().having(
            (e) => e.error,
            'error',
            isA<AudioSupersededException>(),
          ),
        );
        expect(session.calls.last, 'deactivate');
        expect(player.restarts, 0, reason: 'only iOS restarts the engine');

        await capture();
        expect(session.calls.last, 'activate');
        expect(session.configureCalls, 1, reason: 'still prepared');
      },
    );

    test('close gives a held focus back', () async {
      await capture();
      await audio.close();
      expect(session.calls.last, 'deactivate');
    });
  });

  group('iOS: the session stays active between turns', () {
    setUp(() => build(HostPlatform.ios));

    test('prepare configures and activates once; presses, replies and the '
        'ends of turns never deactivate it', () async {
      expect(await audio.prepare(), isA<Ok<void>>());
      expect(session.calls, ['configure', 'activate']);

      final first = await capture();
      await first.stop();
      expect(await audio.activate(), isA<Ok<void>>());
      play();
      await audio.deactivate();
      await capture();
      await audio.deactivate();

      expect(session.calls, ['configure', 'activate']);
      expect(player.inits, 1);
    });

    test('an interruption during playback: the reply stops, the listeners '
        'hear it after the stop, playback fails until the next prepare, '
        'which activates again; the engine is kept (soloud starts its '
        'stopped device again by itself at the next play)', () async {
      await audio.prepare();
      final playback = play()..enqueue(tone(const Duration(milliseconds: 50)));
      final output = player.outputs.single;
      var stoppedFirst = false;
      audio.interruptions.listen((_) => stoppedFirst = output.stopped);

      session.interrupt(AudioInterruption.taken);

      expect(heard, [AudioInterruption.taken]);
      expect(stoppedFirst, isTrue, reason: 'silenced before anyone hears');
      await playback.drained.timeout(const Duration(seconds: 1));
      expect(
        audio.beginPlayback(24000),
        isA<Error<PlaybackHandle>>().having(
          (e) => e.error.toString(),
          'error',
          contains('interrupted'),
        ),
      );

      expect(await audio.activate(), isA<Ok<void>>());
      expect(session.calls, ['configure', 'activate', 'configure', 'activate']);
      expect(player.restarts, 0);
      expect(player.inits, 2, reason: 'init again: a no-op on a running one');
      play();
    });

    test('an interruption during a capture cancels it; the next press '
        'prepares again', () async {
      await capture();
      session.interrupt(AudioInterruption.taken);
      await pumpEventQueue();
      expect(mic.stops, 1);

      await capture();
      expect(session.activations, 2);
      expect(player.restarts, 0);
    });

    test('an activation that fails (a call holds the session) fails prepare '
        'with the focus message and the platform\'s reason', () async {
      session.activateError = Exception(
        'AVAudioSessionErrorCodeCannotInterruptOthers',
      );
      final prepared = await audio.prepare();
      expect(
        prepared,
        isA<Error<void>>().having(
          (e) => e.error,
          'error',
          isA<AudioFocusException>().having(
            (e) => e.message,
            'message',
            allOf(startsWith(kAudioFocusMessage), contains('CannotInterrupt')),
          ),
        ),
      );
      expect(player.inits, 0, reason: 'the engine waits for the session');

      session.activateError = null;
      expect(await audio.prepare(), isA<Ok<void>>());
    });

    test('an interruption while it prepares: not ready, the interrupted '
        'message; the next prepare activates again', () async {
      player.initGate = Completer<void>();
      final preparing = audio.prepare();
      await pumpEventQueue();
      session.interrupt(AudioInterruption.taken);
      player.initGate!.complete();

      expect(
        await preparing,
        isA<Error<void>>().having(
          (e) => e.error,
          'error',
          isA<AudioFocusException>().having(
            (e) => e.message,
            'message',
            kAudioInterruptedMessage,
          ),
        ),
      );
      expect(await audio.prepare(), isA<Ok<void>>());
      expect(session.activations, 2);
      expect(player.restarts, 0);
    });

    test('the app hidden: the next prepare activates again and restarts the '
        'engine (an interruption in the background may never be reported: a '
        'precaution)', () async {
      await audio.prepare();
      audio.appHidden();
      expect(heard, isEmpty, reason: 'not an interruption');

      await audio.prepare();
      expect(session.activations, 2);
      expect(player.restarts, 1);
    });

    test(
      'the app hidden stops a playback still running (the self-test has '
      'no turn to stop), so the restart never tears down a live one',
      () async {
        await audio.prepare();
        final playback = play()
          ..enqueue(tone(const Duration(milliseconds: 50)));

        audio.appHidden();

        expect(player.outputs.single.stopped, isTrue);
        await playback.drained.timeout(const Duration(seconds: 1));
        await audio.prepare();
        expect(player.restarts, 1);
      },
    );

    test('hidden again while the prepare after a hide activates the session: '
        'not ready, and the next prepare restarts the engine again', () async {
      await audio.prepare();
      audio.appHidden();
      session.activateGate = Completer<void>();
      final preparing = audio.prepare();
      await pumpEventQueue();
      audio.appHidden();
      session.activateGate!.complete();
      session.activateGate = null;

      expect(
        await preparing,
        isA<Error<void>>().having(
          (e) => e.error,
          'error',
          isA<AudioFocusException>().having(
            (e) => e.message,
            'message',
            kAudioInterruptedMessage,
          ),
        ),
      );
      expect(player.restarts, 1);

      expect(await audio.prepare(), isA<Ok<void>>());
      expect(player.restarts, 2);
    });

    test('the output unplugged stops the reply and is heard, but the '
        'session stays prepared', () async {
      await audio.prepare();
      final playback = play()..enqueue(tone(const Duration(milliseconds: 50)));

      session.interrupt(AudioInterruption.outputRemoved);

      expect(heard, [AudioInterruption.outputRemoved]);
      expect(player.outputs.single.stopped, isTrue);
      await playback.drained.timeout(const Duration(seconds: 1));
      play();
      expect(session.activations, 1);
    });
  });

  test('Android: the app hidden changes nothing', () async {
    build(HostPlatform.android);
    await audio.prepare();
    audio.appHidden();
    await audio.prepare();
    expect(session.configureCalls, 1);
    expect(player.restarts, 0);
  });

  test('nothing listens to the session before the first prepare, and close '
      'stops listening', () async {
    build(HostPlatform.ios);
    expect(session.hasListener, isFalse);
    await audio.prepare();
    expect(session.hasListener, isTrue);
    await audio.close();
    expect(session.hasListener, isFalse);
  });

  test(
    'macOS activates at prepare (a no-op there) and never per turn',
    () async {
      build(HostPlatform.macos);
      await capture();
      await audio.deactivate();
      expect(session.calls, ['configure', 'activate']);
    },
  );
}
