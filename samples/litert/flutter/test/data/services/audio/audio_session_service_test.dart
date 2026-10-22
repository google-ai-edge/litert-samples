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

import 'package:audio_session/audio_session.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/data/services/audio/audio_session_service.dart';
import 'package:litert_edge_demos/domain/models/audio_interruption.dart';
import 'package:litert_edge_demos/domain/models/hardware_profile.dart';

/// `AudioSession` as audio_session 0.2.4 behaves where the service touches
/// it: [setActive] answers [grant] for true; the test raises the events.
class _FakeSession implements AudioSession {
  final List<String> calls = [];
  bool grant = true;
  AudioSessionConfiguration? configured;
  final StreamController<AudioInterruptionEvent> interruptionEvents =
      StreamController.broadcast();
  final StreamController<void> noisyEvents = StreamController.broadcast();

  @override
  Future<void> configure(AudioSessionConfiguration configuration) async {
    calls.add('configure');
    configured = configuration;
  }

  @override
  Future<bool> setActive(
    bool active, {
    AVAudioSessionSetActiveOptions? avAudioSessionSetActiveOptions,
    AndroidAudioFocusGainType? androidAudioFocusGainType,
    AndroidAudioAttributes? androidAudioAttributes,
    bool? androidWillPauseWhenDucked,
    AudioSessionConfiguration fallbackConfiguration =
        const AudioSessionConfiguration.music(),
  }) async {
    calls.add('setActive($active)');
    return active ? grant : true;
  }

  @override
  Stream<AudioInterruptionEvent> get interruptionEventStream =>
      interruptionEvents.stream;

  @override
  Stream<void> get becomingNoisyEventStream => noisyEvents.stream;

  Future<void> close() async {
    await interruptionEvents.close();
    await noisyEvents.close();
  }

  @override
  Object? noSuchMethod(Invocation invocation) =>
      throw UnimplementedError('${invocation.memberName}');
}

void main() {
  test('Linux gets the explicit no-op session, which says so', () async {
    expect(
      audioSessionServiceFor(HostPlatform.linux),
      isA<NoAudioSessionService>(),
    );
    final lines = <String>[];
    final none = NoAudioSessionService(log: lines.add);
    await none.configureHalfDuplex();
    expect(lines, ['[AudioSession] none on Linux (PulseAudio/PipeWire)']);
    expect(await none.activate(), isTrue);
    await none.deactivate();
    expect(await none.interruptions.isEmpty, isTrue);
  });

  test('Windows gets the no-op session too: audio_session has no Windows '
      'plugin, so its first call would throw MissingPluginException', () async {
    final service = audioSessionServiceFor(HostPlatform.windows);
    expect(service, isA<NoAudioSessionService>());
    expect(await service.activate(), isTrue);
    await service.deactivate();
    expect(await service.interruptions.isEmpty, isTrue);

    final lines = <String>[];
    await NoAudioSessionService.windows(log: lines.add).configureHalfDuplex();
    expect(lines, [
      '[AudioSession] none on Windows (it has no app audio session)',
    ]);
  });

  test('the other platforms keep audio_session', () {
    for (final platform in [
      HostPlatform.macos,
      HostPlatform.ios,
      HostPlatform.android,
    ]) {
      expect(
        audioSessionServiceFor(platform),
        isA<PlatformAudioSessionService>(),
        reason: platform.name,
      );
    }
  });

  group('PlatformAudioSessionService', () {
    late _FakeSession session;
    late PlatformAudioSessionService service;
    late List<String> log;

    setUp(() {
      session = _FakeSession();
      log = [];
      service = PlatformAudioSessionService(
        session: () async => session,
        log: log.add,
      );
    });

    tearDown(() => session.close());

    test('configure sets half-duplex voice and activates nothing (Android: '
        'no focus taken at a demo\'s warm-up)', () async {
      await service.configureHalfDuplex();
      expect(session.calls, ['configure']);
      final config = session.configured!;
      expect(
        config.avAudioSessionCategory,
        AVAudioSessionCategory.playAndRecord,
      );
      expect(config.avAudioSessionMode, AVAudioSessionMode.defaultMode);
      expect(
        config.androidAudioFocusGainType,
        AndroidAudioFocusGainType.gainTransient,
      );
      expect(config.androidAudioAttributes?.usage, AndroidAudioUsage.assistant);
    });

    test(
      'activate and deactivate set the session active and inactive',
      () async {
        expect(await service.activate(), isTrue);
        await service.deactivate();
        expect(session.calls, ['setActive(true)', 'setActive(false)']);
      },
    );

    test(
      'a refused activation is dropped at once: audio_session would keep '
      'the request and answer the next activate true without asking',
      () async {
        session.grant = false;
        expect(await service.activate(), isFalse);
        expect(session.calls, ['setActive(true)', 'setActive(false)']);
        expect(log.last, contains('refused'));
      },
    );

    test('interruptions: a begin that pauses or ends playback is "taken"; a '
        'duck and the end of an interruption are not reported; becoming '
        'noisy is "output removed"', () async {
      final heard = <AudioInterruption>[];
      final sub = service.interruptions.listen(heard.add);
      await pumpEventQueue();

      session.interruptionEvents
        ..add(AudioInterruptionEvent(true, AudioInterruptionType.pause))
        ..add(AudioInterruptionEvent(true, AudioInterruptionType.unknown))
        ..add(AudioInterruptionEvent(true, AudioInterruptionType.duck))
        ..add(AudioInterruptionEvent(false, AudioInterruptionType.pause));
      await pumpEventQueue();
      session.noisyEvents.add(null);
      await pumpEventQueue();

      expect(heard, [
        AudioInterruption.taken,
        AudioInterruption.taken,
        AudioInterruption.outputRemoved,
      ]);
      expect(log.where((l) => l.contains('nothing to do')), hasLength(2));

      await sub.cancel();
      expect(session.interruptionEvents.hasListener, isFalse);
      expect(session.noisyEvents.hasListener, isFalse);
    });

    test('a listener that cancels before the session is there subscribes to '
        'nothing', () async {
      final instance = Completer<AudioSession>();
      service = PlatformAudioSessionService(
        session: () => instance.future,
        log: log.add,
      );
      final sub = service.interruptions.listen((_) {});
      await sub.cancel();
      instance.complete(session);
      await pumpEventQueue();
      expect(session.interruptionEvents.hasListener, isFalse);
    });
  });

  test('interruptionFrom', () {
    expect(
      interruptionFrom(
        AudioInterruptionEvent(true, AudioInterruptionType.pause),
      ),
      AudioInterruption.taken,
    );
    expect(
      interruptionFrom(
        AudioInterruptionEvent(true, AudioInterruptionType.unknown),
      ),
      AudioInterruption.taken,
    );
    expect(
      interruptionFrom(
        AudioInterruptionEvent(true, AudioInterruptionType.duck),
      ),
      isNull,
    );
    expect(
      interruptionFrom(
        AudioInterruptionEvent(false, AudioInterruptionType.unknown),
      ),
      isNull,
    );
  });
}
