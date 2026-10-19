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
import 'package:flutter/foundation.dart';

import '../../../domain/models/audio_interruption.dart';
import '../../../domain/models/hardware_profile.dart' show HostPlatform;

/// The one owner of the platform audio session: its configuration, its
/// activation (iOS) or audio focus (Android), and its interruptions. `record`
/// is told not to manage it (on Android: not to take the focus either). The
/// one other caller is soloud on iOS: it never configures the session, but
/// re-activates it (`setActive:YES`) whenever it starts its stopped output
/// device again, at the end of an interruption or at the next play after one
/// (see `SoloudPcmPlayerService.restart`). Other layers learn about
/// interruptions from the audio repository, never from `AudioSession` itself.
abstract interface class AudioSessionService {
  /// Sets up half-duplex voice; activates nothing. Before soloud starts and
  /// before the first capture; again (harmless) after an interruption.
  Future<void> configureHalfDuplex();

  /// Activates the session: iOS makes it active, Android requests transient
  /// audio focus (other apps' audio pauses until [deactivate]). False when the
  /// OS refuses (Android: a call holds the audio). Throws when the platform
  /// fails (iOS: `setActive`'s error, e.g. during a call).
  Future<bool> activate();

  /// Gives the audio back: Android abandons the focus, so other apps' audio
  /// resumes and no interruption arrives until the next [activate]; iOS
  /// deactivates the session.
  Future<void> deactivate();

  /// The interruptions as they arrive (broadcast; only iOS and Android have
  /// any). Ducks and the ends of interruptions are not reported: nothing
  /// resumes by itself, the user presses again.
  Stream<AudioInterruption> get interruptions;
}

/// The session owner for [platform]: none on Linux (see
/// [NoAudioSessionService]), `audio_session` elsewhere.
AudioSessionService audioSessionServiceFor(HostPlatform platform) =>
    platform == HostPlatform.linux
    ? const NoAudioSessionService()
    : PlatformAudioSessionService();

/// Linux has no audio session: PulseAudio/PipeWire route per stream. Said
/// explicitly, because `audio_session` 0.2.4 swallows its missing Linux plugin
/// (`AudioSession.instance`, `AudioSession.configure`) and
/// `AudioSession.setActive` answers true, which would log a session that does
/// not exist.
final class NoAudioSessionService implements AudioSessionService {
  const NoAudioSessionService({this._log});

  final void Function(String line)? _log;

  @override
  Future<void> configureHalfDuplex() async => (_log ?? debugPrint)(
    '[AudioSession] none on Linux (PulseAudio/PipeWire)',
  );

  @override
  Future<bool> activate() async => true;

  @override
  Future<void> deactivate() async {}

  @override
  Stream<AudioInterruption> get interruptions => const Stream.empty();
}

/// [AudioSessionService] over `audio_session`. On macOS `configure` and
/// `setActive` are no-ops and nothing is interrupted; on iOS this is
/// `.playAndRecord` with the default mode (no voice processing: the mic is
/// closed while the reply plays); on Android speech for an assistant with
/// transient focus.
final class PlatformAudioSessionService implements AudioSessionService {
  /// [session]: `AudioSession.instance` (tests pass a fake).
  PlatformAudioSessionService({
    Future<AudioSession> Function()? session,
    void Function(String line)? log,
  }) : _instance = session ?? (() => AudioSession.instance),
       _log = log ?? debugPrint;

  final Future<AudioSession> Function() _instance;
  final void Function(String line) _log;

  @override
  Future<void> configureHalfDuplex() async {
    final session = await _instance();
    await session.configure(
      AudioSessionConfiguration(
        avAudioSessionCategory: AVAudioSessionCategory.playAndRecord,
        avAudioSessionCategoryOptions:
            AVAudioSessionCategoryOptions.defaultToSpeaker |
            AVAudioSessionCategoryOptions.allowBluetoothA2dp,
        avAudioSessionMode: AVAudioSessionMode.defaultMode,
        androidAudioAttributes: const AndroidAudioAttributes(
          contentType: AndroidAudioContentType.speech,
          usage: AndroidAudioUsage.assistant,
        ),
        androidAudioFocusGainType: AndroidAudioFocusGainType.gainTransient,
      ),
    );
    _log('[AudioSession] half-duplex configured');
  }

  @override
  Future<bool> activate() async {
    final session = await _instance();
    final active = await session.setActive(true);
    if (!active) {
      // audio_session 0.2.4 keeps a refused focus request
      // (AndroidAudioManager.kt requestAudioFocus) and would answer true to
      // the next activate without asking: drop it.
      await session.setActive(false);
    }
    _log('[AudioSession] activate: ${active ? 'active' : 'refused'}');
    return active;
  }

  @override
  Future<void> deactivate() async {
    final session = await _instance();
    final done = await session.setActive(false);
    _log('[AudioSession] deactivated${done ? '' : ' (the platform said no)'}');
  }

  /// audio_session's interruption and becoming-noisy streams as one, each
  /// listener subscribed to both until it cancels.
  @override
  late final Stream<AudioInterruption> interruptions =
      Stream<AudioInterruption>.multi((listener) {
        var cancelled = false;
        final subs = <StreamSubscription<Object?>>[];
        listener.onCancel = () async {
          cancelled = true;
          for (final sub in subs) {
            await sub.cancel();
          }
        };
        unawaited(
          _instance().then(
            (session) {
              if (cancelled) return;
              subs
                ..add(
                  session.interruptionEventStream.listen((event) {
                    final what = interruptionFrom(event);
                    _log(
                      '[AudioSession] interruption '
                      '${event.begin ? 'began' : 'ended'} (${event.type.name})'
                      '${what == null ? ': nothing to do' : ''}',
                    );
                    if (what != null) listener.add(what);
                  }),
                )
                ..add(
                  session.becomingNoisyEventStream.listen((_) {
                    _log(
                      '[AudioSession] the output went away (becoming noisy)',
                    );
                    listener.add(AudioInterruption.outputRemoved);
                  }),
                );
            },
            onError: (Object e, StackTrace st) {
              _log('[AudioSession] interruptions unavailable: $e\n$st');
            },
          ),
        );
      }, isBroadcast: true);
}

/// What an `audio_session` interruption means for the voice demos; null when
/// there is nothing to do: an interruption that ended (nothing resumes by
/// itself, the user presses again) or a duck (Android: a notification sound
/// or a navigation prompt plays over the app for a moment; the reply goes
/// on).
AudioInterruption? interruptionFrom(AudioInterruptionEvent event) =>
    switch (event) {
      AudioInterruptionEvent(begin: false) ||
      AudioInterruptionEvent(type: AudioInterruptionType.duck) => null,
      _ => AudioInterruption.taken,
    };
