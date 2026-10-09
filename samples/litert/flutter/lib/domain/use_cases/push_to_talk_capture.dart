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

import '../../data/repositories/audio_repository.dart';
import '../../utils/result.dart';

/// How a press's mic start ended ([PushToTalkCapture.open]).
sealed class const CaptureStart();

/// The mic captures. [released]: the press was released while the mic
/// opened (and not pressed again); the release cancels the capture, so it
/// never shows as listening.
final class const CaptureOpened({required final bool released})
    extends CaptureStart;

/// A newer action (Stop, a typed send, a submitted utterance, dispose)
/// dropped the press while the mic started; the capture it got was
/// cancelled.
final class const CaptureDropped() extends CaptureStart;

/// The mic did not start.
final class const CaptureFailed(final Exception error) extends CaptureStart;

/// What a release found ([PushToTalkCapture.release]).
sealed class const PressRelease();

/// Released with the mic open. [handOver] completes with the capture, for
/// the turn to close — or with null when a newer action took the press over
/// first.
final class const ReleasedListening(final Future<CaptureHandle?> handOver)
    extends PressRelease;

/// Released while the mic was still opening: nothing was recorded.
/// [cancelled] completes once the start has ended: true when its capture
/// was then cancelled (the user should wait for Listening), false when the
/// press went on without this release — pressed again, dropped, or the
/// start failed (which the press reports).
final class const ReleasedWhileOpening(final Future<bool> cancelled)
    extends PressRelease;

/// The push-to-talk mic, one press at a time, from the
/// press to the capture handed over at release:
///
/// - [open] starts the mic for a press; the start waits for the audio
///   warm-up, and words spoken before it are never recorded;
/// - a release while the mic still opens records nothing: once the start
///   ends, its capture is cancelled — unless the button was pressed again
///   meanwhile ([pressAgain]: the finger slipped and pressed again; that
///   press keeps the capture) or the press was dropped;
/// - [drop] (Stop, a typed send, a submitted utterance, dispose) ends the
///   press: an open capture is cancelled now, one still starting as soon as
///   its start ends, and a release waiting for either gets nothing.
///
/// The turn machine that owns it keeps the phases, the events and the
/// figures; this class only says how each step ended.
final class PushToTalkCapture {
  PushToTalkCapture({required this._audio, required this._maxLength});

  final AudioRepository _audio;

  /// The STT window: the longest capture kept.
  final Duration _maxLength;

  _Press? _current;

  /// A press is in progress: its capture is starting or open.
  bool get pressed => _current != null;

  /// A press while one is in progress: returns true. After a release while
  /// the mic still starts it takes that start over, and the release waiting
  /// for it gets nothing; any other is a no-op. Returns false when no press
  /// is in progress: the caller [open]s one.
  bool pressAgain() {
    final press = _current;
    if (press == null) return false;
    if (press.released && press.handle == null) {
      press
        ..released = false
        ..presses += 1;
    }
    return true;
  }

  /// Starts the mic for a new press. [onLimit]: the STT window ended the
  /// press (called only while this press is the current one).
  Future<CaptureStart> open({required void Function() onLimit}) {
    assert(_current == null, 'a press is in progress: pressAgain first');
    final press = _current = _Press();
    return _start(press, onLimit);
  }

  Future<CaptureStart> _start(_Press press, void Function() onLimit) async {
    try {
      final started = await _audio.startCapture(
        maxLength: _maxLength,
        onLimit: () {
          if (identical(_current, press)) onLimit();
        },
      );
      if (!identical(_current, press)) {
        // Stop, a typed send or dispose took over while the mic started.
        if (started case Ok(:final value)) await value.cancel();
        return const CaptureDropped();
      }
      switch (started) {
        case Ok(:final value):
          press.handle = value;
          return CaptureOpened(released: press.released);
        case Error(:final error):
          _current = null;
          return CaptureFailed(error);
      }
    } finally {
      // After the caller of [open] has seen the result: a release waiting
      // for the start resumes only then.
      press.ready.complete();
    }
  }

  /// The release of the press in progress; null when there is none (or the
  /// STT window already ended it).
  PressRelease? release() {
    final press = _current;
    if (press == null) return null;
    press.released = true;
    if (press.handle == null) {
      return ReleasedWhileOpening(_cancelOnceStarted(press));
    }
    // One turn of the event loop first: an action right after the release
    // (Stop, a typed send) still takes the press over.
    return ReleasedListening(press.ready.future.then((_) => _handOver(press)));
  }

  CaptureHandle? _handOver(_Press press) {
    if (!identical(_current, press)) return null;
    _current = null;
    return press.handle;
  }

  Future<bool> _cancelOnceStarted(_Press press) async {
    final presses = press.presses;
    await press.ready.future;
    final handle = press.handle;
    if (!identical(_current, press) ||
        press.presses != presses ||
        handle == null) {
      // Pressed again, dropped, or a failed start.
      return false;
    }
    _current = null;
    await handle.cancel();
    return true;
  }

  /// Ends the press in progress without a turn. Completes when its open
  /// capture is cancelled; at once when the capture is still starting
  /// (cancelled when the start ends) or no press is in progress.
  Future<void> drop() {
    final press = _current;
    _current = null;
    return press?.handle?.cancel() ?? Future<void>.value();
  }
}

/// One press in progress.
final class _Press {
  /// Completes when the mic start has ended, however it ended.
  final Completer<void> ready = Completer();

  /// Set once the mic has started; null while starting or after a failure.
  CaptureHandle? handle;

  /// The button was released (or the STT window ended the press), and not
  /// pressed again since.
  bool released = false;

  /// Presses this capture serves: a press while it is still starting after
  /// a release takes it over, which a release waiting for it detects.
  int presses = 1;
}
