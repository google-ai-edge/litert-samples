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
import 'package:litert_edge_demos/data/repositories/audio_repository.dart';
import 'package:litert_edge_demos/domain/models/audio_devices.dart';
import 'package:litert_edge_demos/domain/models/voice.dart';
import 'package:litert_edge_demos/utils/result.dart';

import '../support/pcm.dart';

/// Records every audio call into [log] (shared with other fakes so tests can
/// check ordering). Captures return [nextUtterance]; playbacks drain when
/// the test calls [FakePlayback.completeDrain], or at `end()` with
/// [autoDrain].
///
/// Holds to the real contract (`DeviceAudioRepository`) where a caller can
/// get it wrong: [beginPlayback] fails until a [prepare] (or a capture,
/// which prepares) succeeded, and stops the playback still running; a
/// [startCapture] cancels a capture still open and supersedes one still
/// starting ([startGate]) with [AudioSupersededException]; a capture
/// stopped after a cancel or a supersede answers that exception, and a
/// cancel during its stop does nothing; [close] cancels the open capture
/// and stops the playback (completing its drain), after it every call
/// returns the closed error, and it is idempotent.
class FakeAudioRepository implements AudioRepository {
  FakeAudioRepository({List<String>? log}) : log = log ?? [];

  final List<String> log;
  final ValueNotifier<double> level = ValueNotifier(0);
  Utterance nextUtterance = speechUtterance();
  Result<Utterance>? stopResult;
  Exception? startError;
  Result<void> prepareResult = const Result.ok(null);
  bool autoDrain = false;
  int prepareCalls = 0;
  void Function()? lastOnLimit;
  final List<FakeCapture> captures = [];
  final List<FakePlayback> playbacks = [];

  /// When set, [startCapture] waits for it before it hands the capture out
  /// (a permission dialog, a slow mic start, the audio warm-up).
  Completer<void>? startGate;

  /// When set, a capture's `held` is measured with it, from the moment
  /// [startCapture] hands the capture out to its stop, as the real one
  /// counts the hold from the mic start (a slow start is not part of the
  /// press). Otherwise [nextUtterance]'s.
  Duration Function()? clock;

  bool _closed = false;
  bool _ready = false;

  /// The capture open or starting; a newer [startCapture] replaces it.
  FakeCapture? _current;

  /// The playback not yet drained or stopped; a newer [beginPlayback]
  /// stops it.
  FakePlayback? _playing;

  static Result<T> _closedError<T>() =>
      Result.error(asException(StateError('AudioRepository closed')));

  /// The real prepare's rules: closed is an error, once ready it stays
  /// ready, a failure is retried by the next call.
  Result<void> _prepare() {
    if (_closed) return _closedError();
    if (_ready) return const Result.ok(null);
    final result = prepareResult;
    if (result is Ok<void>) _ready = true;
    return result;
  }

  void _release(FakeCapture capture) {
    if (identical(_current, capture)) _current = null;
  }

  void _releasePlayback(FakePlayback playback) {
    if (identical(_playing, playback)) _playing = null;
  }

  FakePlayback? get lastPlayback => playbacks.isEmpty ? null : playbacks.last;

  @override
  ValueListenable<double> get inputLevel => level;

  final ValueNotifier<AudioDeviceStatus> deviceStatus = ValueNotifier(
    const AudioDeviceStatus(),
  );

  @override
  ValueListenable<AudioDeviceStatus> get devices => deviceStatus;

  @override
  Future<Result<void>> prepare() async {
    prepareCalls++;
    return _prepare();
  }

  /// What [requestMicAccess] answers.
  Result<void> micAccess = const Result.ok(null);
  int micAccessRequests = 0;

  /// Holds [requestMicAccess] until completed (an OS dialog left open).
  Completer<void>? micAccessGate;

  /// Prepares first, like the real one.
  @override
  Future<Result<void>> requestMicAccess() async {
    micAccessRequests++;
    log.add('requestMicAccess');
    if (_prepare() case Error(:final error)) return Result.error(error);
    await micAccessGate?.future;
    if (_closed) return _closedError();
    return micAccess;
  }

  @override
  Future<Result<CaptureHandle>> startCapture({
    required Duration maxLength,
    required void Function() onLimit,
  }) async {
    log.add('startCapture');
    if (_closed) return _closedError();
    if (_current case final open? when open.isOpen) {
      // A capture that is still open is cancelled first.
      open.supersede();
    }
    if (_prepare() case Error(:final error)) return Result.error(error);
    lastOnLimit = onLimit;
    final capture = FakeCapture(this);
    _current = capture;
    await startGate?.future;
    if (_closed) {
      _release(capture);
      return _closedError();
    }
    if (!identical(_current, capture)) {
      return const Result.error(AudioSupersededException('capture'));
    }
    if (startError case final error?) {
      _release(capture);
      return Result.error(error);
    }
    capture.startedAt = clock?.call();
    captures.add(capture);
    return Result.ok(capture);
  }

  @override
  Result<PlaybackHandle> beginPlayback(int sampleRate) {
    log.add('beginPlayback($sampleRate)');
    if (_closed) return _closedError();
    if (!_ready) {
      return const Result.error(
        PlaybackException('prepare() has not succeeded'),
      );
    }
    // Like the real one: the reply still playing stops (barge-in and the
    // next reply rely on it).
    if (_playing case final previous?) unawaited(previous.stop());
    final playback = _playing = FakePlayback(this, sampleRate);
    playbacks.add(playback);
    return Result.ok(playback);
  }

  /// Like the real one: the open capture is cancelled (one being stopped
  /// finishes its stop) and the playback stopped; idempotent.
  @override
  Future<void> close() async {
    if (_closed) return;
    _closed = true;
    if (_current case final capture? when capture.isOpen) {
      await capture.cancel();
    }
    await _playing?.stop();
    level.dispose();
    deviceStatus.dispose();
  }
}

class FakeCapture implements CaptureHandle {
  FakeCapture(this._repo);

  final FakeAudioRepository _repo;
  bool cancelled = false;
  bool stopped = false;

  /// A newer capture replaced this one while it was open.
  bool superseded = false;

  /// [FakeAudioRepository.clock] when the capture was handed out.
  Duration? startedAt;

  bool get isOpen => !cancelled && !stopped;

  /// The repository cancels it for a newer capture.
  void supersede() {
    _repo.log.add('capture.superseded');
    superseded = true;
    cancelled = true;
    _repo._release(this);
  }

  /// Like the real one: after a cancel or a supersede there is no
  /// utterance.
  @override
  Future<Result<Utterance>> stop() async {
    _repo.log.add('capture.stop');
    if (cancelled) {
      return const Result.error(AudioSupersededException('capture'));
    }
    stopped = true;
    _repo._release(this);
    if (_repo.stopResult case final result?) return result;
    final next = _repo.nextUtterance;
    final (clock, started) = (_repo.clock, startedAt);
    if (clock == null || started == null) return Result.ok(next);
    return Result.ok(Utterance(pcm: next.pcm, held: clock() - started));
  }

  /// Like the real one: nothing during (or after) a stop.
  @override
  Future<void> cancel() async {
    _repo.log.add('capture.cancel');
    if (stopped) return;
    cancelled = true;
    _repo._release(this);
  }
}

class FakePlayback implements PlaybackHandle {
  FakePlayback(this._repo, this.sampleRate);

  final FakeAudioRepository _repo;
  final int sampleRate;
  final List<int> chunks = [];
  final Completer<void> _drained = Completer();
  bool ended = false;
  bool stopped = false;

  @override
  Future<void> get drained => _drained.future;

  @override
  void enqueue(Uint8List pcm) {
    _repo.log.add(stopped ? 'enqueue-after-stop' : 'enqueue');
    chunks.add(pcm.length);
  }

  @override
  void end() {
    if (ended) return;
    ended = true;
    _repo.log.add('end');
    if (_repo.autoDrain) completeDrain();
  }

  @override
  Future<void> stop() async {
    if (!stopped) {
      stopped = true;
      _repo.log.add('playback.stop');
    }
    completeDrain();
  }

  void completeDrain() {
    if (!_drained.isCompleted) _drained.complete();
    _repo._releasePlayback(this);
  }
}
