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
import 'dart:io' show Platform;
import 'dart:typed_data';

import 'package:flutter/foundation.dart';

import '../../config/voice_config.dart';
import '../../domain/audio/audio_device_checks.dart';
import '../../domain/models/audio_devices.dart';
import '../../domain/models/hardware_profile.dart' show HostPlatform;
import '../../domain/models/voice.dart';
import '../../utils/pcm.dart';
import '../../utils/result.dart';
import '../services/audio/audio_device_service.dart';
import '../services/audio/audio_session_service.dart';
import '../services/audio/mic_service.dart';
import '../services/audio/pcm_player_service.dart';
import 'audio_repository.dart';

/// [AudioRepository] over the platform session, the mic and soloud.
/// Half-duplex: the caller never captures and plays at once (barge-in stops
/// playback before it opens the mic). The input device is checked when a voice
/// demo asks for access (and again on a press after a failed check); errors say
/// what to do.
class DeviceAudioRepository implements AudioRepository {
  DeviceAudioRepository({
    required this._session,
    required this._mic,
    required this._player,
    required this._deviceService,
    HostPlatform? platform,
    this._sampleRate = 16000,
    this._drainSlack = kPlaybackDrainSlack,
    this._streamEndTimeout = const Duration(seconds: 1),
    this._emptyCaptureCheckAfter = const Duration(milliseconds: 500),
    this._closeWait = const Duration(seconds: 5),
    int Function()? clockMicros,
  }) : _platform =
           platform ??
           HostPlatform.fromOperatingSystem(Platform.operatingSystem),
       _clock = clockMicros ?? _monotonicMicros;

  static final Stopwatch _monotonic = Stopwatch()..start();
  static int _monotonicMicros() => _monotonic.elapsedMicroseconds;

  /// Monotonic µs; tests inject a fake.
  final int Function() _clock;

  final AudioSessionService _session;
  final MicService _mic;
  final PcmPlayerService _player;
  final AudioDeviceService _deviceService;
  final HostPlatform _platform;
  final int _sampleRate;
  final Duration _drainSlack;

  /// How long [CaptureHandle.stop] waits for the mic stream to close.
  final Duration _streamEndTimeout;

  /// Linux: a press at least this long that delivered no bytes is checked
  /// (parecord sends a chunk every ~100 ms; when it exits, record 7.1.1 does
  /// not close the app's stream, `record_stream.dart:11-22` forwards no
  /// `onDone`, so the silence is the only sign).
  final Duration _emptyCaptureCheckAfter;

  /// How long [close] waits for a prepare, a capture start or a mic access
  /// request in flight, and then for each shutdown step after a wait that
  /// ran out (a permission dialog left open must not hold the app's exit
  /// forever).
  final Duration _closeWait;

  final ValueNotifier<double> _level = ValueNotifier(0);
  final ValueNotifier<AudioDeviceStatus> _devices = ValueNotifier(
    const AudioDeviceStatus(),
  );
  Future<Result<void>>? _preparing;
  bool _ready = false;
  _DeviceCapture? _capture;
  _DevicePlayback? _playback;
  bool _closed = false;
  Future<void>? _closing;

  /// The [startCapture] and [requestMicAccess] calls in flight. [close]
  /// waits for them (bounded) before it disposes the mic: a start that finds
  /// the repository closed stops the mic it opened, and that stop must come
  /// before the dispose; and record runs every call after the one before it
  /// (record.dart `_safeCall`), so a dispose queues behind their permission
  /// request anyway.
  final Set<Future<void>> _starting = {};

  /// Adds [run] to [_starting] until it ends. Tracked for [close] only: the
  /// caller gets the result, or the error, from [run] itself.
  void _track(Future<Object?> run) {
    final tracked = run.then<void>((_) {}, onError: (Object _) {});
    _starting.add(tracked);
    unawaited(tracked.whenComplete(() => _starting.remove(tracked)));
  }

  @override
  ValueListenable<double> get inputLevel => _level;

  @override
  ValueListenable<AudioDeviceStatus> get devices => _devices;

  @override
  Future<Result<void>> prepare() {
    if (_closed) return Future.value(_closedError());
    if (_ready) return Future.value(const Result.ok(null));
    return _preparing ??= _prepare().whenComplete(() => _preparing = null);
  }

  Future<Result<void>> _prepare() async {
    final watch = Stopwatch()..start();
    // iOS order: configure + activate the session, then soloud; the recorder is
    // created later, on the first capture.
    try {
      await _session.configureHalfDuplex();
    } catch (e, st) {
      debugPrint('[Audio] session setup failed: $e\n$st');
      return Result.error(PlaybackException('audio session setup: $e'));
    }
    try {
      await _player.init();
    } catch (e, st) {
      debugPrint('[Audio] output engine failed to start: $e\n$st');
      return Result.error(PlaybackException('output engine start: $e'));
    }
    if (_closed) return _closedError();
    // An engine that started on miniaudio's Null device (or nothing) plays
    // in silence: fail, and shut it down so the next prepare starts over
    // (a server started meanwhile would otherwise not be used).
    if (await _checkOutput() case DeviceUnavailable(:final message)) {
      await _guard('player dispose', _player.dispose);
      return Result.error(PlaybackException(message));
    }
    if (_closed) return _closedError();
    _ready = true;
    debugPrint('[Audio] ready in ${watch.elapsedMilliseconds}ms');
    return const Result.ok(null);
  }

  /// Prepares first (the iOS session must be configured before `record`
  /// creates its recorder), then asks `record`, which shows the OS dialog
  /// the first time.
  @override
  Future<Result<void>> requestMicAccess() {
    if (_closed) return Future.value(_closedError());
    final run = _requestMicAccess();
    _track(run);
    return run;
  }

  Future<Result<void>> _requestMicAccess() async {
    final prepared = await prepare();
    if (prepared case Error(:final error)) return Result.error(error);
    // A prepare that had finished returns at once, but the await still lets
    // a close run first.
    if (_closed) return _closedError();
    final bool permitted;
    try {
      permitted = await _mic.hasPermission();
    } catch (e, st) {
      debugPrint('[Audio] permission check failed: $e\n$st');
      if (_closed) return _closedError();
      return Result.error(MicAccessException('$kMicAccessMessage ($e)'));
    }
    // The dialog may have stayed open while the app closed.
    if (_closed) return _closedError();
    if (!permitted) {
      debugPrint('[Audio] microphone access denied');
      return Result.error(MicAccessException(kMicAccessMessage));
    }
    if (await _checkInput() case DeviceUnavailable(:final message)) {
      return Result.error(MicAccessException(message));
    }
    return const Result.ok(null);
  }

  /// Names the output the engine plays to and publishes the answer.
  Future<DeviceCheck> _checkOutput() async {
    DeviceCheck check;
    try {
      final playback = canListPlaybackDevices(_platform)
          ? _player.listOutputs()
          : null;
      if (playback != null) {
        debugPrint(
          '[Audio] playback devices: ${playback.isEmpty ? 'none' : [for (final d in playback) '"${d.name}"${d.isDefault ? ' (default)' : ''}'].join(', ')}',
        );
      }
      check = outputCheck(
        platform: _platform,
        playback: playback,
        linux: _platform == HostPlatform.linux
            ? await _deviceService.audioSystem()
            : null,
      );
    } catch (e, st) {
      debugPrint('[Audio] listing the playback devices failed: $e\n$st');
      check = DeviceUnavailable(
        'The output device could not be checked (the audio engine did not '
        'list its devices: $e)',
      );
    }
    if (!_closed) {
      debugPrint('[Audio] output: ${deviceCheckText(check)}');
      _devices.value = _devices.value.copyWith(output: check);
    }
    return check;
  }

  /// Asks the OS which input a capture would use and publishes the answer.
  Future<DeviceCheck> _checkInput() async {
    final check = await _deviceService.checkInput();
    _setInput(check);
    return check;
  }

  void _setInput(DeviceCheck check) {
    if (_closed) return;
    debugPrint('[Audio] input: ${deviceCheckText(check)}');
    _devices.value = _devices.value.copyWith(input: check);
  }

  /// Whether a capture that delivered nothing in [held] (or whose stream
  /// [endedEarly]) is checked rather than handed on as an empty utterance.
  bool _checksEmptyCapture(Duration held, {required bool endedEarly}) =>
      endedEarly ||
      (_platform == HostPlatform.linux && held >= _emptyCaptureCheckAfter);

  /// Why a capture delivered no audio (`parecord` exits at once when the
  /// sound server went away), from a fresh input check.
  Future<MicAccessException> _emptyCaptureFailure(Duration held) async {
    final input = await _checkInput();
    return MicAccessException(switch (input) {
      DeviceUnavailable(:final message) => message,
      _ =>
        'No audio arrived from the microphone in ${held.inMilliseconds} ms '
            '(${deviceCheckText(input)}); press again',
    });
  }

  @override
  Future<Result<CaptureHandle>> startCapture({
    required Duration maxLength,
    required void Function() onLimit,
  }) {
    if (_closed) return Future.value(_closedError());
    final run = _startCapture(maxLength: maxLength, onLimit: onLimit);
    _track(run);
    return run;
  }

  /// [startCapture]'s body. After every await it checks for a [close] (the
  /// closed error, nothing started; a mic it opened is stopped) and then for
  /// a newer capture (superseded).
  Future<Result<CaptureHandle>> _startCapture({
    required Duration maxLength,
    required void Function() onLimit,
  }) async {
    final previous = _capture;
    if (previous != null) await previous.cancel();
    final prepared = await prepare();
    if (prepared case Error(:final error)) return Result.error(error);
    if (_closed) return _closedError();

    final capture = _DeviceCapture(
      this,
      maxBytes: maxLength.inMicroseconds * _sampleRate * 2 ~/ 1000000,
      maxLength: maxLength,
      onLimit: onLimit,
    );
    _capture = capture;
    final bool permitted;
    try {
      permitted = await _mic.hasPermission();
    } catch (e, st) {
      debugPrint('[Audio] permission check failed: $e\n$st');
      _release(capture);
      if (_closed) return _closedError();
      return Result.error(MicAccessException('$kMicAccessMessage ($e)'));
    }
    if (_closed) {
      _release(capture);
      return _closedError();
    }
    if (!identical(_capture, capture)) {
      return const Result.error(AudioSupersededException('capture'));
    }
    if (!permitted) {
      _release(capture);
      return Result.error(MicAccessException(kMicAccessMessage));
    }
    // The last check failed: check once more (the user may have fixed it)
    // rather than start a capture that cannot work. One check per press, no
    // retry loop.
    if (_devices.value.input is DeviceUnavailable) {
      final input = await _checkInput();
      if (_closed) {
        _release(capture);
        return _closedError();
      }
      if (!identical(_capture, capture)) {
        return const Result.error(AudioSupersededException('capture'));
      }
      if (input case DeviceUnavailable(:final message)) {
        _release(capture);
        return Result.error(MicAccessException(message));
      }
    }
    final Stream<Uint8List> stream;
    try {
      stream = await _mic.startPcm16(
        sampleRate: _sampleRate,
        onFormatChanged: capture._onFormatChanged,
      );
    } catch (e, st) {
      debugPrint('[Audio] mic start failed: $e\n$st');
      _release(capture);
      if (_closed) return _closedError();
      final message = micStartErrorMessage(e, _platform);
      if (message == kParecordMissing) {
        _setInput(const DeviceUnavailable(kParecordMissing));
      }
      return Result.error(MicAccessException(message));
    }
    if (_closed || !identical(_capture, capture)) {
      // close(), cancel() or a newer capture ran while the mic was starting.
      // close() waits for this stop before it disposes the mic.
      await _stopMicQuietly();
      _release(capture);
      return _closed
          ? _closedError()
          : const Result.error(AudioSupersededException('capture'));
    }
    capture._listen(stream);
    return Result.ok(capture);
  }

  @override
  Result<PlaybackHandle> beginPlayback(int sampleRate) {
    if (_closed) return _closedError();
    if (!_ready) {
      return const Result.error(
        PlaybackException('prepare() has not succeeded'),
      );
    }
    final previous = _playback;
    if (previous != null) unawaited(previous.stop());
    final PcmOutput output;
    try {
      output = _player.open(sampleRate);
    } catch (e, st) {
      debugPrint('[Audio] opening a playback stream failed: $e\n$st');
      return Result.error(PlaybackException('$e'));
    }
    final playback = _playback = _DevicePlayback(
      this,
      output,
      sampleRate: sampleRate,
      drainSlack: _drainSlack,
    );
    return Result.ok(playback);
  }

  /// Closed first, so every entry point returns the closed error from here
  /// on and an operation in flight stops at its next check. Then waits
  /// (bounded) for a prepare, capture starts and mic access requests in
  /// flight, so nothing is created after the capture and playback taken
  /// here and the engines are not disposed under a call. When that wait ran
  /// out, each shutdown step is bounded too and left to finish on its own:
  /// record queues a mic call behind the one still running (a permission
  /// dialog left open), and the app's exit must not wait for an answer.
  /// Every call gets the same shutdown.
  @override
  Future<void> close() => _closing ??= _close();

  Future<void> _close() async {
    _closed = true;
    var settled = true;
    final inFlight = <Future<Object?>>[?_preparing, ..._starting];
    if (inFlight.isNotEmpty) {
      try {
        await Future.wait(inFlight).timeout(
          _closeWait,
          onTimeout: () {
            settled = false;
            debugPrint(
              '[Audio] close: a prepare, capture start or mic access request '
              'did not finish within ${_closeWait.inMilliseconds}ms; shutting '
              'down anyway',
            );
            return const [];
          },
        );
      } catch (e, st) {
        // Their callers get these errors; here they only must have ended.
        debugPrint('[Audio] close: an operation in flight failed: $e\n$st');
      }
    }
    Future<void> step(String what, Future<void> Function() op) =>
        settled ? _guard(what, op) : _bounded(what, op);
    final capture = _capture;
    if (capture != null) {
      // A press being stopped finishes its stop (its mic call is in flight);
      // any other capture is cancelled.
      final stopping = capture._stopping;
      await step('capture stop', () async {
        if (stopping != null) {
          await stopping;
        } else {
          await capture.cancel();
        }
      });
    }
    final playback = _playback;
    if (playback != null) await step('playback stop', playback.stop);
    await step('mic dispose', _mic.dispose);
    await step('player dispose', _player.dispose);
    _level.dispose();
    _devices.dispose();
  }

  /// [op], guarded, given up on after [_closeWait] (logged): it is left to
  /// finish on its own.
  Future<void> _bounded(String what, Future<void> Function() op) =>
      _guard(what, op).timeout(
        _closeWait,
        onTimeout: () => debugPrint(
          '[Audio] close: $what did not finish within '
          '${_closeWait.inMilliseconds}ms; left to finish on its own',
        ),
      );

  bool _isCurrent(_DeviceCapture capture) =>
      !_closed && identical(_capture, capture);

  void _setLevel(_DeviceCapture capture, double level) {
    if (_isCurrent(capture)) _level.value = level;
  }

  void _release(_DeviceCapture capture) {
    if (!identical(_capture, capture)) return;
    _capture = null;
    if (!_closed) _level.value = 0;
  }

  void _releasePlayback(_DevicePlayback playback) {
    if (identical(_playback, playback)) _playback = null;
  }

  Future<void> _stopMicQuietly() => _guard('mic stop', _mic.stop);

  static Future<void> _guard(String what, Future<void> Function() op) async {
    try {
      await op();
    } catch (e, st) {
      debugPrint('[Audio] $what failed: $e\n$st');
    }
  }

  static Result<T> _closedError<T>() =>
      Result.error(asException(StateError('AudioRepository closed')));
}

final class _DeviceCapture implements CaptureHandle {
  _DeviceCapture(
    this._repo, {
    required this._maxBytes,
    required this._maxLength,
    required this._onLimit,
  });

  final DeviceAudioRepository _repo;
  final int _maxBytes;
  final Duration _maxLength;
  final void Function() _onLimit;
  final BytesBuilder _bytes = BytesBuilder();

  /// Started when the mic stream is listened to, not at the request: a
  /// permission dialog or a slow mic start is not part of the press.
  final Stopwatch _held = Stopwatch();
  final Completer<void> _streamDone = Completer();
  StreamSubscription<Uint8List>? _sub;
  Timer? _limitTimer;
  Object? _failure;
  bool _limitHit = false;
  bool _cancelled = false;

  /// The stream closed before [stop] (a plugin that forwards the capture
  /// process's end; record 7.1.1 does not).
  bool _endedEarly = false;
  Future<Result<Utterance>>? _stopping;

  void _listen(Stream<Uint8List> stream) {
    _sub = stream.listen(
      _onData,
      onError: (Object e, StackTrace st) {
        debugPrint('[Audio] mic stream error: $e\n$st');
        _failure ??= e;
      },
      onDone: () {
        if (_stopping == null && !_cancelled) {
          _endedEarly = true;
          debugPrint('[Audio] the mic stream ended before stop');
        }
        if (!_streamDone.isCompleted) _streamDone.complete();
      },
    );
    _held.start();
    _limitTimer = Timer(_maxLength, _hitLimit);
  }

  /// Keeps chunks until the stream ends — including the ones the mic flushes
  /// after stop, which hold the last word — up to the cap.
  void _onData(Uint8List chunk) {
    if (_cancelled) return;
    if (_stopping == null) {
      _repo._setLevel(this, levelFromDbfs(rmsDbfs(chunk)));
    }
    final room = _maxBytes - _bytes.length;
    if (room > 0) {
      _bytes.add(
        chunk.length <= room ? chunk : Uint8List.sublistView(chunk, 0, room),
      );
    }
    if (_bytes.length >= _maxBytes) _hitLimit();
  }

  void _hitLimit() {
    if (_limitHit || _cancelled || _stopping != null) return;
    _limitHit = true;
    debugPrint(
      '[Audio] capture reached ${_maxLength.inSeconds}s (the STT window); '
      'ending the press',
    );
    _onLimit();
  }

  void _onFormatChanged(String actual) {
    debugPrint('[Audio] mic format changed: $actual');
    _failure ??= MicFormatException(actual);
  }

  @override
  Future<Result<Utterance>> stop() {
    if (_cancelled) {
      return Future.value(
        const Result.error(AudioSupersededException('capture')),
      );
    }
    return _stopping ??= _stop();
  }

  Future<Result<Utterance>> _stop() async {
    _limitTimer?.cancel();
    final held = _held.elapsed;
    try {
      await _repo._mic.stop();
    } catch (e, st) {
      debugPrint('[Audio] mic stop failed: $e\n$st');
      _failure ??= e;
    }
    // record delivers the tail before its stream closes.
    await _streamDone.future.timeout(
      _repo._streamEndTimeout,
      onTimeout: () => debugPrint(
        '[Audio] the mic stream did not close within '
        '${_repo._streamEndTimeout.inMilliseconds}ms of stop',
      ),
    );
    await _sub?.cancel();
    if (_failure == null &&
        _bytes.isEmpty &&
        _repo._checksEmptyCapture(held, endedEarly: _endedEarly)) {
      _failure = await _repo._emptyCaptureFailure(held);
    }
    _repo._release(this);
    final failure = _failure;
    if (failure != null) return Result.error(asException(failure));
    return Result.ok(Utterance(pcm: _bytes.takeBytes(), held: held));
  }

  @override
  Future<void> cancel() async {
    if (_cancelled || _stopping != null) return;
    _cancelled = true;
    _limitTimer?.cancel();
    final wasCurrent = identical(_repo._capture, this);
    _repo._release(this);
    if (_sub != null && wasCurrent) await _repo._stopMicQuietly();
    await _sub?.cancel();
  }
}

final class _DevicePlayback implements PlaybackHandle {
  _DevicePlayback(
    this._repo,
    this._output, {
    required this._sampleRate,
    required this._drainSlack,
  }) {
    unawaited(_output.finished.then((_) => _complete()));
  }

  final DeviceAudioRepository _repo;
  final PcmOutput _output;
  final int _sampleRate;
  final Duration _drainSlack;
  final Completer<void> _drained = Completer();

  /// When the queued audio ends at the latest ([_clock] µs). soloud pauses
  /// when its buffer runs dry, so a chunk that arrives after the queue
  /// emptied plays from its arrival, not from the end of the previous
  /// one.
  int? _endAt;
  Timer? _watchdog;
  int _bytes = 0;
  bool _ended = false;
  Future<void>? _stopping;

  @override
  Future<void> get drained => _drained.future;

  @override
  void enqueue(Uint8List pcm) {
    if (_stopping != null || _drained.isCompleted || pcm.isEmpty) return;
    if (_ended) {
      debugPrint('[Audio] a chunk arrived after end(); dropped');
      return;
    }
    try {
      _output.add(pcm);
    } catch (e, st) {
      debugPrint('[Audio] queueing reply audio failed: $e\n$st');
      throw PlaybackException('$e');
    }
    _bytes += pcm.length;
    final now = _repo._clock();
    final queuedEnd = _endAt;
    _endAt =
        (queuedEnd == null || queuedEnd < now ? now : queuedEnd) +
        pcm16Duration(pcm.length, _sampleRate).inMicroseconds;
  }

  @override
  void end() {
    if (_ended || _stopping != null || _drained.isCompleted) return;
    _ended = true;
    try {
      _output.end();
    } catch (e, st) {
      debugPrint('[Audio] ending the reply stream failed: $e\n$st');
      _complete();
      return;
    }
    if (_bytes == 0) {
      // Nothing was queued (a reply without audible clauses): nothing plays.
      _complete();
      return;
    }
    final remaining = Duration(microseconds: _endAt! - _repo._clock());
    final bound =
        (remaining.isNegative ? Duration.zero : remaining) + _drainSlack;
    _watchdog = Timer(bound, () {
      if (_drained.isCompleted) return;
      debugPrint(
        '[Audio] playback did not report its end within '
        '${bound.inMilliseconds}ms of the last chunk; stopping it and '
        'treating it as drained',
      );
      // Released below, this handle can no longer be stopped by a barge-in
      // or the next reply: silence the voice before letting go of it.
      unawaited(
        _output.stop().catchError(
          (Object e, StackTrace st) =>
              debugPrint('[Audio] stopping a timed-out playback failed: $e'),
        ),
      );
      _complete();
    });
  }

  @override
  Future<void> stop() => _stopping ??= _stop();

  Future<void> _stop() {
    _watchdog?.cancel();
    // PcmOutput.stop runs the native stop synchronously.
    return _output.stop().whenComplete(_complete);
  }

  void _complete() {
    _watchdog?.cancel();
    if (!_drained.isCompleted) _drained.complete();
    _repo._releasePlayback(this);
  }
}
