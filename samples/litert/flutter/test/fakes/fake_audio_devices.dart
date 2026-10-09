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
import 'dart:typed_data';

import 'package:litert_edge_demos/data/services/audio/audio_device_service.dart';
import 'package:litert_edge_demos/data/services/audio/audio_session_service.dart';
import 'package:litert_edge_demos/data/services/audio/mic_service.dart';
import 'package:litert_edge_demos/data/services/audio/pcm_player_service.dart';
import 'package:litert_edge_demos/domain/models/audio_devices.dart';

class FakeAudioSession implements AudioSessionService {
  int configureCalls = 0;
  Exception? error;

  @override
  Future<void> configureHalfDuplex() async {
    configureCalls++;
    if (error case final e?) throw e;
  }
}

/// Answers [input] (and [system]) and counts the input checks.
class FakeAudioDeviceService implements AudioDeviceService {
  DeviceCheck input = const DeviceReady('Fake Mic', detail: 'test');
  AudioSystem? system;
  int inputChecks = 0;

  @override
  Future<DeviceCheck> checkInput() async {
    inputChecks++;
    return input;
  }

  @override
  Future<AudioSystem?> audioSystem() async => system;
}

/// A mic whose chunks the test delivers with [deliver]. Like record 7.1.1,
/// whose recorder runs every call through one semaphore (`_safeCall`,
/// record.dart:216-224), its calls run one at a time in call order: a
/// [dispose] behind a [hasPermission] held by [permissionGate] (a dialog
/// left open) waits for the answer.
class FakeMicService implements MicService {
  bool permission = true;
  bool closeOnStop = true;

  /// Thrown by [startPcm16] (record_linux throws `ProcessException` without
  /// `parecord`).
  Exception? startError;

  /// Delivered by [stop] before the stream closes: record flushes the last
  /// buffer after stop.
  Uint8List? tailOnStop;

  /// When set, [hasPermission] waits for it (the permission dialog).
  Completer<void>? permissionGate;

  /// When set, [startPcm16] waits for it (a slow mic start).
  Completer<void>? startGate;
  int starts = 0;
  int stops = 0;
  int disposes = 0;

  /// Calls made after [dispose] was called: the recorder is gone, each one
  /// is a bug.
  final List<String> usedAfterDispose = [];
  void Function(String actual)? onFormatChanged;
  StreamController<Uint8List>? _stream;
  bool _disposeCalled = false;

  /// The end of the last call queued.
  Future<void> _lock = Future.value();

  /// Runs [call] after every call made before it, like record's semaphore.
  Future<T> _serial<T>(Future<T> Function() call) {
    final previous = _lock;
    final done = Completer<void>();
    _lock = done.future;
    return () async {
      await previous;
      try {
        return await call();
      } finally {
        done.complete();
      }
    }();
  }

  /// Closes the current stream. A stream nobody listens to (a capture that
  /// stopped before it listened) completes its close only once listened to:
  /// that close is not awaited, as record does not wait for a listener.
  Future<void> _closeStream() async {
    final stream = _stream;
    if (stream == null) return;
    if (stream.hasListener) {
      await stream.close();
    } else {
      unawaited(stream.close());
    }
  }

  void _use(String call) {
    if (_disposeCalled) usedAfterDispose.add(call);
  }

  @override
  Future<bool> hasPermission() {
    _use('hasPermission');
    return _serial(() async {
      await permissionGate?.future;
      return permission;
    });
  }

  @override
  Future<Stream<Uint8List>> startPcm16({
    required int sampleRate,
    required void Function(String actual) onFormatChanged,
  }) {
    _use('startPcm16');
    return _serial(() async {
      starts++;
      await startGate?.future;
      if (startError case final e?) throw e;
      this.onFormatChanged = onFormatChanged;
      await _closeStream();
      _stream = StreamController<Uint8List>();
      return _stream!.stream;
    });
  }

  void deliver(Uint8List chunk) => _stream!.add(chunk);

  /// The capture process exited on its own: the stream closes before stop.
  Future<void> endStream() async => _stream?.close();

  @override
  Future<void> stop() {
    _use('stop');
    return _serial(() async {
      stops++;
      if (tailOnStop case final tail?) _stream?.add(tail);
      if (closeOnStop) await _closeStream();
    });
  }

  /// [disposes] counts the disposes that ran (one queued behind a held
  /// call has not).
  @override
  Future<void> dispose() {
    _disposeCalled = true;
    return _serial(() async {
      disposes++;
      await _closeStream();
    });
  }
}

class FakePcmPlayer implements PcmPlayerService {
  int inits = 0;
  int disposes = 0;
  Error? initError;

  /// When set, [init] waits for it (a slow engine start).
  Completer<void>? initGate;

  /// Every call in order (`init`, `init done`, `open`, `dispose`), to check
  /// that the engine is never disposed while it starts.
  final List<String> calls = [];
  final List<FakePcmOutput> outputs = [];

  /// What [listOutputs] answers; [listError] is thrown instead when set.
  List<AudioDevice> playbackDevices = const [
    AudioDevice(id: '0', name: 'Fake Speakers', isDefault: true),
  ];
  Error? listError;

  @override
  List<AudioDevice> listOutputs() {
    if (listError case final e?) throw e;
    return playbackDevices;
  }

  @override
  Future<void> init() async {
    inits++;
    calls.add('init');
    await initGate?.future;
    calls.add('init done');
    if (initError case final e?) throw e;
  }

  @override
  PcmOutput open(int sampleRate) {
    calls.add('open');
    final output = FakePcmOutput(sampleRate);
    outputs.add(output);
    return output;
  }

  @override
  Future<void> dispose() async {
    disposes++;
    calls.add('dispose');
  }
}

/// Reports its end only when the test calls [finishPlaying] (or on stop).
class FakePcmOutput implements PcmOutput {
  FakePcmOutput(this.sampleRate);

  final int sampleRate;
  final List<Uint8List> chunks = [];
  final Completer<void> _finished = Completer();
  bool ended = false;
  bool stopped = false;

  @override
  Future<void> get finished => _finished.future;

  @override
  void add(Uint8List pcm) => chunks.add(pcm);

  @override
  void end() => ended = true;

  @override
  Future<void> stop() async {
    stopped = true;
    finishPlaying();
  }

  void finishPlaying() {
    if (!_finished.isCompleted) _finished.complete();
  }
}
