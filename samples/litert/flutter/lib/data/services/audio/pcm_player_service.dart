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
import 'package:flutter_soloud/flutter_soloud.dart';

import '../../../config/voice_config.dart';
import '../../../domain/models/audio_devices.dart';

/// Streamed PCM16 mono playback: one [PcmOutput] per reply.
abstract interface class PcmPlayerService {
  /// Starts the output engine. Call once, after the audio session is active.
  Future<void> init();

  /// The playback devices the engine's audio backend lists, the default
  /// marked (callable before [init]). Throws when the engine cannot list.
  List<AudioDevice> listOutputs();

  /// A new output stream at [sampleRate]. Throws when the engine cannot make
  /// one.
  PcmOutput open(int sampleRate);

  /// Stops everything and shuts the engine down.
  Future<void> dispose();
}

/// One reply's audio: chunks in order, then [end].
abstract interface class PcmOutput {
  /// Queues [pcm]; the first chunk starts playback. Throws if the engine
  /// rejects it. Ignored after [stop].
  void add(Uint8List pcm);

  /// No more chunks: [finished] completes once the queued audio has played.
  void end();

  /// Completes when playback ended naturally or was stopped, or at [end]
  /// when nothing was ever queued.
  Future<void> get finished;

  /// Silences the output now and releases it. The native stop happens
  /// synchronously, before the returned future's first await.
  Future<void> stop();
}

/// [PcmPlayerService] over flutter_soloud 4.x buffer streams:
/// `BufferingType.released`, s16le mono, and a 0.2 s refill threshold instead
/// of the 2 s default.
final class SoloudPcmPlayerService implements PcmPlayerService {
  SoloudPcmPlayerService([SoLoud? soloud])
    : _soloud = soloud ?? SoLoud.instance;

  final SoLoud _soloud;

  @override
  Future<void> init() async {
    if (_soloud.isInitialized) return;
    await _soloud.init(
      sampleRate: kPlaybackEngineSampleRate,
      bufferSize: kPlaybackEngineBufferFrames,
    );
    debugPrint(
      '[Player] soloud ready: $kPlaybackEngineSampleRate Hz, '
      '$kPlaybackEngineBufferFrames-frame buffer',
    );
  }

  @override
  List<AudioDevice> listOutputs() => [
    for (final d in _soloud.listPlaybackDevices())
      AudioDevice(id: '${d.id}', name: d.name, isDefault: d.isDefault),
  ];

  @override
  PcmOutput open(int sampleRate) {
    final source = _soloud.setBufferStream(
      bufferingType: BufferingType.released,
      bufferingTimeNeeds: kPlaybackBufferingSeconds,
      sampleRate: sampleRate,
      channels: Channels.mono,
      format: BufferType.s16le,
    );
    return _SoloudOutput(_soloud, source);
  }

  @override
  Future<void> dispose() async {
    if (_soloud.isInitialized) _soloud.deinit();
  }
}

final class _SoloudOutput implements PcmOutput {
  _SoloudOutput(this._soloud, this._source);

  final SoLoud _soloud;
  final AudioSource _source;
  final Completer<void> _finished = Completer();
  SoundHandle? _handle;
  StreamSubscription<void>? _endSub;
  bool _ended = false;
  bool _stopped = false;

  @override
  Future<void> get finished => _finished.future;

  @override
  void add(Uint8List pcm) {
    if (_stopped || _finished.isCompleted) return;
    _soloud.addAudioDataStream(_source, pcm);
    if (_handle != null) return;
    // Fires on a natural end and on stop (soloud.dart voice-ended events).
    _endSub = _source.allInstancesFinished.listen((_) => _complete());
    final handle = _soloud.play(_source);
    if (handle.isError || !_soloud.getIsValidVoiceHandle(handle)) {
      _complete();
      throw StateError('soloud did not start a voice for the reply audio');
    }
    _handle = handle;
  }

  @override
  void end() {
    if (_ended || _stopped || _finished.isCompleted) return;
    _ended = true;
    _soloud.setDataIsEnded(_source);
    if (_handle == null) _complete();
  }

  @override
  Future<void> stop() async {
    if (_stopped) return _finished.future;
    _stopped = true;
    final handle = _handle;
    try {
      // SoLoud.stop runs the native stop before its first await.
      if (handle != null) await _soloud.stop(handle);
    } catch (e, st) {
      debugPrint('[Player] stop failed: $e\n$st');
    } finally {
      _complete();
    }
  }

  void _complete() {
    if (_finished.isCompleted) return;
    _finished.complete();
    unawaited(_endSub?.cancel());
    _endSub = null;
    unawaited(
      _soloud
          .disposeSource(_source)
          .catchError(
            (Object e, StackTrace st) => debugPrint(
              '[Player] disposing the reply stream failed: $e\n$st',
            ),
          ),
    );
  }
}
