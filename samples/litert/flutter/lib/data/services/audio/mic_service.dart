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

import 'package:flutter/foundation.dart';
import 'package:record/record.dart';

/// The microphone as a stream of headerless PCM16 mono chunks. One capture
/// at a time; the audio repository owns the policy (cap, level, stop).
abstract interface class MicService {
  /// Whether the app may record; asks the user the first time.
  Future<bool> hasPermission();

  /// Starts capturing PCM16 mono at [sampleRate]. [onFormatChanged] runs when
  /// the platform delivers another format than requested (the STT would get
  /// wrong audio), with a description of what it chose.
  Future<Stream<Uint8List>> startPcm16({
    required int sampleRate,
    required void Function(String actual) onFormatChanged,
  });

  /// Stops the capture; the stream from [startPcm16] then closes.
  Future<void> stop();

  /// Releases the recorder. Safe to call more than once.
  Future<void> dispose();
}

/// [MicService] over `record` 7.x: PCM16, the voice-recognition source on
/// Android, no echo cancellation (half-duplex). On iOS it is told not to manage
/// the audio session, which `audio_session` owns; macOS resamples from the
/// device rate itself.
final class RecordMicService implements MicService {
  AudioRecorder? _recorder;

  Future<AudioRecorder> _ensure() async {
    final existing = _recorder;
    if (existing != null) return existing;
    final recorder = _recorder = AudioRecorder();
    // Null off iOS. Per recorder; dispose() resets it, so it is set once
    // here, after the audio repository configured the session and soloud.
    await recorder.ios?.manageAudioSession(false);
    return recorder;
  }

  @override
  Future<bool> hasPermission() async => (await _ensure()).hasPermission();

  @override
  Future<Stream<Uint8List>> startPcm16({
    required int sampleRate,
    required void Function(String actual) onFormatChanged,
  }) async {
    final recorder = await _ensure();
    await recorder.setOnConfigChanged(
      (c) => onFormatChanged(
        '${c.encoder.name} ${c.sampleRate} Hz ${c.numChannels} ch',
      ),
    );
    final stream = await recorder.startStream(
      RecordConfig(
        encoder: AudioEncoder.pcm16bits,
        sampleRate: sampleRate,
        numChannels: 1,
        androidConfig: const AndroidRecordConfig(
          audioSource: AndroidAudioSource.voiceRecognition,
        ),
      ),
    );
    debugPrint('[Mic] capture started: pcm16 $sampleRate Hz mono');
    return stream;
  }

  @override
  Future<void> stop() async {
    await _recorder?.stop();
  }

  @override
  Future<void> dispose() async {
    final recorder = _recorder;
    _recorder = null;
    await recorder?.dispose();
  }
}
