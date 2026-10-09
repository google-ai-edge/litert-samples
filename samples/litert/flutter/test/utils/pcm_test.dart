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

import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:litert_edge_demos/config/voice_config.dart';
import 'package:litert_edge_demos/utils/pcm.dart';

import '../support/pcm.dart';

void main() {
  test('a tone: loud peak frame, not all zero', () {
    final stats = analyzePcm16(tone(const Duration(milliseconds: 500)));
    expect(stats.samples, 8000);
    expect(stats.allZero, isFalse);
    // 0.3 amplitude sine: RMS = 0.3 / √2 ≈ −13.5 dBFS.
    expect(stats.peakFrameDbfs, closeTo(-13.5, 0.5));
  });

  test('room noise is quiet but not digital zero', () {
    final stats = analyzePcm16(quiet(const Duration(seconds: 1)));
    expect(stats.allZero, isFalse);
    expect(stats.peakFrameDbfs, lessThan(-60));
  });

  test('digital silence is all zero at the floor; empty input too', () {
    final silent = analyzePcm16(zeros(const Duration(seconds: 1)));
    expect(silent.allZero, isTrue);
    expect(silent.peakFrameDbfs, kPcmFloorDbfs);
    final empty = analyzePcm16(Uint8List(0));
    expect(empty.samples, 0);
    expect(empty.peakFrameDbfs, kPcmFloorDbfs);
  });

  test('the peak is per frame: a short loud burst in silence is found', () {
    final pcm = BytesBuilder()
      ..add(zeros(const Duration(milliseconds: 500)))
      ..add(tone(const Duration(milliseconds: 40)))
      ..add(zeros(const Duration(milliseconds: 500)));
    final stats = analyzePcm16(pcm.takeBytes());
    expect(stats.peakFrameDbfs, greaterThan(-20));
  });

  test('reads any byte offset and ignores a trailing odd byte', () {
    final base = tone(const Duration(milliseconds: 100));
    final shifted = Uint8List(base.length + 2)
      ..setRange(1, base.length + 1, base);
    final view = Uint8List.sublistView(shifted, 1, base.length + 2);
    final stats = analyzePcm16(view);
    expect(stats.samples, base.length ~/ 2);
    expect(
      stats.peakFrameDbfs,
      closeTo(analyzePcm16(base).peakFrameDbfs, 0.01),
    );
  });

  test('meter mapping and durations', () {
    expect(levelFromDbfs(-60), 0);
    expect(levelFromDbfs(-30), closeTo(0.5, 1e-9));
    expect(levelFromDbfs(0), 1);
    expect(levelFromDbfs(-96), 0);
    expect(
      rmsDbfs(tone(const Duration(milliseconds: 100))),
      closeTo(-13.5, 0.5),
    );
    expect(pcm16Duration(32000, 16000), const Duration(seconds: 1));
    expect(pcm16Duration(48000, 24000), const Duration(seconds: 1));
  });

  group('measureVoice', () {
    test('speech in a quiet room: the gate is the threshold', () {
      final pcm = BytesBuilder()
        ..add(quiet(const Duration(milliseconds: 500)))
        ..add(tone(const Duration(milliseconds: 800)))
        ..add(quiet(const Duration(milliseconds: 500)));
      final v = measureVoice(pcm.takeBytes(), gateDbfs: -45);
      expect(v.thresholdDbfs, -45);
      expect(v.voiced.inMilliseconds, closeTo(800, 40));
    });

    test('a click is a frame or two', () {
      final v = measureVoice(click(const Duration(seconds: 1)), gateDbfs: -45);
      expect(v.voiced, lessThanOrEqualTo(const Duration(milliseconds: 40)));
    });

    test('constant room noise above the gate sits at its own floor', () {
      final v = measureVoice(noise(const Duration(seconds: 1)), gateDbfs: -45);
      expect(v.floorDbfs, greaterThan(-45));
      expect(v.thresholdDbfs, greaterThan(v.floorDbfs));
      expect(v.voiced, Duration.zero);
    });

    test('speech from end to end still counts (the floor part is capped)', () {
      final v = measureVoice(tone(const Duration(seconds: 1)), gateDbfs: -45);
      expect(v.voiced.inMilliseconds, closeTo(1000, 20));
    });

    test('the real fixture ("What is the capital of France?") passes', () {
      final pcm = File('test_assets/france_16k.pcm').readAsBytesSync();
      final v = measureVoice(pcm, gateDbfs: kDefaultSilenceGateDbfs);
      expect(v.voiced, greaterThan(const Duration(milliseconds: 500)));
    });

    test('empty input', () {
      final v = measureVoice(Uint8List(0), gateDbfs: -45);
      expect(v.voiced, Duration.zero);
      expect(v.peakFrameDbfs, kPcmFloorDbfs);
    });
  });

  group('VOICE_GATE_DBFS', () {
    test('default, a valid value, and a bad flag fails loudly', () {
      expect(voiceConfigFromEnvironment(gate: '').silenceGateDbfs, -45);
      expect(voiceConfigFromEnvironment(gate: ' -38 ').silenceGateDbfs, -38);
      expect(
        () => voiceConfigFromEnvironment(gate: 'loud'),
        throwsA(isA<FormatException>()),
      );
      expect(
        () => voiceConfigFromEnvironment(gate: '3'),
        throwsA(isA<FormatException>()),
      );
    });
  });

  test('sine tone: length, level, no click at the ends', () {
    final tone = sineTonePcm16(
      hz: 440,
      duration: const Duration(seconds: 1),
      sampleRate: 24000,
    );
    expect(tone.length, 48000);
    expect(rmsDbfs(tone), closeTo(-9.1, 0.1));
    final view = ByteData.sublistView(tone);
    expect(view.getInt16(0, Endian.little), 0);
    expect(view.getInt16(tone.length - 2, Endian.little).abs(), lessThan(10));
  });
}

/// A minimal WAV: RIFF header, `fmt ` and `data` chunks.
