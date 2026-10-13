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

import '../../../integration_test/support/wav.dart';

void main() {
  group('pcm16FromWav (any header size, never a fixed offset)', () {
    test("the clip generator's 44-byte header (test_assets/q_cats.wav)", () {
      final wav = File('test_assets/q_cats.wav').readAsBytesSync();
      final pcm = pcm16FromWav(wav);
      // 2.1 s with the silence (tool/make_question_audio.sh): the whole data
      // chunk, and nothing of the header.
      expect(pcm.length, greaterThan(40000));
      expect(pcm.offsetInBytes, 44);
      expect(pcm.length, wav.length - 44);
    });

    test('a plain 44-byte header', () {
      final samples = Uint8List.fromList(List.generate(64, (i) => i));
      final pcm = pcm16FromWav(_wav(samples));
      expect(pcm, samples);
    });

    test('a chunk before the data (a 4096-byte header, as afconvert pads '
        'it): skipped, not read as audio', () {
      final samples = Uint8List.fromList(List.generate(64, (i) => i));
      // 'FLLR' with 4096 - 44 - 8 bytes: the data starts at byte 4096.
      final wav = _wav(samples, padding: 4096 - 44 - 8);
      final pcm = pcm16FromWav(wav);
      expect(pcm, samples);
      expect(pcm.offsetInBytes, 4096);
    });

    test('anything but 16 kHz mono PCM16 is refused', () {
      expect(
        () => pcm16FromWav(_wav(Uint8List(8), channels: 2)),
        throwsA(isA<FormatException>()),
      );
      expect(
        () => pcm16FromWav(_wav(Uint8List(8), rate: 44100)),
        throwsA(isA<FormatException>()),
      );
      expect(
        () => pcm16FromWav(Uint8List.fromList('RIFF0000WAVE'.codeUnits)),
        throwsA(isA<FormatException>()),
      );
    });
  });
}

/// A 16-bit PCM WAV of [samples]; [padding] > 0 puts a `FLLR` chunk of that
/// many bytes between `fmt ` and `data`.
Uint8List _wav(
  Uint8List samples, {
  int channels = 1,
  int rate = 16000,
  int padding = 0,
}) {
  final b = BytesBuilder();
  void u32(int v) =>
      b.add((ByteData(4)..setUint32(0, v, Endian.little)).buffer.asUint8List());
  void u16(int v) =>
      b.add((ByteData(2)..setUint16(0, v, Endian.little)).buffer.asUint8List());
  final filler = padding > 0 ? 8 + padding : 0;
  b.add('RIFF'.codeUnits);
  u32(36 + filler + samples.length);
  b.add('WAVE'.codeUnits);
  b.add('fmt '.codeUnits);
  u32(16);
  u16(1);
  u16(channels);
  u32(rate);
  u32(rate * channels * 2);
  u16(channels * 2);
  u16(16);
  if (padding > 0) {
    b.add('FLLR'.codeUnits);
    u32(padding);
    b.add(Uint8List(padding));
  }
  b.add('data'.codeUnits);
  u32(samples.length);
  b.add(samples);
  return b.toBytes();
}
