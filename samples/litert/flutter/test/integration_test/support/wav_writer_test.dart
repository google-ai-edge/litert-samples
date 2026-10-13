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

import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';

import '../../../integration_test/support/wav.dart';
import '../../../integration_test/support/wav_writer.dart';

void main() {
  group('wavFromPcm16', () {
    test('pcm16FromWav reads back exactly the samples written', () {
      final samples = Uint8List.fromList(List.generate(64, (i) => i));
      final wav = wavFromPcm16(samples, sampleRate: 16000);

      expect(wav.length, kWavHeaderBytes + samples.length);
      final pcm = pcm16FromWav(wav);
      expect(pcm, samples);
      expect(pcm.offsetInBytes, kWavHeaderBytes);
    });

    test('the canonical header: RIFF size, PCM, mono, rate, 16 bits', () {
      final wav = wavFromPcm16(Uint8List(100), sampleRate: 24000);
      final view = ByteData.sublistView(wav);
      String id(int at) => String.fromCharCodes(wav, at, at + 4);

      expect(id(0), 'RIFF');
      expect(view.getUint32(4, Endian.little), 36 + 100);
      expect(id(8), 'WAVE');
      expect(id(12), 'fmt ');
      expect(view.getUint32(16, Endian.little), 16);
      expect(view.getUint16(20, Endian.little), 1, reason: 'PCM');
      expect(view.getUint16(22, Endian.little), 1, reason: 'mono');
      expect(view.getUint32(24, Endian.little), 24000);
      expect(view.getUint32(28, Endian.little), 48000, reason: 'byte rate');
      expect(view.getUint16(32, Endian.little), 2, reason: 'block align');
      expect(view.getUint16(34, Endian.little), 16, reason: 'bits');
      expect(id(36), 'data');
      expect(view.getUint32(40, Endian.little), 100);
    });

    test('an odd byte count or a rate that is not positive is refused', () {
      expect(
        () => wavFromPcm16(Uint8List(3), sampleRate: 16000),
        throwsArgumentError,
      );
      expect(
        () => wavFromPcm16(Uint8List(4), sampleRate: 0),
        throwsArgumentError,
      );
    });
  });
}
