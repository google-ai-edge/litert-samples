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

// The WAV writer for the regenerated question clips
// (tools/make_question_audio_test.dart): the counterpart of wav.dart's
// pcm16FromWav. The app itself never writes a WAV.

import 'dart:typed_data';

/// The size of the header [wavFromPcm16] writes: RIFF/WAVE, a 16-byte
/// `fmt ` chunk and the `data` chunk's id and size.
const kWavHeaderBytes = 44;

/// [pcm], 16-bit little-endian mono samples at [sampleRate] Hz, as a WAV
/// file: the canonical [kWavHeaderBytes]-byte header (PCM format 1, one
/// channel, 16 bits) and the samples as the `data` chunk, nothing else.
/// Throws an [ArgumentError] for an odd byte count or a rate that is not
/// positive.
Uint8List wavFromPcm16(Uint8List pcm, {required int sampleRate}) {
  if (pcm.length.isOdd) {
    throw ArgumentError.value(
      pcm.length,
      'pcm',
      '16-bit PCM has an even number of bytes',
    );
  }
  if (sampleRate <= 0) {
    throw ArgumentError.value(sampleRate, 'sampleRate', 'must be positive');
  }
  const channels = 1;
  const bytesPerSample = 2;
  final wav = Uint8List(kWavHeaderBytes + pcm.length);
  final view = ByteData.sublistView(wav);
  void id(int offset, String fourCc) =>
      wav.setRange(offset, offset + 4, fourCc.codeUnits);
  id(0, 'RIFF');
  view.setUint32(4, kWavHeaderBytes - 8 + pcm.length, Endian.little);
  id(8, 'WAVE');
  id(12, 'fmt ');
  view.setUint32(16, 16, Endian.little); // fmt chunk size
  view.setUint16(20, 1, Endian.little); // PCM
  view.setUint16(22, channels, Endian.little);
  view.setUint32(24, sampleRate, Endian.little);
  view.setUint32(
    28,
    sampleRate * channels * bytesPerSample,
    Endian.little,
  ); // byte rate
  view.setUint16(32, channels * bytesPerSample, Endian.little); // block align
  view.setUint16(34, bytesPerSample * 8, Endian.little); // bits per sample
  id(36, 'data');
  view.setUint32(40, pcm.length, Endian.little);
  wav.setRange(kWavHeaderBytes, wav.length, pcm);
  return wav;
}
