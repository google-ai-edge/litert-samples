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

/// 16 kHz mono PCM16 in a RIFF/WAVE container: a form LiteRT-LM decodes
/// (an audio question to EmbeddingGemma 2 or to a chat model that hears
/// audio), the self-test's bundled clip and the integration tests' spoken
/// questions.
library;

import 'dart:math' as math;
import 'dart:typed_data';

/// The size of the header [wavFromPcm16] writes: RIFF/WAVE, a 16-byte
/// `fmt ` chunk and the `data` chunk's id and size.
const kWavHeaderBytes = 44;

/// The one format every WAV in the app has: what the microphone records.
const kWavSampleRate = 16000;

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

/// The samples of a 16 kHz mono PCM16 WAV: the `data` chunk, found by
/// walking the chunks (afconvert writes a 4096-byte header, so a fixed
/// 44-byte offset reads header bytes as audio, and a fixed 4096 breaks on
/// every other writer). A view into [wav], no copy. Throws a
/// [FormatException] for another format or a file without a `data` chunk.
Uint8List pcm16FromWav(Uint8List wav) {
  if (wav.length < 12 ||
      String.fromCharCodes(wav, 0, 4) != 'RIFF' ||
      String.fromCharCodes(wav, 8, 12) != 'WAVE') {
    throw const FormatException('Not a RIFF/WAVE file');
  }
  final view = ByteData.sublistView(wav);
  var offset = 12;
  var formatChecked = false;
  while (offset + 8 <= wav.length) {
    final id = String.fromCharCodes(wav, offset, offset + 4);
    final size = view.getUint32(offset + 4, Endian.little);
    final start = offset + 8;
    if (id == 'fmt ') {
      if (start + 16 > wav.length) {
        throw const FormatException('WAV fmt chunk is truncated');
      }
      final format = view.getUint16(start, Endian.little);
      final channels = view.getUint16(start + 2, Endian.little);
      final rate = view.getUint32(start + 4, Endian.little);
      final bits = view.getUint16(start + 14, Endian.little);
      if (format != 1 ||
          channels != 1 ||
          rate != kWavSampleRate ||
          bits != 16) {
        throw FormatException(
          'WAV must be 16 kHz mono PCM16, got format $format, $channels '
          'channel(s), $rate Hz, $bits bits',
        );
      }
      formatChecked = true;
    }
    if (id == 'data') {
      if (!formatChecked) {
        throw const FormatException('WAV data chunk before its fmt chunk');
      }
      final end = math.min(start + size, wav.length);
      return Uint8List.sublistView(wav, start, end);
    }
    offset = start + size + (size & 1);
  }
  throw const FormatException('WAV file has no data chunk');
}

/// How long the speech in a 16 kHz mono PCM16 [wav] lasts (its `data`
/// chunk). Throws a [FormatException] like [pcm16FromWav].
Duration wavDuration(Uint8List wav) => Duration(
  microseconds: pcm16FromWav(wav).length * 1000000 ~/ (kWavSampleRate * 2),
);
