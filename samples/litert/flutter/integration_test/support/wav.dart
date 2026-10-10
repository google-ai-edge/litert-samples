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

// The WAV reader for the integration tests' recorded questions: the app
// itself never reads a WAV (its audio is the microphone's PCM).

import 'dart:math' as math;
import 'dart:typed_data';

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
      if (format != 1 || channels != 1 || rate != 16000 || bits != 16) {
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
