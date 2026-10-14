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

import 'dart:io' show ZLibEncoder;
import 'dart:typed_data';

/// A small PNG writer in pure Dart, for isolates without the engine's codecs
/// (the detector worker writes the visual memory's frames with it): 8-bit
/// truecolour without alpha (colour type 2), one IDAT, every scanline with
/// filter 0 (none), deflated by `dart:io`'s zlib. LiteRT-LM's stb decoder
/// and the engine's codec both read it.
///
/// [rgb] is tightly packed R G B, `width × height × 3` bytes. [level] is
/// zlib's (1 fastest … 9 smallest).
Uint8List encodePngRgb(int width, int height, Uint8List rgb, {int level = 6}) {
  if (width <= 0 || height <= 0) {
    throw ArgumentError('a PNG needs a positive size, got ${width}x$height');
  }
  final stride = width * 3;
  if (rgb.length != stride * height) {
    throw ArgumentError(
      '${rgb.length} bytes for ${width}x$height RGB (expected '
      '${stride * height})',
    );
  }
  // Each scanline: its filter byte (0) then the row as it is.
  final raw = Uint8List((stride + 1) * height);
  for (var y = 0; y < height; y++) {
    raw.setRange(y * (stride + 1) + 1, (y + 1) * (stride + 1), rgb, y * stride);
  }
  final idat = ZLibEncoder(level: level).convert(raw);

  final header = ByteData(13)
    ..setUint32(0, width)
    ..setUint32(4, height)
    ..setUint8(8, 8) // bit depth
    ..setUint8(9, 2) // colour type: truecolour
    ..setUint8(10, 0) // compression: deflate
    ..setUint8(11, 0) // filter method: adaptive (every row uses type 0)
    ..setUint8(12, 0); // no interlace

  final out = BytesBuilder(copy: false)..add(_signature);
  _chunk(out, 'IHDR', header.buffer.asUint8List());
  _chunk(out, 'IDAT', idat is Uint8List ? idat : Uint8List.fromList(idat));
  _chunk(out, 'IEND', Uint8List(0));
  return out.takeBytes();
}

const _signature = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

/// Length, type, data, then the CRC-32 of type and data (big-endian).
void _chunk(BytesBuilder out, String type, Uint8List data) {
  final typeBytes = Uint8List.fromList(type.codeUnits);
  final length = ByteData(4)..setUint32(0, data.length);
  var crc = _crcUpdate(0xFFFFFFFF, typeBytes);
  crc = _crcUpdate(crc, data) ^ 0xFFFFFFFF;
  final crcBytes = ByteData(4)..setUint32(0, crc);
  out
    ..add(length.buffer.asUint8List())
    ..add(typeBytes)
    ..add(data)
    ..add(crcBytes.buffer.asUint8List());
}

/// The CRC-32 of [bytes] (ISO 3309, the polynomial PNG and zip use).
int crc32(List<int> bytes) => _crcUpdate(0xFFFFFFFF, bytes) ^ 0xFFFFFFFF;

int _crcUpdate(int crc, List<int> bytes) {
  var c = crc;
  for (final b in bytes) {
    c = _crcTable[(c ^ b) & 0xFF] ^ (c >>> 8);
  }
  return c;
}

final Uint32List _crcTable = () {
  final table = Uint32List(256);
  for (var n = 0; n < 256; n++) {
    var c = n;
    for (var k = 0; k < 8; k++) {
      c = (c & 1) != 0 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1;
    }
    table[n] = c;
  }
  return table;
}();
