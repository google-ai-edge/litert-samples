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

import 'dart:convert' show latin1;
import 'dart:typed_data';

/// The stream is not MJPEG, or is broken beyond resynchronising.
final class MjpegStreamException implements Exception {
  const MjpegStreamException(this.message);

  final String message;

  @override
  String toString() => message;
}

/// A `Content-Type` header split into its media type (lower case) and its
/// parameters (names lower case, values unquoted).
({String mediaType, Map<String, String> parameters}) parseContentType(
  String value,
) {
  final parts = value.split(';');
  final parameters = <String, String>{};
  for (final part in parts.skip(1)) {
    final eq = part.indexOf('=');
    if (eq <= 0) continue;
    final name = part.substring(0, eq).trim().toLowerCase();
    var v = part.substring(eq + 1).trim();
    if (v.length >= 2 && v.startsWith('"') && v.endsWith('"')) {
      v = v.substring(1, v.length - 1);
    }
    parameters[name] = v;
  }
  return (mediaType: parts.first.trim().toLowerCase(), parameters: parameters);
}

/// Largest part (one JPEG) the parser accepts; a 4K JPEG is ~2 MB.
const kMjpegMaxPartBytes = 16 * 1024 * 1024;

/// Bytes the parser skips while looking for a boundary before it decides the
/// stream is not multipart at all.
const kMjpegMaxGarbageBytes = 1024 * 1024;

/// Longest header block of one part.
const kMjpegMaxHeaderBytes = 8 * 1024;

enum _State { boundary, headers, body }

/// Splits a `multipart/x-mixed-replace` byte stream (MJPEG over HTTP, as the
/// IP Webcam app, ffmpeg's `mpjpeg` muxer and most IP cameras serve it) into
/// JPEG images, incrementally: feed each network chunk to [add], which
/// returns the images it completed.
///
/// - A part starts after a line with the boundary. The parameter value is
///   matched with its leading dashes stripped, so `boundary=--abc` and
///   `boundary=abc` both find `--abc` lines (servers differ). Anything
///   before the first boundary (a garbage prefix) is skipped, up to
///   [kMjpegMaxGarbageBytes].
/// - Header names are case-insensitive (`Content-length` from ffmpeg).
/// - With `Content-Length`, the body is exactly that many bytes. Without it,
///   the body is the JPEG from its SOI marker to its EOI, found by walking
///   the JPEG segments (so an EXIF thumbnail's own EOI inside APP1 does not
///   end the image early).
/// - Parts whose `Content-Type` is not `image/jpeg` are skipped and counted
///   in [skippedParts].
///
/// [add] throws [MjpegStreamException] when the stream cannot be MJPEG (no
/// boundary in the first megabyte, an oversized part or header block). The
/// parser keeps one partial part in memory; emitted images are copies.
final class MjpegParser {
  MjpegParser(String boundary)
    : _token = latin1.encode(_stripDashes(boundary)) {
    if (_token!.isEmpty) {
      throw MjpegStreamException('Empty multipart boundary "$boundary"');
    }
  }

  /// For a server that does not name the boundary in its `Content-Type`
  /// (ffmpeg's `-listen 1` sends `application/octet-stream`): the boundary
  /// is the first line of the body, which must start with `--`. Anything
  /// else fails fast.
  MjpegParser.detect() : _token = null;

  static String _stripDashes(String boundary) {
    var i = 0;
    while (i < boundary.length && boundary.codeUnitAt(i) == 0x2D) {
      i++;
    }
    return boundary.substring(i);
  }

  /// The boundary without its leading dashes; null until [MjpegParser.detect]
  /// has read it from the first line.
  Uint8List? _token;
  Uint8List _buf = Uint8List(64 * 1024);
  int _len = 0;

  /// Start of the unconsumed bytes in [_buf].
  int _pos = 0;
  _State _state = _State.boundary;

  // The current part.
  int? _contentLength;
  bool _isJpeg = true;

  /// Bytes skipped before the first boundary (and between parts) since the
  /// last boundary was found.
  int _garbage = 0;

  // Incremental JPEG walk (no Content-Length), relative to [_pos].
  int _jpegStart = -1;
  int _jpegCursor = 0;
  bool _inEntropy = false;

  /// Parts skipped because their `Content-Type` is not `image/jpeg`.
  int skippedParts = 0;

  /// Images emitted so far.
  int images = 0;

  /// Bytes skipped outside parts (prefix, padding, resync after a wrong
  /// `Content-Length`).
  int garbageBytes = 0;

  /// Feeds one chunk; returns the JPEG images it completed, oldest first.
  List<Uint8List> add(List<int> chunk) {
    _append(chunk);
    final out = <Uint8List>[];
    while (_step(out)) {}
    _compact();
    return out;
  }

  void _append(List<int> chunk) {
    if (_len + chunk.length > _buf.length) {
      // Drop what is consumed before growing.
      _compact(force: true);
      if (_len + chunk.length > _buf.length) {
        var size = _buf.length * 2;
        while (size < _len + chunk.length) {
          size *= 2;
        }
        final grown = Uint8List(size)..setRange(0, _len, _buf);
        _buf = grown;
      }
    }
    _buf.setRange(_len, _len + chunk.length, chunk);
    _len += chunk.length;
  }

  void _compact({bool force = false}) {
    if (_pos == 0) return;
    if (!force && _pos < _buf.length ~/ 2) return;
    final remaining = _len - _pos;
    _buf.setRange(0, remaining, _buf, _pos);
    _len = remaining;
    _pos = 0;
  }

  /// One state transition; false when more bytes are needed.
  bool _step(List<Uint8List> out) => switch (_state) {
    _State.boundary => _findBoundary(),
    _State.headers => _readHeaders(),
    _State.body => _readBody(out),
  };

  /// The boundary from the body's first non-blank line (`--boundary`).
  bool _detectBoundary() {
    var start = _pos;
    while (start < _len && (_buf[start] == 0x0D || _buf[start] == 0x0A)) {
      start++;
    }
    final eol = _indexOfByte(0x0A, start);
    if (eol < 0) {
      if (_len - start > 1024) {
        throw const MjpegStreamException(
          'The stream does not start with a multipart boundary line: not an '
          'MJPEG stream',
        );
      }
      return false;
    }
    var end = eol;
    if (end > start && _buf[end - 1] == 0x0D) end--;
    final line = latin1.decode(Uint8List.sublistView(_buf, start, end)).trim();
    final token = _stripDashes(line);
    if (!line.startsWith('--') || token.isEmpty || token.contains(' ')) {
      throw const MjpegStreamException(
        'The stream does not start with a multipart boundary line: not an '
        'MJPEG stream',
      );
    }
    _token = latin1.encode(token);
    detectedBoundary = token;
    return true;
  }

  /// The boundary [MjpegParser.detect] found; null otherwise.
  String? detectedBoundary;

  bool _findBoundary() {
    final token = _token;
    if (token == null) return _detectBoundary();
    final at = _indexOf(token, _pos);
    if (at < 0) {
      // Keep a possible partial token at the end.
      final keep = token.length - 1;
      final skip = (_len - _pos) - keep;
      if (skip > 0) {
        _skipGarbage(_pos, skip);
        _pos += skip;
      }
      return false;
    }
    final eol = _indexOfByte(0x0A, at + token.length);
    if (eol < 0) return false; // the rest of the boundary line is not here
    _skipGarbage(_pos, at - _pos);
    _garbage = 0;
    // `--token--` closes the multipart body; the HTTP stream ends after it.
    final closing =
        eol - (at + token.length) >= 2 &&
        _buf[at + token.length] == 0x2D &&
        _buf[at + token.length + 1] == 0x2D;
    _pos = eol + 1;
    if (closing) return true; // look for another boundary (or the end)
    _state = _State.headers;
    _contentLength = null;
    _isJpeg = true;
    return true;
  }

  /// Counts the [count] skipped bytes at [from] against the garbage limit.
  /// Line breaks, blanks and the boundary's own dashes are not counted in
  /// [garbageBytes].
  void _skipGarbage(int from, int count) {
    for (var i = from; i < from + count; i++) {
      final b = _buf[i];
      if (b != 0x0D && b != 0x0A && b != 0x2D && b != 0x20 && b != 0x09) {
        garbageBytes++;
      }
    }
    _garbage += count;
    if (_garbage > kMjpegMaxGarbageBytes) {
      throw const MjpegStreamException(
        'No multipart boundary in ${kMjpegMaxGarbageBytes ~/ 1024} KB of '
        'data: not an MJPEG stream',
      );
    }
  }

  bool _readHeaders() {
    while (true) {
      final eol = _indexOfByte(0x0A, _pos);
      if (eol < 0) {
        if (_len - _pos > kMjpegMaxHeaderBytes) {
          throw const MjpegStreamException(
            'A multipart header block is over 8 KB: not an MJPEG stream',
          );
        }
        return false;
      }
      var end = eol;
      if (end > _pos && _buf[end - 1] == 0x0D) end--;
      final line = latin1.decode(Uint8List.sublistView(_buf, _pos, end));
      _pos = eol + 1;
      if (line.isEmpty) {
        _state = _State.body;
        _jpegStart = -1;
        _jpegCursor = 0;
        _inEntropy = false;
        return true;
      }
      final colon = line.indexOf(':');
      if (colon <= 0) continue;
      final name = line.substring(0, colon).trim().toLowerCase();
      final value = line.substring(colon + 1).trim();
      switch (name) {
        case 'content-length':
          final length = int.tryParse(value);
          if (length == null || length < 0) {
            throw MjpegStreamException('Bad part Content-Length "$value"');
          }
          if (length > kMjpegMaxPartBytes) {
            throw MjpegStreamException(
              'A part of $length bytes is over the '
              '${kMjpegMaxPartBytes ~/ (1024 * 1024)} MB limit',
            );
          }
          _contentLength = length;
        case 'content-type':
          final type = parseContentType(value).mediaType;
          _isJpeg = type == 'image/jpeg' || type == 'image/jpg';
      }
    }
  }

  bool _readBody(List<Uint8List> out) {
    final length = _contentLength;
    if (length != null) {
      if (_len - _pos < length) return false;
      if (_isJpeg) {
        out.add(_buf.sublist(_pos, _pos + length));
        images++;
      } else {
        skippedParts++;
      }
      _pos += length;
      _state = _State.boundary;
      return true;
    }
    final end = _scanJpeg();
    if (end == null) {
      if (_len - _pos > kMjpegMaxPartBytes) {
        throw const MjpegStreamException(
          'A part without Content-Length has no JPEG end within 16 MB',
        );
      }
      return false;
    }
    if (end < 0) {
      // Not a JPEG: skip to the next boundary.
      skippedParts++;
      _state = _State.boundary;
      return true;
    }
    final start = _pos + _jpegStart;
    if (_isJpeg) {
      out.add(_buf.sublist(start, end));
      images++;
    } else {
      skippedParts++;
    }
    _pos = end;
    _state = _State.boundary;
    return true;
  }

  /// Walks the JPEG at [_pos] from where the last call stopped. Returns the
  /// absolute end (after EOI), null when more bytes are needed, or -1 when
  /// the part does not hold a JPEG.
  int? _scanJpeg() {
    final base = _pos;
    final n = _len;
    if (_jpegStart < 0) {
      // Skip line breaks and padding before SOI.
      var i = base;
      while (i < n &&
          (_buf[i] == 0x0D ||
              _buf[i] == 0x0A ||
              _buf[i] == 0x20 ||
              _buf[i] == 0x09)) {
        i++;
      }
      if (i + 1 >= n) return null;
      if (_buf[i] != 0xFF || _buf[i + 1] != 0xD8) return -1;
      _jpegStart = i - base;
      _jpegCursor = i - base + 2;
      _inEntropy = false;
    }
    var i = base + _jpegCursor;
    while (true) {
      if (_inEntropy) {
        // Entropy-coded data: FF 00 is a stuffed byte, FF D0–D7 a restart
        // marker, FF FF fill; any other FF xx is the next marker.
        while (true) {
          if (i + 1 >= n) {
            _jpegCursor = i - base;
            return null;
          }
          if (_buf[i] != 0xFF) {
            i++;
            continue;
          }
          final next = _buf[i + 1];
          if (next == 0x00 || (next >= 0xD0 && next <= 0xD7)) {
            i += 2;
          } else if (next == 0xFF) {
            i++;
          } else {
            _inEntropy = false;
            break;
          }
        }
      }
      // A marker segment at i.
      if (i + 1 >= n) {
        _jpegCursor = i - base;
        return null;
      }
      if (_buf[i] != 0xFF) return -1;
      var code = _buf[i + 1];
      while (code == 0xFF) {
        i++; // fill bytes before a marker
        if (i + 1 >= n) {
          _jpegCursor = i - base;
          return null;
        }
        code = _buf[i + 1];
      }
      if (code == 0xD9) return i + 2; // EOI
      if ((code >= 0xD0 && code <= 0xD7) || code == 0x01) {
        i += 2; // standalone markers
        continue;
      }
      if (i + 3 >= n) {
        _jpegCursor = i - base;
        return null;
      }
      final segment = (_buf[i + 2] << 8) | _buf[i + 3];
      if (segment < 2) return -1;
      i += 2 + segment;
      if (code == 0xDA) _inEntropy = true; // SOS: entropy data follows
      if (i > n) {
        _jpegCursor = i - base;
        // The segment's end is past the data we have: wait (the cursor may
        // point past _len; the next call resumes there).
        return null;
      }
    }
  }

  int _indexOf(Uint8List needle, int from) {
    final first = needle[0];
    final last = _len - needle.length;
    outer:
    for (var i = from; i <= last; i++) {
      if (_buf[i] != first) continue;
      for (var j = 1; j < needle.length; j++) {
        if (_buf[i + j] != needle[j]) continue outer;
      }
      return i;
    }
    return -1;
  }

  int _indexOfByte(int byte, int from) {
    for (var i = from; i < _len; i++) {
      if (_buf[i] == byte) return i;
    }
    return -1;
  }
}
