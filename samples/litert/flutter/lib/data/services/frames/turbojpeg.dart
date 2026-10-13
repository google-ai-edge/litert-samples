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

import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

/// libjpeg-turbo's TurboJPEG API (the 2.x functions, also exported by 3.x) over
/// FFI: a CPU JPEG decoder for the network camera. Synchronous: use it on a
/// worker isolate (`TurboJpegDecoder` in `jpeg_decoder.dart`).

/// `TJPF_RGBA` (turbojpeg.h): R G B A, alpha 255. The fixture's and the
/// engine decoder's layout, so the detector's gather needs nothing new.
const kTjpfRgba = 7;

/// `TJERR_WARNING` (turbojpeg.h): the image decoded, maybe with damage.
const kTjErrWarning = 0;

/// The largest decoded frame accepted (pixels after the decode-time
/// scaling): 4096×4096. A corrupt header claiming 65535×65535 would
/// otherwise ask for gigabytes.
const kTurboJpegMaxPixels = 4096 * 4096;

typedef _InitC = Pointer<Void> Function();
typedef _HeaderC = Int Function(
  Pointer<Void>,
  Pointer<Uint8>,
  UnsignedLong,
  Pointer<Int>,
  Pointer<Int>,
  Pointer<Int>,
  Pointer<Int>,
);
typedef _HeaderDart = int Function(
  Pointer<Void>,
  Pointer<Uint8>,
  int,
  Pointer<Int>,
  Pointer<Int>,
  Pointer<Int>,
  Pointer<Int>,
);
typedef _DecompressC = Int Function(
  Pointer<Void>,
  Pointer<Uint8>,
  UnsignedLong,
  Pointer<Uint8>,
  Int,
  Int,
  Int,
  Int,
  Int,
);
typedef _DecompressDart = int Function(
  Pointer<Void>,
  Pointer<Uint8>,
  int,
  Pointer<Uint8>,
  int,
  int,
  int,
  int,
  int,
);
typedef _HandleIntC = Int Function(Pointer<Void>);
typedef _HandleIntDart = int Function(Pointer<Void>);
typedef _ErrorStrC = Pointer<Utf8> Function(Pointer<Void>);

/// Where libturbojpeg is looked for, in order, for [operatingSystem]:
/// - Linux: the copy shipped in the bundle's `lib/` (next to the
///   executable, installed by `linux/CMakeLists.txt` from the build
///   machine), then the system's `libturbojpeg.so.0` (Debian/Ubuntu
///   package `libturbojpeg`; Raspberry Pi OS `libturbojpeg0`).
/// - macOS: Homebrew's `jpeg-turbo` (readable from tests and unsandboxed
///   runs; the sandboxed app falls back to the engine decoder).
/// - Elsewhere: none (Android and iOS use the engine decoder).
List<String> turboJpegCandidates({
  String? operatingSystem,
  String? executable,
}) {
  final os = operatingSystem ?? Platform.operatingSystem;
  final exe = executable ?? Platform.resolvedExecutable;
  final dir = exe.substring(0, exe.lastIndexOf('/') + 1);
  return switch (os) {
    'linux' => ['${dir}lib/libturbojpeg.so.0', 'libturbojpeg.so.0'],
    'macos' => [
      '/opt/homebrew/opt/jpeg-turbo/lib/libturbojpeg.0.dylib',
      '/usr/local/opt/jpeg-turbo/lib/libturbojpeg.0.dylib',
    ],
    _ => const [],
  };
}

/// The TurboJPEG functions the decoder uses, from one opened library.
final class TurboJpegLibrary {
  TurboJpegLibrary._(this.path, DynamicLibrary lib)
    : _init = lib.lookupFunction<_InitC, _InitC>('tjInitDecompress'),
      _header = lib.lookupFunction<_HeaderC, _HeaderDart>(
        'tjDecompressHeader3',
      ),
      _decompress = lib.lookupFunction<_DecompressC, _DecompressDart>(
        'tjDecompress2',
      ),
      _destroy = lib.lookupFunction<_HandleIntC, _HandleIntDart>('tjDestroy'),
      _errorStr = lib.lookupFunction<_ErrorStrC, _ErrorStrC>('tjGetErrorStr2'),
      _errorCode = lib.lookupFunction<_HandleIntC, _HandleIntDart>(
        'tjGetErrorCode',
      );

  /// The candidate that opened.
  final String path;
  final Pointer<Void> Function() _init;
  final _HeaderDart _header;
  final _DecompressDart _decompress;
  final _HandleIntDart _destroy;
  final Pointer<Utf8> Function(Pointer<Void>) _errorStr;
  final _HandleIntDart _errorCode;

  /// Opens the first of [candidates] that loads and has every function.
  /// [failures] gets one line per candidate that did not (for the log and
  /// the "slow JPEG decoder" note); null when none loaded.
  static TurboJpegLibrary? open(
    List<String> candidates, {
    List<String>? failures,
  }) {
    for (final path in candidates) {
      try {
        return TurboJpegLibrary._(path, DynamicLibrary.open(path));
      } on Object catch (e) {
        failures?.add('$path: ${_firstLine('$e')}');
      }
    }
    return null;
  }

  /// A decompressor with its own reusable buffers.
  TurboJpegSession session() {
    final handle = _init();
    if (handle == nullptr) {
      throw const TurboJpegException('tjInitDecompress failed');
    }
    return TurboJpegSession._(this, handle);
  }
}

/// The loader's reason, one line, at most 200 characters (macOS's dlopen
/// lists every path it tried).
String _firstLine(String text) {
  final i = text.indexOf('\n');
  final line = i < 0 ? text : text.substring(0, i);
  // Dart's own prefix repeats the path.
  final reason = line.replaceFirst(
    RegExp(r"^Invalid argument\(s\): Failed to load dynamic library '[^']*': "),
    '',
  );
  return reason.length <= 200 ? reason : '${reason.substring(0, 200)}…';
}

/// A JPEG TurboJPEG could not decode.
final class TurboJpegException implements Exception {
  const TurboJpegException(this.message);

  final String message;

  @override
  String toString() => message;
}

/// A decoded image: packed RGBA rows, its size, and the JPEG's own size
/// before the decode-time scaling.
final class const TurboJpegImage({
  required final int width,
  required final int height,
  required final int sourceWidth,
  required final int sourceHeight,

  /// `1/2` etc.; `1/1` when not scaled.
  required final int scaleNum,
  required final int scaleDenom,

  /// A view of the session's output buffer: valid until the session's next
  /// decode or close. Copy it (or wrap it in a TransferableTypedData) first.
  required final Uint8List rgba,
});

/// The scale TurboJPEG decodes [width]×[height] at: the largest reduction
/// among 1/2, 1/4 and 1/8 (libjpeg's fast DCT-domain scalings) that keeps
/// the long side at or above [minLongSide]; 1/1 otherwise. The detector
/// letterboxes to 640 and Gemma's image is at most 1024 on its long side,
/// so ≥ 960 loses nothing either needs (1920×1080 → 960×540).
({int num, int denom}) turboJpegScale(
  int width,
  int height, {
  required int minLongSide,
}) {
  final longSide = width > height ? width : height;
  var best = (num: 1, denom: 1);
  for (final denom in const [2, 4, 8]) {
    if (_scaled(longSide, 1, denom) >= minLongSide) {
      best = (num: 1, denom: denom);
    }
  }
  return best;
}

/// TJSCALED(dimension, scalingFactor) from turbojpeg.h.
int _scaled(int dimension, int num, int denom) =>
    (dimension * num + denom - 1) ~/ denom;

/// One `tjhandle` and its input/output buffers, grown as needed and reused
/// frame after frame (no allocation per frame once warm).
final class TurboJpegSession {
  TurboJpegSession._(this._lib, this._handle);

  final TurboJpegLibrary _lib;
  Pointer<Void> _handle;
  Pointer<Uint8> _in = nullptr;
  int _inCapacity = 0;
  Pointer<Uint8> _out = nullptr;
  int _outCapacity = 0;
  final Pointer<Int> _dims = calloc<Int>(4);

  /// Decodes [jpeg] to RGBA at [turboJpegScale]. Throws
  /// [TurboJpegException] when the data is not a decodable JPEG; a
  /// recoverable warning (a slightly damaged frame) still returns it.
  TurboJpegImage decode(Uint8List jpeg, {required int minLongSide}) {
    if (_handle == nullptr) {
      throw const TurboJpegException('The TurboJPEG session is closed');
    }
    if (jpeg.isEmpty) throw const TurboJpegException('An empty JPEG');
    if (jpeg.length > _inCapacity) {
      // The new buffer first: a failed allocation leaves the old one valid.
      final capacity = jpeg.length + (jpeg.length >> 2);
      final grown = malloc<Uint8>(capacity);
      if (_in != nullptr) malloc.free(_in);
      _in = grown;
      _inCapacity = capacity;
    }
    _in.asTypedList(jpeg.length).setAll(0, jpeg);
    _dims[0] = 0;
    _dims[1] = 0;
    // The header read reports warnings (stray bytes before a marker, a JFIF
    // revision it does not know) as -1 too: only a fatal error or no size is
    // "not a JPEG".
    if (_lib._header(
              _handle,
              _in,
              jpeg.length,
              _dims,
              _dims + 1,
              _dims + 2,
              _dims + 3,
            ) !=
            0 &&
        (_lib._errorCode(_handle) != kTjErrWarning ||
            _dims[0] <= 0 ||
            _dims[1] <= 0)) {
      throw TurboJpegException('Not a JPEG: ${_error()}');
    }
    final srcW = _dims[0];
    final srcH = _dims[1];
    final scale = turboJpegScale(srcW, srcH, minLongSide: minLongSide);
    final w = _scaled(srcW, scale.num, scale.denom);
    final h = _scaled(srcH, scale.num, scale.denom);
    if (w * h > kTurboJpegMaxPixels) {
      throw TurboJpegException(
        'A ${srcW}x$srcH JPEG decodes to ${w}x$h, over the 4096×4096 limit',
      );
    }
    final bytes = w * h * 4;
    if (bytes > _outCapacity) {
      final grown = malloc<Uint8>(bytes);
      if (_out != nullptr) malloc.free(_out);
      _out = grown;
      _outCapacity = bytes;
    }
    if (_lib._decompress(
              _handle,
              _in,
              jpeg.length,
              _out,
              w,
              w * 4,
              h,
              kTjpfRgba,
              0,
            ) !=
            0 &&
        _lib._errorCode(_handle) != kTjErrWarning) {
      throw TurboJpegException('JPEG decode failed: ${_error()}');
    }
    return TurboJpegImage(
      width: w,
      height: h,
      sourceWidth: srcW,
      sourceHeight: srcH,
      scaleNum: scale.num,
      scaleDenom: scale.denom,
      rgba: _out.asTypedList(bytes),
    );
  }

  String _error() {
    final text = _lib._errorStr(_handle);
    return text == nullptr ? 'unknown error' : text.toDartString();
  }

  /// Frees the handle and the buffers. Safe to call twice.
  void close() {
    if (_handle == nullptr) return;
    _lib._destroy(_handle);
    _handle = nullptr;
    if (_in != nullptr) malloc.free(_in);
    if (_out != nullptr) malloc.free(_out);
    _in = _out = nullptr;
    _inCapacity = _outCapacity = 0;
    calloc.free(_dims);
  }
}
