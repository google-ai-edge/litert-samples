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

import 'dart:async';
import 'dart:isolate';
import 'dart:ui' as ui;

import 'package:flutter/foundation.dart';

import '../../../config/live_camera_config.dart';
import '../../../utils/result.dart';
import '../../../utils/worker_channel.dart';
import 'engine_image_decoder.dart';
import 'turbojpeg.dart';

/// One network frame decoded to what the detector reads.
final class const RgbaFrame({
  required final int width,
  required final int height,

  /// `width × height × 4`, packed rows, R G B A.
  required final Uint8List rgba,

  /// The engine decoder's preview image (owned by the receiver, who must
  /// dispose it); null for a CPU decoder, whose preview is made from [rgba].
  final ui.Image? image,

  /// The JPEG's own size, before any decode-time scaling.
  required final int sourceWidth,
  required final int sourceHeight,

  /// The decode itself, as the decoder measured it.
  required final Duration decodeTime,
});

/// Decodes the network camera's JPEGs.
abstract interface class JpegFrameDecoder {
  /// `TurboJPEG (worker isolate) /usr/lib/…/libturbojpeg.so.0` or
  /// `engine codec`: the log line, the integration test's report.
  String get description;

  /// The engine codec where a CPU decoder was expected (Linux without
  /// libturbojpeg): the source labels itself "slow JPEG decoder".
  bool get slow;

  /// Why the CPU decoder is not used; null when it is, or not expected.
  String? get fallbackReason;

  /// Why the decoder itself stopped working (its isolate died); null while
  /// it works. Every later frame fails: the source reports this, not "the
  /// camera's frames do not decode".
  String? get failure;

  /// Decodes [jpeg]. [maxSide] bounds the engine decoder's output; the CPU
  /// decoder scales down only while the long side stays ≥
  /// [kNetworkDecodeMinSide]. Throws when the data does not decode.
  Future<RgbaFrame> decode(Uint8List jpeg, {required int maxSide});

  /// Releases the decoder (the CPU decoder's isolate and buffers).
  Future<void> close();
}

/// Opens the decoder for this platform.
typedef JpegDecoderFactory = Future<JpegFrameDecoder> Function();

/// The engine's codecs ([decodeEncodedImage]): the decode and the RGBA
/// read-back go through the engine's IO and raster threads and a GPU
/// texture. Fast with a real GPU (macOS: 4–9 ms per frame), very slow under
/// a software GL (Xvfb: ~900 ms per 640×480 frame).
final class EngineJpegDecoder implements JpegFrameDecoder {
  const EngineJpegDecoder({this.slow = false, this.fallbackReason});

  @override
  final bool slow;

  @override
  final String? fallbackReason;

  @override
  String get description => 'engine codec';

  @override
  String? get failure => null;

  @override
  Future<RgbaFrame> decode(Uint8List jpeg, {required int maxSide}) async {
    final watch = Stopwatch()..start();
    final decoded = await decodeEncodedImage(jpeg, maxSide: maxSide);
    return RgbaFrame(
      width: decoded.image.width,
      height: decoded.image.height,
      rgba: decoded.rgba,
      image: decoded.image,
      // The engine does not say; the decoded size is the best there is.
      sourceWidth: decoded.image.width,
      sourceHeight: decoded.image.height,
      decodeTime: watch.elapsed,
    );
  }

  @override
  Future<void> close() async {}
}

/// The platform's decoder: TurboJPEG on a worker isolate when the library
/// loads (Linux: the bundle's copy or the system's; macOS: Homebrew's when
/// the app may read it), else the engine codec. On Linux the engine codec
/// is the slow fallback and says so (log line, [JpegFrameDecoder.slow]);
/// elsewhere it is the designed decoder.
Future<JpegFrameDecoder> openJpegDecoder({
  TargetPlatform? platform,
  List<String>? candidates,
  int minLongSide = kNetworkDecodeMinSide,
}) async {
  final target = platform ?? defaultTargetPlatform;
  final paths = candidates ?? turboJpegCandidates();
  final failures = <String>[];
  final lib = paths.isEmpty
      ? null
      : TurboJpegLibrary.open(paths, failures: failures);
  final expected = target == TargetPlatform.linux;
  if (lib != null) {
    switch (await TurboJpegDecoder.start(
      libraryPath: lib.path,
      minLongSide: minLongSide,
    )) {
      case Ok(value: final decoder):
        debugPrint('[NetworkCamera] JPEG decoder: ${decoder.description}');
        return decoder;
      case Error(:final error):
        failures.add('${lib.path}: $error');
    }
  }
  final reason = paths.isEmpty
      ? 'no TurboJPEG on ${target.name}'
      : 'libturbojpeg did not load (${failures.join('; ')})';
  if (expected) {
    debugPrint(
      '[NetworkCamera] SLOW JPEG decoder: the engine codec, because $reason. '
      'Install libturbojpeg (Debian/Ubuntu: sudo apt install libturbojpeg; '
      'Raspberry Pi OS: libturbojpeg0) or use a bundle that ships it.',
    );
  } else {
    debugPrint('[NetworkCamera] JPEG decoder: engine codec ($reason)');
  }
  return EngineJpegDecoder(slow: expected, fallbackReason: reason);
}

/// TurboJPEG on a long-lived worker isolate: JPEG bytes go in and RGBA
/// comes back as [TransferableTypedData] (one copy out of the native
/// buffer, none on arrival). The main isolate never runs the decode.
final class TurboJpegDecoder implements JpegFrameDecoder {
  TurboJpegDecoder._(this._libraryPath, this._minLongSide, this._worker);

  final String _libraryPath;
  final int _minLongSide;
  final WorkerChannel _worker;

  @override
  String get description =>
      'TurboJPEG (worker isolate, scale to ≥$_minLongSide px) $_libraryPath';

  @override
  bool get slow => false;

  @override
  String? get fallbackReason => null;

  @override
  String? get failure => _worker.failure;

  /// Spawns the worker and opens [libraryPath] there; the error says why it
  /// could not start.
  static Future<Result<TurboJpegDecoder>> start({
    required String libraryPath,
    required int minLongSide,
  }) async {
    try {
      final worker = await WorkerChannel.spawn(
        protocol: workerProtocol,
        entryPoint: _turboJpegWorker,
        boot: (replyTo) => (replyTo, libraryPath, minLongSide),
        handshakeTimeout: const Duration(seconds: 5),
      );
      return Result.ok(TurboJpegDecoder._(libraryPath, minLongSide, worker));
    } on TurboJpegException catch (e) {
      return Result.error(e);
    }
  }

  @override
  Future<RgbaFrame> decode(Uint8List jpeg, {required int maxSide}) async {
    final reply = await _worker.request((id) => (id, transferBytes(jpeg)));
    // _parseReply passes no other value.
    final (_, width, height, sourceWidth, sourceHeight, rgba, micros) =
        reply as _DecodedReply;
    return RgbaFrame(
      width: width,
      height: height,
      rgba: receiveBytes(rgba),
      sourceWidth: sourceWidth,
      sourceHeight: sourceHeight,
      decodeTime: Duration(microseconds: micros),
    );
  }

  /// Asks the worker to free its handle and buffers and exit; kills it if
  /// it has not within 2 s (a decode stuck in native code). Safe to call
  /// twice.
  @override
  Future<void> close() => _worker.close(timeout: const Duration(seconds: 2));

  /// [_turboJpegWorker]'s wire format, the [TurboJpegException] texts and
  /// the `[NetworkCamera]` log lines.
  @visibleForTesting
  static const workerProtocol = WorkerProtocol(
    name: 'turbojpeg',
    noun: 'the decoder isolate',
    parseReply: _parseReply,
    closeMessage: _closeMessage,
    failure: _failure,
    log: _log,
  );

  static WorkerReply? _parseReply(Object? message) => switch (message) {
    final _DecodedReply decoded => (
      id: decoded.$1,
      value: decoded,
      error: null,
    ),
    (final int id, final String error) => (id: id, value: null, error: error),
    _ => null,
  };

  /// `null`: the worker frees its session and exits without answering.
  static Object? _closeMessage(int id) => null;

  static Exception _failure(WorkerFailure kind, String reason) =>
      TurboJpegException(switch (kind) {
        WorkerFailure.notRunning => 'The decoder is not running ($reason)',
        WorkerFailure.startFailed ||
        WorkerFailure.replyError ||
        WorkerFailure.lost => reason,
      });

  static void _log(WorkerEvent event, String detail) {
    final line = switch (event) {
      WorkerEvent.unexpectedMessage => 'unexpected decoder message: $detail',
      WorkerEvent.crashed => 'decoder isolate error: $detail',
      WorkerEvent.closeTimedOut => 'the decoder isolate did not exit: kill',
      WorkerEvent.died || WorkerEvent.closeFailed => null,
    };
    if (line != null) debugPrint('[NetworkCamera] $line');
  }
}

/// A decoded frame, worker → main: the request id, width, height, the
/// JPEG's own width and height, the RGBA rows, the decode time in µs.
typedef _DecodedReply = (int, int, int, int, int, TransferableTypedData, int);

/// The worker: opens the library, then decodes until a `null` message.
void _turboJpegWorker((SendPort, String, int) boot) {
  final (replies, libraryPath, minLongSide) = boot;
  final failures = <String>[];
  final lib = TurboJpegLibrary.open([libraryPath], failures: failures);
  if (lib == null) {
    replies.send(failures.join('; '));
    return;
  }
  final TurboJpegSession session;
  try {
    session = lib.session();
  } catch (e) {
    replies.send('$e');
    return;
  }
  final requests = ReceivePort('turbojpeg requests');
  replies.send(requests.sendPort);
  requests.listen((message) {
    switch (message) {
      case null:
        session.close();
        requests.close();
      case (final int id, final TransferableTypedData jpeg):
        final watch = Stopwatch()..start();
        try {
          final image = session.decode(
            receiveBytes(jpeg),
            minLongSide: minLongSide,
          );
          final micros = watch.elapsedMicroseconds;
          replies.send((
            id,
            image.width,
            image.height,
            image.sourceWidth,
            image.sourceHeight,
            transferBytes(image.rgba),
            micros,
          ));
        } catch (e) {
          replies.send((id, '$e'));
        }
    }
  });
}
