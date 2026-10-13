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
import 'dart:io';
import 'dart:ui' as ui;

import 'package:flutter/foundation.dart';

import '../../../config/live_camera_config.dart';
import '../../../domain/models/camera_source.dart' show networkCameraLabel;
import '../../../domain/models/frame_source_info.dart';
import '../../../domain/models/frame_source_spec.dart';
import '../../../domain/models/preview_source.dart';
import '../../../utils/redact_url.dart';
import '../../../utils/result.dart';
import '../../../utils/stall_watchdog.dart';
import 'engine_image_decoder.dart';
import 'frame_source.dart';
import 'jpeg_decoder.dart';
import 'mjpeg_parser.dart';

/// Builds the HTTP client for one connection.
typedef HttpClientFactory = HttpClient Function();

/// The boundary of an HTTP response that is an MJPEG stream, or why it is not
/// one, worded for the user. An empty boundary means "read it from the body"
/// (`application/octet-stream`, as ffmpeg's HTTP server sends;
/// [MjpegParser.detect] then fails fast unless the body starts with a boundary
/// line).
Result<String> mjpegBoundaryOf({
  required Uri url,
  required int status,
  String reason = '',
  String? contentType,
  String? authenticate,
}) {
  final label = networkCameraLabel(url);
  final path = url.path.isEmpty ? '/' : url.path;
  FrameSourceUnavailableException fail(String message) =>
      FrameSourceUnavailableException(message);
  if (status == HttpStatus.unauthorized) {
    final realm = authenticate == null ? '' : ' ($authenticate)';
    return Result.error(
      fail(
        url.userInfo.isEmpty
            ? '$label asks for a login (HTTP 401$realm). Turn the login '
                  'off in IP Webcam, or put the user and password in the URL '
                  '(http://user:password@host:port$path: saved on this '
                  'device, and sent in clear over http), then Reconnect.'
            : '$label rejected the user and password in the URL (HTTP '
                  '401$realm). Check them, then Reconnect.',
      ),
    );
  }
  if (status == HttpStatus.forbidden) {
    return Result.error(fail('$label refuses access to $path (HTTP 403).'));
  }
  if (status == HttpStatus.notFound) {
    return Result.error(
      fail(
        '$label has nothing at $path (HTTP 404). IP Webcam serves its MJPEG '
        'stream at /video.',
      ),
    );
  }
  if (status < 200 || status > 299) {
    return Result.error(
      fail('$label answered HTTP $status${reason.isEmpty ? '' : ' $reason'}.'),
    );
  }
  if (contentType == null || contentType.trim().isEmpty) {
    return Result.error(
      fail(
        '$label sent no Content-Type at $path: not an MJPEG stream '
        '(multipart/x-mixed-replace).',
      ),
    );
  }
  final (:mediaType, :parameters) = parseContentType(contentType);
  switch (mediaType) {
    case 'multipart/x-mixed-replace' || 'multipart/mixed':
      final boundary = parameters['boundary'];
      if (boundary == null || boundary.replaceAll('-', '').isEmpty) {
        return Result.error(
          fail('$label sent $mediaType without a boundary: cannot split it.'),
        );
      }
      return Result.ok(boundary);
    case 'text/html':
      return Result.error(
        fail(
          '$label serves a web page (text/html) at $path, not an MJPEG '
          'stream. IP Webcam serves the stream at http://<phone>:8080/video.',
        ),
      );
    case 'application/octet-stream':
      return const Result.ok('');
    case 'image/jpeg' || 'image/jpg':
      return Result.error(
        fail(
          '$label serves a single JPEG image at $path, not a stream. IP '
          'Webcam serves the stream at /video.',
        ),
      );
    default:
      return Result.error(
        fail(
          '$label serves $mediaType at $path, not an MJPEG stream '
          '(multipart/x-mixed-replace).',
        ),
      );
  }
}

/// A failure to reach [url] as the message the user sees.
FrameSourceUnavailableException networkCameraFailure(
  Object error,
  Uri url, {
  Duration timeout = kNetworkConnectTimeout,
}) {
  final label = networkCameraLabel(url);
  final where = '${url.host}:${url.port}';
  final seconds = timeout.inMilliseconds / 1000;
  if (error is TimeoutException) {
    return FrameSourceUnavailableException(
      'No answer from $where within ${seconds.toStringAsFixed(0)} s: is the '
      'camera on the same network as this device, and is the address right?',
    );
  }
  if (error is SocketException) {
    final message = '${error.message} ${error.osError?.message ?? ''}'
        .toLowerCase();
    final code = error.osError?.errorCode;
    // errno: ECONNREFUSED 61 (Apple) / 111 (Linux, Android); EHOSTUNREACH
    // 65 / 113; EHOSTDOWN 64 / 112; ENETUNREACH 51 / 101; ETIMEDOUT 60 / 110.
    if (code == 61 || code == 111 || message.contains('refused')) {
      return FrameSourceUnavailableException(
        'Nothing answers at $where (connection refused): is IP Webcam '
        'running ("Start server") and is the port right? Then Reconnect.',
      );
    }
    if (message.contains('failed host lookup') ||
        message.contains('nodename nor servname') ||
        message.contains('name or service not known')) {
      return FrameSourceUnavailableException(
        'Cannot find the host "${url.host}": check the address in the URL.',
      );
    }
    if (code == 65 ||
        code == 113 ||
        code == 64 ||
        code == 112 ||
        code == 51 ||
        code == 101 ||
        message.contains('no route') ||
        message.contains('host is down') ||
        message.contains('unreachable')) {
      return FrameSourceUnavailableException(
        '$where is unreachable from this device (no route): is it on the '
        'same Wi-Fi? On macOS and iOS also allow Local Network for this app '
        '(System Settings › Privacy & Security › Local Network).',
      );
    }
    if (code == 60 || code == 110 || message.contains('timed out')) {
      return FrameSourceUnavailableException(
        'No answer from $where within ${seconds.toStringAsFixed(0)} s: is '
        'the camera on the same network as this device, and is the address '
        'right?',
      );
    }
    return FrameSourceUnavailableException(
      'Cannot reach $label: ${networkErrorText(error)}',
    );
  }
  if (error is HandshakeException) {
    return FrameSourceUnavailableException(
      'The TLS handshake with $where failed (${error.message}). IP Webcam '
      'serves plain http://.',
    );
  }
  if (error is HttpException) {
    return FrameSourceUnavailableException(
      '$where closed the connection before a valid HTTP answer '
      '(${error.message}): is it an HTTP camera?',
    );
  }
  return FrameSourceUnavailableException(
    'Cannot open $label: ${networkErrorText(error)}',
  );
}

/// [error] for a message the user sees: the exception's own message, never
/// its URI (`HttpException.toString` adds it, password included).
String networkErrorText(Object error) => switch (error) {
  HttpException(:final message) => message,
  SocketException(:final message, :final osError) =>
    osError == null ? message : '$message: ${osError.message}',
  TlsException(:final message) => message,
  _ => '$error',
};

/// A network camera as a [FrameSource]: an MJPEG stream over HTTP
/// (`multipart/x-mixed-replace`), e.g. a phone running the free IP Webcam app
/// (`http://<phone>:8080/video`). See docs/architecture.md, Live camera
/// pipeline.
///
/// - **Latest frame wins**: the parser runs on the main isolate (it only
///   copies parts; with `Content-Length` it never scans a body). One JPEG is
///   decoded at a time, off the UI thread: by TurboJPEG on a worker isolate
///   where libturbojpeg loads (always expected on Linux), else by the engine
///   codec ([openJpegDecoder]; on Linux that fallback labels the source
///   "slow JPEG decoder"). While a decode runs only the newest JPEG waits,
///   older ones are dropped.
/// - The preview is made from the decoded pixels separately (latest wins
///   too), so a slow preview upload never holds up the detector.
/// - Frames are RGBA ([FramePixelFormat.rgba8888]), upright, unmirrored —
///   the fixture's format, so the detector and the question snapshot need
///   nothing new.
/// - **Fail fast**: unreachable host, a non-MJPEG answer (an HTML page, a
///   single JPEG, HTTP 401/404), no frame within [kNetworkStallTimeout] of
///   connecting or while running, the server closing the stream, or
///   [kNetworkMaxDecodeFailures] undecodable frames in a row are each one
///   clear error; there is no switch to another camera. Reconnect is a new
///   source (single-use, like every source).
final class NetworkFrameSource with SingleUseStart {
  NetworkFrameSource({
    required this._url,
    HttpClientFactory? httpClient,
    JpegDecoderFactory? decoder,
    this._connectTimeout = kNetworkConnectTimeout,
    this._stallTimeout = kNetworkStallTimeout,
    this._maxSide = kNetworkMaxSide,
    this._maxDecodeFailures = kNetworkMaxDecodeFailures,
    this._watchInterval = const Duration(seconds: 1),
    this._logInterval = kNetworkLogInterval,
  }) : _httpClient = httpClient ?? HttpClient.new,
       _decoderFactory = decoder ?? openJpegDecoder;

  final Uri _url;
  final HttpClientFactory _httpClient;
  final JpegDecoderFactory _decoderFactory;

  /// Open from [start] until [stop].
  JpegFrameDecoder? _decoder;
  final Duration _connectTimeout;
  final Duration _stallTimeout;
  final int _maxSide;
  final int _maxDecodeFailures;
  final Duration _watchInterval;
  final Duration _logInterval;

  /// `Network camera · 192.168.1.23:8080`.
  late final String label = networkCameraLabel(_url);

  final ValueNotifier<ui.Image?> _image = ValueNotifier(null);
  late final ImagePreviewSource _preview = ImagePreviewSource(_image);

  /// The image shown before the current one: `RawImage` holds its own clone,
  /// but it is disposed only on the next replacement, one frame later.
  ui.Image? _previous;

  HttpClient? _client;
  StreamSubscription<List<int>>? _subscription;
  MjpegParser? _parser;

  /// Until the first frame has decoded (inside [start]).
  Completer<RgbaFrame>? _first;

  /// The preview made from a CPU decoder's pixels: one at a time, the
  /// newest waiting frame next.
  bool _previewBusy = false;
  RgbaFrame? _previewPending;
  bool _previewFailureLogged = false;
  Uint8List? _pending;
  bool _decoding = false;
  int _decodeFailures = 0;

  final Stopwatch _clock = Stopwatch();

  /// No JPEG for the stall timeout while running fails the source.
  late final StallWatchdog _watchdog = StallWatchdog(
    interval: _watchInterval,
    clockMicros: () => _clock.elapsedMicroseconds,
  );

  // Counters since start, and at the last log line.
  int _received = 0;
  int _decoded = 0;
  int _droppedBusy = 0;
  int _decodeErrors = 0;
  int _bytes = 0;
  int _lastDecodeMicros = 0;
  int _decodeMicrosSum = 0;
  ({int at, int received, int decoded, int decodeMicros})? _logMark;
  int _width = 0;
  int _height = 0;

  @override
  PreviewSource get preview => _preview;

  @override
  String get sourceKind => 'network camera';

  @override
  FrameSourceUnavailableException get stoppedWhileStarting =>
      FrameSourceUnavailableException('Stopped while connecting to $label');

  /// Start, phase by phase: open the decoder, [_connect], [_parseHeader],
  /// wait for the [_firstFrame], then [_pump]. A phase that fails has
  /// closed what it opened and throws the error start returns. Each phase
  /// after the first begins with [_throwIfStopped].
  @override
  Future<Result<FrameSourceInfo>> acquire(
    void Function(FrameView) onFrame, {
    void Function(Exception error)? onError,
  }) async {
    _clock.start();
    try {
      final decoder = await _openDecoder();
      final response = await _connect();
      final contentType = await _parseHeader(response);
      final frame = await _firstFrame(response);
      return Result.ok(_pump(decoder, contentType, frame, onFrame, onError));
    } on FrameSourceUnavailableException catch (e) {
      return Result.error(e);
    }
  }

  /// The JPEG decoder, open from here until [release].
  Future<JpegFrameDecoder> _openDecoder() async {
    final JpegFrameDecoder decoder;
    try {
      decoder = await _decoderFactory();
    } catch (e) {
      throw FrameSourceUnavailableException('No JPEG decoder for $label: $e');
    }
    if (stopped) {
      // stop() ran while the decoder opened and could not close it.
      await decoder.close();
      throw stoppedWhileStarting;
    }
    return _decoder = decoder;
  }

  /// Where a phase begins. stop() may have run between two phases: an
  /// async phase that never suspended resumes [acquire] only after the
  /// microtasks already queued. [release] has then closed what the earlier
  /// phases opened, and the next phase must open nothing.
  void _throwIfStopped() {
    if (stopped) throw stoppedWhileStarting;
  }

  /// Connects and sends the GET; the response's headers have arrived.
  Future<HttpClientResponse> _connect() async {
    _throwIfStopped();
    final client = _client = _httpClient()
      ..connectionTimeout = _connectTimeout
      // A camera on the local network is reached directly, never through
      // an http_proxy from the environment.
      ..findProxy = ((_) => 'DIRECT');
    final HttpClientResponse response;
    var connected = false;
    try {
      final request = await client.getUrl(_url);
      connected = true;
      request.headers.set(
        HttpHeaders.acceptHeader,
        'multipart/x-mixed-replace, image/jpeg;q=0.5, */*;q=0.1',
      );
      response = await request.close().timeout(_connectTimeout);
    } catch (e) {
      if (stopped) throw stoppedWhileStarting;
      await _closeNetwork();
      debugPrint(
        '[NetworkCamera] connecting to $label failed: '
        '${e.runtimeType}: ${networkErrorText(e)}',
      );
      if (connected && e is TimeoutException) {
        throw FrameSourceUnavailableException(
          'Connected to ${_url.host}:${_url.port}, but it sent no HTTP '
          'answer within ${_connectTimeout.inSeconds} s: is it an HTTP '
          'camera?',
        );
      }
      throw networkCameraFailure(e, _url, timeout: _connectTimeout);
    }
    if (stopped) {
      await _closeNetwork();
      throw stoppedWhileStarting;
    }
    return response;
  }

  /// Checks that the answer is an MJPEG stream and makes the parser for its
  /// boundary. Returns the Content-Type, for the log.
  Future<String?> _parseHeader(HttpClientResponse response) async {
    _throwIfStopped();
    final contentType = response.headers.value(HttpHeaders.contentTypeHeader);
    final String boundary;
    switch (mjpegBoundaryOf(
      url: _url,
      status: response.statusCode,
      reason: response.reasonPhrase,
      contentType: contentType,
      authenticate: response.headers.value(HttpHeaders.wwwAuthenticateHeader),
    )) {
      case Ok(:final value):
        boundary = value;
      case Error(:final error):
        debugPrint(
          '[NetworkCamera] $label HTTP ${response.statusCode} '
          'content-type="$contentType": $error',
        );
        await _closeNetwork();
        throw error;
    }
    try {
      _parser = boundary.isEmpty ? MjpegParser.detect() : MjpegParser(boundary);
    } on MjpegStreamException catch (e) {
      await _closeNetwork();
      throw FrameSourceUnavailableException('$label: $e');
    }
    return contentType;
  }

  /// Reads the body and waits, at most the stall timeout, for the first
  /// JPEG to decode ([_decodeOne] completes [_first] with its pixels, the
  /// preview keeps its image; a failure or stop fails it).
  Future<RgbaFrame> _firstFrame(HttpClientResponse response) async {
    _throwIfStopped();
    final first = _first = Completer<RgbaFrame>();
    _subscription = response.listen(
      _onChunk,
      onError: _onStreamError,
      onDone: _onStreamDone,
      cancelOnError: true,
    );
    final RgbaFrame frame;
    try {
      frame = await first.future.timeout(_stallTimeout);
    } on TimeoutException {
      markFailed();
      await _closeNetwork();
      throw FrameSourceUnavailableException(
        'Connected to $label, but no JPEG frame arrived within '
        '${_stallTimeout.inSeconds} s'
        '${_received > 0 ? ' ($_received received, none decoded)' : ''}.',
      );
    } finally {
      _first = null;
    }
    // Nothing to dispose on a stop: the frame has no image.
    _throwIfStopped();
    return frame;
  }

  /// Running: frames go to [onFrame] and the one later error to [onError],
  /// the stall watchdog is armed, and [first] goes out once start has
  /// returned; the chunks and decodes that follow pump the rest.
  FrameSourceInfo _pump(
    JpegFrameDecoder decoder,
    String? contentType,
    RgbaFrame first,
    void Function(FrameView) onFrame,
    void Function(Exception error)? onError,
  ) {
    // [first] has no image (the preview's, [_decodeOne]).
    _throwIfStopped();
    _width = first.width;
    _height = first.height;
    debugPrint(
      'NETCAM url=${redactUrl(_url)} '
      'content-type="$contentType" ${_width}x$_height '
      '(jpeg ${first.sourceWidth}x${first.sourceHeight}) first frame in '
      '${_clock.elapsedMilliseconds} ms '
      'decode=${(first.decodeTime.inMicroseconds / 1000).toStringAsFixed(1)}ms '
      'decoder="${decoder.description}"'
      '${decoder.slow ? ' SLOW: ${decoder.fallbackReason}' : ''}',
    );
    attach(onFrame, onError: onError);
    _armWatchdog();
    // The first frame goes out after start() has returned.
    Timer(Duration.zero, () {
      if (stopped || failed) return;
      _deliver(first, showPreview: false);
    });
    return FrameSourceInfo(
      // The engine codec where TurboJPEG was expected (Linux) is a
      // degraded mode: the chip and the overlay say so.
      label: decoder.slow ? '$label · slow JPEG decoder' : label,
      decoder: decoder.description,
      width: _width,
      height: _height,
      format: FramePixelFormat.rgba8888,
      mirrored: false,
      // Backstop only: this source reports its own stall first.
      stallTimeout: _stallTimeout + _watchInterval * 2,
    );
  }

  @override
  Future<void> release() async {
    _pending = null;
    _previewPending = null;
    final first = _first;
    if (first != null && !first.isCompleted) {
      first.completeError(stoppedWhileStarting);
    }
    await _closeNetwork();
    if (_received > 0) _log(force: true);
    final decoder = _decoder;
    _decoder = null;
    try {
      await decoder?.close();
    } catch (e) {
      debugPrint('[NetworkCamera] closing the decoder: $e');
    }
    final shown = _image.value;
    _image.value = null;
    _image.dispose();
    shown?.dispose();
    _previous?.dispose();
    _previous = null;
  }

  void _onChunk(List<int> chunk) {
    if (stopped || failed) return;
    _bytes += chunk.length;
    final List<Uint8List> jpegs;
    try {
      jpegs = _parser!.add(chunk);
    } on MjpegStreamException catch (e) {
      _fail('$label is not sending MJPEG: $e');
      return;
    }
    if (jpegs.isEmpty) return;
    _watchdog.activity(_clock.elapsedMicroseconds);
    _received += jpegs.length;
    // Only the newest JPEG of a chunk can be shown.
    _droppedBusy += jpegs.length - 1;
    _submit(jpegs.last);
  }

  /// Latest frame wins: decode now, or wait as the only pending JPEG.
  void _submit(Uint8List jpeg) {
    if (_decoding) {
      if (_pending != null) _droppedBusy++;
      _pending = jpeg;
      return;
    }
    unawaited(_decodeOne(jpeg));
  }

  /// Never throws: a failure is counted and, repeated, fails the source.
  Future<void> _decodeOne(Uint8List jpeg) async {
    _decoding = true;
    try {
      final decoder = _decoder;
      if (decoder == null) return; // stopped
      final frame = await decoder.decode(jpeg, maxSide: _maxSide);
      _lastDecodeMicros = frame.decodeTime.inMicroseconds;
      _decodeMicrosSum += _lastDecodeMicros;
      if (stopped || failed) {
        frame.image?.dispose();
        return;
      }
      _decodeFailures = 0;
      _decoded++;
      final first = _first;
      if (first != null && !first.isCompleted) {
        // One owner: the image goes to the preview ([release] disposes it),
        // start gets the pixels only and delivers them.
        _updatePreview(frame);
        first.complete(_pixelsOnly(frame));
      } else {
        _deliver(frame);
      }
    } catch (e) {
      if (stopped || failed) return;
      if (_decoder?.failure case final reason?) {
        // The decoder died, not the camera's JPEGs: say so, once.
        _fail('The JPEG decoder stopped ($reason). Press Reconnect.');
        return;
      }
      _decodeErrors++;
      _decodeFailures++;
      debugPrint(
        '[NetworkCamera] a frame did not decode: ${networkErrorText(e)}',
      );
      if (_decodeFailures >= _maxDecodeFailures) {
        _fail(
          '$label sends frames that do not decode as JPEG '
          '($_decodeFailures in a row): $e',
        );
      }
    } finally {
      _decoding = false;
      final next = _pending;
      _pending = null;
      if (next != null && !stopped && !failed) unawaited(_decodeOne(next));
    }
  }

  /// The detector first (the view is copied there), then the preview.
  void _deliver(RgbaFrame frame, {bool showPreview = true}) {
    final onFrame = frameCallback;
    if (onFrame != null) {
      if (frame.width != _width || frame.height != _height) {
        debugPrint(
          '[NetworkCamera] resolution ${_width}x$_height → '
          '${frame.width}x${frame.height}',
        );
        _width = frame.width;
        _height = frame.height;
      }
      try {
        onFrame(_NetworkFrame(frame));
      } catch (e, st) {
        FlutterError.reportError(
          FlutterErrorDetails(
            exception: e,
            stack: st,
            library: 'network camera',
          ),
        );
      }
    }
    if (showPreview) _updatePreview(frame);
  }

  /// The engine decoder's image as it is; a CPU decoder's pixels become an
  /// image on the engine (an upload, latest wins), never in the way of
  /// the next decode.
  void _updatePreview(RgbaFrame frame) {
    if (frame.image case final image?) {
      _showPreview(image);
      return;
    }
    if (_previewBusy) {
      _previewPending = frame;
      return;
    }
    unawaited(_makePreview(frame));
  }

  Future<void> _makePreview(RgbaFrame frame) async {
    _previewBusy = true;
    try {
      final image = await imageFromRgba(frame.rgba, frame.width, frame.height);
      if (stopped) {
        image.dispose();
        return;
      }
      _showPreview(image);
    } catch (e) {
      // The detector still gets every frame; only the picture is stale.
      if (!_previewFailureLogged) {
        _previewFailureLogged = true;
        debugPrint('[NetworkCamera] the preview image failed: $e');
      }
    } finally {
      _previewBusy = false;
      final next = _previewPending;
      _previewPending = null;
      if (next != null && !stopped) unawaited(_makePreview(next));
    }
  }

  void _showPreview(ui.Image image) {
    final older = _previous;
    _previous = _image.value;
    _image.value = image;
    older?.dispose();
  }

  void _onStreamError(Object error) {
    _fail(
      'Lost the connection to $label (${networkErrorText(error)}). Press '
      'Reconnect.',
    );
  }

  void _onStreamDone() {
    _fail(
      '$label closed the stream (IP Webcam stopped, or the phone went to '
      'sleep?). Press Reconnect.',
    );
  }

  void _armWatchdog() {
    final armedAt = _watchdog.arm(
      timeout: () => _stallTimeout,
      cancelIf: () => stopped || failed,
      onStall: (silent) => _fail(
        '$label stopped sending frames (none for '
        '${(silent / 1e6).toStringAsFixed(1)} s): is the phone awake, IP '
        'Webcam still running and the Wi-Fi up? Press Reconnect.',
      ),
      onHealthy: _log,
    );
    _logMark = (
      at: armedAt,
      received: _received,
      decoded: _decoded,
      decodeMicros: _decodeMicrosSum,
    );
  }

  /// One line every [_logInterval]: what arrived, what was decoded, what
  /// was dropped while the decoder was busy.
  void _log({bool force = false}) {
    final now = _clock.elapsedMicroseconds;
    final mark = _logMark ?? (at: 0, received: 0, decoded: 0, decodeMicros: 0);
    if (!force && now - mark.at < _logInterval.inMicroseconds) return;
    final seconds = (now - mark.at) / 1e6;
    String fps(int n) => seconds <= 0 ? '–' : (n / seconds).toStringAsFixed(1);
    final decodedNow = _decoded - mark.decoded;
    final avgDecode = decodedNow == 0
        ? '–'
        : ((_decodeMicrosSum - mark.decodeMicros) / decodedNow / 1000)
              .toStringAsFixed(1);
    debugPrint(
      '[NetworkCamera] ${_url.host}:${_url.port} ${_width}x$_height '
      'received=$_received (${fps(_received - mark.received)} fps) '
      'decoded=$_decoded (${fps(decodedNow)} fps) '
      'dropped=$_droppedBusy decode_errors=$_decodeErrors '
      'avg_jpeg=${_received == 0 ? 0 : _bytes ~/ _received ~/ 1024} KB '
      'decode_avg=${avgDecode}ms decoder="${_decoder?.description ?? '–'}"',
    );
    _logMark = (
      at: now,
      received: _received,
      decoded: _decoded,
      decodeMicros: _decodeMicrosSum,
    );
  }

  /// Average decode time per frame since start; null before the first.
  @visibleForTesting
  Duration? get averageDecodeTime => _decoded == 0
      ? null
      : Duration(microseconds: _decodeMicrosSum ~/ _decoded);

  /// JPEGs parsed from the stream since start (decoded or dropped).
  @visibleForTesting
  int get receivedFrames => _received;

  /// Reports [message] once (through start's wait, or `onError`) and closes
  /// the connection; the owner still calls [stop].
  void _fail(String message) {
    if (!markFailed()) return;
    debugPrint('[NetworkCamera] $message');
    final error = FrameSourceUnavailableException(message);
    final first = _first;
    if (first != null && !first.isCompleted) {
      first.completeError(error);
    } else {
      errorCallback?.call(error);
    }
    unawaited(_closeNetwork());
  }

  /// Cancels the response stream and closes the client. Never throws.
  Future<void> _closeNetwork() async {
    _watchdog.disarm();
    final client = _client;
    _client = null;
    try {
      // Cancelling twice (a failure, then stop) is harmless.
      await _subscription?.cancel();
    } catch (e) {
      debugPrint('[NetworkCamera] cancelling the stream: $e');
    }
    _subscription = null;
    try {
      client?.close(force: true);
    } catch (e) {
      debugPrint('[NetworkCamera] closing the client: $e');
    }
  }
}

/// [frame] without its image: what start delivers once the preview owns
/// the image.
RgbaFrame _pixelsOnly(RgbaFrame frame) => RgbaFrame(
  width: frame.width,
  height: frame.height,
  rgba: frame.rgba,
  sourceWidth: frame.sourceWidth,
  sourceHeight: frame.sourceHeight,
  decodeTime: frame.decodeTime,
);

/// One decoded network frame as a [FrameView].
final class _NetworkFrame implements FrameView {
  _NetworkFrame(RgbaFrame frame)
    : width = frame.width,
      height = frame.height,
      planes = [
        FramePlane(
          bytes: frame.rgba,
          bytesPerRow: frame.width * 4,
          bytesPerPixel: 4,
        ),
      ];

  @override
  final int width;

  @override
  final int height;

  @override
  final List<FramePlane> planes;

  @override
  FramePixelFormat get format => FramePixelFormat.rgba8888;

  @override
  int get rotationDeg => 0;
}
