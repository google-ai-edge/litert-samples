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
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/rendering.dart';
import 'package:flutter/widgets.dart';

/// Records the act to a video from inside the app (`ACT3_RECORD`): grabs the
/// stage's repaint boundary [fps] times a second and pipes raw RGBA frames to
/// ffmpeg. A terminal without the Screen Recording permission cannot film the
/// window, but the app can always read back its own frames.
///
/// Real time is kept: a tick that finds [maxInFlight] grabs still running
/// repeats the last frame, so the video runs at the speed the act did. The
/// read-back costs GPU time, so a recorded run is for showing, not measuring.
///
/// Grabs may overlap ([maxInFlight]) and are written in tick order. Two help
/// when the GPU has headroom: a 1440p grab takes ~21 ms but its read-back
/// queues behind the narrator's work now and then, and one at a time repeated
/// 5.7 % of frames where two repeated 2.2 %. On a saturated GPU more grabs
/// only add load (four at render scale 1.0: 28 % repeated, scene 75 fps).
///
/// A recording never takes the act down with it: if ffmpeg is missing, the
/// encoder is unknown or the pipe breaks, recording stops and [error] says
/// why; the act and its results carry on.
class FrameRecorder {
  FrameRecorder({required this.path, this.fps = 30, this.pixelRatio = 1.0, this.maxInFlight = 2});

  final String path;
  final int fps;

  /// Video pixels per logical pixel. The scene already renders at the
  /// screen's device pixel ratio, so on a Retina Mac 2.0 records the frame as
  /// drawn (a 1280×720 window → 2560×1440) and 1.0 halves it.
  final double pixelRatio;

  final int maxInFlight;

  /// Frames waiting for ffmpeg, beyond which ticks repeat the last frame
  /// instead of holding new ones (14.7 MB each at 1440p).
  static const maxQueued = 30;

  Process? _ffmpeg;
  Timer? _timer;
  Uint8List? _last;
  int _inFlight = 0;
  int _queued = 0;
  Future<void> _written = Future.value();
  final List<int> _grabMs = [];
  final List<String> _ffmpegLog = [];
  int? _width;
  int? _height;
  int? get width => _width;
  int? get height => _height;
  int frames = 0;
  int repeated = 0;

  /// Why recording stopped early, or why ffmpeg failed. Null for a good file.
  String? error;

  /// Grabbing runs from [start]; frames reach the file only once writing is
  /// on. Grabbing before writing lets a calibration pass see the read-back's
  /// GPU cost, so a recorded "100 % of the frame budget" is the real thing.
  bool writing = true;

  /// Milliseconds from a tick to its frame's bytes: p50, p95 and max.
  String get grabLatency {
    if (_grabMs.isEmpty) return 'no grabs';
    final s = [..._grabMs]..sort();
    int at(double q) => s[math.min(s.length - 1, (s.length * q).floor())];
    return 'grab ms p50 ${at(0.5)} p95 ${at(0.95)} max ${s.last}';
  }

  void start(GlobalKey boundaryKey) {
    _timer = Timer.periodic(Duration(microseconds: 1000000 ~/ fps), (_) => _tick(boundaryKey));
  }

  void _tick(GlobalKey key) {
    if (error != null) return;
    final write = writing;
    if (_inFlight >= maxInFlight || _queued >= maxQueued) {
      _written = _written.then((_) => _repeat(write));
      return;
    }
    _inFlight++;
    _queued++;
    final grab = _grab(key).whenComplete(() => _inFlight--);
    _written = _written.then((_) async {
      try {
        await _write(await grab, write);
      } finally {
        _queued--;
      }
    });
  }

  /// The layer tree is snapshotted when toImage is called, so a frame shows
  /// its tick's moment however long the read-back then waits for the GPU.
  Future<(Uint8List, int, int)?> _grab(GlobalKey key) async {
    final boundary = key.currentContext?.findRenderObject() as RenderRepaintBoundary?;
    if (boundary == null) return null;
    final clock = Stopwatch()..start();
    try {
      final image = await boundary.toImage(pixelRatio: pixelRatio);
      final data = await image.toByteData(format: ui.ImageByteFormat.rawRgba);
      final w = image.width, h = image.height;
      image.dispose();
      if (data == null) return null;
      _grabMs.add(clock.elapsedMilliseconds);
      return (data.buffer.asUint8List(), w, h);
    } catch (_) {
      // A failed grab is a repeated frame; it must not break the write chain.
      return null;
    }
  }

  Future<void> _write((Uint8List, int, int)? grabbed, bool write) async {
    if (error != null) return;
    if (grabbed == null) return _repeat(write);
    final (bytes, w, h) = grabbed;
    if (_ffmpeg == null) {
      try {
        _ffmpeg = await _startEncoder(w, h);
      } catch (e) {
        _fail('ffmpeg did not start: $e');
        return;
      }
    }
    // A resized window would break the raw stream's frame size.
    if (w != _width || h != _height) return _repeat(write);
    _last = bytes;
    if (!write) return;
    _ffmpeg!.stdin.add(bytes);
    frames++;
    // Backpressure: wait for the pipe before the next frame is written.
    try {
      await _ffmpeg!.stdin.flush();
    } catch (e) {
      _fail('ffmpeg stopped reading: $e');
    }
  }

  void _repeat(bool write) {
    final last = _last;
    final ffmpeg = _ffmpeg;
    if (last == null || !write || ffmpeg == null || error != null) return;
    ffmpeg.stdin.add(last);
    repeated++;
  }

  void _fail(String why) {
    error ??= why;
    _timer?.cancel();
    _timer = null;
  }

  /// The hardware H.264 encoder each desktop OS ships with.
  static String get _encoder {
    if (Platform.isMacOS) return 'h264_videotoolbox';
    if (Platform.isWindows) return 'h264_mf';
    return 'libx264';
  }

  /// `FFMPEG` in the environment, Homebrew's locations on macOS (an app
  /// started from Finder has no shell PATH), then whatever PATH finds.
  static String get _ffmpegExe {
    final env = Platform.environment['FFMPEG'];
    if (env != null && env.isNotEmpty) return env;
    if (Platform.isMacOS) {
      for (final p in ['/opt/homebrew/bin/ffmpeg', '/usr/local/bin/ffmpeg']) {
        if (File(p).existsSync()) return p;
      }
    }
    return 'ffmpeg';
  }

  Future<Process> _startEncoder(int w, int h) async {
    _width = w;
    _height = h;
    // ~0.35 bits per pixel: 10 Mbit/s at 928×580, ~39 at 1440p30, ~87 at
    // 4K30, above what YouTube asks of an upload at each size.
    final mbps = math.max(10, (w * h * fps * 0.35 / 1e6).round());
    final p = await Process.start(_ffmpegExe, [
      '-y', '-loglevel', 'error',
      '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', '${w}x$h', '-r', '$fps', '-i', '-',
      // H.264 needs even dimensions.
      '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
      '-c:v', _encoder, '-b:v', '${mbps}M', '-pix_fmt', 'yuv420p',
      path,
    ]);
    // Drained so ffmpeg never blocks on a full stderr pipe; kept for [error].
    p.stderr.transform(utf8.decoder).transform(const LineSplitter()).listen((line) {
      _ffmpegLog.add(line);
      if (_ffmpegLog.length > 20) _ffmpegLog.removeAt(0);
    });
    p.stdout.drain<void>();
    unawaited(p.stdin.done.catchError((Object e) => _fail('ffmpeg pipe closed: $e')));
    return p;
  }

  /// Finishes the file. Never throws; check [error] afterwards.
  Future<void> stop() async {
    _timer?.cancel();
    _timer = null;
    try {
      await _written;
    } catch (_) {}
    final ffmpeg = _ffmpeg;
    if (ffmpeg == null) return;
    try {
      await ffmpeg.stdin.close();
    } catch (_) {}
    final code = await ffmpeg.exitCode.timeout(const Duration(seconds: 60), onTimeout: () {
      ffmpeg.kill();
      return -1;
    });
    if (code != 0) {
      error ??= 'ffmpeg exited with $code${_ffmpegLog.isEmpty ? '' : ': ${_ffmpegLog.join(' / ')}'}';
    }
  }
}
