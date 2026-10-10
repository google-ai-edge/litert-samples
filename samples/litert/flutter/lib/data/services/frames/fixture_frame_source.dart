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
import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:flutter/foundation.dart';

import '../../../config/live_camera_config.dart';
import '../../../domain/models/frame_source_info.dart';
import '../../../domain/models/frame_source_spec.dart';
import '../../../domain/models/preview_source.dart';
import '../../../utils/result.dart';
import 'engine_image_decoder.dart';
import 'frame_source.dart';

/// A deterministic stand-in for the camera: a slideshow of still images. Each
/// image is decoded once by the engine (off the UI thread) to RGBA and to a
/// `ui.Image` for the preview; frames are emitted at [fps] on a drift-free
/// schedule, each image held for [hold].
///
/// [mirrored] plays camera_desktop on macOS, which mirrors both its frames
/// and its preview: each image is flipped once at decode, and the source
/// reports `mirrored` and `previewMirrored` (the un-mirroring test).
///
/// Single-use: start once, stop once.
///
/// [clockMicros] and [timer] (a monotonic clock and `Timer.new` by default)
/// let a test drive the schedule.
final class FixtureFrameSource with SingleUseStart {
  FixtureFrameSource({
    required this._paths,
    int fps = kFixtureFps,
    Duration hold = kFixtureHold,
    this._maxSide = kFixtureMaxSide,
    this._maxImages = kFixtureMaxImages,
    this._mirrored = false,
    int Function()? clockMicros,
    Timer Function(Duration delay, void Function() callback)? timer,
  }) : _periodMicros = 1000000 ~/ fps,
       _holdMicros = hold.inMicroseconds,
       _now = clockMicros ?? _monotonicMicros,
       _schedule = timer ?? Timer.new;

  final List<String> _paths;
  final bool _mirrored;
  final int _periodMicros;
  final int _holdMicros;
  final int _maxSide;
  final int _maxImages;
  final int Function() _now;
  final Timer Function(Duration delay, void Function() callback) _schedule;

  static final Stopwatch _monotonic = Stopwatch()..start();
  static int _monotonicMicros() => _monotonic.elapsedMicroseconds;

  final ValueNotifier<ui.Image?> _image = ValueNotifier(null);
  late final ImagePreviewSource _preview = ImagePreviewSource(_image);
  List<_Slide> _slides = const [];

  /// [_now] at start: the schedule counts from here.
  int _startedAt = 0;
  Timer? _timer;
  int _tick = 0;
  int _shown = -1;

  @override
  PreviewSource get preview => _preview;

  /// The image files the slideshow shows, in order.
  @visibleForTesting
  List<String> get imagePaths => [for (final s in _slides) s.path];

  @override
  String get sourceKind => 'fixture';

  @override
  FrameSourceUnavailableException get stoppedWhileStarting =>
      const FrameSourceUnavailableException(
        'Stopped while decoding the fixtures',
      );

  /// Decodes every image, then starts the schedule. Never fails after
  /// start: there is no `onError`.
  @override
  Future<Result<FrameSourceInfo>> acquire(
    void Function(FrameView) onFrame, {
    void Function(Exception error)? onError,
  }) async {
    final List<File> files;
    try {
      files = _listImages();
    } on FrameSourceUnavailableException catch (e) {
      return Result.error(e);
    }
    final slides = <_Slide>[];
    for (final file in files) {
      try {
        slides.add(await _decode(file));
      } catch (e) {
        _dispose(slides);
        return Result.error(
          FrameSourceUnavailableException('Could not decode ${file.path}: $e'),
        );
      }
      if (stopped) {
        _dispose(slides);
        return Result.error(stoppedWhileStarting);
      }
    }
    _slides = slides;
    attach(onFrame);
    _startedAt = _now();
    // The first frame goes out after start() has returned.
    _timer = _schedule(Duration.zero, _onTick);
    final first = slides.first;
    return Result.ok(
      FrameSourceInfo(
        label:
            'fixture (${slides.length} image${slides.length == 1 ? '' : 's'}'
            '${_mirrored ? ', mirrored' : ''})',
        width: first.width,
        height: first.height,
        format: FramePixelFormat.rgba8888,
        mirrored: _mirrored,
        previewMirrored: _mirrored,
      ),
    );
  }

  @override
  Future<void> release() async {
    _timer?.cancel();
    _timer = null;
    _image.value = null;
    _image.dispose();
    _dispose(_slides); // RawImage holds its own clone
    _slides = const [];
  }

  static void _dispose(List<_Slide> slides) {
    for (final slide in slides) {
      slide.image.dispose();
    }
  }

  void _onTick() {
    _timer = null;
    final onFrame = frameCallback;
    if (stopped || onFrame == null) return;
    final now = _now() - _startedAt;
    final index = (now ~/ _holdMicros) % _slides.length;
    final slide = _slides[index];
    if (index != _shown) {
      _shown = index;
      _image.value = slide.image;
    }
    try {
      onFrame(slide);
    } catch (e, st) {
      FlutterError.reportError(
        FlutterErrorDetails(exception: e, stack: st, library: 'fixture source'),
      );
    }
    if (stopped) return; // stopped from inside the callback
    // Frame k is due at k × period from the start: no drift. After a stall
    // (debugger, long GC) skip the missed frames instead of bursting.
    _tick++;
    final after = _now() - _startedAt;
    if (_tick * _periodMicros < after - _periodMicros) {
      _tick = after ~/ _periodMicros + 1;
    }
    _timer = _schedule(
      Duration(microseconds: math.max(0, _tick * _periodMicros - after)),
      _onTick,
    );
  }

  List<File> _listImages() {
    final files = <File>[];
    for (final path in _paths) {
      final type = FileSystemEntity.typeSync(path);
      if (type == FileSystemEntityType.directory) {
        final images =
            Directory(path)
                .listSync()
                .whereType<File>()
                .where((f) => _isImage(f.path))
                .toList()
              ..sort((a, b) => a.path.compareTo(b.path));
        files.addAll(images);
      } else if (type == FileSystemEntityType.file) {
        files.add(File(path));
      } else {
        throw FrameSourceUnavailableException('Fixture path not found: $path');
      }
    }
    if (files.isEmpty) {
      throw FrameSourceUnavailableException(
        'No images in ${_paths.join(', ')}',
      );
    }
    if (files.length > _maxImages) {
      debugPrint(
        '[FixtureFrameSource] ${files.length} images, using the first '
        '$_maxImages',
      );
      return files.sublist(0, _maxImages);
    }
    return files;
  }

  static bool _isImage(String path) {
    final lower = path.toLowerCase();
    return lower.endsWith('.jpg') ||
        lower.endsWith('.jpeg') ||
        lower.endsWith('.png') ||
        lower.endsWith('.webp') ||
        lower.endsWith('.bmp');
  }

  /// Decodes one image (downscaled to [_maxSide]) to a `ui.Image` and its
  /// RGBA bytes.
  Future<_Slide> _decode(File file) async {
    final decoded = await decodeEncodedImage(
      await file.readAsBytes(),
      maxSide: _maxSide,
    );
    var image = decoded.image;
    var bytes = decoded.rgba;
    if (_mirrored) {
      try {
        bytes = mirrorRgba(bytes, image.width, image.height);
        final flipped = await imageFromRgba(bytes, image.width, image.height);
        image.dispose();
        image = flipped;
      } catch (_) {
        image.dispose();
        rethrow;
      }
    }
    return _Slide(path: file.path, image: image, rgba: bytes);
  }
}

/// [rgba] ([width]×[height], packed) flipped left to right, as a new buffer.
@visibleForTesting
Uint8List mirrorRgba(Uint8List rgba, int width, int height) {
  final src = rgba.buffer.asUint32List(rgba.offsetInBytes, width * height);
  final out = Uint8List(width * height * 4);
  final dst = out.buffer.asUint32List();
  for (var y = 0; y < height; y++) {
    final row = y * width;
    for (var x = 0; x < width; x++) {
      dst[row + x] = src[row + width - 1 - x];
    }
  }
  return out;
}

/// One decoded image; also the [FrameView] for every frame it is shown in.
final class _Slide implements FrameView {
  _Slide({required this.path, required this.image, required Uint8List rgba})
    : width = image.width,
      height = image.height,
      planes = [
        FramePlane(bytes: rgba, bytesPerRow: image.width * 4, bytesPerPixel: 4),
      ];

  final String path;
  final ui.Image image;

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
