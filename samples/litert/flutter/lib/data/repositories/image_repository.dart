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

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart' show PlatformException;

import '../../domain/models/llm_image.dart';
import '../../utils/result.dart';
import '../services/images/image_input_service.dart';
import '../services/images/image_normalizer.dart';

/// Turns encoded bytes into a [NormalizedImage]; [normalizeForLlm] in the app.
typedef ImageNormalizer = Future<NormalizedImage> Function(Uint8List encoded);

/// The platform could not deliver a picture (access denied, no camera, the
/// picker failed).
final class ImagePickFailedException implements Exception {
  const ImagePickFailedException(this.message);

  final String message;

  @override
  String toString() => message;
}

/// Images for the chat: picks from the gallery or the camera and
/// normalizes to the PNG Gemma gets ([LlmImage]).
class ImageRepository {
  ImageRepository({required this._input, this._normalize = normalizeForLlm});

  final ImageInputService _input;
  final ImageNormalizer _normalize;

  /// Whether [pick] can use [source] here (the camera: iPhone and Android).
  bool supports(ImageSourceKind source) => _input.supports(source);

  /// Ok(null) when the user cancelled. Every failure is an error the UI
  /// shows: an unsupported source, denied access, an unreadable file.
  Future<Result<LlmImage?>> pick(ImageSourceKind source) async {
    if (!_input.supports(source)) {
      return Result.error(UnsupportedImageSourceException(source));
    }
    final Uint8List? bytes;
    try {
      bytes = await _input.pick(source);
    } on UnsupportedImageSourceException catch (e) {
      return Result.error(e);
    } on PlatformException catch (e) {
      debugPrint('[Images] pick(${source.name}) failed: $e');
      return Result.error(
        ImagePickFailedException(
          'The ${source.name} did not return a picture: '
          '${e.message ?? e.code}',
        ),
      );
    } catch (e, st) {
      debugPrint('[Images] pick(${source.name}) failed: $e\n$st');
      return Result.error(asException(e));
    }
    if (bytes == null) return const Result.ok(null);
    final watch = Stopwatch()..start();
    try {
      final normalized = await _normalize(bytes);
      watch.stop();
      final image = LlmImage(
        png: normalized.png,
        width: normalized.width,
        height: normalized.height,
        sourceWidth: normalized.sourceWidth,
        sourceHeight: normalized.sourceHeight,
        sourceBytes: bytes.length,
        normalizeTime: watch.elapsed,
      );
      debugPrint(
        '[Images] ${source.name}: ${bytes.length} B '
        '${normalized.sourceWidth}x${normalized.sourceHeight} → PNG '
        '${normalized.width}x${normalized.height} ${normalized.png.length} B '
        'in ${watch.elapsedMilliseconds} ms',
      );
      return Result.ok(image);
    } catch (e, st) {
      debugPrint('[Images] normalizing failed: $e\n$st');
      return Result.error(asException(e));
    }
  }
}
