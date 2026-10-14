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
import 'dart:typed_data';

import 'package:litert_edge_demos/domain/models/detection.dart';
import 'package:litert_edge_demos/domain/models/visual_memory.dart';
import 'package:litert_edge_demos/domain/ports/visual_memory_embedder.dart';
import 'package:litert_edge_demos/utils/result.dart';

/// A [VisualMemoryEmbedder] the test drives: an image's vector is
/// [imageVectors] for its first byte (the scene id the fake shots carry),
/// a phrase's [textVectors]; [gate] holds image embeddings until completed.
final class FakeMemoryEmbedder implements VisualMemoryEmbedder {
  FakeMemoryEmbedder({
    Map<int, List<double>>? imageVectors,
    Map<String, List<double>>? textVectors,
  }) : imageVectors = imageVectors ?? {},
       textVectors = textVectors ?? {};

  final Map<int, List<double>> imageVectors;
  final Map<String, List<double>> textVectors;

  /// What [imageUnavailableReason] says; null: images work.
  String? unavailable;
  Exception? imageError;
  Exception? textError;
  Completer<void>? gate;
  final List<Uint8List> images = [];
  final List<String> texts = [];

  @override
  String? get imageUnavailableReason => unavailable;

  /// [enableImages] calls; it fails with [enableError], waits for
  /// [enableGate], and otherwise makes images available.
  int enables = 0;
  Exception? enableError;
  Completer<void>? enableGate;

  /// What the memory gets as its `EnableMemoryImages`
  /// (`ModelRepository.enableEmbedderImages` in the app).
  Future<Result<void>> enableImages() async {
    enables++;
    await enableGate?.future;
    if (enableError case final error?) return Result.error(error);
    unavailable = null;
    return const Result.ok(null);
  }

  @override
  Future<Result<List<double>>> embedImage(Uint8List encoded) async {
    images.add(encoded);
    await gate?.future;
    if (imageError case final error?) return Result.error(error);
    return Result.ok(imageVectors[encoded.first] ?? const [0.1, 0.1, 0.1]);
  }

  @override
  Future<Result<List<double>>> embedText(String text) async {
    texts.add(text);
    if (textError case final error?) return Result.error(error);
    return Result.ok(textVectors[text] ?? const [0.1, 0.1, 0.1]);
  }
}

/// A memory shot of scene [scene] (its PNG's first byte and its thumbnail's
/// value), with [cats] cat boxes; [png] false: the scene did not change.
/// [encoded] replaces the fake PNG bytes (a real PNG, for a widget that
/// decodes it); [grey], the thumbnail.
MemoryShot fakeShot({
  int frameId = 1,
  int scene = 1,
  bool png = true,
  int cats = 0,
  int pngBytes = 100,
  Uint8List? encoded,
  Uint8List? grey,
}) => MemoryShot(
  frameId: frameId,
  detections: DetectionFrame(
    frameId: frameId,
    width: 640,
    height: 360,
    boxes: Float32List.fromList([
      for (var i = 0; i < cats; i++) ...[10.0, 20, 110, 220, 0.9, 15],
    ]),
    preMicros: 1000,
    runMicros: 4000,
    postMicros: 1000,
    backend: DetectorBackend.gpu,
  ),
  grey: grey ?? (Uint8List(1024)..fillRange(0, 1024, scene)),
  difference: null,
  png: png ? (encoded ?? (Uint8List(pngBytes)..[0] = scene)) : null,
  width: 320,
  height: 180,
  workerTime: const Duration(milliseconds: 5),
);

/// The live pipeline's memory capture as the test drives it: each call is
/// recorded and answered from [results] in order (or held by [gate]).
final class FakeMemoryCapture {
  final List<Result<MemoryShot>> results = [];
  final List<({Uint8List? reference})> calls = [];
  Completer<void>? gate;

  Future<Result<MemoryShot>> call({Uint8List? reference}) async {
    calls.add((reference: reference));
    await gate?.future;
    if (results.isEmpty) {
      return const Result.error(_NoShotException());
    }
    return results.removeAt(0);
  }
}

final class _NoShotException implements Exception {
  const _NoShotException();

  @override
  String toString() => 'no shot queued';
}
