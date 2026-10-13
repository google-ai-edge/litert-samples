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

import 'package:flutter/foundation.dart';
import 'package:litert_edge_demos/data/repositories/image_repository.dart';
import 'package:litert_edge_demos/data/services/images/image_input_service.dart';
import 'package:litert_edge_demos/data/services/images/image_normalizer.dart';
import 'package:litert_edge_demos/domain/models/llm_image.dart';

/// [ImageInputService] played by the test: [nextPick] is what the next pick
/// returns (null = the user cancelled), [nextError] makes it throw instead.
/// The real `PlatformImageInputService` refuses an unsupported source; this
/// one lets the test check that the repository refuses it first.
class FakeImageInputService implements ImageInputService {
  FakeImageInputService({Set<ImageSourceKind>? supported})
    : supported = supported ?? {ImageSourceKind.gallery};

  final Set<ImageSourceKind> supported;
  Uint8List? nextPick = Uint8List.fromList([0xFF, 0xD8, 0xFF, 0xE0]);
  Exception? nextError;

  /// When set, [pick] waits for it (a picker dialog that is still open).
  Completer<void>? gate;
  final List<ImageSourceKind> picks = [];

  @override
  bool supports(ImageSourceKind source) => supported.contains(source);

  @override
  Future<Uint8List?> pick(ImageSourceKind source) async {
    picks.add(source);
    await gate?.future;
    if (nextError case final error?) throw error;
    return nextPick;
  }
}

/// A normalizer without the engine's codecs: a fresh "PNG" per call, sized
/// like a 4032×3024 photo shrunk to 1024×768.
Future<NormalizedImage> fakeNormalize(Uint8List encoded) async =>
    NormalizedImage(
      png: Uint8List.fromList([0x89, 0x50, 0x4E, 0x47, ...encoded]),
      width: 1024,
      height: 768,
      sourceWidth: 4032,
      sourceHeight: 3024,
    );

/// The real [ImageRepository] over [input] and [fakeNormalize].
ImageRepository fakeImageRepository([FakeImageInputService? input]) =>
    ImageRepository(
      input: input ?? FakeImageInputService(),
      normalize: fakeNormalize,
    );
