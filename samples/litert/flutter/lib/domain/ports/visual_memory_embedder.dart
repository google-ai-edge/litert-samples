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

// A port: an interface that the domain and the view models depend on and an
// outer layer implements. Every port lives in lib/domain/ports/;
// function-typed dependencies stay next to their one consumer.

import 'dart:typed_data';

import '../../utils/result.dart';

/// What the visual memory needs from EmbeddingGemma 2: frames as images and
/// questions as raw text, in one embedding space. `EmbedderService`
/// implements it (the package's input and task types stay there); tests pass
/// a fake.
abstract interface class VisualMemoryEmbedder {
  /// Why images cannot be embedded now (the embedder is not loaded, or was
  /// loaded without images); null when they can.
  String? get imageUnavailableReason;

  /// The vector of an encoded image (PNG or JPEG, upright and small). Never
  /// throws: every failure is an Error.
  Future<Result<List<double>>> embedImage(Uint8List encoded);

  /// The vector of a memory question's phrase, embedded raw (no task
  /// prefix, as `docs/design/eg2-multimodal.md` chose and
  /// flutter_edge_ai_litertlm's README embeds a text query over media; not
  /// compared with a prefixed query). Never throws.
  Future<Result<List<double>>> embedText(String text);
}
