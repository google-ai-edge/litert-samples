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

import 'dart:typed_data';

/// Where a picked image came from.
enum ImageSourceKind {
  /// The photo library (macOS: a file dialog).
  gallery,

  /// The system camera UI (iPhone and Android only).
  camera,
}

/// A picture ready for Gemma: re-encoded as PNG by `normalizeForLlm`, EXIF
/// orientation applied, at most `kLlmImageMaxSide` on the long side.
///
/// [png] is also the image's identity: the conversation compares it with
/// `identical` and sends it only when the live chat cannot see it. Pass this
/// same object on every turn while the image stays attached.
final class const LlmImage({
  required final Uint8List png,
  required final int width,
  required final int height,

  /// The picked file, before normalization (EXIF-oriented size).
  required final int sourceWidth,
  required final int sourceHeight,
  required final int sourceBytes,

  /// Decode, resize and PNG encode.
  required final Duration normalizeTime,
});
