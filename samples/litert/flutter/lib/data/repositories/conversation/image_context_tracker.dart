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

import '../../../domain/models/assistant_event.dart';

/// Which image the live chat can see, and why it lost the last one. LiteRT-LM
/// keeps an image in the native conversation until the conversation is
/// rebuilt — after a stopped turn it is rebuilt from text only — so the
/// repository re-sends the caller's image whenever the chat cannot see it.
///
/// Images are compared with `identical`: the caller passes the same object
/// every turn while it stays attached, and a new object is a new image.
final class ImageContextTracker {
  Uint8List? _inContext;

  /// The image the chat saw last before it lost it, and why: a re-send of
  /// this same object is logged and reported as "image resent".
  ({Uint8List image, ImageLoss reason})? _lost;

  /// The image the live chat can see.
  Uint8List? get inContext => _inContext;

  /// What a turn with [image] attached must send: the image, unless the live
  /// chat already sees it. Null when nothing is attached.
  Uint8List? toSend(Uint8List? image) =>
      identical(image, _inContext) ? null : image;

  /// Why the chat lost [image], when it is the image it saw last; null for a
  /// new image.
  ImageLoss? resendReason(Uint8List image) {
    final lost = _lost;
    return lost != null && identical(lost.image, image) ? lost.reason : null;
  }

  /// After a turn that reached the model: a stop loses the image (the
  /// conversation is rebuilt from text); a normal end with a sent image
  /// makes it the one in context.
  void settle({required bool stopped, required Uint8List? sentImage}) {
    if (stopped) {
      lose(ImageLoss.stop, sentImage: sentImage);
    } else if (sentImage != null) {
      _inContext = sentImage;
      _lost = null;
    }
  }

  /// The chat can no longer see its image ([sentImage]: the one this turn
  /// sent, which supersedes the one in context).
  void lose(ImageLoss reason, {Uint8List? sentImage}) {
    final lost = sentImage ?? _inContext;
    _inContext = null;
    if (lost != null) _lost = (image: lost, reason: reason);
  }
}
