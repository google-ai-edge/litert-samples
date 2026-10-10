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

import 'package:flutter/widgets.dart';

/// Keys the tests use to find the chat controls.
abstract final class ChatKeys {
  static const input = ValueKey('chat-input');
  static const send = ValueKey('chat-send');
  static const stop = ValueKey('chat-stop');
  static const newConversation = ValueKey('chat-new-conversation');
  static const streamingBubble = ValueKey('chat-streaming-bubble');
  static const mic = ValueKey('chat-mic');
  static const sttError = ValueKey('chat-stt-error');
  static const retryStt = ValueKey('chat-retry-stt');
  static const accessError = ValueKey('chat-access-error');
  static const retryAccess = ValueKey('chat-retry-access');
  static const speakReplies = ValueKey('chat-speak-replies');
  static const phase = ValueKey('chat-phase');
  static const attachGallery = ValueKey('chat-attach-gallery');
  static const attachCamera = ValueKey('chat-attach-camera');
  static const attachmentThumbnail = ValueKey('chat-attachment-thumbnail');
  static const removeAttachment = ValueKey('chat-remove-attachment');
  static const photosOff = ValueKey('chat-photos-off');
  static const skillsOff = ValueKey('chat-skills-off');
  static const entryImage = ValueKey('chat-entry-image');

  /// An item of the app bar's "More" menu ([more]), like
  /// [newConversation].
  static const skills = ValueKey('chat-skills');

  /// The app bar's "More" menu.
  static const more = ValueKey('chat-more');
}
