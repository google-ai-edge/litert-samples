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

import '../../domain/models/voice.dart';

/// The notice both voice demos show for a capture that made no LLM call.
String notHeardText(NotHeardReason reason) => switch (reason) {
  NotHeardReason.tooShort =>
    "Didn't catch that — hold the mic button while you speak.",
  NotHeardReason.releasedBeforeListening =>
    'The mic was still opening — wait for Listening… before you speak.',
  NotHeardReason.silent => "Didn't catch that — it was too quiet.",
  NotHeardReason.emptyTranscript =>
    "Didn't catch that — no words were recognized.",
};
