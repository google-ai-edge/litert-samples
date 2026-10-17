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

/// The OS or another app took the voice demos' audio, as the audio-session
/// owner reports it. Only iOS and Android report any.
enum AudioInterruption {
  /// The session was interrupted: a call, Siri or an alarm, or another app's
  /// audio (iOS: an interruption began; Android: the audio focus was lost,
  /// for good or for a while). A duck (Android lowering this app for a
  /// notification sound) is not one.
  taken,

  /// The output went away (headphones unplugged, a Bluetooth headset
  /// disconnected): a reply would go on through the speaker.
  outputRemoved,
}
