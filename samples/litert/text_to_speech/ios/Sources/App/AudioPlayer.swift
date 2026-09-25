// Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import AVFoundation

/// Plays the generated mono Float32 samples through the system output mixer.
final class AudioPlayer {
  private let engine = AVAudioEngine()
  private let player = AVAudioPlayerNode()
  private let format = AVAudioFormat(
    commonFormat: .pcmFormatFloat32, sampleRate: 22_050, channels: 1, interleaved: false)

  /// Attaches the player; format creation failures are reported by play.
  init() {
    engine.attach(player)
    if let format { engine.connect(player, to: engine.mainMixerNode, format: format) }
  }

  /// Starts playback after configuring the shared audio session.
  func play(_ audio: [Float]) throws {
    guard !audio.isEmpty else { return }
    guard let format,
      let buffer = AVAudioPCMBuffer(
        pcmFormat: format, frameCapacity: AVAudioFrameCount(audio.count)),
      let channels = buffer.floatChannelData
    else {
      throw AppSpeechError.unavailableAudioBuffer
    }
    let session = AVAudioSession.sharedInstance()
    try session.setCategory(.playback, mode: .spokenAudio)
    try session.setActive(true)
    player.stop()
    buffer.frameLength = AVAudioFrameCount(audio.count)
    try audio.withUnsafeBufferPointer { source in
      guard let base = source.baseAddress else { throw AppSpeechError.unavailableAudioBuffer }
      channels[0].update(from: base, count: source.count)
    }
    if !engine.isRunning { try engine.start() }
    player.scheduleBuffer(buffer, completionHandler: nil)
    player.play()
  }

  /// Stops playback without discarding the reusable audio engine.
  func stop() { player.stop() }
}
