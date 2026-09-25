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

import Foundation
import SwiftUI

@MainActor
/// Main-actor state for synthesis controls, results, and playback.
final class SpeechController: ObservableObject {
  @Published var text = "The rain is soft today."
  @Published var seed = "0"
  @Published var steps = 10
  @Published var configuration = AppRunConfiguration()
  @Published private(set) var isBusy = false
  @Published private(set) var message: String?
  @Published private(set) var result: AppSpeechResult?
  private let worker: SpeechWorker
  private let player = AudioPlayer()

  /// Locates prepared resources and reports any missing files.
  init() {
    let assets = (Bundle.main.resourceURL ?? Bundle.main.bundleURL).appendingPathComponent(
      "Models", isDirectory: true)
    worker = SpeechWorker(assets: assets)
    let missing = worker.missingResources
    if !missing.isEmpty {
      message = AppSpeechError.missingResources(missing).localizedDescription
    }
  }

  /// Starts synthesis with the current text, seed, and graph settings.
  func speak() {
    guard !isBusy else { return }
    guard let numericSeed = UInt64(seed) else {
      message = AppSpeechError.invalidSeed.localizedDescription
      return
    }
    player.stop()
    isBusy = true
    message = nil
    worker.speak(text: text, seed: numericSeed, steps: steps, configuration: configuration) {
      outcome in
      DispatchQueue.main.async {
        self.isBusy = false
        switch outcome {
        case .success(let result):
          self.result = result
          do { try self.player.play(result.audio) } catch {
            self.message = error.localizedDescription
          }
        case .failure(let error): self.message = error.localizedDescription
        }
      }
    }
  }

  /// Stops the current playback without changing synthesis settings.
  func stopAudio() { player.stop() }
}
