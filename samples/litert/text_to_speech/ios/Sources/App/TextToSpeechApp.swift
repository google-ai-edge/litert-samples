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

import SwiftUI

/// Selects interactive speech controls or the launch-argument synthesis workflow.
@main
struct TextToSpeechApp: App {
  var body: some Scene {
    WindowGroup {
      if ProcessInfo.processInfo.arguments.contains("-speak") {
        HeadlessView()
      } else {
        SpeechView()
      }
    }
  }
}

/// Starts command-line work without constructing the interactive speech interface.
private struct HeadlessView: View {
  var body: some View {
    Color.clear.task {
      do {
        guard let request = try HeadlessRequest.parse(ProcessInfo.processInfo.arguments),
          let resourceURL = Bundle.main.resourceURL,
          let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
            .first
        else {
          throw AppSpeechError.invalidArgument("-speak")
        }
        let worker = SpeechWorker(assets: resourceURL.appendingPathComponent("Models"))
        worker.runHeadless(request, documents: documents) { outcome in
          switch outcome {
          case .success:
            print("SPEECH_COMPLETE")
            exit(0)
          case .failure(let error):
            print("SPEECH_ERROR: \(error.localizedDescription)")
            exit(1)
          }
        }
      } catch {
        print("SPEECH_ERROR: \(error.localizedDescription)")
        exit(1)
      }
    }
  }
}
