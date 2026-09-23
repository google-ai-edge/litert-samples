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

/// Interactive text, placement, and synthesis result controls.
struct SpeechView: View {
  @StateObject private var controller = SpeechController()

  var body: some View {
    NavigationStack {
      Form {
        Section("Text") {
          TextField("Enter text", text: $controller.text, axis: .vertical)
            .lineLimit(3...8)
          HStack {
            TextField("Seed", text: $controller.seed).keyboardType(.numberPad)
            Stepper("Steps: \(controller.steps)", value: $controller.steps, in: 1...30)
          }
          Button(action: controller.speak) {
            HStack {
              Text(controller.isBusy ? "Working…" : "Speak")
              if controller.isBusy { ProgressView() }
            }
          }.disabled(
            controller.isBusy
              || controller.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
          Button("Stop audio", action: controller.stopAudio)
        }
        Section("Graph backends") {
          Button("Use vocoder Metal") { controller.configuration = AppRunConfiguration() }
          Button("Use encoder and vocoder Metal FP32") { controller.configuration = .highPrecision }
          LabeledContent("Pronunciation", value: "CPU")
          graphPicker("Text encoder", configuration: $controller.configuration.textEncoder)
          LabeledContent("Decoder", value: "CPU")
          graphPicker("Vocoder", configuration: $controller.configuration.vocoder)
        }.disabled(controller.isBusy)
        if let message = controller.message {
          Section { Text(message).font(.callout).textSelection(.enabled) }
        }
        if let result = controller.result {
          Section("Results") {
            ForEach(result.rows) { row in
              VStack(alignment: .leading, spacing: 4) {
                HStack {
                  Text(row.name).font(.headline)
                  Spacer()
                  Text(String(format: "%.2f ms", row.milliseconds)).monospacedDigit()
                }
                Text("Requested: \(row.requested) · Effective: \(row.effective)").font(.caption)
              }
            }
            Text("Graph times include output readback; decoder time sums all steps.").font(.caption)
            LabeledContent("Total", value: String(format: "%.2f ms", result.totalMilliseconds))
            LabeledContent("Audio", value: String(format: "%.2f s", result.audioSeconds))
            LabeledContent("RTF", value: String(format: "%.3f", result.realTimeFactor))
          }
        }
      }
      .navigationTitle("Speech synthesis")
    }
  }

  /// Binds one graph configuration to its accelerator and precision controls.
  private func graphPicker(_ title: String, configuration: Binding<MatchaGraphConfiguration>)
    -> some View
  {
    VStack(alignment: .leading) {
      Picker(title, selection: configuration.backend) {
        Text("CPU").tag(MatchaBackend.cpu)
        Text("Metal").tag(MatchaBackend.gpu)
      }
      if configuration.wrappedValue.backend == .gpu {
        Picker("\(title) precision", selection: configuration.precision) {
          Text("Default").tag(MatchaGPUPrecision.defaultPrecision)
          Text("FP32").tag(MatchaGPUPrecision.fp32)
        }
      }
    }
  }
}
