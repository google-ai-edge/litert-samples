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

/// A model file the app can load. The app looks for it in its Documents folder.
struct ModelChoice: Identifiable, Hashable {
  let id: String
  let name: String
  let fileName: String
  let repository: String
  /// The size of the file on Hugging Face that this sample was tested with. A file of another
  /// size on the phone is most often a copy that stopped early.
  let bytes: Int64
  /// How much the app's footprint grew with this model, from before loading it to the most it
  /// used while answering a question about a picture, measured on an iPhone 18 Pro with
  /// LiteRT-LM 0.19.1 (the README has the table). The check before loading asks iOS for this
  /// much plus `Models.margin`.
  let growthGPU: Int64
  let growthCPU: Int64

  var file: URL { Models.documents.appendingPathComponent(fileName) }

  /// The size of the file on the phone; nil when it is not there.
  var bytesOnPhone: Int64? {
    guard let size = try? file.resourceValues(forKeys: [.fileSizeKey]).fileSize else { return nil }
    return Int64(size)
  }

  func growth(gpu: Bool) -> Int64 { gpu ? growthGPU : growthCPU }
}

enum Models {
  static let e2b = ModelChoice(
    id: "e2b", name: "Gemma 4 E2B", fileName: "gemma-4-E2B-it.litertlm",
    repository: "litert-community/gemma-4-E2B-it-litert-lm", bytes: 2_588_147_712,
    growthGPU: mb(1700), growthCPU: mb(1600))
  static let e4b = ModelChoice(
    id: "e4b", name: "Gemma 4 E4B", fileName: "gemma-4-E4B-it.litertlm",
    repository: "litert-community/gemma-4-E4B-it-litert-lm", bytes: 3_659_530_240,
    growthGPU: mb(1900), growthCPU: mb(1800))
  static let all = [e2b, e4b]

  /// Kept free on top of a model's measured growth.
  static let margin = mb(300)

  static let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]

  static func mb(_ n: Int64) -> Int64 { n * 1_048_576 }
}
