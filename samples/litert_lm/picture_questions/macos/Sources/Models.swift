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

/// The two model files this sample was built and measured with. Another `.litertlm` vision-language model also
/// opens; its memory need is then estimated from its file size.
struct KnownModel {
  var name: String
  var fileName: String
  var fileBytes: UInt64
  var huggingFace: String
  /// The most memory the engine process used in this app, rounded up to a whole GB (macOS footprint, the Memory
  /// column of Activity Monitor; measured on an M4 Max: 25.9 GB and 45.6 GB).
  var peakFootprintBytes: UInt64
  /// The most tokens one reply may have. The file's context is 2,048 tokens: a picture (256 tokens), the question
  /// and the reply share it.
  var maxOutputTokens: Int

  static let all = [
    KnownModel(
      name: "Muse-Glimmer-30B", fileName: "Muse-Glimmer-30B_q4_ekv2048.litertlm", fileBytes: 23_393_996_524,
      huggingFace: "litert-community/Muse-Glimmer-30B-LiteRT", peakFootprintBytes: 26 * gib, maxOutputTokens: 1536),
    KnownModel(
      name: "Qwen2.5-VL-72B-Instruct", fileName: "Qwen2.5-VL-72B-Instruct_q4_ekv2048.litertlm",
      fileBytes: 44_789_536_688, huggingFace: "litert-community/Qwen2.5-VL-72B-Instruct-LiteRT",
      peakFootprintBytes: 46 * gib, maxOutputTokens: 1024),
  ]

  static func matching(bytes: UInt64) -> KnownModel? {
    all.first { $0.fileBytes == bytes }
  }
}

let gib: UInt64 = 1 << 30

enum MemoryNeed {
  /// What another file needs: 1.2 x its size (the two files above peaked at 1.09 and 1.19 x on the GPU).
  static let perFileByte = 1.2
  /// Memory left for macOS and the other apps while the engine runs.
  static let reserveBytes: UInt64 = 6 * gib

  static func engineBytes(fileBytes: UInt64) -> UInt64 {
    KnownModel.matching(bytes: fileBytes)?.peakFootprintBytes ?? UInt64(Double(fileBytes) * perFileByte)
  }
}

/// Muse-Glimmer-30B reads its reasoning strength from the system message ("Reasoning strength: <value>.", the
/// model card's Best Practices).
enum ReasoningStrength: String, CaseIterable, Identifiable {
  case low, medium, high, xhigh
  var id: String { rawValue }
  var systemMessage: String { "Reasoning strength: \(rawValue)." }
}
