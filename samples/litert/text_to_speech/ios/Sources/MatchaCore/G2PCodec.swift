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

/// Character IDs, phoneme IDs, and sequence limits from the published metadata.
public struct G2PMetadata: Decodable {
  public let char2idx: [String: Int]
  public let idx2ph: [String: String]
  public let charRepeats: Int
  public let start: Int
  public let end: Int
  public let maxTokens: Int
  public let phonemeCount: Int
  public let special: [String]

  /// Preserves the published JSON field names while exposing Swift naming conventions.
  private enum CodingKeys: String, CodingKey {
    case char2idx, idx2ph, start, end, special
    case charRepeats = "char_repeats"
    case maxTokens = "MAXT"
    case phonemeCount = "n_phonemes"
  }
}

/// Fixed-size neural pronunciation input and its valid length.
public struct G2PInput {
  public let values: [Float]
  public let length: Int
  public let untruncatedLength: Int
}

/// Invalid pronunciation tensor dimensions or sequence lengths.
public enum G2PCodecError: Error {
  case invalidLogitCount(expected: Int, actual: Int)
  case invalidLength(Int)
}

/// Encodes UTF-16 character units and decodes token-major pronunciation logits.
public struct G2PCodec {
  public let metadata: G2PMetadata
  private let charToIndex: [UInt16: Int]
  private let indexToPhoneme: [Int: String]
  private let special: Set<String>

  /// Builds UTF-16 character and phoneme maps from metadata.
  public init(metadata: G2PMetadata) {
    self.metadata = metadata
    var characters = [UInt16: Int]()
    for (key, value) in metadata.char2idx {
      let units = Array(key.utf16)
      if units.count == 1 {
        characters[units[0]] = value
      }
    }
    charToIndex = characters
    indexToPhoneme = Dictionary(
      uniqueKeysWithValues: metadata.idx2ph.compactMap { key, value in
        Int(key).map { ($0, value) }
      })
    special = Set(metadata.special)
  }

  /// Builds the capped repeated-character input, including start and end tokens.
  public func input(for word: String) -> G2PInput {
    var ids = [metadata.start]
    for unit in word.utf16 {
      if let id = charToIndex[unit] {
        ids.append(contentsOf: repeatElement(id, count: metadata.charRepeats))
      }
    }
    ids.append(metadata.end)
    let length = min(ids.count, metadata.maxTokens)
    var values = [Float](repeating: 0, count: metadata.maxTokens)
    for index in 0..<length {
      values[index] = Float(ids[index])
    }
    return G2PInput(values: values, length: length, untruncatedLength: ids.count)
  }

  /// Selects the first maximum by using strict-greater comparisons.
  public func argmax(_ logits: [Float]) throws -> [Int] {
    let expected = metadata.maxTokens * metadata.phonemeCount
    guard logits.count == expected else {
      throw G2PCodecError.invalidLogitCount(expected: expected, actual: logits.count)
    }
    return (0..<metadata.maxTokens).map { token in
      var best = 0
      var bestScore = logits[token * metadata.phonemeCount]
      for index in 1..<metadata.phonemeCount {
        let score = logits[token * metadata.phonemeCount + index]
        if score > bestScore {
          best = index
          bestScore = score
        }
      }
      return best
    }
  }

  /// Collapses repeats before removing special tokens and hyphens.
  public func decode(_ logits: [Float], length: Int) throws -> String {
    guard length >= 0 && length <= metadata.maxTokens else {
      throw G2PCodecError.invalidLength(length)
    }
    let best = try argmax(logits)
    var previous = -1
    var result = [UInt16]()
    for token in 0..<length {
      let index = best[token]
      if index == previous { continue }
      previous = index
      guard let phoneme = indexToPhoneme[index], !special.contains(phoneme), index != 0 else {
        continue
      }
      result.append(contentsOf: phoneme.utf16.filter { $0 != 45 })
    }
    return String(decoding: result, as: UTF16.self)
  }
}
