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

/// Invalid shapes, table data, or lengths supplied to host tensor operations.
public enum MatchaMathError: Error {
  case invalidEmbeddingBytes(Int)
  case invalidEmbeddingTable(Int)
  case invalidSymbolID(Int32)
  case invalidTensorSize(String, expected: Int, actual: Int)
  case invalidMelLength(Int)
}

/// Float host operations matching the acoustic graph layouts and duration rules.
public enum MatchaMath {
  public static let maxText = 256
  public static let maxMel = 512
  public static let features = 80
  public static let channels = 192
  public static let timeDimension = 160
  public static let hop = 256
  public static let sampleRate = 22_050
  public static let defaultSteps = 10
  public static let lengthScale: Float = 0.95
  public static let melMean: Float = -5.536622
  public static let melStandardDeviation: Float = 2.116101

  /// Blank-interspersed token IDs, embeddings, and the text mask.
  public struct TextInput {
    public let textLength: Int
    public let paddedIDs: [Int32]
    public let embedded: [Float]
    public let mask: [Float]
  }

  /// Integer length regulation and the resulting acoustic conditioning tensors.
  public struct RegulatedInput {
    public let durations: [Float]
    public let cumulative: [Float]
    public let melLength: Int
    public let muY: [Float]
    public let mask: [Float]
  }

  /// Decodes Float32 data without assuming pointer alignment.
  public static func readLittleEndianFloats(_ data: Data) throws -> [Float] {
    guard data.count % 4 == 0 else {
      throw MatchaMathError.invalidEmbeddingBytes(data.count)
    }
    return data.withUnsafeBytes { raw in
      (0..<(data.count / 4)).map { index in
        let bits = raw.loadUnaligned(fromByteOffset: index * 4, as: UInt32.self)
        return Float(bitPattern: UInt32(littleEndian: bits))
      }
    }
  }

  /// Intersperses blank IDs, truncates tokens, looks up embeddings, and builds the mask.
  public static func prepareText(ids: [Int32], embedding: [Float]) throws -> TextInput {
    guard !embedding.isEmpty && embedding.count % channels == 0 else {
      throw MatchaMathError.invalidEmbeddingTable(embedding.count)
    }
    let textLength = ids.count >= maxText / 2 ? maxText : ids.count * 2 + 1
    var padded = [Int32](repeating: 0, count: maxText)
    let symbolCount = embedding.count / channels
    for (index, id) in ids.prefix(maxText / 2).enumerated() {
      guard id >= 0 && Int(id) < symbolCount else {
        throw MatchaMathError.invalidSymbolID(id)
      }
      padded[index * 2 + 1] = id
    }
    var embedded = [Float](repeating: 0, count: maxText * channels)
    for token in 0..<maxText {
      let source = Int(padded[token]) * channels
      let destination = token * channels
      for channel in 0..<channels {
        embedded[destination + channel] = embedding[source + channel]
      }
    }
    return TextInput(
      textLength: textLength, paddedIDs: padded, embedded: embedded,
      mask: (0..<maxText).map { $0 < textLength ? 1 : 0 })
  }

  /// Rounds durations and performs sequential Float accumulation and integer regulation.
  public static func regulate(mu: [Float], logw: [Float], textMask: [Float]) throws
    -> RegulatedInput
  {
    try requireCount(mu, features * maxText, "mu")
    try requireCount(logw, maxText, "logw")
    try requireCount(textMask, maxText, "text mask")
    var durations = [Float](repeating: 0, count: maxText)
    var cumulative = durations
    var total: Float = 0
    for index in 0..<maxText {
      // Kotlin's Float transcendental overload rounds the Double result.
      let exponential = Float(exp(Double(logw[index])))
      let masked: Float = exponential * textMask[index]
      let ceiling = Float(ceil(Double(masked)))
      durations[index] = ceiling * lengthScale
      total = total + durations[index]
      cumulative[index] = total
    }
    let melLength: Int
    if total.isNaN || total < 1 {
      melLength = 1
    } else if total >= Float(maxMel) {
      melLength = maxMel
    } else {
      melLength = Int(total)
    }
    var muY = [Float](repeating: 0, count: features * maxMel)
    var token = 0
    for frame in 0..<melLength {
      while token < maxText - 1 && cumulative[token] <= Float(frame) {
        token += 1
      }
      for channel in 0..<features {
        muY[channel * maxMel + frame] = mu[channel * maxText + token]
      }
    }
    return RegulatedInput(
      durations: durations, cumulative: cumulative, melLength: melLength, muY: muY,
      mask: (0..<maxMel).map { $0 < melLength ? 1 : 0 })
  }

  /// Builds the sinusoidal embedding with a Float-rounded angle.
  public static func timeEmbedding(_ time: Float) -> [Float] {
    let half = timeDimension / 2
    let logScale = -log(10_000.0) / Double(half - 1)
    var values = [Float](repeating: 0, count: timeDimension)
    for index in 0..<half {
      let frequency = Float(exp(Double(index) * logScale))
      let scaledTime: Float = 1_000 * time
      let angle: Float = scaledTime * frequency
      values[index] = Float(sin(Double(angle)))
      values[half + index] = Float(cos(Double(angle)))
    }
    return values
  }

  /// Applies separate Float multiplication and addition for one Euler step.
  public static func eulerUpdate(x: inout [Float], velocity: [Float], dt: Float) throws {
    try requireCount(velocity, x.count, "velocity")
    for index in x.indices {
      // Keep two Float operations; do not replace with addingProduct.
      let delta: Float = dt * velocity[index]
      x[index] = x[index] + delta
    }
  }

  /// Zeros noise padding outside the valid acoustic frame range.
  public static func maskNoise(_ noise: [Float], melLength: Int) throws -> [Float] {
    try requireCount(noise, features * maxMel, "noise")
    guard (1...maxMel).contains(melLength) else {
      throw MatchaMathError.invalidMelLength(melLength)
    }
    var masked = noise
    for channel in 0..<features {
      for frame in melLength..<maxMel {
        masked[channel * maxMel + frame] = 0
      }
    }
    return masked
  }

  /// Scales valid mel frames and leaves the padded frames zero.
  public static func denormalize(_ x: [Float], melLength: Int) throws -> [Float] {
    try requireCount(x, features * maxMel, "x")
    guard (1...maxMel).contains(melLength) else {
      throw MatchaMathError.invalidMelLength(melLength)
    }
    var mel = [Float](repeating: 0, count: features * maxMel)
    for channel in 0..<features {
      for frame in 0..<melLength {
        let index = channel * maxMel + frame
        let scaled: Float = x[index] * melStandardDeviation
        mel[index] = scaled + melMean
      }
    }
    return mel
  }

  /// Trims the fixed vocoder output to the valid sample count.
  public static func crop(_ waveform: [Float], melLength: Int) throws -> [Float] {
    try requireCount(waveform, maxMel * hop, "vocoder output")
    guard (1...maxMel).contains(melLength) else {
      throw MatchaMathError.invalidMelLength(melLength)
    }
    return Array(waveform.prefix(melLength * hop))
  }

  /// Clamps samples to the normalized audio range.
  public static func clip(_ waveform: [Float]) -> [Float] {
    waveform.map { value in
      if value < -1 { return -1 }
      if value > 1 { return 1 }
      return value
    }
  }

  /// Rejects tensors whose element count differs from the required shape.
  public static func requireCount(_ values: [Float], _ expected: Int, _ name: String) throws {
    guard values.count == expected else {
      throw MatchaMathError.invalidTensorSize(name, expected: expected, actual: values.count)
    }
  }
}
