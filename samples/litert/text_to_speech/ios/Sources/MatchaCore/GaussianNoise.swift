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

/// Seeded SplitMix64 and Box–Muller noise with zero-filled acoustic padding.
public struct GaussianNoise {
  private var state: UInt64
  private var spare: Double?

  /// Initializes the deterministic generator with the requested seed.
  public init(seed: UInt64) {
    state = seed
  }

  /// Advances the SplitMix64 state with wrapping integer operations.
  private mutating func nextUInt64() -> UInt64 {
    state &+= 0x9E37_79B9_7F4A_7C15
    var value = state
    value = (value ^ (value >> 30)) &* 0xBF58_476D_1CE4_E5B9
    value = (value ^ (value >> 27)) &* 0x94D0_49BB_1331_11EB
    return value ^ (value >> 31)
  }

  /// Returns a uniform value strictly between zero and one.
  private mutating func uniformOpen() -> Double {
    // Fifty-two bits plus a half step excludes both zero and one.
    (Double(nextUInt64() >> 12) + 0.5) * 0x1.0p-52
  }

  /// Returns a Gaussian sample, reusing the spare Box–Muller sample.
  public mutating func next() -> Float {
    if let value = spare {
      spare = nil
      return Float(value)
    }
    let radius = sqrt(-2 * log(uniformOpen()))
    let angle = 2 * Double.pi * uniformOpen()
    spare = radius * sin(angle)
    return Float(radius * cos(angle))
  }

  /// Draws samples only for valid frames, leaving padding zero.
  public mutating func tensor(melLength: Int) throws -> [Float] {
    guard (1...MatchaMath.maxMel).contains(melLength) else {
      throw MatchaMathError.invalidMelLength(melLength)
    }
    var result = [Float](repeating: 0, count: MatchaMath.features * MatchaMath.maxMel)
    for channel in 0..<MatchaMath.features {
      for frame in 0..<melLength {
        result[channel * MatchaMath.maxMel + frame] = next()
      }
    }
    return result
  }
}
