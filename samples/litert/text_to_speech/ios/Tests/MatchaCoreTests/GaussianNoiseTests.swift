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

import XCTest

@testable import MatchaCore

/// Tests deterministic sample generation and padding behavior.
final class GaussianNoiseTests: XCTestCase {
  /// Checks a fixed reference sequence.
  func testSeededSequence() {
    var generator = GaussianNoise(seed: 12_345)
    let expected: [UInt32] = [
      0x3f10_02da, 0x3ff6_c87f, 0x3f6c_3cc4, 0x3feb_e700,
      0xbf1b_2f4e, 0x3f7e_e8af, 0xbfee_4767, 0x3f5a_f07a,
    ]
    XCTAssertEqual(expected.map { _ in generator.next().bitPattern }, expected)
  }

  /// Ensures padding does not advance the generator.
  func testPaddingDoesNotConsumeRandomNumbers() throws {
    var generator = GaussianNoise(seed: 7)
    var sequential = GaussianNoise(seed: 7)
    let tensor = try generator.tensor(melLength: 3)
    XCTAssertEqual(tensor.count, 80 * 512)
    for channel in 0..<80 {
      for frame in 0..<3 {
        XCTAssertEqual(tensor[channel * 512 + frame].bitPattern, sequential.next().bitPattern)
      }
      XCTAssertTrue(
        tensor[(channel * 512 + 3)..<((channel + 1) * 512)].allSatisfy { $0.bitPattern == 0 })
    }
    XCTAssertEqual(generator.next().bitPattern, sequential.next().bitPattern)
    XCTAssertTrue(tensor.allSatisfy(\.isFinite))
  }

  /// Checks seed repeatability and independence.
  func testSameSeedRepeatsAndDifferentSeedChangesValidFrames() throws {
    var first = GaussianNoise(seed: 7)
    var repeated = GaussianNoise(seed: 7)
    var different = GaussianNoise(seed: 8)
    let firstTensor = try first.tensor(melLength: 3)
    let repeatedTensor = try repeated.tensor(melLength: 3)
    let differentTensor = try different.tensor(melLength: 3)
    XCTAssertEqual(firstTensor.map(\.bitPattern), repeatedTensor.map(\.bitPattern))
    XCTAssertNotEqual(firstTensor[0].bitPattern, differentTensor[0].bitPattern)
    for channel in 0..<80 {
      XCTAssertTrue(
        differentTensor[(channel * 512 + 3)..<((channel + 1) * 512)].allSatisfy {
          $0.bitPattern == 0
        })
    }
  }

  /// Rejects lengths outside the model capacity.
  func testInvalidLengthIsRejected() {
    var generator = GaussianNoise(seed: 1)
    XCTAssertThrowsError(try generator.tensor(melLength: 0))
    XCTAssertThrowsError(try generator.tensor(melLength: 513))
  }
}
