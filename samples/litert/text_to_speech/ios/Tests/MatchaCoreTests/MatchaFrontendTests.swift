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
import XCTest

@testable import MatchaCore

/// A frozen text input with its expected pronunciation and symbol IDs.
private struct TextFixture: Decodable {
  let id: String
  let text: String
  let ipa: String
  let ids: [Int32]
}

/// The symbol inventory needed by the text tests.
private struct Configuration: Decodable {
  let symbols: [String]
}

/// Supplies saved neural outputs without a runtime dependency.
private struct LookupPhonemizer: WordPhonemizer {
  let table: [String: String]
  /// Missing test pronunciation data.
  enum LookupError: Error { case missingWord(String) }
  /// Returns the saved pronunciation or identifies missing test data.
  func phonemizeWord(_ word: String) throws -> String {
    guard let result = table[word] else { throw LookupError.missingWord(word) }
    return result
  }
}

/// Tests text behavior independently of model execution.
final class MatchaFrontendTests: XCTestCase {
  /// Locates a small resource included in the test bundle.
  private func resource(_ name: String, _ ext: String) throws -> URL {
    try XCTUnwrap(Bundle.module.url(forResource: name, withExtension: ext))
  }

  /// Checks complete IPA byte sequences and symbol IDs for all saved texts.
  func testAllFrozenTextFixtures() throws {
    let decoder = JSONDecoder()
    let cases = try decoder.decode(
      [TextFixture].self, from: Data(contentsOf: resource("text_fixtures", "json")))
    let config = try decoder.decode(
      Configuration.self, from: Data(contentsOf: resource("config", "json")))
    let neural = try decoder.decode(
      [String: String].self, from: Data(contentsOf: resource("neural_lookup", "json")))
    let dictionary = try MatchaFrontend.readDictionary(at: resource("dictionary_subset", "txt"))
    let frontend = try MatchaFrontend(
      dictionary: dictionary, symbols: config.symbols, phonemizer: LookupPhonemizer(table: neural))
    for fixture in cases {
      let result = try frontend.phonemize(fixture.text)
      XCTAssertEqual(Array(result.ipa.utf8), Array(fixture.ipa.utf8), fixture.id)
      XCTAssertEqual(result.ids, fixture.ids, fixture.id)
    }
    XCTAssertEqual(cases.count, 34)

  }

  /// Checks UTF-16 symbol handling and ASCII digit tokenization.
  func testUTF16SymbolMappingAndAsciiTokenDigits() throws {
    let frontend = try MatchaFrontend(
      dictionary: ["rain": "a\u{0303}🙂"],
      symbols: ["_", "a", "\u{0303}", "🙂", "a\u{0303}"],
      phonemizer: LookupPhonemizer(table: [:]))
    let result = try frontend.phonemize("rain １２٣")
    XCTAssertEqual(result.ipa, "a\u{0303}🙂")
    XCTAssertEqual(result.ids, [1, 2])
  }

  /// Checks signed integer overflow, leading zeros, and large-number reading.
  func testLongParsingBoundaries() {
    XCTAssertEqual(MatchaFrontend.numberToWords("9223372036854775808"), [])
    XCTAssertEqual(
      MatchaFrontend.numberToWords("9223372036854775808.5"), ["zero", "point", "five"])
    XCTAssertEqual(MatchaFrontend.numberToWords("0007"), ["seven"])
    XCTAssertEqual(MatchaFrontend.numberToWords("1234567890123456").count, 16)
  }
}
