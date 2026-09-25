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

/// Invalid text-front-end configuration.
public enum MatchaFrontendError: Error { case invalidExpression(String) }

/// Provides neural pronunciations for words absent from the dictionary.
public protocol WordPhonemizer {
  /// Returns the pronunciation of a single dictionary-missing word.
  func phonemizeWord(_ word: String) throws -> String
}

/// The complete IPA spelling and its model symbol IDs.
public struct Phonemization: Codable, Equatable {
  public let ipa: String
  public let ids: [Int32]
}

/// Tokenizes text and resolves acronyms, numbers, and word pronunciations.
public final class MatchaFrontend {
  private let dictionary: [String: String]
  private let phonemizer: any WordPhonemizer
  private let symbolToID: [UInt16: Int32]

  private let tokenExpression: NSRegularExpression
  private let acronymExpression: NSRegularExpression
  private let wordExpression: NSRegularExpression
  private static let letterIPA: [UInt16: String] = Dictionary(
    uniqueKeysWithValues: zip(
      Array("abcdefghijklmnopqrstuvwxyz".utf16),
      [
        "ˈeɪ", "bˈiː", "sˈiː", "dˈiː", "ˈiː", "ˈɛf", "dʒˈiː", "ˈeɪtʃ", "ˈaɪ",
        "dʒˈeɪ", "kˈeɪ", "ˈɛl", "ˈɛm", "ˈɛn", "ˈoʊ", "pˈiː", "kjˈuː", "ˈɑːɹ",
        "ˈɛs", "tˈiː", "jˈuː", "vˈiː", "dˈʌbəljˌuː", "ˈɛks", "wˈaɪ", "zˈiː",
      ]))
  private static let ones = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen",
  ]
  private static let tens = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
  ]
  private static let scales = ["", "thousand", "million", "billion", "trillion"]

  /// Stores the dictionary and UTF-16 symbol map with an injected phonemizer.
  public init(dictionary: [String: String], symbols: [String], phonemizer: any WordPhonemizer)
    throws
  {
    // Explicit ASCII digit classes match the default JVM regex semantics.
    tokenExpression = try Self.expression(
      "[A-Z]{2,}|[0-9][0-9,]*(?:\\.[0-9]+)?|[A-Za-z']+|[.,!?;:—…\"]")
    acronymExpression = try Self.expression("^[A-Z]{2,}$")
    wordExpression = try Self.expression("^[A-Za-z']+$")
    self.dictionary = dictionary
    self.phonemizer = phonemizer
    var mapping = [UInt16: Int32]()
    for (index, symbol) in symbols.enumerated() {
      let units = Array(symbol.utf16)
      if units.count == 1 {
        mapping[units[0]] = Int32(index)
      }
    }
    symbolToID = mapping
  }

  /// Reads tab-separated UTF-8 word and pronunciation pairs.
  public static func readDictionary(at url: URL) throws -> [String: String] {
    let text = try String(contentsOf: url, encoding: .utf8)
    var result = [String: String]()
    result.reserveCapacity(300_000)
    for line in text.components(separatedBy: .newlines) {
      guard let tab = line.firstIndex(of: "\t"), tab != line.startIndex else { continue }
      result[String(line[..<tab])] = String(line[line.index(after: tab)...])
    }
    return result
  }

  /// Compiles a constant pattern and preserves a typed failure for callers.
  private static func expression(_ pattern: String) throws -> NSRegularExpression {
    do { return try NSRegularExpression(pattern: pattern) } catch {
      throw MatchaFrontendError.invalidExpression(pattern)
    }
  }

  /// Builds IPA and symbol IDs using dictionary-first word lookup.
  public func phonemize(_ text: String) throws -> Phonemization {
    var ipa = ""
    var first = true
    /// Appends a nonempty pronunciation with word spacing.
    func append(_ phonemes: String) {
      if phonemes.isEmpty { return }
      if !first {
        ipa.append(" ")
      }
      ipa.append(phonemes)
      first = false
    }
    /// Uses the dictionary before invoking the injected phonemizer.
    func lookup(_ word: String) throws -> String {
      if let phonemes = dictionary[word] {
        return phonemes
      }
      return try phonemizer.phonemizeWord(word)
    }
    let nsText = text as NSString
    for match in tokenExpression.matches(
      in: text, range: NSRange(location: 0, length: nsText.length))
    {
      let token = nsText.substring(with: match.range)
      let fullRange = NSRange(location: 0, length: token.utf16.count)
      if acronymExpression.firstMatch(in: token, range: fullRange) != nil {
        append(token.lowercased().utf16.compactMap { Self.letterIPA[$0] }.joined())
      } else if let firstUnit = token.utf16.first, firstUnit >= 48 && firstUnit <= 57 {
        for word in Self.numberToWords(token) {
          append(try lookup(word))
        }
      } else if wordExpression.firstMatch(in: token, range: fullRange) != nil {
        append(try lookup(token.lowercased()))
      } else {
        ipa.append(token)
      }
    }
    return Phonemization(ipa: ipa, ids: ipa.utf16.compactMap { symbolToID[$0] })
  }

  /// Expands numeric text with signed-64-bit parsing semantics.
  public static func numberToWords(_ raw: String) -> [String] {
    let token = raw.replacingOccurrences(of: ",", with: "")
    if let dot = token.firstIndex(of: ".") {
      let integer = String(token[..<dot])
      var words = integer.isEmpty ? ["zero"] : integerToWords(Int64(integer) ?? 0)
      words.append("point")
      for digit in token[token.index(after: dot)...].utf16 where digit >= 48 && digit <= 57 {
        words.append(ones[Int(digit - 48)])
      }
      return words
    }
    guard let value = Int64(token) else { return [] }
    return integerToWords(value)
  }

  /// Expands one three-digit group without a scale suffix.
  private static func wordsUnderThousand(_ value: Int) -> [String] {
    var n = value
    var words = [String]()
    if n >= 100 {
      words.append(ones[n / 100])
      words.append("hundred")
      n %= 100
    }
    if n >= 20 {
      words.append(tens[n / 10])
      n %= 10
    }
    if n > 0 {
      words.append(ones[n])
    }
    return words
  }

  /// Handles signed values before expanding their unsigned magnitude.
  private static func integerToWords(_ value: Int64) -> [String] {
    if value == 0 { return ["zero"] }
    // Tokenized input is unsigned; magnitude also makes direct negative calls safe.
    if value < 0 {
      return ["minus"] + unsignedIntegerToWords(value.magnitude)
    }
    return unsignedIntegerToWords(UInt64(value))
  }

  /// Expands scale groups or reads unsupported large values digit by digit.
  private static func unsignedIntegerToWords(_ value: UInt64) -> [String] {
    var groups = [Int]()
    var n = value
    while n > 0 {
      groups.append(Int(n % 1000))
      n /= 1000
    }
    if groups.count > scales.count {
      return String(value).utf16.map { ones[Int($0 - 48)] }
    }
    var words = [String]()
    for index in groups.indices.reversed() {
      if groups[index] == 0 { continue }
      words.append(contentsOf: wordsUnderThousand(groups[index]))
      if !scales[index].isEmpty {
        words.append(scales[index])
      }
    }
    return words
  }
}
