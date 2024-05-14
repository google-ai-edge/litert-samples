// Copyright 2024 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

import Foundation

/// Runs WordPiece tokenziation.
///
/// Name of functions and variables are from
/// [google-research/bert](https://github.com/google-research/bert/blob/d66a146741588fb208450bde15aa7db143baaa69/tokenization.py#L300).
struct WordpieceTokenizer {
  let vocabularyIDs: [String: Int32]

  private static let UNKNOWN_TOKEN = "[UNK]"  // For unknown words.
  private static let MAX_INPUT_CHARS_PER_WORD = 128

  init(with model: Model) {

    guard let vocabPath = model.vocabPath else {
      fatalError("can not load vocabrary")
    }
    switch model {
    case .mobileBert:
      vocabularyIDs = FileLoader.loadBertVocabularies(from: vocabPath)
    case .avgWordClassifier:
      vocabularyIDs = FileLoader.loadAverageVocabularies(from: vocabPath)
    }




  }

  /// Tokenizes a piece of text into its word pieces.
  ///
  /// This uses a greedy longest-match-first algorithm to perform tokenization using the given
  /// vocabulary.
  ///
  /// For example:
  ///   ```
  ///   input = "unaffable"
  ///   output = ["un", "##aff", "##able"]
  ///   ```
  ///
  ///   ```
  ///   input = "unaffableX"
  ///   output = ["[UNK]"]
  ///   ```
  ///
  /// - Parameter text: A single token or whitespace separated tokens. This should have already been
  ///   passed through `BasicTokenizer.
  /// - Returns: A list of wordpiece tokens.
  func tokenize(_ text: String) -> [String] {
    var outputTokens = [String]()
    text.splitByWhitespace().forEach { rawToken in
      let token = rawToken.lowercased()
      if token.count > WordpieceTokenizer.MAX_INPUT_CHARS_PER_WORD {
        outputTokens.append(WordpieceTokenizer.UNKNOWN_TOKEN)
        return
      }

      var start = token.startIndex
      var subWords = [String]()

      // Find all subwords in `token`.
      while start < token.endIndex {
        var end = token.endIndex
        var hasFound = false

        // Find longest known subword in the `token` from its `start` index.
        while start < end {
          var subStr = String(token[start..<end])
          if start > token.startIndex {
            subStr = "##" + subStr
          }

          if vocabularyIDs[subStr] != nil {
            // `subStr` is the longest subword that can be found.
            hasFound = true
            subWords.append(subStr)
            break
          } else {
            end = token.index(before: end)
          }
        }

        if hasFound {
          // Proceed to tokenize the residual string.
          start = end
        } else {
          // The `token` contains unknown subwords. It can't be tokenized into subwords.
          subWords = [WordpieceTokenizer.UNKNOWN_TOKEN]
          break
        }
      }
      outputTokens.append(contentsOf: subWords)
    }
    return outputTokens
  }

  func convertToIDs(tokens: [String]) -> [Int32] {
    return tokens.compactMap { vocabularyIDs[$0] }
  }
}

extension String {

  /// Splits this string on whitespace.
  func splitByWhitespace() -> [String] {
    // Normalize string to NFC(Normalization Form Canonical Composition).
    return self.precomposedStringWithCanonicalMapping
      .unicodeScalars.split { $0.isWhitespace }.map { String($0) }
  }
}

extension UnicodeScalar {

  var isWhitespace: Bool {
    switch self {
    case " ", "\t", "\n", "\r":
      return true
    default:
      return properties.generalCategory == .spaceSeparator
    }
  }
}
