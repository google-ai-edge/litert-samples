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

/// Splits a reply into the answer and, for Muse-Glimmer-30B, its reasoning message.
///
/// LiteRT-LM returns Muse-Glimmer-30B's reply with the format tokens it wrote:
/// ` to=self<|message|>REASONING<|eom|><|start|>assistant to=user<|message|>ANSWER`. The answer is the text after the
/// last `<|message|>`; the reasoning is the message addressed to itself. Format tokens (`<|...|>`) are not shown.
struct ReplyText: Equatable {
  var reasoning: String?
  var answer: String
  /// The model is still writing its reasoning message (no answer yet).
  var stillReasoning: Bool

  static let messageMark = "<|message|>"

  static func parse(_ raw: String, format: ReplyFormat) -> ReplyText {
    guard format == .reasoningMessages else {
      return ReplyText(reasoning: nil, answer: clean(raw), stillReasoning: false)
    }
    guard let firstMark = raw.range(of: messageMark) else {
      // Only the header so far (" to=self" or " to=user").
      return ReplyText(reasoning: nil, answer: "", stillReasoning: !raw.isEmpty)
    }
    let header = raw[raw.startIndex..<firstMark.lowerBound]
    if header.contains("to=user") {
      // An answer without a reasoning message.
      return ReplyText(reasoning: nil, answer: clean(String(raw[firstMark.upperBound...])), stillReasoning: false)
    }
    if answerHasStarted(raw), let lastMark = raw.range(of: messageMark, options: .backwards) {
      let reasoningEnd = raw.range(of: "<|eom|>", range: firstMark.upperBound..<raw.endIndex)?.lowerBound
        ?? raw.range(of: "<|start|>", range: firstMark.upperBound..<raw.endIndex)?.lowerBound
        ?? lastMark.lowerBound
      return ReplyText(
        reasoning: clean(String(raw[firstMark.upperBound..<reasoningEnd])),
        answer: clean(String(raw[lastMark.upperBound...])), stillReasoning: false)
    }
    return ReplyText(reasoning: clean(String(raw[firstMark.upperBound...])), answer: "", stillReasoning: true)
  }

  /// Muse-Glimmer-30B has finished its reasoning message and started the answer.
  static func answerHasStarted(_ raw: String) -> Bool {
    raw.contains("to=user<|message|>")
  }

  /// Every format token (`<|...|>`) out, then the white space at both ends.
  static func clean(_ text: String) -> String {
    text.replacingOccurrences(of: #"<\|[^|<>]*\|>"#, with: " ", options: .regularExpression)
      .trimmingCharacters(in: .whitespacesAndNewlines)
  }
}
