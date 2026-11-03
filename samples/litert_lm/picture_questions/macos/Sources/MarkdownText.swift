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


import SwiftUI

/// A model's answer: fenced code blocks in a monospaced box, everything else as paragraphs with
/// inline Markdown (bold, italics, `code`, links), headings in bold and list dashes as bullets.
/// Enough for chat answers; tables show as text.
struct MarkdownText: View {
  let text: String

  enum Block {
    case prose(String)
    case code(String)
  }

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      ForEach(Array(Self.blocks(text).enumerated()), id: \.offset) { _, block in
        switch block {
        case .code(let code):
          ScrollView(.horizontal, showsIndicators: false) {
            Text(code)
              .font(.system(size: 13, design: .monospaced))
              .textSelection(.enabled)
              .padding(10)
          }
          .background(RoundedRectangle(cornerRadius: 6).fill(Color.secondary.opacity(0.12)))
        case .prose(let prose):
          Text(Self.inline(prose))
            .font(.system(size: 15))
            .textSelection(.enabled)
            .fixedSize(horizontal: false, vertical: true)
        }
      }
    }
  }

  static func blocks(_ text: String) -> [Block] {
    var blocks: [Block] = []
    var prose: [Substring] = []
    var code: [Substring]?
    for line in text.split(separator: "\n", omittingEmptySubsequences: false) {
      if line.trimmingCharacters(in: .whitespaces).hasPrefix("```") {
        if let lines = code {
          blocks.append(.code(lines.joined(separator: "\n")))
          code = nil
        } else {
          blocks.append(.prose(prose.joined(separator: "\n")))
          prose = []
          code = []
        }
      } else if code != nil {
        code?.append(line)
      } else {
        prose.append(line)
      }
    }
    // A fence that is still open is a code block that is still streaming.
    if let lines = code { blocks.append(.code(lines.joined(separator: "\n"))) }
    blocks.append(.prose(prose.joined(separator: "\n")))
    return blocks.filter {
      if case .prose(let text) = $0 { return !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
      return true
    }
  }

  static func inline(_ text: String) -> AttributedString {
    let lines = plainMath(text).split(separator: "\n", omittingEmptySubsequences: false).map { line -> String in
      let s = String(line)
      if let range = s.range(of: #"^#{1,6} +"#, options: .regularExpression) {
        return "**" + s[range.upperBound...] + "**"
      }
      if let range = s.range(of: #"^ *[-*] +"#, options: .regularExpression) {
        let indent = s[range].prefix { $0 == " " }
        return indent + "• " + s[range.upperBound...]
      }
      return s
    }
    let joined = lines.joined(separator: "\n")
    let options = AttributedString.MarkdownParsingOptions(interpretedSyntax: .inlineOnlyPreservingWhitespace)
    return (try? AttributedString(markdown: joined, options: options)) ?? AttributedString(joined)
  }

  // MARK: Math

  /// Models write math in LaTeX: `$x$`, `$$x = \frac{23 - 5y}{3}$$`, `\text{...}`. The window shows
  /// it as plain text: the delimiters go and the common commands become symbols. A `$...$` span
  /// counts as math only when it holds a command or one of `= ^ _ { }`, or only math characters
  /// with no word of three letters or more, so prices ("$2 and $3") stay as written.
  static func plainMath(_ text: String) -> String {
    guard text.contains("$") || text.contains("\\(") || text.contains("\\[") else { return text }
    // An escaped dollar is a dollar sign, never a delimiter.
    let dollar = "\u{E000}"
    var out = text.replacingOccurrences(of: "\\$", with: dollar)
    // Display math, which can span lines: its own line.
    out = replace(#"\$\$(.+?)\$\$"#, in: out, options: [.dotMatchesLineSeparators]) { math($0[1]) }
    out = replace(#"\\\[(.+?)\\\]"#, in: out, options: [.dotMatchesLineSeparators]) { math($0[1]) }
    out = replace(#"\\\((.+?)\\\)"#, in: out) { math($0[1]) }
    out = replace(#"(?<![\\$\w])\$([^$\n]+?)\$(?![\w$])"#, in: out) { groups in
      let body = groups[1]
      let trimmed = body.trimmingCharacters(in: .whitespaces)
      let isMath =
        trimmed == body
        && (body.contains("\\") || body.rangeOfCharacter(from: CharacterSet(charactersIn: "=^_{}")) != nil
          || (body.range(of: #"^[0-9A-Za-z+\-*/().,\s\x{E000}]+$"#, options: .regularExpression) != nil
            && body.range(of: #"[A-Za-z]{3,}"#, options: .regularExpression) == nil))
      return isMath ? math(body) : groups[0]
    }
    return out.replacingOccurrences(of: dollar, with: "$")
  }

  private static let symbols: [(String, String)] = [
    ("\\times", "×"), ("\\cdot", "·"), ("\\div", "÷"), ("\\pm", "±"), ("\\leq", "≤"), ("\\le", "≤"),
    ("\\geq", "≥"), ("\\ge", "≥"), ("\\neq", "≠"), ("\\ne", "≠"), ("\\approx", "≈"), ("\\equiv", "≡"),
    ("\\Rightarrow", "⇒"), ("\\implies", "⇒"), ("\\rightarrow", "→"), ("\\to", "→"), ("\\leftarrow", "←"),
    ("\\infty", "∞"), ("\\pi", "π"), ("\\alpha", "α"), ("\\beta", "β"), ("\\gamma", "γ"), ("\\delta", "δ"),
    ("\\theta", "θ"), ("\\lambda", "λ"), ("\\mu", "μ"), ("\\sigma", "σ"), ("\\Delta", "Δ"), ("\\sum", "Σ"),
    ("\\prod", "Π"), ("\\lfloor", "⌊"), ("\\rfloor", "⌋"), ("\\lceil", "⌈"), ("\\rceil", "⌉"),
    ("\\ldots", "…"), ("\\cdots", "⋯"), ("\\dots", "…"), ("\\in", "∈"), ("\\notin", "∉"), ("\\subseteq", "⊆"),
    ("\\cup", "∪"), ("\\cap", "∩"), ("\\forall", "∀"), ("\\exists", "∃"), ("\\neg", "¬"), ("\\land", "∧"),
    ("\\lor", "∨"), ("\\circ", "°"), ("\\%", "%"), ("\\$", "$"), ("\\{", "{"), ("\\}", "}"),
  ]

  private static let superscripts: [Character: Character] = [
    "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
    "+": "⁺", "-": "⁻", "n": "ⁿ", "i": "ⁱ",
  ]

  /// One math span → plain text.
  private static func math(_ source: String) -> String {
    var s = source
    for name in ["text", "mathrm", "textbf", "mathbf", "textit", "mathit", "operatorname", "boxed"] {
      s = replace("\\\\" + name + #"\{([^{}]*)\}"#, in: s) { $0[1] }
    }
    // Innermost first, until nothing changes: \frac{a}{b} → a/b, \sqrt{x} → √x.
    var previous = ""
    while previous != s {
      previous = s
      s = replace(#"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}"#, in: s) { "\(group($0[1]))/\(group($0[2]))" }
      s = replace(#"\\sqrt\{([^{}]*)\}"#, in: s) { "√\(group($0[1]))" }
    }
    s = replace(#"\\(left|right|big|Big|bigg|Bigg)(?![A-Za-z])\s*"#, in: s) { _ in "" }
    s = replace(#"\\(quad|qquad)(?![A-Za-z])"#, in: s) { _ in "  " }
    s = replace(#"\\[,;:! ]"#, in: s) { _ in " " }
    s = s.replacingOccurrences(of: "\\\\", with: "\n")
    // Longest names first, so \leq is not read as \le + q.
    for (name, symbol) in symbols.sorted(by: { $0.0.count > $1.0.count }) {
      s = replace(NSRegularExpression.escapedPattern(for: name) + #"(?![A-Za-z])"#, in: s) { _ in symbol }
    }
    s = replace(#"\^\{([^{}]*)\}"#, in: s) { raise($0[1]) }
    s = replace(#"\^([0-9A-Za-z])"#, in: s) { raise($0[1]) }
    s = replace(#"_\{([^{}]*)\}"#, in: s) { "_" + $0[1] }
    s = replace(#"\\([A-Za-z]+)"#, in: s) { $0[1] }
    s = s.replacingOccurrences(of: "{", with: "").replacingOccurrences(of: "}", with: "")
    s = replace(#" {2,}"#, in: s) { _ in " " }
    return s.trimmingCharacters(in: .whitespaces)
  }

  /// "23 - 5y" → "(23 - 5y)" where a fraction needs it; "3" stays "3".
  private static func group(_ s: String) -> String {
    s.range(of: #"^[\w.√²³⁰¹⁴⁵⁶⁷⁸⁹]+$"#, options: .regularExpression) != nil ? s : "(\(s))"
  }

  private static func raise(_ s: String) -> String {
    let mapped = s.compactMap { superscripts[$0] }
    return mapped.count == s.count ? String(mapped) : "^(\(s))"
  }

  private static func replace(
    _ pattern: String, in text: String, options: NSRegularExpression.Options = [], with transform: ([String]) -> String
  ) -> String {
    guard let regex = try? NSRegularExpression(pattern: pattern, options: options) else { return text }
    let ns = text as NSString
    var result = ""
    var last = 0
    for match in regex.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
      result += ns.substring(with: NSRange(location: last, length: match.range.location - last))
      let groups = (0..<match.numberOfRanges).map {
        match.range(at: $0).location == NSNotFound ? "" : ns.substring(with: match.range(at: $0))
      }
      result += transform(groups)
      last = match.range.location + match.range.length
    }
    result += ns.substring(from: last)
    return result
  }
}
