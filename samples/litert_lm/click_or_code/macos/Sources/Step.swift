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

/// One `<tool_call>` of a reply, as the window shows it.
struct ToolCall: Equatable {
  /// `computer_use` or `bash`, as the model wrote it.
  var name: String
  var action: String?
  /// The point in the model's 0-999 coordinates, for a click-like action.
  var coordinate: Point?
  var start: Point?
  var end: Point?
  var text: String?
  var keys: [String] = []
  var command: String?
  /// False when the call's JSON did not parse and the call was read by the fallback.
  var strictJSON = true

  var isCode: Bool { name == "bash" }

  struct Point: Equatable {
    var x: Double
    var y: Double
  }
}

/// What a reply says: the reasoning, the one-sentence action, the tool calls.
struct StepReply: Equatable {
  var thinking: String
  var actionLine: String?
  var calls: [ToolCall]

  var first: ToolCall? { calls.first }

  static let empty = StepReply(thinking: "", actionLine: nil, calls: [])
}

/// The agent's steps, as the model's own hybrid agent sends them (microsoft/Mage-VL-Cowork,
/// osworld_eval/mm_agents/mage_hybrid_agent.py), and the reading of a reply. The same text and the same reading as the
/// Android app's Step.kt.
///
/// The first inference is the agent's first step. Its reply is often a call for a screenshot; the app then answers
/// that call with the same picture and asks again, the agent's second step. The two take separate conversations: the
/// file has room for one screenshot.
enum Step {
  /// The tokens the file holds: the system prompt, the screenshot, the request and the reply.
  static let fileTokens = 4096

  /// The cap of one reply: what is left of `fileTokens` after the 1,095 tokens of the system prompt, the 2,040 of the
  /// screenshot and 170 of the request and the template's framing. A second inference also holds the first reply, so
  /// it can reach the end of the file first.
  static let maxOutputTokens = fileTokens - 1095 - 2040 - 170

  /// SHA-256 of the bundled system prompt, the verbatim MAGE_HYBRID_SYSTEM_PROMPT (4,230 bytes).
  static let systemSHA256 = "db6aefb8923bfae10d6548d1f968b0b08be02825a6b171643bec8731351ecb36"

  /// What the runtime says when a prompt does not fit the file.
  private static let tooLongErrors = ["Input token ids are too long", "Prefill input length exceeds"]

  static func isTooLong(_ message: String?) -> Bool {
    guard let message else { return false }
    return tooLongErrors.contains { message.contains($0) }
  }

  /// The agent's step text, after the screenshot in the first step's user turn.
  static func stepText(_ request: String) -> String {
    "\nPlease generate the next move according to the UI screenshot, instruction and previous actions.\n\n"
      + "Instruction: \(request.trimmingCharacters(in: .whitespacesAndNewlines))\n\nPrevious actions:\nNone"
  }

  /// The first step's user turn in the second step's history: its screenshot collapsed.
  static func requestTurn(_ request: String) -> String {
    "[This screenshot has been collapsed.]" + stepText(request)
  }

  /// Around the screenshot in the second step's user turn.
  static let resultBefore = "<tool_response>\n"
  static let resultAfter = "\n</tool_response>"

  /// The reply is the agent's call for a screenshot: the app answers it with the same picture.
  static func asksForScreenshot(_ reply: StepReply) -> Bool {
    guard let call = reply.first else { return false }
    return call.name == "computer_use" && call.action == "screenshot"
  }

  private static func regex(_ pattern: String, _ options: NSRegularExpression.Options = []) -> NSRegularExpression {
    try! NSRegularExpression(pattern: pattern, options: options)
  }

  private static let toolCallRegex = regex(#"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)"#, [.dotMatchesLineSeparators])
  private static let actionRegex = regex(#"^\s*Action:\s*(.+)$"#, [.anchorsMatchLines])
  private static let coordRegex = regex(#""coordinate"\s*:\s*\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]"#)
  private static let actionArgRegex = regex(#""action"\s*:\s*"(\w+)""#)
  private static let nameRegex = regex(#""name"\s*:\s*"(\w+)""#)
  private static let commandRegex = regex(#""command"\s*:\s*"(.*)"\s*\}\s*\}?\s*$"#, [.dotMatchesLineSeparators])

  private static func group(_ regex: NSRegularExpression, _ text: String, _ index: Int = 1) -> String? {
    guard let match = regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)),
      let range = Range(match.range(at: index), in: text)
    else { return nil }
    return String(text[range])
  }

  /// Reads `text`, a whole reply or the part streamed so far.
  static func parse(_ text: String) -> StepReply {
    let end = text.range(of: "</think>")
    var thinking = end.map { String(text[..<$0.lowerBound]) } ?? text
    if thinking.hasPrefix("<think>") { thinking.removeFirst("<think>".count) }
    thinking = thinking.trimmingCharacters(in: .whitespacesAndNewlines)
    let after = end.map { String(text[$0.upperBound...]) } ?? ""
    let action = group(actionRegex, after)?.trimmingCharacters(in: .whitespacesAndNewlines)
    var calls: [ToolCall] = []
    toolCallRegex.enumerateMatches(in: after, range: NSRange(after.startIndex..., in: after)) { match, _, _ in
      guard let match, let range = Range(match.range(at: 1), in: after) else { return }
      if let call = readCall(String(after[range])) { calls.append(call) }
    }
    return StepReply(thinking: thinking, actionLine: action, calls: calls)
  }

  /// One call's JSON; when it does not parse, the fields the app shows, read one by one.
  static func readCall(_ body: String) -> ToolCall? {
    // The model can write a typographic quote where JSON needs a straight one.
    let json = body.replacingOccurrences(of: "“", with: "\"").replacingOccurrences(of: "”", with: "\"")
    if let object = (try? JSONSerialization.jsonObject(with: Data(json.utf8))) as? [String: Any] {
      guard let name = object["name"] as? String, !name.isEmpty else { return nil }
      let args = object["arguments"] as? [String: Any] ?? [:]
      return ToolCall(
        name: name, action: nonEmpty(args["action"]), coordinate: point(args["coordinate"]), start: point(args["start"]),
        end: point(args["end"]), text: nonEmpty(args["text"]), keys: (args["keys"] as? [Any])?.map { "\($0)" } ?? [],
        command: nonEmpty(args["command"]))
    }
    let name = group(nameRegex, json)
    if let command = group(commandRegex, json), let name {
      let plain = command.replacingOccurrences(of: "\\n", with: "\n").replacingOccurrences(of: "\\\"", with: "\"")
        .replacingOccurrences(of: "\\\\", with: "\\")
      return ToolCall(name: name, command: plain, strictJSON: false)
    }
    let x = group(coordRegex, json, 1).flatMap(Double.init)
    let y = group(coordRegex, json, 2).flatMap(Double.init)
    let action = group(actionArgRegex, json)
    let hasPoint = group(coordRegex, json) != nil
    if !hasPoint && action == nil { return nil }
    var coordinate: ToolCall.Point?
    if let x, let y { coordinate = ToolCall.Point(x: x, y: y) }
    return ToolCall(name: name ?? "computer_use", action: action, coordinate: coordinate, strictJSON: false)
  }

  private static func nonEmpty(_ value: Any?) -> String? {
    guard let text = value as? String, !text.isEmpty else { return nil }
    return text
  }

  private static func point(_ value: Any?) -> ToolCall.Point? {
    guard let list = value as? [Any], list.count == 2, let x = number(list[0]), let y = number(list[1]) else { return nil }
    return ToolCall.Point(x: x, y: y)
  }

  private static func number(_ value: Any) -> Double? {
    if let n = value as? NSNumber { return n.doubleValue }
    return (value as? String).flatMap(Double.init)
  }

  /// The model's 0-999 point as pixels of a `width` x `height` screenshot, the way the vendor's agent converts it:
  /// round(x / 999 * (width - 1)), halves to even as in Python.
  static func toPixels(_ point: ToolCall.Point, width: Int, height: Int) -> (x: Int, y: Int) {
    let x = min(max(point.x, 0), 999)
    let y = min(max(point.y, 0), 999)
    return (
      Int((x / 999 * Double(width - 1)).rounded(.toNearestOrEven)),
      Int((y / 999 * Double(height - 1)).rounded(.toNearestOrEven))
    )
  }
}
