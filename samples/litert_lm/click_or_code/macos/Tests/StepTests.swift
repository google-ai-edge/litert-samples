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

import CryptoKit
import XCTest

/// The same cases as the Android app's StepTest.kt and PicturesTest.kt.
final class StepTests: XCTestCase {
  func testReadsAClick() {
    let r = Step.parse(
      "The sailboat is the third picture.\n</think>\n\nAction: Double-click the sailboat photo.\n\n"
        + "<tool_call>\n{\"name\": \"computer_use\", \"arguments\": {\"action\": \"double_click\", "
        + "\"coordinate\": [247, 391]}}\n</tool_call>")
    XCTAssertEqual(r.thinking, "The sailboat is the third picture.")
    XCTAssertEqual(r.actionLine, "Double-click the sailboat photo.")
    let c = r.first!
    XCTAssertFalse(c.isCode)
    XCTAssertEqual(c.action, "double_click")
    XCTAssertEqual(c.coordinate, ToolCall.Point(x: 247, y: 391))
    XCTAssertTrue(c.strictJSON)
  }

  func testReadsACommand() {
    let r = Step.parse(
      "Shell is reliable here.</think>\nAction: Rename the files.\n<tool_call>\n{\"name\": \"bash\", "
        + "\"arguments\": {\"command\": \"cd ~/Pictures/trip && for f in *.JPG; do mv \\\"$f\\\" "
        + "\\\"${f%.JPG}.jpg\\\"; done\"}}\n</tool_call>")
    let c = r.first!
    XCTAssertTrue(c.isCode)
    XCTAssertEqual(c.command, "cd ~/Pictures/trip && for f in *.JPG; do mv \"$f\" \"${f%.JPG}.jpg\"; done")
  }

  func testReadsTypographicQuotes() {
    let r = Step.parse(
      "x</think>\nAction: Click it.\n<tool_call>\n{“name”: “computer_use”, "
        + "“arguments”: {“action”: “click”, “coordinate”: [5, 6]}}\n</tool_call>")
    XCTAssertEqual(r.first?.coordinate, ToolCall.Point(x: 5, y: 6))
  }

  func testReadsABrokenCallFieldByField() {
    let r = Step.parse(
      "x</think>\nAction: Click it.\n<tool_call>\n{\"name\": \"computer_use\", \"arguments\": "
        + "{\"action\": \"click\", \"coordinate\": [12, 34]}\n</tool_call>")
    let c = r.first!
    XCTAssertFalse(c.strictJSON)
    XCTAssertEqual(c.action, "click")
    XCTAssertEqual(c.coordinate, ToolCall.Point(x: 12, y: 34))
  }

  func testAReplyCutBeforeItsCallHasNoStep() {
    let r = Step.parse("Still thinking about the screen")
    XCTAssertEqual(r.thinking, "Still thinking about the screen")
    XCTAssertNil(r.first)
    XCTAssertNil(r.actionLine)
  }

  func testACallCutInsideIsReadAsFarAsItGoes() {
    let r = Step.parse("x</think>\nAction: Run it.\n<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"comm")
    XCTAssertNil(r.first?.command)
  }

  func testReadsEveryCallOfAStep() {
    let r = Step.parse(
      "x</think>\nAction: Type and press enter.\n<tool_call>\n{\"name\": \"computer_use\", \"arguments\": "
        + "{\"action\": \"type\", \"text\": \"hi\"}}\n</tool_call>\n<tool_call>\n{\"name\": \"computer_use\", "
        + "\"arguments\": {\"action\": \"keypress\", \"keys\": [\"enter\"]}}\n</tool_call>")
    XCTAssertEqual(r.calls.count, 2)
    XCTAssertEqual(r.calls[0].text, "hi")
    XCTAssertEqual(r.calls[1].keys, ["enter"])
  }

  func testPointBecomesPixelsAsTheVendorsAgentComputesIt() {
    // round(x / 999 * (w - 1)), halves to even.
    func px(_ x: Double, _ y: Double, _ w: Int, _ h: Int) -> [Int] {
      let p = Step.toPixels(ToolCall.Point(x: x, y: y), width: w, height: h)
      return [p.x, p.y]
    }
    XCTAssertEqual(px(0, 0, 1920, 1080), [0, 0])
    XCTAssertEqual(px(999, 999, 1920, 1080), [1919, 1079])
    XCTAssertEqual(px(499.5, 499.5, 1921, 1081), [960, 540])
    XCTAssertEqual(px(5000, 5000, 1920, 1080), [1919, 1079])
  }

  func testRequestTurnIsTheAgentsStepTextWithItsScreenshotCollapsed() {
    XCTAssertEqual(
      Step.requestTurn("  Open it. "),
      "[This screenshot has been collapsed.]\nPlease generate the next move according to the UI screenshot, "
        + "instruction and previous actions.\n\nInstruction: Open it.\n\nPrevious actions:\nNone")
  }

  func testStepTextIsTheAgentsFirstStep() {
    XCTAssertEqual(
      Step.stepText("  Open it. "),
      "\nPlease generate the next move according to the UI screenshot, instruction and previous "
        + "actions.\n\nInstruction: Open it.\n\nPrevious actions:\nNone")
  }

  func testOnlyTheCallForAScreenshotAsksForTheSecondStep() {
    func reply(_ call: String) -> StepReply {
      Step.parse("Reasoning.\n</think>\n\nAction: Do it.\n\n<tool_call>\n\(call)\n</tool_call>")
    }
    XCTAssertTrue(Step.asksForScreenshot(reply(#"{"name": "computer_use", "arguments": {"action": "screenshot"}}"#)))
    XCTAssertFalse(Step.asksForScreenshot(reply(#"{"name": "computer_use", "arguments": {"action": "click", "coordinate": [1, 2]}}"#)))
    XCTAssertFalse(Step.asksForScreenshot(reply(#"{"name": "bash", "arguments": {"command": "ls"}}"#)))
    XCTAssertFalse(Step.asksForScreenshot(Step.parse("Reasoning, cut before a call")))
  }

  func testBundledSystemPromptIsTheOneThisAppWasCheckedWith() throws {
    let url = try XCTUnwrap(Bundle(for: Self.self).url(forResource: "mage_hybrid_system_prompt", withExtension: "txt"))
    let sha = SHA256.hash(data: try Data(contentsOf: url)).map { String(format: "%02x", $0) }.joined()
    XCTAssertEqual(sha, Step.systemSHA256)
  }

  func testAPromptTooLongIsReadFromEitherMessage() {
    XCTAssertTrue(
      Step.isTooLong(
        "Status Code: 3. Message: Input token ids are too long. Exceeding the maximum number of tokens allowed: 4375 >= 4096"))
    XCTAssertTrue(Step.isTooLong("Prefill input length exceeds the file's signatures"))
    XCTAssertFalse(Step.isTooLong("Status Code: 13. Message: something else"))
    XCTAssertFalse(Step.isTooLong(nil))
  }
}

/// The app's picture preparation against Pillow 13.0.0 (../android/tools/make_picture_rows.py writes the expected rows).
final class PictureTests: XCTestCase {
  private func rows() throws -> [String: Any] {
    let url = try XCTUnwrap(Bundle(for: Self.self).url(forResource: "picture_rows", withExtension: "json"))
    return try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
  }

  /// The pattern of make_picture_rows.py: r = 7x + 3y, g = x*x + 5y, b = x xor y (mod 256).
  private func pattern(_ w: Int, _ h: Int) -> Rgb {
    var bytes = [UInt8](repeating: 0, count: w * h * 3)
    for y in 0..<h {
      for x in 0..<w {
        let i = (y * w + x) * 3
        bytes[i] = UInt8((7 * x + 3 * y) % 256)
        bytes[i + 1] = UInt8((x * x + 5 * y) % 256)
        bytes[i + 2] = UInt8((x ^ y) % 256)
      }
    }
    return Rgb(width: w, height: h, bytes: bytes)
  }

  func testStretchMatchesPillow() throws {
    for row in try XCTUnwrap(try rows()["stretch"] as? [[String: Any]]) {
      let w = try XCTUnwrap(row["w"] as? Int)
      let h = try XCTUnwrap(row["h"] as? Int)
      let out = Pictures.stretch(pattern(w, h))
      XCTAssertEqual(out.width, 1920)
      XCTAssertEqual(out.height, 1088)
      XCTAssertEqual(out.sha256, row["sha256"] as? String, "\(w)x\(h)")
    }
  }

  func testOrientMatchesExifTranspose() throws {
    for row in try XCTUnwrap(try rows()["orient"] as? [[String: Any]]) {
      let orientation = try XCTUnwrap(row["orientation"] as? Int)
      let out = pattern(try XCTUnwrap(row["w"] as? Int), try XCTUnwrap(row["h"] as? Int)).oriented(orientation)
      XCTAssertEqual(out.sha256, row["sha256"] as? String, "orientation \(orientation)")
    }
  }
}
