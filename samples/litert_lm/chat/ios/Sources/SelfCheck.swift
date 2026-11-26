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

import LiteRTLM
import UIKit

/// A scripted run of the app for checking it on a phone. It presses the same buttons a person does
/// (through `ChatModel`) and prints one `CHECK:` line per step to the console and to
/// Documents/selfcheck.log. Start the app with launch arguments, for example:
///
///     -selfcheck load,text,image,stop,long,newchat -model e2b -backend gpu
///
/// in the scheme's Run arguments, or from the Mac (devicectl needs the `--`):
///
///     xcrun devicectl device process launch --console --device <iPhone> <bundle identifier> \
///       -- -selfcheck load,text -model e2b
///
/// Steps: `launch` (only what the app shows at launch), `load`; `usefile`, `anyway` and `usecpu`
/// (the buttons of a file-size, a not-enough-memory or a GPU message); `text`, `image` (the
/// example picture), `image2` (a second picture in the same conversation, then a question only
/// the first one answers), `stop` (Stop during a reply, then ask again), `long` (a message longer
/// than the context, then ask again), `newchat`, `background` (waits while the app is sent to the
/// background during a reply, then asks again), `switch` (loads the other model), `essay` (a
/// reply of about 150 words, long enough to time the decoding).
/// `-benchmark YES` also prints LiteRT-LM's own timing of each reply (an experimental API).
/// `-failGPU YES` makes the GPU engine fail before LiteRT-LM is called, to show the GPU error
/// message on a phone whose GPU works.
@MainActor
enum SelfCheck {
  nonisolated static var isRequested: Bool { LaunchArgument.value("selfcheck") != nil }
  nonisolated static var failGPU: Bool { LaunchArgument.value("failGPU") == "YES" }

  static func run(_ chat: ChatModel) async {
    let steps = (LaunchArgument.value("selfcheck") ?? "").split(separator: ",").map(String.init)
    chat.choice = LaunchArgument.value("model") == "e4b" ? Models.e4b : Models.e2b
    chat.useGPU = LaunchArgument.value("backend") != "cpu"
    let benchmark = LaunchArgument.value("benchmark") == "YES"
    if benchmark {
      // LiteRT-LM's own timing (prefill and decode speed) is an experimental API.
      ExperimentalFlags.optIntoExperimentalAPIs()
      ExperimentalFlags.enableBenchmark = true
    }

    log(
      "start steps=\(steps.joined(separator: ",")) model=\(chat.choice.id) gpu=\(chat.useGPU) "
        + "benchmark=\(benchmark)")
    if let notice = chat.notice { log("notice at launch: \(notice.text)") }
    logMemory("launch")

    for step in steps {
      switch step {
      case "launch":
        break
      case "load", "usefile", "anyway", "usecpu":
        // `usefile`, `anyway` and `usecpu` press the button of the message the previous step left.
        let start = Date()
        switch step {
        case "usefile": await chat.perform(.useThisFile)
        case "anyway": await chat.perform(.loadAnyway)
        case "usecpu": await chat.perform(.useCPU)
        default: await chat.load()
        }
        log("\(step) \(chat.loaded?.label ?? "not loaded") seconds=\(seconds(since: start))")
        if let notice = chat.notice {
          log("notice: \(notice.text) [button: \(notice.action?.rawValue ?? "none")]")
        }
        logMemory("after \(step)")
      case "text":
        await ask(chat, "What is the capital of France? Answer in one word.", expect: "paris")
      case "image":
        chat.attachExample()
        await ask(chat, "How much is a latte?", expect: "4.50")
      case "image2":
        // A second picture in the same conversation; then a question only the menu answers.
        chat.attach(signPicture())
        await ask(chat, "What number is on this sign?", expect: "12")
        await ask(chat, "How much is a cappuccino?", expect: "4.25")
      case "stop":
        await stopDuringReply(chat)
        await ask(chat, "What is 2 + 2? Answer with a number.", expect: "4")
      case "long":
        let sentence = "The quick brown fox jumps over the lazy dog near the river bank. "
        await ask(chat, String(repeating: sentence, count: 400) + "Summarize this.", expect: nil)
        await ask(chat, "What is 2 + 2? Answer with a number.", expect: "4")
      case "essay":
        await ask(chat, "Explain in about 150 words why the sky is blue.", expect: nil)
      case "newchat":
        await chat.newChat()
        await ask(chat, "What is the capital of Japan? Answer in one word.", expect: "tokyo")
      case "background":
        await backgroundDuringReply(chat)
        await ask(chat, "What is 2 + 2? Answer with a number.", expect: "4")
      case "switch":
        chat.choice = chat.loaded?.model == Models.e4b ? Models.e2b : Models.e4b
        logMemory("before switch")
        let start = Date()
        await chat.load()
        log("switch \(chat.loaded?.label ?? "failed") seconds=\(seconds(since: start))")
        if let notice = chat.notice { log("notice: \(notice.text)") }
        logMemory("after switch")
      default:
        log("unknown step \(step)")
      }
    }
    log("DONE")
  }

  /// Sends a message the way the Send button does and prints the reply.
  private static func ask(_ chat: ChatModel, _ text: String, expect: String?) async {
    guard chat.phase == .ready else {
      log("ask skipped: phase=\(chat.phase)")
      return
    }
    chat.draft = text
    let withPicture = chat.picture != nil
    let firstNewLine = chat.lines.count
    let start = Date()
    await chat.send()
    let added = chat.lines.dropFirst(firstNewLine)
    let reply = added.first(where: { $0.role == .model })?.text ?? ""
    var result = "ask chars=\(text.count) picture=\(withPicture) seconds=\(seconds(since: start))"
    if let expect {
      result += reply.lowercased().contains(expect) ? " PASS" : " FAIL expected=\(expect)"
    }
    log(result)
    log("reply: \(reply.prefix(300))")
    if let notice = chat.notice { log("notice: \(notice.text)") }
    for note in added where note.role == .note { log("note: \(note.text)") }
    await logBenchmark(chat)
    logMemory("after reply")
  }

  /// Presses Stop once the reply has some text.
  private static func stopDuringReply(_ chat: ChatModel) async {
    chat.draft = "Write a long story about a lighthouse keeper."
    let reply = Task { await chat.send() }
    for _ in 0..<600 {
      if let last = chat.lines.last, last.role == .model, last.text.count >= 120 { break }
      try? await Task.sleep(for: .milliseconds(50))
    }
    log("stop pressed after chars=\(chat.lines.last?.text.count ?? 0)")
    await chat.stop()
    await reply.value
    log("stop ended phase=\(chat.phase) note=\(chat.lines.last?.text ?? "")")
  }

  /// Starts a long reply and waits for the app to come back from the background (the harness
  /// opens another app when it reads "CHECK: replying").
  private static func backgroundDuringReply(_ chat: ChatModel) async {
    chat.draft = "Write a long story about a lighthouse keeper."
    let reply = Task { await chat.send() }
    for _ in 0..<600 {
      if let last = chat.lines.last, last.role == .model, last.text.count >= 40 { break }
      try? await Task.sleep(for: .milliseconds(50))
    }
    log("replying: send the app to the background now")
    let start = Date()
    await reply.value
    log("background reply ended seconds=\(seconds(since: start)) phase=\(chat.phase) "
      + "note=\(chat.lines.last?.text ?? "")")
    // Wait (at most 2 minutes) until the app is in front again before the next step.
    var waited = 0
    while !isActive, waited < 1200 {
      try? await Task.sleep(for: .milliseconds(100))
      waited += 1
    }
    log("active=\(isActive)")
  }

  private static var isActive: Bool {
    UIApplication.shared.applicationState == .active
  }

  /// A second picture, drawn here: a blue sign that says GATE 12.
  private static func signPicture() -> Data {
    let size = CGSize(width: 800, height: 600)
    let format = UIGraphicsImageRendererFormat()
    format.scale = 1
    return UIGraphicsImageRenderer(size: size, format: format).pngData { context in
      UIColor(red: 0.05, green: 0.25, blue: 0.6, alpha: 1).setFill()
      context.fill(CGRect(origin: .zero, size: size))
      let text = "GATE 12" as NSString
      let attributes: [NSAttributedString.Key: Any] = [
        .font: UIFont.boldSystemFont(ofSize: 150), .foregroundColor: UIColor.white,
      ]
      let textSize = text.size(withAttributes: attributes)
      text.draw(
        at: CGPoint(x: (size.width - textSize.width) / 2, y: (size.height - textSize.height) / 2),
        withAttributes: attributes)
    }
  }

  private static func logBenchmark(_ chat: ChatModel) async {
    guard let info = await chat.benchmark() else { return }
    let ttft = String(format: "%.2f", info.timeToFirstTokenInSecond)
    let prefill = String(format: "%.1f", info.lastPrefillTokensPerSecond)
    let decode = String(format: "%.1f", info.lastDecodeTokensPerSecond)
    log("benchmark ttft=\(ttft)s prefill=\(info.lastPrefillTokenCount) tok \(prefill) tok/s "
      + "decode=\(info.lastDecodeTokenCount) tok \(decode) tok/s")
  }

  private static func logMemory(_ when: String) {
    let footprint = Memory.footprint()
    log("memory \(when): footprint=\(mb(footprint.now)) MB peak=\(mb(footprint.peak)) MB "
      + "available=\(mb(Memory.available())) MB")
  }

  private static func mb(_ bytes: Int64) -> Int64 { bytes / 1_048_576 }

  private static func seconds(since start: Date) -> String {
    String(format: "%.1f", Date().timeIntervalSince(start))
  }

  private static func log(_ text: String) {
    let line = "CHECK: \(text)"
    print(line)
    fflush(stdout)
    let url = Models.documents.appendingPathComponent("selfcheck.log")
    let data = Data((ISO8601DateFormatter().string(from: Date()) + " " + line + "\n").utf8)
    if let handle = try? FileHandle(forWritingTo: url) {
      _ = try? handle.seekToEnd()
      try? handle.write(contentsOf: data)
      try? handle.close()
    } else {
      try? data.write(to: url)
    }
  }
}

/// The launch arguments the self-check and `-assumeAvailableMB` read: `-<name> <value>`. They are
/// read from the argument list itself, so an argument before them (such as a `--`) does not matter.
enum LaunchArgument {
  static func value(_ name: String) -> String? {
    let arguments = ProcessInfo.processInfo.arguments
    guard let index = arguments.firstIndex(of: "-" + name), index + 1 < arguments.count else {
      return nil
    }
    return arguments[index + 1]
  }
}

/// The error `-failGPU` makes `ModelRunner.load` throw instead of starting the GPU engine.
struct SimulatedGPUFailure: LocalizedError {
  var errorDescription: String? { "GPU failure simulated with -failGPU (no LiteRT-LM call)." }
}
