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

import AppKit
import Foundation

/// The recorder: runs a script through the same actions the window's buttons call (open the model, click an
/// example, open or paste a picture, type into the field, Ask, Stop, New conversation), checks each answer, and
/// writes a run JSON.
///
///   PictureQuestions --model <file.litertlm> --autoplay <script.json> --run-out <run.json> [--snapshot-dir <dir>]
///
/// A script:
///   {"name": "...", "typing_characters_per_second": 0, "pause_before_ask_seconds": 0, "pause_after_answer_seconds": 0,
///    "pause_before_open_seconds": 0, "hold_seconds_at_end": 0, "quit_when_done": true,
///    "conversations": [
///      {"example": "<example id>", "expect": "<regex>", "turns": [...follow-ups...]},
///      {"picture": "<path, relative to the script>", "reasoning": "low", "turns": [{"text": "...", "expect": "..."}]},
///      {"paste": "<path>", "turns": [...]},
///      {"picture": "<a file that is not a picture>", "expect_picture_problem": "<regex>"},
///      {"new_conversation": true, "turns": [...]}  (the same picture, no earlier questions)
///    ]}
/// A turn may also hold "stop_after_chunks" (press Stop), "kill_engine_after_chunks" (end the engine process with
/// SIGKILL, as a force quit or the out-of-memory killer would) or "pause_engine_after_chunks" with
/// "pause_engine_seconds" (stop the engine process with SIGSTOP and continue it after that many seconds).
@MainActor
enum Autoplay {
  struct Script: Decodable {
    struct Turn: Decodable {
      var text: String
      var expect: String?
      var stopAfterChunks: Int?
      var killEngineAfterChunks: Int?
      var pauseEngineAfterChunks: Int?
      var pauseEngineSeconds: Double?
    }

    struct Conversation: Decodable {
      var example: String?
      var picture: String?
      var paste: String?
      var newConversation: Bool?
      var reasoning: String?
      /// For an example: the regex its answer has to contain.
      var expect: String?
      var expectPictureProblem: String?
      var restartEngine: Bool?
      var turns: [Turn]?
    }

    var name: String
    var typingCharactersPerSecond: Double?
    var pauseBeforeAskSeconds: Double?
    var pauseAfterAnswerSeconds: Double?
    var pauseBeforeOpenSeconds: Double?
    var holdSecondsAtEnd: Double?
    var quitWhenDone: Bool?
    var conversations: [Conversation]
  }

  struct Check: Encodable {
    var step: String
    var question: String?
    var expect: String?
    var answer: String
    var matched: Bool?
  }

  struct Run: Encodable {
    var script: String
    var scriptPath: String
    var started: String
    var finished: String
    var app: [String: String]
    var machine: [String: String]
    var model: [String: String]
    var fileInfo: FileInfo?
    var replyFormat: String?
    var windowFrameFromTopLeft: [Double]?
    var engines: [AppModel.EngineRecord]
    var turns: [AppModel.TurnRecord]
    var checks: [Check]
    var verdict: [String: Int]
    var aborted: String?
    var abortedDetail: String?
    var events: [String]
    var snapshots: [String]
  }

  /// One wait (an engine build, an answer) longer than this ends the run.
  static let waitLimit: TimeInterval = 3600

  static func run(model: AppModel, scriptPath: String, modelPath: String, outPath: String?) async {
    let started = Date()
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    let script: Script
    do {
      script = try decoder.decode(Script.self, from: Data(contentsOf: URL(fileURLWithPath: scriptPath)))
    } catch {
      FileHandle.standardError.write(Data("cannot read the autoplay script \(scriptPath): \(error)\n".utf8))
      return
    }
    let base = URL(fileURLWithPath: scriptPath).deletingLastPathComponent()
    NSApp.activate(ignoringOtherApps: true)
    let snapshots = Snapshots(directory: Launch.value(after: "--snapshot-dir"))
    var checks: [Check] = []
    var events: [String] = []
    var aborted: AppModel.Problem?

    await pause(script.pauseBeforeOpenSeconds)
    model.openModel(URL(fileURLWithPath: modelPath))
    if await !waitForEngine(model, snapshots: snapshots, tag: "open") {
      aborted = model.stoppedProblem ?? AppModel.Problem(title: "timed out waiting for the engine", detail: "", canRetry: false)
    }

    conversations: for (index, conversation) in script.conversations.enumerated() where aborted == nil {
      let c = index + 1
      if conversation.restartEngine == true {
        model.restartEngine()
        if await !waitForEngine(model, snapshots: snapshots, tag: "c\(c)-restart") {
          aborted = model.stoppedProblem ?? AppModel.Problem(title: "timed out waiting for the engine", detail: "", canRetry: false)
          break
        }
      }
      if let reasoning = conversation.reasoning, let value = ReasoningStrength(rawValue: reasoning) {
        model.strength = value
      }
      var turns = conversation.turns ?? []
      if let id = conversation.example {
        guard let example = Example.all.first(where: { $0.id == id }) else {
          events.append("conversation \(c): no example \(id)")
          continue
        }
        // Clicking an example sets its picture and asks its question.
        model.useExample(example)
        if let answered = await waitForAnswer(model, turn: Script.Turn(text: example.question), snapshots: snapshots, tag: "c\(c)-t1") {
          checks.append(Check(step: "c\(c)-t1", question: example.question, expect: conversation.expect, answer: answered,
                              matched: conversation.expect.map { AnswerCheck.matches(answered, $0) }))
        } else {
          aborted = AppModel.Problem(title: "timed out waiting for an answer", detail: "", canRetry: false)
          break
        }
        await snapshots.takeAndWait("c\(c)-t1-answer")
        if let problem = model.stoppedProblem {
          events.append("conversation \(c): \(problem.title)")
          if !problem.canRetry { aborted = problem; break }
          continue conversations
        }
      } else if let path = conversation.picture ?? conversation.paste {
        let url = path.hasPrefix("/") ? URL(fileURLWithPath: path) : base.appendingPathComponent(path)
        if conversation.paste != nil {
          // A pasteboard of the recorder's own: the same paste code, and the user's clipboard stays as it was.
          let board = NSPasteboard(name: NSPasteboard.Name("PictureQuestions.recorder"))
          board.clearContents()
          if let data = try? Data(contentsOf: url) { board.setData(data, forType: .png) }
          model.pastePicture(from: board)
        } else {
          model.setPicture(url)
        }
        await pause(0.5)
        if let expected = conversation.expectPictureProblem {
          let shown = model.pictureProblem ?? ""
          checks.append(Check(step: "c\(c)-picture", question: nil, expect: expected, answer: shown,
                              matched: AnswerCheck.matches(shown, expected)))
          events.append("conversation \(c): picture problem shown: \(shown)")
          await snapshots.takeAndWait("c\(c)-picture-problem")
          continue
        }
        await snapshots.takeAndWait("c\(c)-picture")
      } else if conversation.newConversation == true {
        model.newConversation()
      }

      let first = conversation.example != nil ? 2 : 1
      for (t, turn) in turns.enumerated() {
        let n = t + first
        await type(turn.text, into: model, perSecond: script.typingCharactersPerSecond ?? 0)
        await pause(script.pauseBeforeAskSeconds)
        guard model.canAsk else {
          events.append("conversation \(c) turn \(n): Ask was not enabled (phase \(model.phase))")
          continue
        }
        model.ask()
        guard let answer = await waitForAnswer(model, turn: turn, snapshots: snapshots, tag: "c\(c)-t\(n)") else {
          aborted = AppModel.Problem(title: "timed out waiting for an answer", detail: "", canRetry: false)
          break conversations
        }
        await snapshots.takeAndWait("c\(c)-t\(n)-answer")
        checks.append(Check(step: "c\(c)-t\(n)", question: turn.text, expect: turn.expect, answer: answer,
                            matched: turn.expect.map { AnswerCheck.matches(answer, $0) }))
        if let problem = model.stoppedProblem {
          events.append("conversation \(c) turn \(n): \(problem.title)")
          if !problem.canRetry { aborted = problem; break conversations }
          continue conversations
        }
        await pause(script.pauseAfterAnswerSeconds)
      }
      turns.removeAll()
    }
    await pause(script.holdSecondsAtEnd)

    let frame = WindowConfigurator.frameFromTopLeft()
    // Close the engine too, so its exit and peak memory are in the record.
    model.closeEngine()
    _ = await waitUntil(model, { !$0.engineRunning })

    let expected = checks.filter { $0.expect != nil }
    let run = Run(
      script: script.name, scriptPath: scriptPath,
      started: ISO8601DateFormatter().string(from: started),
      finished: ISO8601DateFormatter().string(from: Date()),
      app: [
        "version": Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "",
        "runtime": "LiteRT-LM 0.19.0 (CLiteRTLM_mac.xcframework of the v0.19.0 release, C API)",
        "backend": "gpu (decoder and vision encoder)",
        "cache_dir": ":nocache",
        "sampler": "top-k 1 (greedy)",
        "warm_up": UserDefaults.standard.bool(forKey: "noWarmUp") ? "off" : "on",
        "log_directory": model.logDirectory.path,
      ],
      machine: machine(),
      model: [
        "path": modelPath,
        "size_bytes": model.file.map { String($0.size) } ?? "",
        "known_as": model.file?.knownName ?? "",
      ],
      fileInfo: model.fileInfo, replyFormat: model.format?.rawValue, windowFrameFromTopLeft: frame,
      engines: model.allEngineRecords, turns: model.turnRecords, checks: checks,
      verdict: [
        "checks": checks.count, "expected": expected.count,
        "matched": expected.filter { $0.matched == true }.count,
      ],
      aborted: aborted?.title, abortedDetail: aborted?.detail, events: events, snapshots: snapshots.taken)
    if let outPath {
      let encoder = JSONEncoder()
      encoder.keyEncodingStrategy = .convertToSnakeCase
      encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
      do {
        try encoder.encode(run).write(to: URL(fileURLWithPath: outPath))
      } catch {
        FileHandle.standardError.write(Data("cannot write \(outPath): \(error)\n".utf8))
      }
    }
    if script.quitWhenDone == true { NSApp.terminate(nil) }
  }

  /// Waits for a ready engine (or a stopped one); takes a picture of the build screen once.
  private static func waitForEngine(_ model: AppModel, snapshots: Snapshots, tag: String) async -> Bool {
    var took = false
    let ok = await waitUntil(model, { $0.phase == .ready || $0.isStopped }) {
      if !took, case .building(let since) = model.phase, Date().timeIntervalSince(since) >= 5 {
        took = true
        snapshots.take("\(tag)-building")
      }
    }
    await pause(0.5)
    await snapshots.takeAndWait(model.isStopped ? "\(tag)-stopped" : "\(tag)-ready")
    return ok && !model.isStopped
  }

  /// Waits until the reply of the turn that was just asked ends, doing what the turn says along the way; returns the
  /// answer shown (empty for a failure), or nil when the wait timed out.
  private static func waitForAnswer(_ model: AppModel, turn: Script.Turn, snapshots: Snapshots, tag: String) async
    -> String?
  {
    var tookStreaming = false
    var tookQuiet = false
    var pressedStop = false
    var killed = false
    var paused = false
    let answered = await waitUntil(model, { $0.phase != .answering }) {
      let chunks = model.turns.last?.chunks ?? 0
      if !tookStreaming, chunks >= 30 {
        tookStreaming = true
        snapshots.take("\(tag)-streaming")
      }
      // The window's "No new text" line, once it has shown for a while.
      if !tookQuiet, let last = model.turns.last,
        Date().timeIntervalSince(last.lastChunkAt ?? last.sentAt) >= 60
      {
        tookQuiet = true
        snapshots.take("\(tag)-no-new-text")
      }
      if !pressedStop, let limit = turn.stopAfterChunks, chunks >= limit {
        pressedStop = true
        model.stopAnswer()
      }
      if !killed, let limit = turn.killEngineAfterChunks, chunks >= limit, let pid = model.enginePid {
        killed = true
        kill(pid, SIGKILL)
      }
      if !paused, let limit = turn.pauseEngineAfterChunks, chunks >= limit, let pid = model.enginePid {
        paused = true
        kill(pid, SIGSTOP)
        let seconds = turn.pauseEngineSeconds ?? 660
        DispatchQueue.global().asyncAfter(deadline: .now() + seconds) { kill(pid, SIGCONT) }
      }
    }
    guard answered else { return nil }
    // A reply that ended the engine: wait for the window to settle.
    _ = await waitUntil(model, { $0.phase == .ready || $0.isStopped })
    await pause(0.5)
    guard let last = model.turns.last else { return "" }
    return ReplyText.parse(last.raw, format: model.format ?? .plain).answer
  }

  private static func type(_ text: String, into model: AppModel, perSecond: Double) async {
    guard perSecond > 0 else {
      model.draft = text
      return
    }
    model.draft = ""
    for character in text {
      model.draft.append(character)
      try? await Task.sleep(nanoseconds: UInt64(1e9 / perSecond))
    }
  }

  private static func pause(_ seconds: Double?) async {
    guard let seconds, seconds > 0 else { return }
    try? await Task.sleep(nanoseconds: UInt64(seconds * 1e9))
  }

  private static func waitUntil(_ model: AppModel, _ condition: (AppModel) -> Bool, onTick: () -> Void = {}) async -> Bool {
    let deadline = Date().addingTimeInterval(waitLimit)
    while !condition(model) {
      if Date() > deadline { return false }
      onTick()
      try? await Task.sleep(nanoseconds: 100_000_000)
    }
    return true
  }

  private static func machine() -> [String: String] {
    var size = 0
    sysctlbyname("hw.model", nil, &size, nil, 0)
    var name = [CChar](repeating: 0, count: max(size, 1))
    sysctlbyname("hw.model", &name, &size, nil, 0)
    return [
      "model": String(cString: name),
      "os": ProcessInfo.processInfo.operatingSystemVersionString,
      "physical_memory_bytes": String(ProcessInfo.processInfo.physicalMemory),
      "assumed_physical_memory_bytes": SystemMemory.physicalIsAssumed ? String(SystemMemory.physicalBytes) : "",
    ]
  }
}

/// Saves a picture of the window at named moments of a recorder run (`--snapshot-dir <dir>`) with macOS's
/// `screencapture`, the window only. The Screen Recording permission is the one of the app that started this one.
@MainActor
final class Snapshots {
  private let directory: URL?
  private(set) var taken: [String] = []

  init(directory: String?) {
    self.directory = directory.map { URL(fileURLWithPath: $0) }
  }

  func take(_ name: String) {
    _ = start(name)
  }

  func takeAndWait(_ name: String) async {
    guard let process = start(name) else { return }
    await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
      DispatchQueue.global().async {
        process.waitUntilExit()
        done.resume()
      }
    }
  }

  private func start(_ name: String) -> Process? {
    guard let directory,
      let window = NSApp.windows.first(where: { $0.isVisible && $0.title == "Picture Questions" })
    else { return nil }
    try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/sbin/screencapture")
    process.arguments = ["-x", "-o", "-l", String(window.windowNumber), directory.appendingPathComponent("\(name).png").path]
    guard (try? process.run()) != nil else { return nil }
    taken.append(name)
    return process
  }
}

enum AnswerCheck {
  static func matches(_ answer: String, _ pattern: String) -> Bool {
    guard let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) else { return false }
    return regex.firstMatch(in: answer, range: NSRange(answer.startIndex..., in: answer)) != nil
  }
}

extension AppModel {
  var stoppedProblem: Problem? {
    if case .stopped(let problem) = phase { return problem }
    return nil
  }
}
