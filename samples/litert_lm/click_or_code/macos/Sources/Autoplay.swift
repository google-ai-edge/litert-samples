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

/// The recorder: runs a script through the same actions the window's buttons call (open or capture a screenshot, type
/// into the field, Decide, Stop, Copy), checks each step, and writes a run JSON.
///
///   ClickOrCode --autoplay <script.json> --run-out <run.json> [--model <file.litertlm>] [--snapshot-dir <dir>]
///
/// A script:
///   {"name": "...", "typing_characters_per_second": 0, "pause_before_decide_seconds": 0,
///    "pause_after_step_seconds": 0, "hold_seconds_at_end": 0, "quit_when_done": true,
///    "steps": [
///      {"tag": "t1", "example": "<id>", "request": "...", "expect_route": "computer_use", "expect_box": [x, y, w, h]},
///      {"tag": "t2", "picture": "<path, relative to the script>", "request": "...", "expect_route": "bash",
///       "expect_command": "<regex>", "copy": "own", "copy_to_file": "<path>"},
///      {"tag": "t3", "paste": "<path of a PNG>", "expect_picture": "<regex>", "request": "..."},
///      {"tag": "t4", "capture_window": {"owner": "<app>", "title": "<window title>"}, "request": "..."},
///      {"tag": "t5", "capture_screen": true, "request": "..."},
///      {"tag": "t6", "picture": "<not a picture>", "expect_picture_problem": "<regex>"},
///      {"tag": "t7", "request": "", "expect_message": "<regex>"},
///      {"tag": "t8", "request": "...", "stop_after_seconds": 1, "expect_stopped": true},
///      {"tag": "t9", "request": "...", "kill_engine_after_chunks": 5},
///      {"tag": "t10", "restart_engine": true},
///      {"tag": "t11", "wait_for_file": "<path>"},
///      {"tag": "t12", "download": true, "cancel_download_at_bytes": 1000000, "expect_file": "<regex>"},
///      {"tag": "t13", "import_file": "<path>", "expect_file": "<regex>", "expect_problem": "<regex>"},
///      {"tag": "t14", "request": "...", "repeat_request": 900, "expect_error": "<regex>"}
///    ]}
/// A script whose first step is `download` or `import_file` starts without the model file; the others wait for the
/// engine first.
/// `copy` presses Copy with a pasteboard of the recorder's own, so that a run leaves the clipboard as it was
/// (`"copy": "general"` uses the clipboard itself); `copy_to_file` also writes the command to a file for a recording
/// driver. `wait_for_file` holds the run until the file exists (a recording driver arms its recorder meanwhile).
/// `marks_out` in the script names a file that gets one line per moment of the run as it happens, for that driver.
@MainActor
enum Autoplay {
  struct Script: Decodable {
    struct WindowName: Decodable {
      var owner: String
      var title: String?
    }

    struct Item: Decodable {
      var tag: String
      var example: String?
      var picture: String?
      var paste: String?
      var captureWindow: WindowName?
      var captureScreen: Bool?
      var waitForFile: String?
      var restartEngine: Bool?
      var download: Bool?
      var cancelDownloadAtBytes: UInt64?
      var importFile: String?
      var expectFile: String?
      var expectProblem: String?
      var expectError: String?
      var repeatRequest: Int?
      var request: String?
      var expectRoute: String?
      var expectBox: [Double]?
      var expectCommand: String?
      var expectPicture: String?
      var expectPictureProblem: String?
      var expectMessage: String?
      var expectStopped: Bool?
      var copy: String?
      var copyToFile: String?
      var stopAfterSeconds: Double?
      var killEngineAfterChunks: Int?
      var holdSeconds: Double?
    }

    var name: String
    var typingCharactersPerSecond: Double?
    var pauseBeforeDecideSeconds: Double?
    var pauseAfterStepSeconds: Double?
    var holdSecondsAtEnd: Double?
    var quitWhenDone: Bool?
    var marksOut: String?
    var steps: [Item]
  }

  struct Check: Encodable {
    var tag: String
    var what: String
    var expect: String
    var got: String
    var matched: Bool
  }

  struct Mark: Encodable {
    var name: String
    var at: Double
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
    var windowFrameFromTopLeft: [Double]?
    var engines: [AppModel.EngineRecord]
    var decisions: [AppModel.DecisionRecord]
    var checks: [Check]
    var verdict: [String: Int]
    var aborted: String?
    var events: [String]
    var marks: [Mark]
    var snapshots: [String]
  }

  /// One wait (a file, an engine build, a step) longer than this ends the run.
  static let waitLimit: TimeInterval = 3600

  private static var marks: [Mark] = []
  private static var marksHandle: FileHandle?

  static func mark(_ name: String) {
    let at = Date().timeIntervalSince1970
    marks.append(Mark(name: name, at: at))
    try? marksHandle?.write(contentsOf: Data("\(String(format: "%.3f", at)) \(name)\n".utf8))
  }

  static func run(model: AppModel, scriptPath: String, outPath: String?) async {
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
    if let path = script.marksOut {
      FileManager.default.createFile(atPath: path, contents: nil)
      marksHandle = try? FileHandle(forWritingTo: URL(fileURLWithPath: path))
    }
    let base = URL(fileURLWithPath: scriptPath).deletingLastPathComponent()
    let snapshots = Snapshots(directory: Launch.value(after: "--snapshot-dir"))
    var checks: [Check] = []
    var events: [String] = []
    var aborted: String?

    await model.checkModelFile()
    if script.steps.first?.download != true && script.steps.first?.importFile == nil {
      if await !waitForEngine(model) {
        aborted = model.stoppedProblem?.title ?? "the engine did not get ready: \(model.fileText)"
      }
      mark("ready")
      await snapshots.takeAndWait("ready")
    }

    steps: for item in script.steps where aborted == nil {
      let tag = item.tag
      if let path = item.waitForFile {
        mark("\(tag)-waiting")
        if await !waitUntil({ FileManager.default.fileExists(atPath: path) }) {
          aborted = "timed out waiting for \(path)"
          break
        }
        mark("\(tag)-go")
        continue
      }
      if item.restartEngine == true {
        model.restartEngine()
        if await !waitForEngine(model) {
          aborted = model.stoppedProblem?.title ?? "the engine did not start again"
          break
        }
        continue
      }
      if item.download == true || item.importFile != nil {
        if let path = item.importFile {
          model.importFile(URL(fileURLWithPath: path))
        } else {
          model.downloadModel()
        }
        var cancelled = false
        await pause(0.05)
        _ = await waitUntil({ model.fileIsSettled }) {
          if !cancelled, let at = item.cancelDownloadAtBytes, case .downloading(let received) = model.file, received >= at {
            cancelled = true
            model.cancelDownload()
          }
        }
        // A file that checks out goes on to the engine.
        if case .ready = model.file { _ = await waitForEngine(model) }
        mark("\(tag)-file")
        await snapshots.takeAndWait("\(tag)-file")
        if let expected = item.expectFile {
          let got = model.fileText
          checks.append(Check(tag: tag, what: "model file", expect: expected, got: got, matched: matches(got, expected)))
        }
        if let expected = item.expectProblem {
          let got = model.stoppedProblem.map { "\($0.title): \($0.detail)" } ?? "-"
          checks.append(Check(tag: tag, what: "problem", expect: expected, got: got, matched: matches(got, expected)))
        }
        continue
      }
      if let id = item.example {
        guard let example = Example.all.first(where: { $0.id == id }) else {
          events.append("\(tag): no example \(id)")
          continue
        }
        model.useExample(example)
      } else if let path = item.picture {
        model.setPicture(path.hasPrefix("/") ? URL(fileURLWithPath: path) : base.appendingPathComponent(path))
      } else if let path = item.paste {
        // A pasteboard of the recorder's own: the same paste code, and the user's clipboard stays as it was.
        let board = NSPasteboard(name: NSPasteboard.Name("ClickOrCode.recorder"))
        board.clearContents()
        let file = path.hasPrefix("/") ? URL(fileURLWithPath: path) : base.appendingPathComponent(path)
        if let data = try? Data(contentsOf: file) { board.setData(data, forType: .png) }
        model.pastePicture(from: board)
      } else if let window = item.captureWindow {
        guard let id = Capture.windowID(owner: window.owner, title: window.title) else {
          aborted = "\(tag): no window of \(window.owner)"
          break
        }
        await model.captureWindow(id: id, name: window.title ?? window.owner)
      } else if item.captureScreen == true {
        model.captureScreen()
        await pause(1.0)
      }
      if item.example != nil || item.picture != nil || item.paste != nil || item.captureWindow != nil || item.captureScreen == true {
        await pause(0.05)
        _ = await waitUntil({ !model.preparingPicture })
        mark("\(tag)-picture")
        await pause(0.2)
        await snapshots.takeAndWait("\(tag)-picture")
      }
      if let expected = item.expectPicture {
        let shown = model.picture.map { "\($0.name) \($0.width) x \($0.height)" } ?? "-"
        checks.append(Check(tag: tag, what: "picture", expect: expected, got: shown, matched: matches(shown, expected)))
      }
      if let expected = item.expectPictureProblem {
        let shown = model.pictureProblem ?? ""
        checks.append(Check(tag: tag, what: "picture problem", expect: expected, got: shown, matched: matches(shown, expected)))
        await snapshots.takeAndWait("\(tag)-picture-problem")
        continue
      }
      guard var request = item.request else { continue }
      if let times = item.repeatRequest { request = String(repeating: request, count: times) }
      await type(request, into: model, perSecond: item.repeatRequest == nil ? script.typingCharactersPerSecond ?? 0 : 0)
      mark("\(tag)-typed")
      await pause(script.pauseBeforeDecideSeconds)
      if let expected = item.expectMessage {
        model.decide()
        await pause(0.2)
        let shown = model.message ?? ""
        checks.append(Check(tag: tag, what: "message", expect: expected, got: shown, matched: matches(shown, expected)))
        await snapshots.takeAndWait("\(tag)-message")
        continue
      }
      guard model.canDecide else {
        events.append("\(tag): Decide was not enabled (phase \(model.phase), deciding \(model.deciding))")
        continue
      }
      let before = model.records.count
      mark("\(tag)-decide")
      model.decide()
      var stopped = false
      var killed = false
      let pressed = Date()
      let ended = await waitUntil({ !model.deciding }) {
        if !stopped, let seconds = item.stopAfterSeconds, Date().timeIntervalSince(pressed) >= seconds {
          stopped = true
          model.stopDecision()
          mark("\(tag)-stop")
        }
        if !killed, let limit = item.killEngineAfterChunks, (model.decision?.text.count ?? 0) >= limit, let pid = model.enginePid {
          killed = true
          kill(pid, SIGKILL)
        }
      }
      guard ended else {
        aborted = "\(tag): timed out waiting for the step"
        break steps
      }
      mark("\(tag)-done")
      await pause(0.6)
      await snapshots.takeAndWait("\(tag)-answer")
      let record = model.records.count > before ? model.records.last : nil
      if let expected = item.expectRoute {
        let got = record?.route ?? "-"
        checks.append(Check(tag: tag, what: "route", expect: expected, got: got, matched: got == expected))
      }
      if let box = item.expectBox, box.count == 4 {
        let got = record?.pointPixels ?? []
        let inside =
          got.count == 2 && Double(got[0]) >= box[0] && Double(got[0]) < box[0] + box[2] && Double(got[1]) >= box[1]
          && Double(got[1]) < box[1] + box[3]
        checks.append(Check(tag: tag, what: "point in box", expect: "\(box)", got: "\(got)", matched: inside))
      }
      if let expected = item.expectCommand {
        let got = record?.command ?? ""
        checks.append(Check(tag: tag, what: "command", expect: expected, got: got, matched: matches(got, expected)))
      }
      if let expected = item.expectError {
        let got = record?.error ?? "-"
        checks.append(Check(tag: tag, what: "error", expect: expected, got: got, matched: matches(got, expected)))
      }
      if let expected = item.expectStopped {
        let got = record?.stopped ?? false
        checks.append(Check(tag: tag, what: "stopped", expect: "\(expected)", got: "\(got)", matched: got == expected))
      }
      if let copy = item.copy, let command = record?.command {
        await pause(item.holdSeconds)
        model.copyCommand(command, to: copy == "general" ? .general : NSPasteboard(name: NSPasteboard.Name("ClickOrCode.recorder")))
        if let path = item.copyToFile { try? command.write(toFile: path, atomically: true, encoding: .utf8) }
        mark("\(tag)-copied")
        await pause(0.3)
        await snapshots.takeAndWait("\(tag)-copied")
      } else {
        await pause(item.holdSeconds)
      }
      if let problem = model.stoppedProblem {
        events.append("\(tag): \(problem.title)")
        if !problem.canRetry { aborted = problem.title; break steps }
      }
      await pause(script.pauseAfterStepSeconds)
    }
    await pause(script.holdSecondsAtEnd)
    mark("end")

    let frame = Launch.frameFromTopLeft()
    // Close the engine too, so its exit and peak memory are in the record.
    model.closeEngine()
    _ = await waitUntil({ !model.engineRunning })

    let expected = checks.count
    let run = Run(
      script: script.name, scriptPath: scriptPath, started: ISO8601DateFormatter().string(from: started),
      finished: ISO8601DateFormatter().string(from: Date()),
      app: [
        "version": Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "",
        "runtime": "LiteRT-LM 0.19.0 (CLiteRTLM_mac.xcframework of the v0.19.0 release, C API)",
        "backend": "gpu (decoder and vision encoder)",
        "cache_dir": UserDefaults.standard.bool(forKey: "noEngineCache") ? ":nocache" : model.cacheDirectory.path,
        "warm_up": UserDefaults.standard.bool(forKey: "noWarmUp") ? "off" : "on",
        "log_directory": model.logDirectory.path,
      ],
      machine: machine(),
      model: {
        var info = ["expected_bytes": String(model.fileBytes), "expected_sha256": model.store?.entry.sha256 ?? ""]
        if case .ready(let url) = model.file { info["path"] = url.path }
        return info
      }(),
      fileInfo: model.fileInfo, windowFrameFromTopLeft: frame, engines: model.allEngineRecords, decisions: model.records,
      checks: checks, verdict: ["checks": expected, "matched": checks.filter(\.matched).count], aborted: aborted,
      events: events, marks: marks, snapshots: snapshots.taken)
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

  private static func waitForEngine(_ model: AppModel) async -> Bool {
    let ok = await waitUntil({
      if case .failed = model.file { return true }
      return model.phase == .ready || model.isStopped
    })
    await pause(0.3)
    return ok && model.phase == .ready
  }

  private static func type(_ text: String, into model: AppModel, perSecond: Double) async {
    guard perSecond > 0 else {
      model.request = text
      return
    }
    model.request = ""
    for character in text {
      model.request.append(character)
      try? await Task.sleep(nanoseconds: UInt64(1e9 / perSecond))
    }
  }

  private static func pause(_ seconds: Double?) async {
    guard let seconds, seconds > 0 else { return }
    try? await Task.sleep(nanoseconds: UInt64(seconds * 1e9))
  }

  private static func waitUntil(_ condition: () -> Bool, onTick: () -> Void = {}) async -> Bool {
    let deadline = Date().addingTimeInterval(waitLimit)
    while !condition() {
      if Date() > deadline { return false }
      onTick()
      try? await Task.sleep(nanoseconds: 50_000_000)
    }
    return true
  }

  private static func matches(_ text: String, _ pattern: String) -> Bool {
    guard let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) else { return false }
    return regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)) != nil
  }

  private static func machine() -> [String: String] {
    var size = 0
    sysctlbyname("hw.model", nil, &size, nil, 0)
    var name = [CChar](repeating: 0, count: max(size, 1))
    sysctlbyname("hw.model", &name, &size, nil, 0)
    return [
      "model": String(cString: name),
      "chip": Machine.chip,
      "os": ProcessInfo.processInfo.operatingSystemVersionString,
      "physical_memory_bytes": String(ProcessInfo.processInfo.physicalMemory),
      "assumed_physical_memory_bytes": SystemMemory.physicalIsAssumed ? String(SystemMemory.physicalBytes) : "",
    ]
  }
}

/// Saves a picture of the window at named moments of a recorder run (`--snapshot-dir <dir>`) with macOS's
/// `screencapture`, the window only.
@MainActor
final class Snapshots {
  private let directory: URL?
  private(set) var taken: [String] = []

  init(directory: String?) {
    self.directory = directory.map { URL(fileURLWithPath: $0) }
  }

  func takeAndWait(_ name: String) async {
    guard let directory,
      let window = NSApp.windows.first(where: { $0.isVisible && $0.title == "Click or Code" })
    else { return }
    try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/sbin/screencapture")
    process.arguments = ["-x", "-o", "-l", String(window.windowNumber), directory.appendingPathComponent("\(name).png").path]
    guard (try? process.run()) != nil else { return }
    taken.append(name)
    await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
      DispatchQueue.global().async {
        process.waitUntilExit()
        done.resume()
      }
    }
  }
}

extension AppModel {
  var stoppedProblem: Problem? {
    if case .stopped(let problem) = phase { return problem }
    return nil
  }
}
