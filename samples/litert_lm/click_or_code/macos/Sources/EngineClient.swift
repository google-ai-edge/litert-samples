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

import Darwin
import Foundation

/// The window's side of one engine process: starts it, sends commands, relays its events, writes its stderr to a
/// log file, and reports how it ended. Callbacks arrive on the main queue.
final class EngineClient {
  struct Ending {
    var status: Int32
    var signaled: Bool
  }

  let logURL: URL
  var pid: Int32 { process.processIdentifier }

  private let process = Process()
  private let input: FileHandle
  private let queue = DispatchQueue(label: "engine-client")
  private let log: FileHandle?
  private var stdoutBuffer = Data()
  private var stderrBuffer = Data()
  private var stdoutClosed = false
  private var stderrClosed = false
  private var exitStatus: (status: Int32, signaled: Bool)?
  private var reported = false

  private let onEvent: (EngineEvent) -> Void
  private let onWebGPUError: (String) -> Void
  private let onEnd: (Ending) -> Void

  /// - Parameter onWebGPUError: called for each log line with `Validation error` or `Shape mismatch`: on LiteRT-LM
  ///   0.19.0 a WebGPU validation error does not stop the engine, but the answers after it can be wrong.
  init(
    modelPath: String, cacheDir: String, logDirectory: URL, runtimeLog: String?, warmUp: Bool,
    onEvent: @escaping (EngineEvent) -> Void,
    onWebGPUError: @escaping (String) -> Void,
    onEnd: @escaping (Ending) -> Void
  ) throws {
    self.onEvent = onEvent
    self.onWebGPUError = onWebGPUError
    self.onEnd = onEnd

    try FileManager.default.createDirectory(at: logDirectory, withIntermediateDirectories: true)
    let stamp = DateFormatter()
    stamp.dateFormat = "yyyyMMdd-HHmmss-SSS"
    logURL = logDirectory.appendingPathComponent("engine-\(stamp.string(from: Date())).log")
    FileManager.default.createFile(atPath: logURL.path, contents: nil)
    log = try? FileHandle(forWritingTo: logURL)

    var arguments = [EngineHost.flag, "--model", modelPath, "--cache-dir", cacheDir]
    if let runtimeLog { arguments += ["--runtime-log", runtimeLog] }
    if !warmUp { arguments.append("--no-warm-up") }
    process.executableURL = Bundle.main.executableURL
    process.arguments = arguments
    let stdinPipe = Pipe()
    let stdoutPipe = Pipe()
    let stderrPipe = Pipe()
    process.standardInput = stdinPipe
    process.standardOutput = stdoutPipe
    process.standardError = stderrPipe
    input = stdinPipe.fileHandleForWriting

    stdoutPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
      let data = handle.availableData
      if data.isEmpty { handle.readabilityHandler = nil }
      self?.queue.async { self?.consumeStdout(data) }
    }
    stderrPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
      let data = handle.availableData
      if data.isEmpty { handle.readabilityHandler = nil }
      self?.queue.async { self?.consumeStderr(data) }
    }
    process.terminationHandler = { [weak self] process in
      let signaled = process.terminationReason == .uncaughtSignal
      self?.queue.async {
        self?.exitStatus = (process.terminationStatus, signaled)
        self?.reportIfFinished()
      }
      // A process that left a child holding the pipes open would never send EOF.
      self?.queue.asyncAfter(deadline: .now() + 3) {
        self?.stdoutClosed = true
        self?.stderrClosed = true
        self?.reportIfFinished()
      }
    }

    try process.run()
    let header = "[app] engine pid \(pid): \(arguments.joined(separator: " "))\n"
    try? log?.write(contentsOf: Data(header.utf8))
  }

  func send(_ command: EngineCommand) {
    guard var data = try? JSONEncoder().encode(command) else { return }
    data.append(0x0A)
    // The app ignores SIGPIPE, so an engine that is already gone makes this throw instead of ending the app.
    try? input.write(contentsOf: data)
  }

  /// Asks the engine to end (it exits when its stdin closes) and kills it if it has not ended after `grace`
  /// seconds, for example while it is inside a GPU call.
  func stop(grace: TimeInterval = 5) {
    try? input.close()
    let pid = self.pid
    DispatchQueue.global().asyncAfter(deadline: .now() + grace) { [weak self] in
      guard let self, self.process.isRunning else { return }
      Darwin.kill(pid, SIGKILL)
    }
  }

  /// Ends the engine at once.
  func kill() {
    let pid = self.pid
    if process.isRunning { Darwin.kill(pid, SIGKILL) }
  }

  private func consumeStdout(_ data: Data) {
    if data.isEmpty {
      stdoutClosed = true
      reportIfFinished()
      return
    }
    stdoutBuffer.append(data)
    while let newline = stdoutBuffer.firstIndex(of: 0x0A) {
      let line = stdoutBuffer[stdoutBuffer.startIndex..<newline]
      stdoutBuffer.removeSubrange(stdoutBuffer.startIndex...newline)
      guard let event = try? JSONDecoder().decode(EngineEvent.self, from: line) else { continue }
      DispatchQueue.main.async { self.onEvent(event) }
    }
  }

  private func consumeStderr(_ data: Data) {
    if data.isEmpty {
      stderrClosed = true
      reportIfFinished()
      return
    }
    try? log?.write(contentsOf: data)
    stderrBuffer.append(data)
    while let newline = stderrBuffer.firstIndex(of: 0x0A) {
      let line = String(decoding: stderrBuffer[stderrBuffer.startIndex..<newline], as: UTF8.self)
      stderrBuffer.removeSubrange(stderrBuffer.startIndex...newline)
      if line.contains("Validation error") || line.contains("Shape mismatch") {
        DispatchQueue.main.async { self.onWebGPUError(line) }
      }
    }
  }

  private func reportIfFinished() {
    guard !reported, stdoutClosed, stderrClosed, let exit = exitStatus else { return }
    reported = true
    let note = "[app] engine \(exit.signaled ? "killed by signal" : "exited with status") \(exit.status)\n"
    try? log?.write(contentsOf: Data(note.utf8))
    try? log?.close()
    let ending = Ending(status: exit.status, signaled: exit.signaled)
    DispatchQueue.main.async { self.onEnd(ending) }
  }
}
