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

import CoreGraphics
import Darwin
import Foundation
import ImageIO

/// The headless half of the app. The window starts `<this executable> --engine-host --model <path>` and talks to it
/// in JSON lines (Protocol.swift). One engine per process, built once and kept; each inference is a new conversation
/// on it, so nothing crosses from one step to the next. An engine that runs the Mac out of memory takes down this
/// process, not the window.
enum EngineHost {
  static let flag = "--engine-host"

  static func main(arguments: [String]) -> Never {
    // Stdout carries the protocol. File descriptor 1 is pointed at stderr so that anything else printed to stdout
    // lands in the engine log instead of in the middle of an event.
    let protocolFD = dup(STDOUT_FILENO)
    dup2(STDERR_FILENO, STDOUT_FILENO)
    let options = HostOptions(arguments: arguments)
    RuntimeLog.setMinimumLevel(options.runtimeLog)
    // No App Nap for the engine: it is the work the user is waiting for.
    let activity = ProcessInfo.processInfo.beginActivity(
      options: .userInitiatedAllowingIdleSystemSleep, reason: "LiteRT-LM engine")
    let host = HostSession(options: options, events: EventWriter(fd: protocolFD))
    Thread.detachNewThread { host.readCommands() }
    Thread.detachNewThread { host.start() }
    withExtendedLifetime(activity) { dispatchMain() }
  }
}

struct HostOptions {
  var modelPath = ""
  var cacheDir = ":nocache"
  var runtimeLog: String?
  var warmUp = true

  init(arguments: [String]) {
    var i = 0
    while i < arguments.count {
      let next = i + 1 < arguments.count ? arguments[i + 1] : nil
      switch arguments[i] {
      case "--model":
        modelPath = next ?? ""
        i += 1
      case "--cache-dir":
        cacheDir = next ?? ":nocache"
        i += 1
      case "--runtime-log":
        runtimeLog = next
        i += 1
      case "--no-warm-up":
        warmUp = false
      default:
        break
      }
      i += 1
    }
  }
}

/// Writes one event per line to the protocol pipe. Several threads send events.
final class EventWriter: @unchecked Sendable {
  private let fd: Int32
  private let lock = NSLock()
  private let encoder = JSONEncoder()

  init(fd: Int32) {
    self.fd = fd
  }

  func send(_ event: EngineEvent) {
    lock.lock()
    defer { lock.unlock() }
    guard var data = try? encoder.encode(event) else { return }
    data.append(0x0A)
    data.withUnsafeBytes { raw in
      var offset = 0
      while offset < raw.count {
        let n = write(fd, raw.baseAddress! + offset, raw.count - offset)
        if n < 0 && errno == EINTR { continue }
        if n <= 0 { return }
        offset += n
      }
    }
  }
}

final class HostSession: @unchecked Sendable {
  private let options: HostOptions
  private let events: EventWriter
  private let lock = NSLock()
  private let completions = DispatchQueue(label: "engine-turns")
  private var engine: Engine?
  private var conversation: Conversation?
  private var busy = false
  private var cancelRequested = false

  init(options: HostOptions, events: EventWriter) {
    self.options = options
    self.events = events
  }

  func start() {
    let path = options.modelPath
    log("model \(path)")

    // 1. The file's header: fast, and where a file that is not a LiteRT-LM file fails.
    do {
      let info = try FileInfo.read(path: path)
      log(
        "model type \(info.modelType), reads pictures \(info.readsPictures), context \(info.maxContextTokens ?? -1) "
          + "tokens, min runtime \(info.minRuntimeVersion ?? "-")")
      events.send(EngineEvent(event: .fileInfo, file: info))
      guard info.modelType == "llm", info.readsPictures else {
        fail(.file, "This file holds no model that reads pictures.", EngineExit.badFile)
      }
    } catch {
      fail(.file, "LiteRT-LM could not read this file: \(Self.describe(error))", EngineExit.badFile)
    }

    // 2. The engine, set to run the decoder and the vision encoder on the GPU; the decoder loads here.
    events.send(EngineEvent(event: .building))
    let started = DispatchTime.now().uptimeNanoseconds
    let engine: Engine
    do {
      engine = try Engine(modelPath: path, cacheDir: options.cacheDir)
    } catch {
      fail(.engine, Self.describe(error), EngineExit.engineFailed)
    }
    let buildSeconds = Self.seconds(since: started)
    log(String(format: "engine built in %.2f s", buildSeconds))

    // 3. The vision encoder. LiteRT-LM 0.19.0 loads it onto the GPU when the engine's first conversation is created;
    //    then one gray picture goes through it, so that the GPU programs it needs are built now and not at the first
    //    step. Both count as the vision encoder's time.
    var warmUpSeconds: Double?
    if options.warmUp {
      events.send(EngineEvent(event: .warming))
      let visionStart = DispatchTime.now().uptimeNanoseconds
      do {
        try warmUp(engine)
        warmUpSeconds = Self.seconds(since: visionStart)
        log(String(format: "warm-up done in %.2f s", warmUpSeconds ?? 0))
      } catch {
        fail(.engine, "The first picture through the vision encoder failed: \(Self.describe(error))", EngineExit.engineFailed)
      }
    }
    lock.lock()
    self.engine = engine
    lock.unlock()
    let peak = SystemMemory.footprint(pid: getpid())?.peak
    events.send(EngineEvent(event: .ready, buildSeconds: buildSeconds, warmUpSeconds: warmUpSeconds, peakFootprintBytes: peak))
  }

  private func warmUp(_ engine: Engine) throws {
    let picture = FileManager.default.temporaryDirectory.appendingPathComponent("click-or-code-warm-up-gray.png")
    try Picture.writeGray(to: picture)
    defer { try? FileManager.default.removeItem(at: picture) }
    let conversation = try Conversation(engine: engine, messagesJSON: nil)
    let done = DispatchSemaphore(value: 0)
    var failure: String?
    try conversation.send(messageJSON: MessageJSON.warmUp(picturePath: picture.path), maxOutputTokens: 1) { event in
      switch event {
      case .text: break
      case .finished: done.signal()
      case .failed(let message):
        // One output token was asked for, so the reply ends at the limit.
        if !Self.isLimit(message) { failure = message }
        done.signal()
      }
    }
    if done.wait(timeout: .now() + 900) == .timedOut { throw HostError("the warm-up did not finish in 15 minutes") }
    if let failure { throw HostError(failure) }
  }

  // MARK: One inference

  func step(_ command: EngineCommand) {
    let id = command.id ?? 0
    lock.lock()
    guard let engine, !busy else {
      lock.unlock()
      events.send(EngineEvent(event: .error, id: id, stage: .generate, message: "The engine is not ready or still deciding."))
      return
    }
    busy = true
    cancelRequested = false
    // The previous conversation is deleted first: one conversation at a time.
    conversation = nil
    lock.unlock()
    let request = command.request ?? ""
    let picture = command.picture ?? ""
    let got = DispatchTime.now().uptimeNanoseconds
    let created: Conversation
    do {
      created = try Conversation(
        engine: engine,
        messagesJSON: MessageJSON.preface(system: command.system ?? "", request: request, firstReply: command.firstReply))
    } catch {
      setIdle()
      events.send(EngineEvent(event: .error, id: id, stage: .generate, message: Self.describe(error)))
      return
    }
    lock.lock()
    conversation = created
    let stopNow = cancelRequested
    lock.unlock()
    var stats = TurnStats()
    stats.conversationSeconds = Self.seconds(since: got)
    if stopNow {
      completions.async { self.finish(id: id, conversation: created, sent: got, stats: stats, error: nil) }
      return
    }
    log("step \(id): \(command.firstReply == nil ? "first" : "second") inference, picture \(picture)")
    let message =
      command.firstReply == nil
      ? MessageJSON.firstStep(picturePath: picture, request: request) : MessageJSON.secondStep(picturePath: picture)
    run(id: id, conversation: created, messageJSON: message, maxOutputTokens: command.maxOutputTokens ?? Step.maxOutputTokens, stats: stats)
  }

  private func run(id: Int, conversation: Conversation, messageJSON: String, maxOutputTokens: Int, stats: TurnStats) {
    let sent = DispatchTime.now().uptimeNanoseconds
    var stats = stats
    let state = NSLock()
    do {
      try conversation.send(messageJSON: messageJSON, maxOutputTokens: maxOutputTokens) { [self] event in
        switch event {
        case .text(let piece):
          let t = Self.seconds(since: sent)
          state.lock()
          stats.chunks += 1
          if stats.firstTextSeconds == nil { stats.firstTextSeconds = t }
          state.unlock()
          events.send(EngineEvent(event: .chunk, id: id, text: piece, t: t))
        case .finished:
          completions.async { self.finish(id: id, conversation: conversation, sent: sent, stats: stats, error: nil) }
        case .failed(let message):
          completions.async { self.finish(id: id, conversation: conversation, sent: sent, stats: stats, error: message) }
        }
      }
    } catch {
      setIdle()
      events.send(EngineEvent(event: .error, id: id, stage: .generate, message: Self.describe(error)))
    }
  }

  private func finish(id: Int, conversation: Conversation, sent: UInt64, stats: TurnStats, error: String?) {
    var stats = stats
    stats.replySeconds = Self.seconds(since: sent)
    let cancelled = isCancelRequested()
    let limit = error.map(Self.isLimit) ?? false
    // A cancelled reply ends with a CANCELLED error; the output-token limit and a full context end with "Max number
    // of tokens reached" (the Python API treats both as the end of the reply, not as failures).
    let failure = (cancelled || limit) ? nil : error
    if !cancelled { stats.counters = conversation.counters() }
    if let memory = SystemMemory.footprint(pid: getpid()) {
      stats.footprintBytes = memory.current
      stats.peakFootprintBytes = memory.peak
    }
    let counters = stats.counters
    log(
      "step \(id) \(cancelled ? "cancelled" : failure == nil ? (limit ? "ended at the token limit" : "done") : "failed"): "
        + "first text \(stats.firstTextSeconds ?? -1) s, reply \(stats.replySeconds ?? -1) s, "
        + "prefill \(counters?.prefillTokens ?? -1) tokens at \(counters?.prefillTokensPerSecond ?? -1) tok/s, "
        + "decode \(counters?.decodeTokens ?? -1) tokens at \(counters?.decodeTokensPerSecond ?? -1) tok/s, "
        + "time to first token \(counters?.timeToFirstToken ?? -1) s, context \(counters?.contextTokens ?? -1), "
        + "peak footprint \(stats.peakFootprintBytes ?? 0) bytes"
        + (error.map { ", runtime: \($0)" } ?? ""))
    lock.lock()
    busy = false
    lock.unlock()
    events.send(
      EngineEvent(event: .turnDone, id: id, message: failure, stats: stats, cancelled: cancelled, limitReached: limit))
    if let failure, EngineExit.isDeadline(failure) {
      // A wait in the runtime passed its time limit: the next conversation of this engine fails ("already has" a
      // pending task), so the process ends and the window offers to start a new one.
      log("exit (the reply passed a time limit of the runtime), peak footprint \(SystemMemory.footprint(pid: getpid())?.peak ?? 0) bytes")
      _exit(EngineExit.deadline)
    }
  }

  func cancel() {
    lock.lock()
    let conversation = busy ? self.conversation : nil
    if busy { cancelRequested = true }
    lock.unlock()
    log("cancel")
    conversation?.cancel()
  }

  private func setIdle() {
    lock.lock()
    busy = false
    lock.unlock()
  }

  private func isCancelRequested() -> Bool {
    lock.lock()
    defer { lock.unlock() }
    return cancelRequested
  }

  /// Runs on its own thread for the life of the process.
  func readCommands() {
    let decoder = JSONDecoder()
    while let line = readLine(strippingNewline: true) {
      guard let data = line.data(using: .utf8), let command = try? decoder.decode(EngineCommand.self, from: data) else {
        continue
      }
      switch command.cmd {
      case .step:
        // On its own thread: creating a conversation blocks, and a cancel must still be read meanwhile.
        Thread.detachNewThread { self.step(command) }
      case .cancel: cancel()
      case .quit: finishProcess("quit")
      }
    }
    finishProcess("the window closed the pipe")
  }

  private func finishProcess(_ why: String) -> Never {
    log("exit (\(why)), peak footprint \(SystemMemory.footprint(pid: getpid())?.peak ?? 0) bytes")
    // _exit: no atexit handlers or C++ static destructors while a runtime thread may still run.
    _exit(0)
  }

  private func fail(_ stage: EngineEvent.Stage, _ message: String, _ code: Int32) -> Never {
    log("error (\(stage.rawValue)): \(message)")
    events.send(EngineEvent(event: .error, stage: stage, message: message))
    _exit(code)
  }

  private func log(_ line: String) {
    FileHandle.standardError.write(Data("[engine] \(line)\n".utf8))
  }

  static func isLimit(_ message: String) -> Bool {
    message.contains("Max number of tokens reached")
  }

  static func seconds(since start: UInt64) -> Double {
    Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
  }

  /// The error's text with anything that is not printable left out (a runtime message can quote raw file bytes).
  static func describe(_ error: Error) -> String {
    let text = (error as? LocalizedError)?.errorDescription ?? "\(error)"
    let printable = text.unicodeScalars.filter { $0 == "\n" || ($0.value >= 0x20 && $0.value < 0x7F) }
    return String(String.UnicodeScalarView(printable)).trimmingCharacters(in: .whitespacesAndNewlines)
  }
}

struct HostError: LocalizedError {
  let message: String
  init(_ message: String) { self.message = message }
  var errorDescription: String? { message }
}
