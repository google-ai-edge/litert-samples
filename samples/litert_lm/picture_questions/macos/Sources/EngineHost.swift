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

/// The headless half of the app. For each model file the window starts `<this executable> --engine-host --model
/// <path>` and talks to it in JSON lines (Protocol.swift). One engine per process, built once and kept: a new
/// picture is a new conversation on the same engine. Both models here are attention-only decoders, so a
/// conversation keeps its own KV cache and leaves nothing behind for the next one. An engine that runs the Mac out
/// of memory takes down this process, not the window.
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
  private var format = ReplyFormat.plain
  private var conversation: Conversation?
  /// After a cancel the conversation must not be used again, only deleted (C API, conversation.h).
  private var conversationRetired = false
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
    var info: FileInfo?
    do {
      info = try FileInfo.read(path: path)
    } catch {
      let message = Self.describe(error)
      if message.contains("section size exceeds maximum allowed limit") {
        // LiteRT-LM 0.19.0's file reader refuses a vision encoder section larger than 2 GB (kMaxVisionSectionSize,
        // schema/model_info/model_info.cc), while its engine loads the same file. Both model files of this sample
        // have one (7.4 GB and 2.6 GB), so the engine reads the file instead.
        log("file header not read (\(message)); the engine reads the file")
      } else {
        fail(
          .file,
          message.contains("Invalid magic number")
            ? "LiteRT-LM could not read this file: it does not begin the way a .litertlm file does (it may be "
              + "another kind of file, or a download that did not finish)."
            : "LiteRT-LM could not read this file: \(message)",
          EngineExit.badFile)
      }
    }
    if let info {
      log(
        "model type \(info.modelType), reads pictures \(info.readsPictures), context \(info.maxContextTokens ?? -1) "
          + "tokens, min runtime \(info.minRuntimeVersion ?? "-")")
      events.send(EngineEvent(event: .fileInfo, file: info))
      guard info.modelType == "llm" else {
        fail(
          .file, "This file holds an \(info.modelType) model, not a model that answers questions.", EngineExit.badFile)
      }
      guard info.readsPictures else {
        fail(
          .file, "This model does not read pictures: its file has no vision encoder. Open a vision-language model.",
          EngineExit.badFile)
      }
    }

    // 2. The engine, set to run the decoder and the vision encoder on the GPU; the decoder loads here.
    events.send(EngineEvent(event: .building))
    let started = DispatchTime.now().uptimeNanoseconds
    let engine: Engine
    do {
      // ":nocache": LiteRT-LM would otherwise write GPU caches as large as the model next to it.
      engine = try Engine(modelPath: path, cacheDir: ":nocache")
    } catch {
      fail(.engine, Self.describe(error), EngineExit.engineFailed)
    }
    let buildSeconds = Self.seconds(since: started)

    // 3. The vision encoder. LiteRT-LM 0.19.0 loads it onto the GPU when the engine's first conversation is created
    //    (the reply-format probe here); then one small picture goes through it, so that the GPU programs it needs are
    //    built now and not at the user's first question (measured: Muse-Glimmer-30B 12.7 s, Qwen2.5-VL-72B 32.7 s on
    //    an M4 Max). Both count as the vision encoder's time.
    if options.warmUp { events.send(EngineEvent(event: .warming)) }
    let visionStart = DispatchTime.now().uptimeNanoseconds
    format = Self.replyFormat(engine)
    log(String(format: "engine built in %.1f s, reply format %@", buildSeconds, format.rawValue))
    var warmUpSeconds: Double?
    if options.warmUp {
      do {
        try warmUp(engine)
        warmUpSeconds = Self.seconds(since: visionStart)
        log(String(format: "warm-up done in %.1f s", warmUpSeconds ?? 0))
      } catch {
        fail(.engine, "The first picture through the vision encoder failed: \(Self.describe(error))", EngineExit.engineFailed)
      }
    }
    lock.lock()
    self.engine = engine
    lock.unlock()
    let peak = SystemMemory.footprint(pid: getpid())?.peak
    events.send(
      EngineEvent(
        event: .ready, format: format, buildSeconds: buildSeconds, warmUpSeconds: warmUpSeconds,
        peakFootprintBytes: peak))
  }

  /// The model's chat format, from its template's rendering of a short user message.
  private static func replyFormat(_ engine: Engine) -> ReplyFormat {
    guard let probe = try? Conversation(engine: engine, messagesJSON: nil, maxOutputTokens: 1),
      let rendered = probe.render(messageJSON: MessageJSON.userTurn(text: "Hello"))
    else { return .plain }
    return rendered.contains("<|start|>user<|message|>") ? .reasoningMessages : .plain
  }

  private func warmUp(_ engine: Engine) throws {
    let picture = FileManager.default.temporaryDirectory.appendingPathComponent("warm-up-gray-448.png")
    try WarmUpPicture.write(to: picture)
    let conversation = try Conversation(engine: engine, messagesJSON: nil, maxOutputTokens: 1)
    let done = DispatchSemaphore(value: 0)
    var failure: String?
    try conversation.send(messageJSON: MessageJSON.userTurn(picturePath: picture.path, text: "Hi")) { event in
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

  // MARK: Questions

  func ask(id: Int, picture: String, text: String, system: String?, maxOutputTokens: Int) {
    lock.lock()
    guard let engine, !busy else {
      lock.unlock()
      events.send(EngineEvent(event: .error, id: id, stage: .generate, message: "The engine is not ready or still answering."))
      return
    }
    busy = true
    cancelRequested = false
    // The previous conversation (another picture, or this one before "New conversation") is deleted first.
    conversation = nil
    conversationRetired = false
    lock.unlock()
    let created: Conversation
    do {
      created = try Conversation(
        engine: engine, messagesJSON: system.map { MessageJSON.systemMessages($0) }, maxOutputTokens: maxOutputTokens)
    } catch {
      setIdle()
      events.send(EngineEvent(event: .error, id: id, stage: .generate, message: Self.describe(error)))
      return
    }
    lock.lock()
    conversation = created
    lock.unlock()
    log("turn \(id): new conversation, picture \(picture), system \(system ?? "-"), max output \(maxOutputTokens)")
    run(id: id, conversation: created, messageJSON: MessageJSON.userTurn(picturePath: picture, text: text))
  }

  func followUp(id: Int, text: String) {
    lock.lock()
    guard let conversation, !conversationRetired, !busy else {
      lock.unlock()
      events.send(
        EngineEvent(
          event: .error, id: id, stage: .generate,
          message: "There is no conversation to continue: ask about the picture again."))
      return
    }
    busy = true
    cancelRequested = false
    lock.unlock()
    log("turn \(id): follow-up")
    run(id: id, conversation: conversation, messageJSON: MessageJSON.userTurn(text: text))
  }

  private func run(id: Int, conversation: Conversation, messageJSON: String) {
    let sent = DispatchTime.now().uptimeNanoseconds
    let format = self.format
    var stats = TurnStats()
    var reply = ""
    let state = NSLock()
    do {
      try conversation.send(messageJSON: messageJSON) { [self] event in
        switch event {
        case .text(let piece):
          let now = DispatchTime.now().uptimeNanoseconds
          let t = Double(now - sent) / 1e9
          state.lock()
          reply += piece
          stats.chunks += 1
          if stats.firstTextSeconds == nil { stats.firstTextSeconds = t }
          if stats.firstAnswerSeconds == nil, format == .plain || ReplyText.answerHasStarted(reply) {
            stats.firstAnswerSeconds = t
          }
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
      "turn \(id) \(cancelled ? "cancelled" : failure == nil ? (limit ? "ended at the token limit" : "done") : "failed"): "
        + "first text \(stats.firstTextSeconds ?? -1) s, first answer \(stats.firstAnswerSeconds ?? -1) s, "
        + "prefill \(counters?.prefillTokens ?? -1) tokens at \(counters?.prefillTokensPerSecond ?? -1) tok/s, "
        + "decode \(counters?.decodeTokens ?? -1) tokens at \(counters?.decodeTokensPerSecond ?? -1) tok/s, "
        + "context \(counters?.contextTokens ?? -1), peak footprint \(stats.peakFootprintBytes ?? 0) bytes"
        + (error.map { ", runtime: \($0)" } ?? ""))
    lock.lock()
    if cancelled { conversationRetired = true }
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
    if conversation != nil { cancelRequested = true }
    lock.unlock()
    guard let conversation else { return }
    log("cancel")
    conversation.cancel()
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
      case .ask:
        ask(
          id: command.id ?? 0, picture: command.picture ?? "", text: command.text ?? "", system: command.system,
          maxOutputTokens: command.maxOutputTokens ?? 1024)
      case .followUp: followUp(id: command.id ?? 0, text: command.text ?? "")
      case .cancel: cancel()
      case .quit: finish("quit")
      }
    }
    finish("the window closed the pipe")
  }

  private func finish(_ why: String) -> Never {
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

/// A plain mid-gray 448 x 448 PNG for the warm-up.
enum WarmUpPicture {
  static func write(to url: URL) throws {
    let side = PreparedPicture.side
    var pixels = [UInt8](repeating: 128, count: side * side * 4)
    let png: Data? = pixels.withUnsafeMutableBytes { raw -> Data? in
      guard let space = CGColorSpace(name: CGColorSpace.sRGB),
        let context = CGContext(
          data: raw.baseAddress, width: side, height: side, bitsPerComponent: 8, bytesPerRow: side * 4, space: space,
          bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue),
        let image = context.makeImage()
      else { return nil }
      let data = NSMutableData()
      guard let destination = CGImageDestinationCreateWithData(data, "public.png" as CFString, 1, nil) else { return nil }
      CGImageDestinationAddImage(destination, image, nil)
      return CGImageDestinationFinalize(destination) ? data as Data : nil
    }
    guard let png else { throw HostError("could not make the warm-up picture") }
    try png.write(to: url, options: .atomic)
  }
}
