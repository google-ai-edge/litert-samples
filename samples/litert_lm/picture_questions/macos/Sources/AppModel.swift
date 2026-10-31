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
import UniformTypeIdentifiers

/// Everything the window shows, and the actions its buttons call. The recorder (Autoplay.swift) calls the same
/// actions.
@MainActor
final class AppModel: ObservableObject {
  struct ModelFile: Equatable {
    var url: URL
    var size: UInt64
    var knownName: String?

    var name: String { url.lastPathComponent }
    /// "Muse-Glimmer-30B_q4_ekv2048.litertlm" -> "Muse-Glimmer-30B".
    var displayName: String {
      if let knownName { return knownName }
      let stem = url.deletingPathExtension().lastPathComponent
      return stem.split(separator: "_").first.map(String.init) ?? stem
    }
  }

  struct Problem: Equatable {
    var title: String
    var detail: String
    /// Whether starting the engine again for the same file makes sense.
    var canRetry: Bool
  }

  enum Phase: Equatable {
    case empty
    case starting
    case building(since: Date)
    case warming(since: Date)
    case ready
    case answering
    case closing
    case stopped(Problem)
  }

  enum TurnState: Equatable {
    case streaming, done, cancelled, limit
    case failed(String)
  }

  struct Turn: Identifiable, Equatable {
    let id: Int
    let question: String
    let sentAt: Date
    /// The question started a conversation (the picture went with it); otherwise it is a follow-up.
    let opensConversation: Bool
    /// What the engine got with this question.
    let system: String?
    let maxOutputTokens: Int
    let picture: PreparedPicture?
    /// The reply as LiteRT-LM returned it (format tokens included).
    var raw = ""
    var state = TurnState.streaming
    var firstChunkAt: Date?
    var lastChunkAt: Date?
    var chunks = 0
    var firstChunkTime: Double?
    var lastChunkTime: Double?
    var stats: TurnStats?

    /// Text pieces per second since the first one, on the engine's clock, while the reply streams.
    var liveRate: Double? {
      guard chunks >= 2, let first = firstChunkTime, let last = lastChunkTime, last > first else { return nil }
      return Double(chunks - 1) / (last - first)
    }
  }

  struct Memory: Equatable {
    var current: UInt64
    var peak: UInt64
  }

  /// One answered (or stopped, or failed) question, as the recorder writes it.
  struct TurnRecord: Codable {
    var turnID: Int
    var kind: String
    var question: String
    var pictureSource: String?
    var picturePNG: String?
    var pictureSHA256: String?
    var pictureNote: String?
    var system: String?
    var maxOutputTokens: Int
    var raw: String
    var answer: String
    var reasoning: String?
    var state: String
    var stats: TurnStats?
    /// Send to the first piece of text as the window saw it.
    var windowFirstTextSeconds: Double?
    var enginePid: Int32
    var sentAt: Double
    var firstChunkAt: Double?
    var doneAt: Double
  }

  struct EngineRecord: Codable {
    var pid: Int32
    var log: String
    var spawnedAt: Double
    var readyAt: Double?
    var spawnToReadySeconds: Double?
    var buildSeconds: Double?
    var warmUpSeconds: Double?
    var peakFootprintAtReadyBytes: UInt64?
    var maxSampledFootprintBytes: UInt64?
    var webgpuErrorLines = 0
    var ending: String?
    var problem: String?
  }

  @Published private(set) var file: ModelFile?
  @Published private(set) var fileInfo: FileInfo?
  @Published private(set) var format: ReplyFormat?
  @Published private(set) var phase = Phase.empty
  @Published private(set) var picture: PreparedPicture?
  @Published private(set) var pictureProblem: String?
  @Published private(set) var turns: [Turn] = []
  @Published var draft = ""
  @Published var strength = ReasoningStrength.low {
    didSet { if strength != oldValue, !turns.isEmpty { newConversation() } }
  }
  @Published private(set) var clock = Date()
  @Published private(set) var buildSeconds: Double?
  @Published private(set) var warmUpSeconds: Double?
  @Published private(set) var memory: Memory?
  @Published private(set) var webGPUErrors = 0
  @Published private(set) var memoryNote: String?

  private(set) var turnRecords: [TurnRecord] = []
  private(set) var engineRecords: [EngineRecord] = []
  private var currentEngine: EngineRecord?
  private var client: EngineClient?
  /// The engine has a conversation about the current picture that can take a follow-up.
  private var conversationOpen = false
  private var nextTurnID = 1
  private var spawnedAt = Date()
  private var lastError: EngineEvent?
  private var expectingEnd = false
  private var afterEnd: (() -> Void)?
  private var closingProblem: Problem?
  private var ticker: Timer?
  private var activity: NSObjectProtocol?
  private var cancelWatchdog: DispatchWorkItem?
  private let defaults = UserDefaults.standard

  let logDirectory: URL
  let pictureDirectory: URL
  private let runtimeLog: String?

  init() {
    let home = FileManager.default.homeDirectoryForCurrentUser
    logDirectory = defaults.string(forKey: "logDirectory").map { URL(fileURLWithPath: $0) }
      ?? home.appendingPathComponent("Library/Logs/PictureQuestions")
    pictureDirectory = home.appendingPathComponent("Library/Caches/PictureQuestions/pictures")
    runtimeLog = defaults.string(forKey: "runtimeLog")
  }

  var lastModelURL: URL? { defaults.string(forKey: "lastModelPath").map { URL(fileURLWithPath: $0) } }
  var engineRunning: Bool { client != nil }
  var enginePid: Int32? { client?.pid }
  var isMuse: Bool { format == .reasoningMessages }
  var knownModel: KnownModel? { file.flatMap { KnownModel.matching(bytes: $0.size) } }
  var maxOutputTokens: Int { knownModel?.maxOutputTokens ?? 1024 }

  var isStopped: Bool {
    if case .stopped = phase { return true }
    return false
  }

  var canRestartEngine: Bool {
    guard file != nil else { return false }
    if case .stopped(let problem) = phase { return problem.canRetry }
    return false
  }

  var canAsk: Bool {
    phase == .ready && picture != nil && !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
  }

  // MARK: The model file

  func chooseModel() {
    let panel = NSOpenPanel()
    panel.allowedContentTypes = [UTType(filenameExtension: "litertlm") ?? .data]
    panel.allowsMultipleSelection = false
    panel.canChooseDirectories = false
    panel.message = "Choose a LiteRT-LM model file (.litertlm)"
    if panel.runModal() == .OK, let url = panel.url { openModel(url) }
  }

  func openModel(_ url: URL) {
    if client != nil {
      stopEngine { [weak self] in self?.load(url) }
    } else {
      load(url)
    }
  }

  func restartEngine() {
    guard file != nil else { return }
    if client != nil {
      stopEngine { [weak self] in self?.startEngine() }
    } else {
      startEngine()
    }
  }

  /// Ends the engine without starting another one (the recorder's last step).
  func closeEngine() {
    guard client != nil else { return }
    stopEngine(then: nil)
  }

  private func load(_ url: URL) {
    file = nil
    fileInfo = nil
    format = nil
    memoryNote = nil
    resetConversation()
    let path = url.path
    guard FileManager.default.fileExists(atPath: path) else {
      phase = .stopped(
        Problem(
          title: "File not found",
          detail: "Nothing is at \(path). The file may have been moved, renamed or deleted; open it from where it is now.",
          canRetry: false))
      return
    }
    guard url.pathExtension.lowercased() == "litertlm" else {
      phase = .stopped(
        Problem(
          title: "Not a .litertlm file",
          detail: "LiteRT-LM opens a model from a file whose name ends in .litertlm, and “\(url.lastPathComponent)” does not.",
          canRetry: false))
      return
    }
    // The size of the file itself, also when the path is a symbolic link to it (a model kept on another disk):
    // attributesOfItem describes a link, not what it points to.
    let attributes = try? FileManager.default.attributesOfItem(atPath: url.resolvingSymlinksInPath().path)
    let size = (attributes?[.size] as? NSNumber)?.uint64Value ?? 0
    let known = KnownModel.matching(bytes: size)
    file = ModelFile(url: url, size: size, knownName: known?.name)
    defaults.set(path, forKey: "lastModelPath")

    // Checked before anything is loaded: the engine of a 45 GB file builds for minutes and then would not fit.
    let need = MemoryNeed.engineBytes(fileBytes: size)
    let physical = SystemMemory.physicalBytes
    if physical < need + MemoryNeed.reserveBytes {
      let minimum = Int((Double(need + MemoryNeed.reserveBytes) / 1_073_741_824).rounded(.up))
      var detail =
        known != nil
        ? "\(known!.name)'s engine used up to \(Format.memory(need)) of memory in this app"
        : "This model's engine would need about \(Format.memory(need)) of memory (estimated at "
          + "\(MemoryNeed.perFileByte)× the file size)"
      detail +=
        ", and macOS needs about \(Format.memory(MemoryNeed.reserveBytes)) besides. This Mac has "
        + "\(Format.ram(physical)). Open it on a Mac with \(minimum) GB of memory or more."
      if SystemMemory.physicalIsAssumed { detail += "\n\n(This Mac's memory size was set with -assumePhysicalMemoryGB.)" }
      phase = .stopped(Problem(title: "Not enough memory", detail: detail, canRetry: false))
      return
    }
    if let available = SystemMemory.availableBytes, available < need {
      memoryNote =
        "macOS has \(Format.memory(available)) of memory free right now and the engine needs about \(Format.memory(need)). "
        + "Quit other apps, or macOS will swap and the engine will be slow."
    }
    startEngine()
  }

  private func startEngine() {
    guard let file else { return }
    buildSeconds = nil
    warmUpSeconds = nil
    memory = nil
    webGPUErrors = 0
    lastError = nil
    expectingEnd = false
    spawnedAt = Date()
    phase = .starting
    resetConversation()
    do {
      let client = try EngineClient(
        modelPath: file.url.path, logDirectory: logDirectory, runtimeLog: runtimeLog,
        warmUp: !defaults.bool(forKey: "noWarmUp"),
        onEvent: { [weak self] event in MainActor.assumeIsolated { self?.handle(event) } },
        onWebGPUError: { [weak self] _ in MainActor.assumeIsolated { self?.noteWebGPUError() } },
        onEnd: { [weak self] ending in MainActor.assumeIsolated { self?.engineEnded(ending) } })
      self.client = client
      currentEngine = EngineRecord(pid: client.pid, log: client.logURL.path, spawnedAt: spawnedAt.timeIntervalSince1970)
      // No App Nap while an engine lives: the window keeps up with the answer even when it is not in front.
      activity =
        activity
        ?? ProcessInfo.processInfo.beginActivity(
          options: .userInitiatedAllowingIdleSystemSleep, reason: "A LiteRT-LM engine is running")
    } catch {
      phase = .stopped(Problem(title: "Could not start the engine", detail: error.localizedDescription, canRetry: true))
      return
    }
    startTicker()
  }

  private func stopEngine(then next: (() -> Void)?, problem: Problem? = nil) {
    afterEnd = next
    closingProblem = problem
    expectingEnd = true
    phase = .closing
    client?.stop()
  }

  // MARK: The picture

  func choosePicture() {
    let panel = NSOpenPanel()
    panel.allowedContentTypes = [.image]
    panel.allowsMultipleSelection = false
    panel.canChooseDirectories = false
    panel.message = "Choose a picture: a screenshot, a chart, a document or a photo"
    if panel.runModal() == .OK, let url = panel.url { setPicture(url) }
  }

  /// The picture stays while an answer is being written: the conversation it started is still running.
  private var pictureLocked: Bool {
    guard phase == .answering else { return false }
    pictureProblem = "Wait for the answer, or press Stop, before you change the picture."
    return true
  }

  func setPicture(_ url: URL) {
    guard !pictureLocked else { return }
    do {
      showPicture(try PreparedPicture.make(from: url, directory: pictureDirectory))
    } catch {
      pictureProblem = error.localizedDescription
    }
  }

  /// ⇧⌘V: a picture file copied in Finder, or picture data (a screenshot copied with ⌃⇧⌘4, an image from a page).
  /// The recorder passes a pasteboard of its own, so that a run does not overwrite what the user copied.
  func pastePicture(from board: NSPasteboard = .general) {
    guard !pictureLocked else { return }
    if let urls = board.readObjects(forClasses: [NSURL.self], options: [.urlReadingFileURLsOnly: true]) as? [URL],
      let url = urls.first
    {
      setPicture(url)
      return
    }
    for type in [NSPasteboard.PasteboardType.png, .tiff] {
      if let data = board.data(forType: type) {
        do {
          showPicture(try PreparedPicture.make(from: data, name: "Pasted picture", directory: pictureDirectory))
        } catch {
          pictureProblem = error.localizedDescription
        }
        return
      }
    }
    pictureProblem = "The clipboard holds no picture. Copy a picture or a screenshot first (⌃⇧⌘4 copies a part of the screen)."
  }

  private func showPicture(_ prepared: PreparedPicture) {
    picture = prepared
    pictureProblem = nil
    resetConversation()
  }

  func useExample(_ example: Example) {
    guard !pictureLocked else { return }
    guard let url = example.pictureURL else {
      pictureProblem = "The example picture \(example.picture) is missing from the app."
      return
    }
    setPicture(url)
    guard picture != nil, pictureProblem == nil else { return }
    if let reasoning = example.reasoning, let value = ReasoningStrength(rawValue: reasoning) { strength = value }
    draft = example.question
    ask()
  }

  // MARK: Questions

  func ask() {
    let text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
    guard phase == .ready, !text.isEmpty, let client, let picture else { return }
    let opens = !conversationOpen
    let turn = Turn(
      id: nextTurnID, question: text, sentAt: Date(), opensConversation: opens, system: opens ? systemMessage : nil,
      maxOutputTokens: maxOutputTokens, picture: picture)
    nextTurnID += 1
    turns.append(turn)
    draft = ""
    phase = .answering
    if opens {
      client.send(
        EngineCommand(
          cmd: .ask, id: turn.id, text: text, picture: picture.modelPNG.path, system: turn.system,
          maxOutputTokens: turn.maxOutputTokens))
      conversationOpen = true
    } else {
      client.send(EngineCommand(cmd: .followUp, id: turn.id, text: text))
    }
  }

  /// The system message of a new conversation: the reasoning strength for Muse-Glimmer-30B, none otherwise (the
  /// model's template then writes its own default).
  var systemMessage: String? { isMuse ? strength.systemMessage : nil }

  func stopAnswer() {
    guard phase == .answering, let client else { return }
    client.send(EngineCommand(cmd: .cancel))
    // A runtime that does not end the reply within 20 s is closed with its engine.
    let watchdog = DispatchWorkItem { [weak self] in
      guard let self, self.phase == .answering else { return }
      self.client?.kill()
    }
    cancelWatchdog = watchdog
    DispatchQueue.main.asyncAfter(deadline: .now() + 20, execute: watchdog)
  }

  /// The same picture, no earlier questions.
  func newConversation() {
    guard phase != .answering else { return }
    turns = []
    conversationOpen = false
  }

  private func resetConversation() {
    turns = []
    conversationOpen = false
  }

  // MARK: Engine events

  private func handle(_ event: EngineEvent) {
    switch event.event {
    case .fileInfo:
      fileInfo = event.file
    case .building:
      phase = .building(since: spawnedAt)
    case .warming:
      phase = .warming(since: Date())
    case .ready:
      buildSeconds = event.buildSeconds
      warmUpSeconds = event.warmUpSeconds
      format = event.format
      currentEngine?.buildSeconds = event.buildSeconds
      currentEngine?.warmUpSeconds = event.warmUpSeconds
      currentEngine?.spawnToReadySeconds = Date().timeIntervalSince(spawnedAt)
      currentEngine?.peakFootprintAtReadyBytes = event.peakFootprintBytes
      currentEngine?.readyAt = Date().timeIntervalSince1970
      phase = .ready
    case .chunk:
      guard let index = turns.lastIndex(where: { $0.id == event.id }) else { return }
      let now = Date()
      if turns[index].firstChunkAt == nil { turns[index].firstChunkAt = now }
      turns[index].lastChunkAt = now
      if let t = event.t {
        if turns[index].firstChunkTime == nil { turns[index].firstChunkTime = t }
        turns[index].lastChunkTime = t
      }
      turns[index].chunks += 1
      turns[index].raw += event.text ?? ""
    case .turnDone:
      cancelWatchdog?.cancel()
      guard let index = turns.lastIndex(where: { $0.id == event.id }) else {
        if phase == .answering { phase = .ready }
        return
      }
      turns[index].stats = event.stats
      if event.cancelled == true {
        // The runtime does not go on with a conversation after a cancel: the next question starts a new one about
        // the same picture.
        turns[index].state = .cancelled
        conversationOpen = false
      } else if let message = event.message {
        turns[index].state = .failed(
          EngineExit.isDeadline(message)
            ? "LiteRT-LM gave up on this reply: a wait passed its time limit (\(message))." : message)
      } else if event.limitReached == true {
        turns[index].state = .limit
      } else {
        turns[index].state = .done
      }
      record(turns[index])
      if phase == .answering { phase = .ready }
    case .error:
      lastError = event
      if event.stage == .generate {
        if let index = turns.lastIndex(where: { $0.id == event.id }) {
          turns[index].state = .failed(event.message ?? "The engine could not answer.")
          record(turns[index])
        }
        conversationOpen = false
        if phase == .answering { phase = .ready }
      }
    }
  }

  private func noteWebGPUError() {
    webGPUErrors += 1
    currentEngine?.webgpuErrorLines += 1
  }

  private func engineEnded(_ ending: EngineClient.Ending) {
    cancelWatchdog?.cancel()
    stopTicker()
    if let activity {
      ProcessInfo.processInfo.endActivity(activity)
      self.activity = nil
    }
    let logPath = client?.logURL.path ?? ""
    client = nil
    conversationOpen = false
    currentEngine?.ending = ending.signaled ? "signal \(ending.status)" : "exit \(ending.status)"
    for index in turns.indices where turns[index].state == .streaming {
      turns[index].state = .failed("The engine stopped before the answer was finished.")
      record(turns[index])
    }

    if expectingEnd {
      expectingEnd = false
      finishEngineRecord(problem: nil)
      let problem = closingProblem
      closingProblem = nil
      if let next = afterEnd {
        afterEnd = nil
        next()
      } else {
        phase = .stopped(
          problem ?? Problem(title: "The engine is closed", detail: "Start it again to ask more.", canRetry: true))
      }
      return
    }

    let problem: Problem
    let logLine = "\n\nEngine log: \(logPath)"
    if let error = lastError, error.stage == .file {
      problem = Problem(title: "Not a model this app can run", detail: error.message ?? "", canRetry: false)
    } else if let error = lastError, error.stage == .engine {
      problem = Problem(title: "The engine could not be built", detail: (error.message ?? "") + logLine, canRetry: true)
    } else if !ending.signaled && ending.status == EngineExit.deadline {
      problem = Problem(
        title: "The engine took too long",
        detail: "LiteRT-LM gave up on a reply when one of its waits passed its time limit, and this engine cannot "
          + "answer again after that. Start it again to ask more."
          + logLine,
        canRetry: true)
    } else if ending.signaled {
      var detail = "The engine process was ended by signal \(ending.status) (\(Self.signalName(ending.status)))."
      if ending.status == SIGKILL {
        detail +=
          " A process ends this way when it is force-quit, when it did not stop an answer in time, or when macOS "
          + "stops it because the Mac ran out of memory. Quit other apps and start the engine again."
      }
      problem = Problem(title: "The engine stopped", detail: detail + logLine, canRetry: true)
    } else {
      problem = Problem(
        title: "The engine stopped", detail: "The engine process exited with status \(ending.status)." + logLine,
        canRetry: true)
    }
    finishEngineRecord(problem: problem.title)
    phase = .stopped(problem)
  }

  private func record(_ turn: Turn) {
    let state: String
    switch turn.state {
    case .streaming: state = "streaming"
    case .done: state = "done"
    case .cancelled: state = "cancelled"
    case .limit: state = "token limit"
    case .failed(let message): state = "failed: \(message)"
    }
    let parsed = ReplyText.parse(turn.raw, format: format ?? .plain)
    turnRecords.append(
      TurnRecord(
        turnID: turn.id, kind: turn.opensConversation ? "ask" : "follow_up", question: turn.question,
        pictureSource: turn.picture?.sourceName, picturePNG: turn.picture?.modelPNG.path,
        pictureSHA256: turn.picture?.sha256, pictureNote: turn.picture?.note, system: turn.system,
        maxOutputTokens: turn.maxOutputTokens,
        raw: turn.raw, answer: parsed.answer, reasoning: parsed.reasoning, state: state, stats: turn.stats,
        windowFirstTextSeconds: turn.firstChunkAt.map { $0.timeIntervalSince(turn.sentAt) },
        enginePid: currentEngine?.pid ?? 0, sentAt: turn.sentAt.timeIntervalSince1970,
        firstChunkAt: turn.firstChunkAt?.timeIntervalSince1970, doneAt: Date().timeIntervalSince1970))
  }

  private func finishEngineRecord(problem: String?) {
    guard var record = currentEngine else { return }
    record.problem = problem
    engineRecords.append(record)
    currentEngine = nil
  }

  var allEngineRecords: [EngineRecord] { engineRecords + (currentEngine.map { [$0] } ?? []) }

  private func startTicker() {
    ticker?.invalidate()
    ticker = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
      MainActor.assumeIsolated { self?.tick() }
    }
  }

  private func stopTicker() {
    ticker?.invalidate()
    ticker = nil
  }

  private func tick() {
    clock = Date()
    // A process that is ending reports 0.
    guard let client, let footprint = SystemMemory.footprint(pid: client.pid), footprint.current > 0 else { return }
    memory = Memory(current: footprint.current, peak: footprint.peak)
    let sampled = max(currentEngine?.maxSampledFootprintBytes ?? 0, footprint.current)
    currentEngine?.maxSampledFootprintBytes = sampled
  }

  private static func signalName(_ signal: Int32) -> String {
    switch signal {
    case SIGKILL: return "SIGKILL"
    case SIGTERM: return "SIGTERM"
    case SIGABRT: return "SIGABRT"
    case SIGSEGV: return "SIGSEGV"
    case SIGBUS: return "SIGBUS"
    default: return "signal \(signal)"
    }
  }
}
