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

/// Everything the window shows, and the actions its buttons call. The recorder (Autoplay.swift) calls the same actions.
@MainActor
final class AppModel: ObservableObject {
  enum FileState: Equatable {
    case checking
    /// `partial`: the bytes of an earlier download that the next one goes on from.
    case missing(partial: UInt64)
    case downloading(received: UInt64)
    case verifying
    case ready(URL)
    case failed(String)
  }

  struct Problem: Equatable {
    var title: String
    var detail: String
    /// Whether starting the engine again for the same file makes sense.
    var canRetry: Bool
  }

  enum Phase: Equatable {
    case idle
    case starting
    case building(since: Date)
    case warming(since: Date)
    case ready
    case closing
    case stopped(Problem)
  }

  /// One inference that has ended: its reasoning and the seconds of its stages.
  struct Inference: Equatable, Codable {
    var text: String
    var thinking: String
    var totalSeconds: Double
    var conversationSeconds: Double?
    var firstTextSeconds: Double?
    var visionSeconds: Double?
    var prefillSeconds: Double?
    var decodeSeconds: Double?
    var prefillTokens: Int?
    var decodeTokens: Int?
    var prefillTokensPerSecond: Double?
    var decodeTokensPerSecond: Double?
    var timeToFirstToken: Double?
    var limitReached: Bool?
    var peakFootprintBytes: UInt64?
  }

  /// The reply of the inference on screen, and the inferences that have ended.
  struct Decision: Equatable {
    var text = ""
    var done = false
    var reply = StepReply.empty
    /// The screenshot's size, for the point's pixels.
    var width: Int
    var height: Int
    /// 1 or 2: the inference this reply is from.
    var inference = 1
    var hasText = false
    /// From pressing Decide to the end of the last inference.
    var totalSeconds: Double?
    var runs: [Inference] = []
    /// The reply stopped at the file's token limit before its tool call ended.
    var cut = false
    /// Stop was pressed: what is on screen is not the model's answer.
    var stopped = false
    var error: String?
  }

  struct Memory: Equatable {
    var current: UInt64
    var peak: UInt64
  }

  /// One decision, as the recorder writes it.
  struct DecisionRecord: Codable {
    var request: String
    var pictureName: String
    var pictureSha256: String
    var pictureRgbSha256: String
    var screenshotSize: [Int]
    var inferences: [Inference]
    var nInferences: Int
    var totalSeconds: Double
    var route: String?
    var action: String?
    var actionLine: String?
    var pointModel: [Double]?
    var pointPixels: [Int]?
    var command: String?
    var strictJSON: Bool?
    var cut: Bool
    var stopped: Bool
    var error: String?
    var enginePid: Int32
    var pressedAt: Double
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

  @Published private(set) var file = FileState.checking
  @Published private(set) var fileInfo: FileInfo?
  @Published private(set) var phase = Phase.idle
  @Published private(set) var picture: Picture?
  @Published private(set) var pictureProblem: String?
  @Published private(set) var preparingPicture = false
  @Published private(set) var requestChips: [String] = []
  @Published var request = ""
  @Published private(set) var deciding = false
  @Published private(set) var stopping = false
  @Published private(set) var elapsed: Double = 0
  @Published private(set) var decision: Decision?
  @Published var message: String?
  @Published private(set) var copied = false
  @Published private(set) var buildSeconds: Double?
  @Published private(set) var warmUpSeconds: Double?
  @Published private(set) var memory: Memory?
  @Published private(set) var webGPUErrors = 0
  @Published private(set) var memoryNote: String?
  @Published private(set) var clock = Date()

  private(set) var records: [DecisionRecord] = []
  private(set) var engineRecords: [EngineRecord] = []
  private var currentEngine: EngineRecord?
  private var client: EngineClient?
  private var spawnedAt = Date()
  private var lastError: EngineEvent?
  private var expectingEnd = false
  private var afterEnd: (() -> Void)?
  private var ticker: Timer?
  private var decideTicker: Timer?
  private var activity: NSObjectProtocol?
  private var cancelWatchdog: DispatchWorkItem?

  // The decision in flight.
  private var currentID = 0
  private var nextID = 1
  private var pressedAt = Date()
  private var decidedRequest = ""
  private var replyText = ""
  private var firstReply: String?
  private var runs: [Inference] = []

  private let defaults = UserDefaults.standard
  let store: ModelStore?
  let logDirectory: URL
  let pictureDirectory: URL
  let cacheDirectory: URL
  private let runtimeLog: String?
  let systemPrompt: String
  /// `--model`: run this file instead of the one the app keeps.
  var modelOverride: URL?
  private var modelURL: URL?

  init() {
    let home = FileManager.default.homeDirectoryForCurrentUser
    logDirectory = defaults.string(forKey: "logDirectory").map { URL(fileURLWithPath: $0) }
      ?? home.appendingPathComponent("Library/Logs/ClickOrCode")
    pictureDirectory = home.appendingPathComponent("Library/Caches/ClickOrCode/pictures")
    cacheDirectory = home.appendingPathComponent("Library/Caches/ClickOrCode/engine")
    runtimeLog = defaults.string(forKey: "runtimeLog")
    let directory = defaults.string(forKey: "modelDirectory").map { URL(fileURLWithPath: $0) }
    store = ModelEntry.load(path: defaults.string(forKey: "modelsJson")).map { ModelStore(entry: $0, directory: directory) }
    systemPrompt =
      Bundle.main.url(forResource: "mage_hybrid_system_prompt", withExtension: "txt")
      .flatMap { try? String(contentsOf: $0, encoding: .utf8) } ?? ""
    try? FileManager.default.removeItem(at: pictureDirectory)
  }

  var engineRunning: Bool { client != nil }
  var enginePid: Int32? { client?.pid }
  var fileBytes: UInt64 { store?.entry.bytes ?? 0 }

  var isStopped: Bool {
    if case .stopped = phase { return true }
    return false
  }

  /// The model file's state in words, for the recorder's checks.
  var fileText: String {
    switch file {
    case .checking: return "checking"
    case .missing(let partial): return "missing, \(partial) bytes here"
    case .downloading(let received): return "downloading, \(received) bytes here"
    case .verifying: return "verifying"
    case .ready: return "ready"
    case .failed(let message): return message
    }
  }

  var fileIsSettled: Bool {
    switch file {
    case .checking, .downloading, .verifying: return false
    case .missing, .ready, .failed: return true
    }
  }

  var canDecide: Bool {
    phase == .ready && !deciding && picture != nil && !request.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
  }

  // MARK: The model file

  /// At launch: the file the app keeps, or the one named on the command line.
  func checkModelFile() async {
    guard let store else {
      file = .failed("The app's model list is missing from the app.")
      return
    }
    if let path = modelOverride {
      file = .verifying
      do {
        try await ModelStore.verify(path, entry: store.entry)
        useFile(path)
      } catch {
        file = .failed(error.localizedDescription)
      }
      return
    }
    if store.hasFile {
      useFile(store.target)
    } else {
      file = .missing(partial: store.partialBytes)
    }
  }

  /// Downloads the file (what an earlier download left goes on), checks it and loads it.
  func downloadModel() {
    guard let store else { return }
    switch file {
    case .downloading, .verifying: return
    default: break
    }
    file = .downloading(received: store.partialBytes)
    Task {
      do {
        try await store.download { [weak self] received in
          DispatchQueue.main.async {
            MainActor.assumeIsolated { self?.file = .downloading(received: received) }
          }
        }
        file = .verifying
        try await store.commit()
        useFile(store.target)
      } catch is CancellationError {
        file = .missing(partial: store.partialBytes)
      } catch {
        file = .failed(error.localizedDescription)
      }
    }
  }

  func cancelDownload() {
    store?.cancelDownload()
  }

  /// A copy already on this Mac, for example from `hf download`.
  func chooseLocalFile() {
    guard let store else { return }
    let panel = NSOpenPanel()
    panel.allowedContentTypes = [UTType(filenameExtension: "litertlm") ?? .data]
    panel.allowsMultipleSelection = false
    panel.canChooseDirectories = false
    panel.message = "Choose \(store.entry.name)"
    guard panel.runModal() == .OK, let url = panel.url else { return }
    importFile(url)
  }

  func importFile(_ url: URL) {
    guard let store else { return }
    file = .verifying
    Task {
      do {
        try await store.importFile(url)
        useFile(store.target)
      } catch {
        file = .failed(error.localizedDescription)
      }
    }
  }

  private func useFile(_ url: URL) {
    file = .ready(url)
    modelURL = url
    // Checked before anything is loaded.
    let physical = SystemMemory.physicalBytes
    if physical < MemoryNeed.engineBytes + MemoryNeed.reserveBytes {
      phase = .stopped(
        Problem(
          title: "Not enough memory",
          detail:
            "The engine used up to \(Format.memory(MemoryNeed.engineBytes)) of memory in this app, and macOS needs about "
            + "\(Format.memory(MemoryNeed.reserveBytes)) besides. This Mac has \(Format.ram(physical)). Nothing was loaded."
            + (SystemMemory.physicalIsAssumed ? "\n\n(This Mac's memory size was set with -assumePhysicalMemoryGB.)" : ""),
          canRetry: false))
      return
    }
    if let available = SystemMemory.availableBytes, available < MemoryNeed.engineBytes {
      memoryNote =
        "macOS has \(Format.memory(available)) of memory free right now and the engine needs about "
        + "\(Format.memory(MemoryNeed.engineBytes)). Quit other apps, or macOS will swap and the engine will be slow."
    }
    startEngine()
  }

  // MARK: The engine

  func restartEngine() {
    guard modelURL != nil else { return }
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

  private func startEngine() {
    guard let modelURL else { return }
    buildSeconds = nil
    warmUpSeconds = nil
    memory = nil
    webGPUErrors = 0
    lastError = nil
    expectingEnd = false
    spawnedAt = Date()
    phase = .starting
    do {
      try? FileManager.default.createDirectory(at: cacheDirectory, withIntermediateDirectories: true)
      let client = try EngineClient(
        modelPath: modelURL.path, cacheDir: defaults.bool(forKey: "noEngineCache") ? ":nocache" : cacheDirectory.path,
        logDirectory: logDirectory, runtimeLog: runtimeLog, warmUp: !defaults.bool(forKey: "noWarmUp"),
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

  private func stopEngine(then next: (() -> Void)?) {
    afterEnd = next
    expectingEnd = true
    phase = .closing
    client?.stop()
  }

  // MARK: The screenshot

  private var pictureLocked: Bool {
    guard deciding else { return false }
    pictureProblem = "Wait for the step, or press Stop, before you change the screenshot."
    return true
  }

  func choosePicture() {
    let panel = NSOpenPanel()
    panel.allowedContentTypes = [.image]
    panel.allowsMultipleSelection = false
    panel.canChooseDirectories = false
    panel.message = "Choose a screenshot"
    if panel.runModal() == .OK, let url = panel.url { setPicture(url) }
  }

  func setPicture(_ url: URL, chips: [String] = []) {
    guard !pictureLocked else { return }
    guard let data = try? Data(contentsOf: url) else {
      pictureProblem = "macOS could not read “\(url.lastPathComponent)” as a picture. It may be damaged, cut short, or not a picture at all."
      return
    }
    show(data: data, name: url.lastPathComponent, chips: chips)
  }

  /// ⇧⌘V: a picture file copied in Finder, or picture data (a screenshot copied with ⌃⇧⌘4). The recorder passes a
  /// pasteboard of its own, so that a run does not overwrite what the user copied.
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
        show(data: data, name: "Pasted picture", chips: [])
        return
      }
    }
    pictureProblem = "The clipboard holds no picture. Copy a screenshot first (⌃⇧⌘4 copies a part of the screen)."
  }

  func useExample(_ example: Example) {
    guard !pictureLocked else { return }
    guard let url = example.pictureURL else {
      pictureProblem = "The example picture \(example.file) is missing from the app."
      return
    }
    request = ""
    setPicture(url, chips: example.requests)
  }

  /// The whole main display, with this window out of the way.
  func captureScreen() {
    guard !pictureLocked else { return }
    Task {
      do {
        try Capture.ensureAllowed()
        NSApp.hide(nil)
        try? await Task.sleep(nanoseconds: 400_000_000)
        defer { NSApp.unhide(nil) }
        let data = try await Capture.screen(to: captureFile())
        show(data: data, name: "This screen", chips: [])
      } catch {
        pictureProblem = error.localizedDescription
      }
    }
  }

  /// macOS's window picker: click a window; Esc leaves everything as it was.
  func captureWindow() {
    guard !pictureLocked else { return }
    Task {
      do {
        try Capture.ensureAllowed()
        if let data = try await Capture.pickWindow(to: captureFile()) { show(data: data, name: "A window", chips: []) }
      } catch {
        pictureProblem = error.localizedDescription
      }
    }
  }

  /// The recorder's capture of one window: what the picker does once the user has clicked it.
  func captureWindow(id: CGWindowID, name: String) async {
    do {
      try Capture.ensureAllowed()
      show(data: try await Capture.window(id: id, to: captureFile()), name: name, chips: [])
    } catch {
      pictureProblem = error.localizedDescription
    }
  }

  private func captureFile() -> URL {
    try? FileManager.default.createDirectory(at: pictureDirectory, withIntermediateDirectories: true)
    return pictureDirectory.appendingPathComponent("capture.png")
  }

  private func show(data: Data, name: String, chips: [String]) {
    preparingPicture = true
    pictureProblem = nil
    let directory = pictureDirectory
    Task {
      do {
        let made = try await Task.detached(priority: .userInitiated) {
          try Picture.make(data: data, name: name, directory: directory)
        }.value
        picture = made
        requestChips = chips
        decision = nil
        message = nil
        copied = false
      } catch {
        pictureProblem = error.localizedDescription
      }
      preparingPicture = false
    }
  }

  // MARK: Decide

  /// `-maxOutputTokens`: a smaller cap, to try a reply that is cut short.
  private var outputCap: Int {
    let cap = defaults.integer(forKey: "maxOutputTokens")
    return cap > 0 ? cap : Step.maxOutputTokens
  }

  /// The next step for the screenshot and the request on screen: the agent's first step and, when its reply calls for
  /// a screenshot, its second, with the same picture as the answer to that call.
  func decide() {
    guard phase == .ready, !deciding, client != nil, let picture else { return }
    let text = request.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !text.isEmpty else {
      message = "Type what you want done on this screen."
      return
    }
    decidedRequest = text
    pressedAt = Date()
    deciding = true
    stopping = false
    elapsed = 0
    message = nil
    copied = false
    firstReply = nil
    runs = []
    decision = Decision(width: picture.width, height: picture.height)
    let timer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
      MainActor.assumeIsolated {
        guard let self else { return }
        self.elapsed = Date().timeIntervalSince(self.pressedAt)
      }
    }
    decideTicker = timer
    sendInference()
  }

  private func sendInference() {
    guard let client, let picture else { return }
    currentID = nextID
    nextID += 1
    replyText = ""
    var pending = Decision(width: picture.width, height: picture.height)
    pending.inference = runs.count + 1
    pending.runs = runs
    decision = pending
    client.send(
      EngineCommand(
        cmd: .step, id: currentID, picture: picture.modelPNG.path, request: decidedRequest, firstReply: firstReply,
        system: systemPrompt, maxOutputTokens: outputCap))
  }

  func stopDecision() {
    guard deciding, let client else { return }
    stopping = true
    client.send(EngineCommand(cmd: .cancel))
    // A runtime that does not end the reply within 20 s is closed with its engine.
    let watchdog = DispatchWorkItem { [weak self] in
      guard let self, self.deciding else { return }
      self.client?.kill()
    }
    cancelWatchdog = watchdog
    DispatchQueue.main.asyncAfter(deadline: .now() + 20, execute: watchdog)
  }

  /// Puts the command on the clipboard. The app runs nothing. The recorder can pass a pasteboard of its own.
  func copyCommand(_ command: String, to board: NSPasteboard = .general) {
    board.clearContents()
    board.setString(command, forType: .string)
    copied = true
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
      currentEngine?.buildSeconds = event.buildSeconds
      currentEngine?.warmUpSeconds = event.warmUpSeconds
      currentEngine?.spawnToReadySeconds = Date().timeIntervalSince(spawnedAt)
      currentEngine?.peakFootprintAtReadyBytes = event.peakFootprintBytes
      currentEngine?.readyAt = Date().timeIntervalSince1970
      phase = .ready
    case .chunk:
      guard event.id == currentID, deciding, var shown = decision else { return }
      replyText += event.text ?? ""
      shown.text = replyText
      shown.reply = Step.parse(replyText)
      shown.hasText = true
      decision = shown
    case .turnDone:
      guard event.id == currentID, deciding else { return }
      inferenceEnded(event)
    case .error:
      lastError = event
      if event.stage == .generate, deciding {
        finishDecision(error: Step.isTooLong(event.message) ? Self.tooLongMessage : event.message ?? "The engine could not answer.")
      }
    }
  }

  private func inferenceEnded(_ event: EngineEvent) {
    cancelWatchdog?.cancel()
    let answer = replyText.trimmingCharacters(in: .whitespacesAndNewlines)
    let reply = Step.parse(answer)
    let stats = event.stats
    let counters = stats?.counters
    // The first piece of the reply comes after the picture's encoder and the prefill.
    var vision: Double?
    if let first = stats?.firstTextSeconds, let ttft = counters?.timeToFirstToken { vision = max(first - ttft, 0) }
    var prefill: Double?
    if let tokens = counters?.prefillTokens, let rate = counters?.prefillTokensPerSecond, rate > 0 { prefill = Double(tokens) / rate }
    var decode: Double?
    if let tokens = counters?.decodeTokens, let rate = counters?.decodeTokensPerSecond, rate > 0 { decode = Double(tokens) / rate }
    runs.append(
      Inference(
        text: answer, thinking: reply.thinking, totalSeconds: (stats?.conversationSeconds ?? 0) + (stats?.replySeconds ?? 0),
        conversationSeconds: stats?.conversationSeconds, firstTextSeconds: stats?.firstTextSeconds, visionSeconds: vision,
        prefillSeconds: prefill, decodeSeconds: decode, prefillTokens: counters?.prefillTokens,
        decodeTokens: counters?.decodeTokens, prefillTokensPerSecond: counters?.prefillTokensPerSecond,
        decodeTokensPerSecond: counters?.decodeTokensPerSecond, timeToFirstToken: counters?.timeToFirstToken,
        limitReached: event.limitReached, peakFootprintBytes: stats?.peakFootprintBytes))
    if let message = event.message {
      finishDecision(
        error: Step.isTooLong(message)
          ? Self.tooLongMessage
          : EngineExit.isDeadline(message) ? "LiteRT-LM gave up on this step: a wait passed its time limit (\(message))." : message)
      return
    }
    let stopped = event.cancelled == true || stopping
    if firstReply == nil, !stopped, Step.asksForScreenshot(reply) {
      firstReply = answer
      sendInference()
      return
    }
    // Cut: no call came to its end, and the reply used up what the file or the cap leaves it.
    let used = (counters?.prefillTokens ?? 0) + (counters?.decodeTokens ?? 0)
    let cut =
      !answer.contains("</tool_call>")
      && (counters == nil || used >= Step.fileTokens - 4 || (counters?.decodeTokens ?? 0) >= outputCap)
    finishDecision(error: nil, stopped: stopped, cut: cut)
  }

  private func finishDecision(error: String?, stopped: Bool = false, cut: Bool = false) {
    decideTicker?.invalidate()
    decideTicker = nil
    cancelWatchdog?.cancel()
    let answer = replyText.trimmingCharacters(in: .whitespacesAndNewlines)
    let reply = Step.parse(answer)
    let total = Date().timeIntervalSince(pressedAt)
    var last = decision ?? Decision(width: picture?.width ?? 1, height: picture?.height ?? 1)
    last.text = answer
    last.reply = reply
    last.done = true
    last.hasText = !answer.isEmpty
    last.inference = max(runs.count, 1)
    last.totalSeconds = total
    last.runs = runs
    last.cut = cut
    last.stopped = stopped
    last.error = error
    decision = last
    deciding = false
    stopping = false
    if let picture {
      let first = reply.first
      let pixels = first?.coordinate.map { Step.toPixels($0, width: picture.width, height: picture.height) }
      records.append(
        DecisionRecord(
          request: decidedRequest, pictureName: picture.name, pictureSha256: picture.sourceSha256,
          pictureRgbSha256: picture.pictureRgbSha256, screenshotSize: [picture.width, picture.height], inferences: runs,
          nInferences: runs.count, totalSeconds: total, route: first?.name, action: first?.action, actionLine: reply.actionLine,
          pointModel: first?.coordinate.map { [$0.x, $0.y] }, pointPixels: pixels.map { [$0.x, $0.y] }, command: first?.command,
          strictJSON: first?.strictJSON, cut: cut, stopped: stopped, error: error, enginePid: currentEngine?.pid ?? 0,
          pressedAt: pressedAt.timeIntervalSince1970, doneAt: Date().timeIntervalSince1970))
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
    currentEngine?.ending = ending.signaled ? "signal \(ending.status)" : "exit \(ending.status)"
    if deciding { finishDecision(error: "The engine stopped before the step was finished.") }

    if expectingEnd {
      expectingEnd = false
      finishEngineRecord(problem: nil)
      if let next = afterEnd {
        afterEnd = nil
        next()
      } else {
        phase = .stopped(Problem(title: "The engine is closed", detail: "Start it again to decide more.", canRetry: true))
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
          + "answer again after that. Start it again to decide more." + logLine,
        canRetry: true)
    } else if ending.signaled {
      var detail = "The engine process was ended by signal \(ending.status) (\(Self.signalName(ending.status)))."
      if ending.status == SIGKILL {
        detail +=
          " A process ends this way when it is force-quit, when it did not stop a step in time, or when macOS stops it "
          + "because the Mac ran out of memory. Quit other apps and start the engine again."
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

  private static let tooLongMessage = "The request is too long for the model file. Shorten it and decide again."

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
