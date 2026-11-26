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
import SwiftUI

/// The chat screen's state and actions. The buttons and the self-check (SelfCheck.swift) call the
/// same methods.
@MainActor
final class ChatModel: ObservableObject {
  enum Phase { case empty, loading, ready, replying, stopping }

  struct Line: Identifiable {
    enum Role { case user, model, note }
    let id = UUID()
    let role: Role
    var text: String
    var image: UIImage? = nil
  }

  /// A message above the input, with at most one action.
  struct Notice {
    enum Action: String {
      case useThisFile = "Use this file", loadAnyway = "Load anyway", useCPU = "Use the CPU"
    }
    let text: String
    var action: Action? = nil
  }

  struct Loaded: Equatable {
    let model: ModelChoice
    let gpu: Bool
    var label: String { "\(model.name) on the \(gpu ? "GPU" : "CPU")" }
  }

  @Published var choice: ModelChoice = Models.e2b
  @Published var useGPU = true
  @Published private(set) var phase: Phase = .empty
  @Published private(set) var loaded: Loaded?
  @Published private(set) var lines: [Line] = []
  @Published var notice: Notice?
  @Published var draft = ""
  @Published var picture: Picture?
  @Published private(set) var memoryLine = ""

  private let runner = ModelRunner()
  /// After Stop or an error the next message starts a new conversation; this sample does not
  /// continue the old one.
  private var needsNewConversation = false
  private var stoppedInBackground = false
  /// Counts Stop presses, so that a late check of an earlier Stop does not report a later one.
  private var stops = 0

  var busy: Bool { phase == .loading || phase == .replying || phase == .stopping }

  // MARK: Start

  func start() async {
    if let last = LoadMarker.take() {
      notice = Notice(
        text: "The last launch ended while \(last) was loading. If iOS stopped the app because it "
          + "ran out of memory, try a smaller model or the CPU. This app can use "
          + "\(Memory.text(Memory.available())) more memory now.")
    }
    updateMemoryLine()
    if SelfCheck.isRequested { await SelfCheck.run(self) }
  }

  // MARK: Load

  /// Loads the chosen model on the chosen backend. Before loading, it checks the file's size and
  /// refuses when iOS would not leave the app enough memory for the model.
  func load(acceptFile: Bool = false, ignoreMemory: Bool = false) async {
    guard !busy else { return }
    let model = choice
    let gpu = useGPU
    let label = Loaded(model: model, gpu: gpu).label
    notice = nil
    guard let size = model.bytesOnPhone else {
      notice = Notice(
        text: "\(model.fileName) is not on this iPhone. Download it from huggingface.co/"
          + "\(model.repository) and copy it into this app's files: in Finder, select the iPhone, "
          + "open Files, and drop it on LiteRT-LM Chat.")
      return
    }
    if size != model.bytes && !acceptFile {
      notice = Notice(
        text: "\(model.fileName) on this iPhone has \(size.formatted()) bytes; the file this "
          + "sample was tested with has \(model.bytes.formatted()). If the copy stopped early, "
          + "copy it again.",
        action: .useThisFile)
      return
    }
    // Free the model in use first, so that the check below sees the memory it leaves.
    if loaded != nil {
      loaded = nil
      phase = .loading
      await runner.unload()
      phase = .empty
    }
    let need = model.growth(gpu: gpu) + Models.margin
    let available = Memory.available()
    if available < need && !ignoreMemory {
      phase = .empty
      notice = Notice(
        text: "Not enough memory for \(label): in this sample it needed "
          + "\(Memory.text(model.growth(gpu: gpu))) (measured on an iPhone 18 Pro) plus "
          + "\(Memory.text(Models.margin)) to spare, and iOS lets this app use "
          + "\(Memory.text(available)) more"
          + (Memory.availableIsAssumed ? " (set with -assumeAvailableMB)" : "")
          + ". Choose a smaller model or the CPU, or keep the app's two memory capabilities "
          + "(see the README).",
        action: .loadAnyway)
      updateMemoryLine()
      return
    }

    phase = .loading
    lines.removeAll()
    LoadMarker.set(label)
    do {
      try await runner.load(file: model.file, gpu: gpu)
      LoadMarker.clear()
      loaded = Loaded(model: model, gpu: gpu)
      needsNewConversation = false
      phase = .ready
    } catch {
      LoadMarker.clear()
      phase = .empty
      notice = Notice(
        text: "\(label) could not start. LiteRT-LM says: \(Self.describe(error))",
        action: gpu ? .useCPU : nil)
    }
    updateMemoryLine()
  }

  /// The button of a notice. A memory or GPU notice comes only after the file was accepted.
  func perform(_ action: Notice.Action) async {
    switch action {
    case .useThisFile:
      await load(acceptFile: true)
    case .loadAnyway:
      await load(acceptFile: true, ignoreMemory: true)
    case .useCPU:
      useGPU = false
      await load(acceptFile: true)
    }
  }

  // MARK: Chat

  /// Sends the draft (and the attached picture) and shows the reply as it arrives.
  func send() async {
    let text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
    guard phase == .ready, !text.isEmpty else { return }
    let picture = self.picture
    draft = ""
    self.picture = nil
    notice = nil
    stoppedInBackground = false
    lines.append(Line(role: .user, text: text, image: picture?.image))
    lines.append(Line(role: .model, text: ""))
    let index = lines.count - 1
    phase = .replying

    do {
      if needsNewConversation {
        try await runner.newConversation()
        needsNewConversation = false
      }
      let stream = try await runner.send(text, picture: picture?.file)
      for try await piece in stream {
        lines[index].text += Self.text(of: piece)
      }
      if lines[index].text.isEmpty && phase != .stopping {
        lines[index].text = "(The model gave an empty reply.)"
      }
    } catch {
      needsNewConversation = true
      if phase != .stopping {
        if lines[index].text.isEmpty { lines.remove(at: index) }
        notice = Notice(
          text: "The reply failed. LiteRT-LM says: \(Self.describe(error)) "
            + "The next message starts a new conversation.")
      }
    }
    if phase == .stopping {
      // Also when the reply ended just before the stop reached LiteRT-LM.
      needsNewConversation = true
      if lines[index].text.isEmpty { lines.remove(at: index) }
      lines.append(
        Line(
          role: .note,
          text: (stoppedInBackground ? "Stopped: the app went to the background." : "Stopped.")
            + " The next message starts a new conversation."))
    }
    // The reply has ended, so LiteRT-LM has read the picture: its file goes.
    if let file = picture?.file { try? FileManager.default.removeItem(at: file) }
    phase = .ready
    updateMemoryLine()
  }

  /// Stops the reply in progress. The text so far stays on screen.
  func stop() async {
    guard phase == .replying else { return }
    phase = .stopping
    stops += 1
    let thisStop = stops
    await runner.stop()
    // The reply's stream normally ends within a second of a stop.
    Task {
      try? await Task.sleep(for: .seconds(30))
      if phase == .stopping && stops == thisStop {
        notice = Notice(
          text: "LiteRT-LM has not ended the reply 30 seconds after Stop. If the app does not "
            + "respond, close it and open it again.")
      }
    }
  }

  /// iOS does not run an app's GPU work in the background, so a reply stops when the app leaves
  /// the screen.
  func enteredBackground() {
    guard phase == .replying else { return }
    stoppedInBackground = true
    Task { await stop() }
  }

  /// Starts over with the same model.
  func newChat() async {
    guard phase == .ready else { return }
    phase = .loading
    do {
      try await runner.newConversation()
      needsNewConversation = false
      lines.removeAll()
      notice = nil
    } catch {
      needsNewConversation = true
      notice = Notice(text: "A new conversation could not start. LiteRT-LM says: \(Self.describe(error))")
    }
    phase = .ready
    updateMemoryLine()
  }

  func attach(_ data: Data) {
    do {
      let new = try Picture.make(from: data)
      removePicture()
      picture = new
    } catch {
      notice = Notice(text: Self.describe(error))
    }
  }

  func attachExample() {
    do {
      let new = try Picture.example()
      removePicture()
      picture = new
      if draft.isEmpty { draft = "How much is a latte?" }
    } catch {
      notice = Notice(text: Self.describe(error))
    }
  }

  /// Takes the attached picture off the next message and deletes its file.
  func removePicture() {
    if let file = picture?.file { try? FileManager.default.removeItem(at: file) }
    picture = nil
  }

  // MARK: Helpers

  func benchmark() async -> BenchmarkInfo? { await runner.benchmark() }

  func updateMemoryLine() {
    let footprint = Memory.footprint()
    memoryLine =
      "Memory: this app \(Memory.text(footprint.now)), at most \(Memory.text(footprint.peak)); "
      + "iOS allows \(Memory.text(Memory.available())) more"
  }

  /// The text of one piece of a reply.
  static func text(of message: Message) -> String {
    message.contents.compactMap { content -> String? in
      if case .text(let text) = content { return text }
      return nil
    }.joined()
  }

  /// The error's text as a sentence: LiteRT-LM's native messages often end without a full stop.
  static func describe(_ error: Error) -> String {
    let text = ((error as? LocalizedError)?.errorDescription ?? String(describing: error))
      .trimmingCharacters(in: .whitespacesAndNewlines)
    return text.hasSuffix(".") ? text : text + "."
  }
}

/// A file that says which model was loading. It is written before loading and removed after, so it
/// is still there at the next launch only when the app ended during the load, which is how iOS
/// ends an app that runs out of memory.
enum LoadMarker {
  private static var file: URL {
    FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
      .appendingPathComponent("loading.txt")
  }

  static func set(_ what: String) {
    try? FileManager.default.createDirectory(
      at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
    try? Data(what.utf8).write(to: file, options: .atomic)
  }

  static func clear() {
    try? FileManager.default.removeItem(at: file)
  }

  /// The marker's text, if the last launch left it; the marker is removed.
  static func take() -> String? {
    guard let data = try? Data(contentsOf: file) else { return nil }
    clear()
    return String(decoding: data, as: UTF8.self)
  }
}
