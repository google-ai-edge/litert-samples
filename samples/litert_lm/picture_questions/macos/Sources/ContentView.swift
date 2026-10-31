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

import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
  @ObservedObject var model: AppModel

  var body: some View {
    VStack(spacing: 0) {
      HeaderBar(model: model)
      Divider()
      if model.webGPUErrors > 0 {
        Banner(
          text: "WebGPU reported an error in this engine (\(model.webGPUErrors) log line"
            + "\(model.webGPUErrors == 1 ? "" : "s")). Answers from here on may be wrong: start the engine again.",
          color: .red)
      }
      if let note = model.memoryNote {
        Banner(text: note, color: .orange)
      }
      HStack(spacing: 0) {
        PicturePane(model: model)
          .frame(width: 420)
        Divider()
        ConversationPane(model: model)
      }
    }
    .frame(minWidth: 1040, minHeight: 700)
  }
}

// MARK: Header

private struct HeaderBar: View {
  @ObservedObject var model: AppModel

  var body: some View {
    HStack(spacing: 12) {
      Button {
        model.chooseModel()
      } label: {
        Label("Open model…", systemImage: "folder")
      }
      if let file = model.file {
        VStack(alignment: .leading, spacing: 2) {
          Text(file.displayName)
            .font(.headline)
          Text("\(file.name) · \(Format.gb(file.size)) · GPU · LiteRT-LM 0.19.0")
            .font(.subheadline)
            .foregroundStyle(.secondary)
            .lineLimit(1)
            .truncationMode(.middle)
        }
      } else {
        Text("No model open")
          .foregroundStyle(.secondary)
      }
      Spacer()
      VStack(alignment: .trailing, spacing: 2) {
        status
        if let memory = model.memory, model.engineRunning, model.phase != .closing {
          Text("Engine memory \(Format.memory(memory.current)) · peak \(Format.memory(memory.peak))")
            .font(.subheadline.monospacedDigit())
            .foregroundStyle(.secondary)
        }
      }
    }
    .padding(.horizontal, 16)
    .padding(.vertical, 10)
  }

  @ViewBuilder private var status: some View {
    switch model.phase {
    case .empty:
      EmptyView()
    case .starting:
      Text("Reading the file…")
    case .building(let since):
      Text("Building the GPU engine · \(Format.clock(model.clock.timeIntervalSince(since)))")
        .monospacedDigit()
    case .warming(let since):
      Text("Preparing the vision encoder · \(Format.clock(model.clock.timeIntervalSince(since)))")
        .monospacedDigit()
    case .ready, .answering:
      Text(readyLine)
        .monospacedDigit()
    case .closing:
      Text("Closing the engine…")
    case .stopped(let problem):
      Text(problem.title)
        .foregroundStyle(.red)
    }
  }

  private var readyLine: String {
    guard let build = model.buildSeconds else { return "Ready" }
    var line = "Ready · engine built in \(Format.clock(build))"
    if let warm = model.warmUpSeconds { line += ", vision encoder in \(Format.clock(warm))" }
    return line
  }
}

private struct Banner: View {
  let text: String
  let color: Color

  var body: some View {
    HStack(alignment: .top, spacing: 8) {
      Image(systemName: "exclamationmark.triangle.fill")
      Text(text)
        .fixedSize(horizontal: false, vertical: true)
      Spacer(minLength: 0)
    }
    .font(.callout)
    .foregroundStyle(color)
    .padding(.horizontal, 16)
    .padding(.vertical, 8)
    .background(color.opacity(0.1))
  }
}

// MARK: Picture

private struct PicturePane: View {
  @ObservedObject var model: AppModel
  @State private var dropTargeted = false

  var body: some View {
    ScrollView {
      VStack(alignment: .leading, spacing: 14) {
        pictureBox
        if let problem = model.pictureProblem {
          Label(problem, systemImage: "exclamationmark.circle.fill")
            .foregroundStyle(.red)
            .fixedSize(horizontal: false, vertical: true)
            .textSelection(.enabled)
        }
        HStack {
          Button("Open picture…") { model.choosePicture() }
          Button("Paste picture") { model.pastePicture() }
            .help("A picture or a screenshot on the clipboard (⇧⌘V)")
        }
        if model.isMuse {
          VStack(alignment: .leading, spacing: 6) {
            Picker("Reasoning strength", selection: $model.strength) {
              ForEach(ReasoningStrength.allCases) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)
            .disabled(model.phase == .answering)
            Text("How much Muse-Glimmer-30B reasons before it answers. Changing it starts a new conversation.")
              .font(.caption)
              .foregroundStyle(.secondary)
              .fixedSize(horizontal: false, vertical: true)
          }
        }
        if !Example.all.isEmpty {
          VStack(alignment: .leading, spacing: 8) {
            Text("Examples")
              .font(.headline)
            ForEach(Example.all) { example in
              Button {
                model.useExample(example)
              } label: {
                VStack(alignment: .leading, spacing: 2) {
                  Text(example.title)
                    .font(.callout.weight(.semibold))
                  Text(example.question)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
              }
              .buttonStyle(.bordered)
              .disabled(model.phase != .ready)
            }
          }
        }
      }
      .padding(16)
    }
    .onDrop(of: [.fileURL], isTargeted: $dropTargeted) { providers in
      guard let provider = providers.first else { return false }
      _ = provider.loadObject(ofClass: URL.self) { url, _ in
        guard let url else { return }
        DispatchQueue.main.async { model.setPicture(url) }
      }
      return true
    }
  }

  @ViewBuilder private var pictureBox: some View {
    if let picture = model.picture, let image = NSImage(contentsOf: picture.modelPNG) {
      VStack(alignment: .leading, spacing: 6) {
        Image(nsImage: image)
          .resizable()
          .interpolation(.high)
          .aspectRatio(1, contentMode: .fit)
          .frame(width: 388, height: 388)
          .overlay(RoundedRectangle(cornerRadius: 4).strokeBorder(dropTargeted ? Color.accentColor : .secondary.opacity(0.3), lineWidth: dropTargeted ? 3 : 1))
        Text("The model sees this 448 × 448 picture.")
          .font(.callout)
        Text("\(picture.sourceName): \(picture.note)")
          .font(.caption)
          .foregroundStyle(.secondary)
          .fixedSize(horizontal: false, vertical: true)
          .textSelection(.enabled)
      }
    } else {
      VStack(spacing: 10) {
        Image(systemName: "photo.on.rectangle.angled")
          .font(.system(size: 44))
          .foregroundStyle(.secondary)
        Text("Drop a picture here")
          .font(.title3)
        Text("A screenshot, a chart, a document or a photo. The model reads it at 448 × 448 pixels, so crop to the part you are asking about.")
          .multilineTextAlignment(.center)
          .foregroundStyle(.secondary)
          .fixedSize(horizontal: false, vertical: true)
      }
      .padding(20)
      .frame(width: 388, height: 388)
      .background(
        RoundedRectangle(cornerRadius: 8)
          .strokeBorder(dropTargeted ? Color.accentColor : .secondary.opacity(0.4), style: StrokeStyle(lineWidth: 2, dash: [6]))
      )
    }
  }
}

// MARK: Conversation

private struct ConversationPane: View {
  @ObservedObject var model: AppModel

  var body: some View {
    VStack(spacing: 0) {
      if model.turns.isEmpty {
        VStack {
          Spacer()
          StatusCard(model: model)
            .frame(maxWidth: 560)
          Spacer()
        }
        .padding(20)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
      } else {
        ScrollViewReader { proxy in
          ScrollView {
            LazyVStack(alignment: .leading, spacing: 22) {
              ForEach(model.turns) { turn in
                TurnView(
                  turn: turn, speaker: model.file?.displayName ?? "Model", format: model.format ?? .plain,
                  now: model.clock)
              }
              StatusCard(model: model)
              Color.clear.frame(height: 1).id("bottom")
            }
            .padding(20)
          }
          .defaultScrollAnchor(.bottom)
          .onChange(of: model.turns.last?.raw) { proxy.scrollTo("bottom", anchor: .bottom) }
          .onChange(of: model.turns.count) { proxy.scrollTo("bottom", anchor: .bottom) }
        }
      }
      Divider()
      Composer(model: model)
    }
  }
}

private struct TurnView: View {
  let turn: AppModel.Turn
  let speaker: String
  let format: ReplyFormat
  /// The window's clock (ticks once a second while an engine runs).
  let now: Date

  /// No new text for this long while a reply streams: say so (the first text comes in a few seconds, and then a
  /// piece every fraction of a second).
  static let quietSeconds: Double = 30

  var body: some View {
    let reply = ReplyText.parse(turn.raw, format: format)
    VStack(alignment: .leading, spacing: 12) {
      VStack(alignment: .leading, spacing: 4) {
        Text("You")
          .font(.caption.weight(.semibold))
          .foregroundStyle(.secondary)
        Text(turn.question)
          .font(.system(size: 15))
          .textSelection(.enabled)
          .padding(10)
          .background(RoundedRectangle(cornerRadius: 8).fill(Color.accentColor.opacity(0.12)))
      }
      VStack(alignment: .leading, spacing: 6) {
        Text(speaker)
          .font(.caption.weight(.semibold))
          .foregroundStyle(.secondary)
        if let reasoning = reply.reasoning, !reasoning.isEmpty {
          DisclosureGroup(reply.stillReasoning ? "Reasoning…" : "Reasoning (\(reasoning.split(separator: " ").count) words)") {
            Text(reasoning)
              .font(.callout)
              .foregroundStyle(.secondary)
              .textSelection(.enabled)
              .frame(maxWidth: .infinity, alignment: .leading)
          }
          .font(.callout)
        }
        if reply.answer.isEmpty && turn.state == .streaming {
          HStack(spacing: 8) {
            ProgressView().controlSize(.small)
            Text(reply.stillReasoning ? "Reasoning" : "Reading the picture")
              .foregroundStyle(.secondary)
          }
        } else {
          MarkdownText(text: reply.answer)
        }
        footer
      }
    }
  }

  @ViewBuilder private var footer: some View {
    switch turn.state {
    case .streaming:
      VStack(alignment: .leading, spacing: 2) {
        if let rate = turn.liveRate, turn.chunks >= 8 {
          Text(String(format: "%.1f tok/s · %d tokens", rate, turn.chunks))
            .foregroundStyle(.secondary)
        }
        let quiet = now.timeIntervalSince(turn.lastChunkAt ?? turn.sentAt)
        if quiet >= Self.quietSeconds {
          Text(
            "No new text for \(Format.clock(quiet)). The engine is not answering; it may be waiting for memory. "
              + "Stop ends this answer."
          )
          .foregroundStyle(.orange)
        }
      }
      .font(.caption.monospacedDigit())
    case .done, .cancelled, .limit:
      VStack(alignment: .leading, spacing: 2) {
        if turn.state == .cancelled {
          Text("Stopped. The next question starts a new conversation about the same picture.")
        }
        if turn.state == .limit {
          Text("The reply reached its token limit or the end of the 2,048-token context before it finished.")
        }
        Text(Self.summary(of: turn))
      }
      .font(.caption.monospacedDigit())
      .foregroundStyle(.secondary)
      .textSelection(.enabled)
    case .failed(let message):
      Text(message)
        .font(.caption)
        .foregroundStyle(.red)
        .textSelection(.enabled)
    }
  }

  /// A rate over a handful of tokens is mostly the fixed cost of a call, so it shows from these counts on.
  static let minPrefillTokensForRate = 64
  static let minDecodeTokensForRate = 16

  static func summary(of turn: AppModel.Turn) -> String {
    guard let stats = turn.stats else { return "" }
    var parts: [String] = []
    if let first = stats.firstTextSeconds { parts.append(String(format: "first text %.2f s", first)) }
    if let answer = stats.firstAnswerSeconds, let first = stats.firstTextSeconds, answer - first > 0.05 {
      parts.append(String(format: "answer from %.1f s", answer))
    }
    if let counters = stats.counters {
      if let tokens = counters.prefillTokens, tokens > 0 {
        if let rate = counters.prefillTokensPerSecond, tokens >= minPrefillTokensForRate {
          parts.append(String(format: "prefill %d tokens at %.1f tok/s", tokens, rate))
        } else {
          parts.append("prefill \(tokens) tokens")
        }
      }
      if let tokens = counters.decodeTokens, tokens > 0 {
        if let rate = counters.decodeTokensPerSecond, tokens >= minDecodeTokensForRate {
          parts.append(String(format: "decode %d tokens at %.1f tok/s", tokens, rate))
        } else {
          parts.append("decode \(tokens) tokens")
        }
      }
    }
    return parts.joined(separator: " · ")
  }
}

private struct StatusCard: View {
  @ObservedObject var model: AppModel

  var body: some View {
    switch model.phase {
    case .empty:
      welcome
    case .starting, .closing:
      HStack(spacing: 10) {
        ProgressView().controlSize(.small)
        Text(model.phase == .closing ? "Closing the engine…" : "Reading the file…")
          .foregroundStyle(.secondary)
      }
    case .building(let since), .warming(let since):
      buildingCard(since: since)
    case .ready where model.turns.isEmpty:
      Text(
        model.picture == nil
          ? "Ready. Add a picture on the left, or click an example."
          : "Ready. Ask about the picture, or click an example."
      )
      .foregroundStyle(.secondary)
    case .stopped(let problem):
      problemCard(problem)
    default:
      EmptyView()
    }
  }

  private var welcome: some View {
    VStack(alignment: .leading, spacing: 14) {
      Text("Ask a large vision-language model about a picture, on this Mac")
        .font(.title2)
      Text(
        "Open a LiteRT-LM model file. The app runs it on the Mac's GPU with Google's LiteRT-LM runtime; the picture "
          + "and the question never leave the Mac."
      )
      .foregroundStyle(.secondary)
      .fixedSize(horizontal: false, vertical: true)
      ForEach(KnownModel.all, id: \.name) { known in
        VStack(alignment: .leading, spacing: 4) {
          Text("\(known.name) (\(Format.gb(known.fileBytes)), uses about \(Format.memory(known.peakFootprintBytes)) of memory):")
            .font(.callout)
            .foregroundStyle(.secondary)
          Text("hf download \(known.huggingFace) \(known.fileName) --local-dir ~/models")
            .font(.system(.callout, design: .monospaced))
            .textSelection(.enabled)
            .padding(8)
            .background(RoundedRectangle(cornerRadius: 6).fill(Color.secondary.opacity(0.12)))
        }
      }
      HStack {
        Button("Open a .litertlm file…") { model.chooseModel() }
          .controlSize(.large)
          .keyboardShortcut(.defaultAction)
        if let last = model.lastModelURL {
          Button("Open \(last.lastPathComponent) again") { model.openModel(last) }
            .controlSize(.large)
        }
      }
    }
  }

  private func buildingCard(since: Date) -> some View {
    VStack(alignment: .leading, spacing: 10) {
      HStack(alignment: .firstTextBaseline, spacing: 12) {
        ProgressView().controlSize(.small)
        Text(Format.clock(model.clock.timeIntervalSince(since)))
          .font(.system(size: 34, weight: .medium, design: .rounded).monospacedDigit())
        Text(model.phase.isWarming ? "Preparing the vision encoder" : "Building the GPU engine")
          .font(.title3)
      }
      Text(
        model.phase.isWarming
          ? "One gray picture goes through the vision encoder now, so that your first question does not wait for its GPU programs."
          : "LiteRT-LM reads the model into memory and prepares it for the GPU, once per launch. You can add a picture and type a question meanwhile."
      )
      .foregroundStyle(.secondary)
      .fixedSize(horizontal: false, vertical: true)
    }
    .padding(16)
    .frame(maxWidth: .infinity, alignment: .leading)
    .background(RoundedRectangle(cornerRadius: 10).fill(Color.secondary.opacity(0.08)))
  }

  private func problemCard(_ problem: AppModel.Problem) -> some View {
    VStack(alignment: .leading, spacing: 10) {
      Label(problem.title, systemImage: "exclamationmark.octagon.fill")
        .font(.title3)
        .foregroundStyle(.red)
      Text(problem.detail)
        .textSelection(.enabled)
        .fixedSize(horizontal: false, vertical: true)
      HStack {
        Button("Open another file…") { model.chooseModel() }
        if model.canRestartEngine {
          Button("Start the engine again") { model.restartEngine() }
        }
      }
    }
    .padding(16)
    .frame(maxWidth: .infinity, alignment: .leading)
    .background(RoundedRectangle(cornerRadius: 10).fill(Color.red.opacity(0.07)))
  }
}

extension AppModel.Phase {
  var isWarming: Bool {
    if case .warming = self { return true }
    return false
  }
}

private struct Composer: View {
  @ObservedObject var model: AppModel
  @FocusState private var focused: Bool

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      HStack(alignment: .bottom, spacing: 10) {
        TextField(placeholder, text: $model.draft, axis: .vertical)
          .textFieldStyle(.plain)
          .font(.system(size: 15))
          .lineLimit(1...6)
          .focused($focused)
          .onSubmit { model.ask() }
          .padding(10)
          .background(RoundedRectangle(cornerRadius: 8).strokeBorder(Color.secondary.opacity(0.35)))
        if model.phase == .answering {
          Button("Stop") { model.stopAnswer() }
            .controlSize(.large)
            .keyboardShortcut(".", modifiers: .command)
        } else {
          Button("Ask") { model.ask() }
            .controlSize(.large)
            .keyboardShortcut(.return, modifiers: .command)
            .disabled(!model.canAsk)
        }
      }
      HStack {
        Button("New conversation") { model.newConversation() }
          .disabled(model.turns.isEmpty || model.phase == .answering)
        Text("Follow-up questions are about the same picture. A new conversation forgets the earlier questions.")
          .font(.caption)
          .foregroundStyle(.secondary)
        Spacer()
      }
    }
    .padding(14)
    .onAppear { focused = true }
  }

  private var placeholder: String {
    if model.picture == nil { return "Add a picture first" }
    if model.turns.isEmpty { return "Ask about this picture" }
    return "Ask a follow-up question"
  }
}
