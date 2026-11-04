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
import SwiftUI
import UniformTypeIdentifiers

private let ink = Color(red: 0x1B / 255, green: 0x1B / 255, blue: 0x1F / 255)
private let accent = Color(red: 0x1A / 255, green: 0x73 / 255, blue: 0xE8 / 255)
private let muted = Color(red: 0x5F / 255, green: 0x63 / 255, blue: 0x68 / 255)
private let paper = Color(red: 0xF6 / 255, green: 0xF7 / 255, blue: 0xF9 / 255)
private let good = Color(red: 0x18 / 255, green: 0x80 / 255, blue: 0x38 / 255)
private let bad = Color(red: 0xB3 / 255, green: 0x26 / 255, blue: 0x1E / 255)
private let mark = Color(red: 0xE5 / 255, green: 0x00 / 255, blue: 0x4F / 255)
private let codeBack = Color(red: 0x14 / 255, green: 0x17 / 255, blue: 0x1F / 255)
private let codeInk = Color(red: 0xE6 / 255, green: 0xED / 255, blue: 0xF3 / 255)
private let line = Color(red: 0xDA / 255, green: 0xDC / 255, blue: 0xE0 / 255)

/// The window: the screenshot, the request, Decide, and the next step: a point or a command.
struct ContentView: View {
  @ObservedObject var model: AppModel

  var body: some View {
    GeometryReader { window in
      content(screenshotHeight: min(ScreenshotPanel.maxHeight, window.size.height * 0.42))
    }
    .frame(minWidth: 520, minHeight: 620)
    .background(Color.white)
    .foregroundStyle(ink)
    .preferredColorScheme(.light)
  }

  private func content(screenshotHeight: CGFloat) -> some View {
    VStack(spacing: 0) {
      Header(model: model)
      Divider()
      if model.webGPUErrors > 0 {
        Banner(
          text: "WebGPU reported an error in this engine (\(model.webGPUErrors) log line"
            + "\(model.webGPUErrors == 1 ? "" : "s")). Steps from here on may be wrong: start the engine again.",
          color: bad)
      }
      if let note = model.memoryNote {
        Banner(text: note, color: .orange)
      }
      ScrollViewReader { proxy in
        ScrollView {
          VStack(alignment: .leading, spacing: 8) {
            FilePanel(model: model)
            ScreenshotPanel(model: model, height: screenshotHeight)
            SourceRow(model: model)
            RequestField(model: model)
            if !model.requestChips.isEmpty { RequestChips(model: model) }
            DecideButton(model: model)
            if let message = model.message {
              Text(message)
                .font(.system(size: 14))
                .foregroundStyle(bad)
                .fixedSize(horizontal: false, vertical: true)
            }
            if let decision = model.decision {
              DecisionPanel(model: model, decision: decision)
                .id("decision")
            }
          }
          .padding(.horizontal, 14)
          .padding(.vertical, 8)
        }
        .onChange(of: model.decision?.done) {
          if model.decision?.done == true { proxy.scrollTo("decision", anchor: .bottom) }
        }
      }
    }
  }
}

enum Machine {
  static let chip: String = {
    var size = 0
    sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
    var name = [CChar](repeating: 0, count: max(size, 1))
    sysctlbyname("machdep.cpu.brand_string", &name, &size, nil, 0)
    return String(cString: name)
  }()
}

private struct Header: View {
  @ObservedObject var model: AppModel

  var body: some View {
    VStack(alignment: .leading, spacing: 2) {
      HStack(spacing: 8) {
        Text("Click or code")
          .font(.system(size: 20, weight: .semibold))
        Spacer()
        Circle().fill(dot).frame(width: 9, height: 9)
        Text(status)
          .font(.system(size: 13).monospacedDigit())
          .foregroundStyle(muted)
          .lineLimit(1)
      }
      ViewThatFits(in: .horizontal) {
        Text(about + memoryText)
        Text(about)
      }
      .font(.system(size: 12).monospacedDigit())
      .foregroundStyle(muted)
      .lineLimit(1)
    }
    .padding(.horizontal, 14)
    .padding(.vertical, 6)
  }

  private var about: String {
    "Mage-VL-Cowork on this Mac · GPU · \(Machine.chip) · LiteRT-LM 0.19.0"
  }

  private var memoryText: String {
    guard let memory = model.memory, model.engineRunning, model.phase != .closing else { return "" }
    return " · engine memory \(Format.memory(memory.current)), peak \(Format.memory(memory.peak))"
  }

  private var dot: Color {
    if model.deciding { return accent }
    switch model.phase {
    case .ready: return good
    case .starting, .building, .warming, .closing: return Color(red: 0xF9 / 255, green: 0xAB / 255, blue: 0)
    case .stopped: return bad
    case .idle: return muted
    }
  }

  private var status: String {
    if model.deciding { return "Deciding" }
    switch model.phase {
    case .idle: return "No model yet"
    case .starting: return "Reading the file"
    case .building(let since): return "Building the GPU engine \(Format.clock(model.clock.timeIntervalSince(since)))"
    case .warming(let since): return "Preparing the vision encoder \(Format.clock(model.clock.timeIntervalSince(since)))"
    case .ready: return "Ready"
    case .closing: return "Closing the engine"
    case .stopped(let problem): return problem.title
    }
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

/// The model file's state, until the file is here and the engine runs.
private struct FilePanel: View {
  @ObservedObject var model: AppModel

  var body: some View {
    switch model.file {
    case .checking:
      Text("Looking for the model file…")
        .font(.system(size: 14))
    case .missing(let partial):
      VStack(alignment: .leading, spacing: 8) {
        Text(
          "The app decides the next step on this Mac, offline, with Mage-VL-Cowork. Its model file ("
            + Format.gb(model.fileBytes) + ") downloads once from Hugging Face."
        )
        .font(.system(size: 14))
        .fixedSize(horizontal: false, vertical: true)
        HStack {
          Button(partial > 0 ? "Go on with the download (\(Format.gb(partial)) of \(Format.gb(model.fileBytes)) is here)" : "Download (\(Format.gb(model.fileBytes))) and load") {
            model.downloadModel()
          }
          .controlSize(.large)
          .keyboardShortcut(.defaultAction)
          Button("Use a copy on this Mac…") { model.chooseLocalFile() }
            .controlSize(.large)
        }
      }
    case .downloading(let received):
      VStack(alignment: .leading, spacing: 6) {
        HStack {
          Text("Downloading \(Format.gb(received)) of \(Format.gb(model.fileBytes))")
            .font(.system(size: 14).monospacedDigit())
          Spacer()
          Button("Stop") { model.cancelDownload() }
        }
        ProgressView(value: Double(received), total: Double(max(model.fileBytes, 1)))
      }
    case .verifying:
      HStack(spacing: 8) {
        ProgressView().controlSize(.small)
        Text("Checking the file (size and SHA-256)…")
          .font(.system(size: 14))
      }
    case .failed(let message):
      VStack(alignment: .leading, spacing: 8) {
        Text(message)
          .font(.system(size: 14))
          .foregroundStyle(bad)
          .fixedSize(horizontal: false, vertical: true)
          .textSelection(.enabled)
        HStack {
          Button("Try again") { model.downloadModel() }
            .controlSize(.large)
          Button("Use a copy on this Mac…") { model.chooseLocalFile() }
            .controlSize(.large)
        }
      }
    case .ready:
      switch model.phase {
      case .stopped(let problem):
        VStack(alignment: .leading, spacing: 8) {
          Text(problem.detail)
            .font(.system(size: 14))
            .foregroundStyle(bad)
            .fixedSize(horizontal: false, vertical: true)
            .textSelection(.enabled)
          if problem.canRetry {
            Button("Start the engine again") { model.restartEngine() }
          }
        }
      case .starting, .building, .warming:
        HStack(spacing: 8) {
          ProgressView().controlSize(.small)
          Text("Loading the model…")
            .font(.system(size: 13))
            .foregroundStyle(muted)
        }
      default:
        EmptyView()
      }
    }
  }
}

/// The screenshot at its own shape, with the click's mark drawn on it.
private struct ScreenshotPanel: View {
  @ObservedObject var model: AppModel
  let height: CGFloat

  static let maxHeight: CGFloat = 430

  var body: some View {
    Group {
      if let picture = model.picture {
        let shown = model.decision?.done == true && model.decision?.stopped != true ? model.decision : nil
        let call = shown?.reply.first
        let points = call?.isCode == true ? [] : [call?.coordinate, call?.start, call?.end].compactMap { $0 }
        Image(nsImage: picture.display)
          .resizable()
          .interpolation(.high)
          .aspectRatio(CGFloat(picture.width) / CGFloat(picture.height), contentMode: .fit)
          .frame(maxWidth: .infinity, maxHeight: height)
          .overlay {
            GeometryReader { geometry in
              let side = min(geometry.size.width, geometry.size.height * CGFloat(picture.width) / CGFloat(picture.height))
              let frame = CGSize(width: side, height: side * CGFloat(picture.height) / CGFloat(picture.width))
              let origin = CGPoint(x: (geometry.size.width - frame.width) / 2, y: (geometry.size.height - frame.height) / 2)
              ZStack(alignment: .topLeading) {
                RoundedRectangle(cornerRadius: 4)
                  .strokeBorder(line, lineWidth: 1)
                  .frame(width: frame.width, height: frame.height)
                ForEach(Array(points.enumerated()), id: \.offset) { _, point in
                  Ring()
                    .id(shown?.text ?? "")
                    .position(
                      x: CGFloat(min(max(point.x, 0), 999) / 999) * frame.width,
                      y: CGFloat(min(max(point.y, 0), 999) / 999) * frame.height)
                    .frame(width: frame.width, height: frame.height)
                }
              }
              .offset(x: origin.x, y: origin.y)
            }
          }
      } else {
        VStack(spacing: 6) {
          Text(model.preparingPicture ? "Preparing the screenshot…" : "Capture a screen or a window, or choose a screenshot or an example.")
            .font(.system(size: 14))
            .foregroundStyle(muted)
          Text("It stays on this Mac.")
            .font(.system(size: 12))
            .foregroundStyle(muted)
        }
        .frame(maxWidth: .infinity, minHeight: 150)
        .background(RoundedRectangle(cornerRadius: 8).fill(paper))
        .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(line, lineWidth: 1))
      }
    }
  }
}

/// The screenshot around the point, at its own pixels: on a small icon the ring on the whole picture covers what it points at.
private struct PointLens: View {
  let picture: NSImage
  /// The point as fractions of the picture.
  let x: Double
  let y: Double

  var body: some View {
    if let image = picture.cgImage(forProposedRect: nil, context: nil, hints: nil) {
      let w = min(520, image.width)
      let h = min(176, image.height)
      let px = Int(x * Double(image.width))
      let py = Int(y * Double(image.height))
      let left = min(max(px - w / 2, 0), image.width - w)
      let top = min(max(py - h / 2, 0), image.height - h)
      if let crop = image.cropping(to: CGRect(x: left, y: top, width: w, height: h)) {
        Image(decorative: crop, scale: 2)
          .overlay(alignment: .topLeading) {
            ZStack {
              Circle().stroke(Color.white, lineWidth: 5)
              Circle().stroke(mark, lineWidth: 2.5)
            }
            .frame(width: 44, height: 44)
            .position(x: CGFloat(px - left) / 2, y: CGFloat(py - top) / 2)
          }
          .clipShape(RoundedRectangle(cornerRadius: 6))
          .overlay(RoundedRectangle(cornerRadius: 6).strokeBorder(line, lineWidth: 1))
          .accessibilityLabel("Close-up of the screenshot around the point")
      }
    }
  }
}

/// A ring that closes on the point.
private struct Ring: View {
  @State private var grow: CGFloat = 0

  var body: some View {
    let radius = 14 * (3 - 2 * grow)
    ZStack {
      Circle().stroke(Color.white, lineWidth: 6).frame(width: radius * 2 + 3, height: radius * 2 + 3)
      Circle().stroke(mark, lineWidth: 3.5).frame(width: radius * 2, height: radius * 2)
      Circle().fill(Color.white).frame(width: 10, height: 10)
      Circle().fill(mark).frame(width: 6, height: 6)
    }
    .onAppear {
      withAnimation(.easeOut(duration: 0.45)) { grow = 1 }
    }
    .allowsHitTesting(false)
  }
}

private struct SourceRow: View {
  @ObservedObject var model: AppModel

  var body: some View {
    VStack(alignment: .leading, spacing: 6) {
      if let problem = model.pictureProblem {
        Text(problem)
          .font(.system(size: 13))
          .foregroundStyle(bad)
          .fixedSize(horizontal: false, vertical: true)
          .textSelection(.enabled)
      }
      HStack(spacing: 8) {
        Button("Capture screen") { model.captureScreen() }
          .help("The whole main display, with this window out of the way (⇧⌘1)")
        Button("Capture window…") { model.captureWindow() }
          .help("Click a window; Esc leaves everything as it was (⇧⌘2)")
        Button("Open picture…") { model.choosePicture() }
        Spacer(minLength: 4)
        Menu("Examples") {
          ForEach(Example.all) { example in
            Button(example.name) { model.useExample(example) }
          }
        }
        .fixedSize()
      }
      .disabled(model.deciding)
    }
  }
}

private struct RequestField: View {
  @ObservedObject var model: AppModel
  @FocusState private var focused: Bool

  var body: some View {
    TextField("What should happen on this screen?", text: $model.request, axis: .vertical)
      .textFieldStyle(.plain)
      .font(.system(size: 15))
      .lineLimit(1...3)
      .focused($focused)
      .onSubmit { model.decide() }
      .padding(8)
      .background(RoundedRectangle(cornerRadius: 8).strokeBorder(line))
      .disabled(model.deciding)
      .onAppear { focused = true }
  }
}

private struct RequestChips: View {
  @ObservedObject var model: AppModel

  var body: some View {
    VStack(alignment: .leading, spacing: 6) {
      ForEach(model.requestChips, id: \.self) { chip in
        Button {
          model.request = chip
        } label: {
          Text(chip)
            .font(.system(size: 13))
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(RoundedRectangle(cornerRadius: 14).fill(paper))
            .overlay(RoundedRectangle(cornerRadius: 14).strokeBorder(line))
        }
        .buttonStyle(.plain)
        .disabled(model.deciding)
      }
    }
  }
}

private struct DecideButton: View {
  @ObservedObject var model: AppModel

  var body: some View {
    if model.deciding {
      Button {
        model.stopDecision()
      } label: {
        Text(model.stopping ? "Stopping…" : String(format: "Stop · %.0f s", model.elapsed))
          .font(.system(size: 16).monospacedDigit())
          .frame(maxWidth: .infinity, minHeight: 34)
          .background(RoundedRectangle(cornerRadius: 8).fill(paper))
          .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(line))
          .contentShape(Rectangle())
      }
      .buttonStyle(.plain)
      .disabled(model.stopping)
      .keyboardShortcut(".", modifiers: .command)
    } else {
      Button {
        model.decide()
      } label: {
        Text("Decide")
          .font(.system(size: 16, weight: .medium))
          .foregroundStyle(Color.white)
          .frame(maxWidth: .infinity, minHeight: 34)
          .background(RoundedRectangle(cornerRadius: 8).fill(model.canDecide ? accent : muted.opacity(0.45)))
          .contentShape(Rectangle())
      }
      .buttonStyle(.plain)
      .disabled(!model.canDecide)
      .keyboardShortcut(.return, modifiers: .command)
    }
  }
}

/// The next step: the word CLICK or CODE, the model's Action sentence, the point or the command.
private struct DecisionPanel: View {
  @ObservedObject var model: AppModel
  let decision: AppModel.Decision

  var body: some View {
    VStack(alignment: .leading, spacing: 6) {
      let call = decision.reply.first
      if !decision.done {
        Text(phase(call))
          .font(.system(size: 15, weight: .medium))
        if !decision.text.isEmpty {
          Text(String(decision.reply.thinking.suffix(170)))
            .font(.system(size: 12))
            .foregroundStyle(muted)
            .lineLimit(3)
        }
      } else {
        if let error = decision.error {
          Text("Stopped: \(error)")
            .font(.system(size: 14))
            .foregroundStyle(bad)
            .textSelection(.enabled)
        } else if decision.stopped {
          Text("Stopped.")
            .font(.system(size: 14))
            .foregroundStyle(muted)
        } else if let call {
          Answer(model: model, call: call, decision: decision)
        } else {
          Text(decision.cut ? "The reply reached its token limit before it gave a step." : "The reply has no step in it.")
            .font(.system(size: 14))
            .foregroundStyle(bad)
        }
        StagesLine(decision: decision)
        Thinking(decision: decision)
      }
    }
    .frame(maxWidth: .infinity, alignment: .leading)
    .padding(10)
    .background(RoundedRectangle(cornerRadius: 8).fill(paper))
  }

  private func phase(_ call: ToolCall?) -> String {
    if !decision.hasText && decision.inference == 2 { return "The model asked for a screenshot. Inference 2: reading the screen…" }
    if !decision.hasText { return "Inference 1: reading the screen…" }
    if call == nil { return "Inference \(decision.inference): thinking…" }
    return "Inference \(decision.inference): writing the step…"
  }
}

private struct Answer: View {
  @ObservedObject var model: AppModel
  let call: ToolCall
  let decision: AppModel.Decision

  var body: some View {
    HStack(spacing: 10) {
      Text(call.isCode ? "CODE" : (call.action ?? call.name).replacingOccurrences(of: "_", with: " ").uppercased())
        .font(.system(size: 15, weight: .bold))
        .foregroundStyle(Color.white)
        .padding(.horizontal, 8)
        .padding(.vertical, 3)
        .background(RoundedRectangle(cornerRadius: 6).fill(call.isCode ? codeBack : mark))
      if !call.isCode, let point = call.coordinate {
        let pixels = Step.toPixels(point, width: decision.width, height: decision.height)
        Text("x \(pixels.x), y \(pixels.y) of \(decision.width) × \(decision.height)")
          .font(.system(size: 13).monospacedDigit())
          .foregroundStyle(muted)
      }
    }
    if let action = decision.reply.actionLine {
      Text(action)
        .font(.system(size: 15, weight: .medium))
        .fixedSize(horizontal: false, vertical: true)
    }
    if !call.isCode, let point = call.coordinate, let picture = model.picture {
      PointLens(picture: picture.display, x: min(max(point.x, 0), 999) / 999, y: min(max(point.y, 0), 999) / 999)
    }
    if call.isCode, let command = call.command {
      Text(command)
        .font(.system(size: 13, design: .monospaced))
        .foregroundStyle(codeInk)
        .textSelection(.enabled)
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(8)
        .background(RoundedRectangle(cornerRadius: 6).fill(codeBack))
      HStack {
        Text("Shown, not run.")
          .font(.system(size: 12))
          .foregroundStyle(muted)
        Spacer()
        Button(model.copied ? "Copied" : "Copy") { model.copyCommand(command) }
      }
    } else {
      let extra = [call.text.map { "type: \($0)" }, call.keys.isEmpty ? nil : "keys: \(call.keys.joined(separator: "+"))"]
        .compactMap { $0 }.joined(separator: "\n")
      if !extra.isEmpty {
        Text(extra)
          .font(.system(size: 13, design: .monospaced))
      }
    }
    if decision.reply.calls.count > 1 {
      Text("+ \(decision.reply.calls.count - 1) more in the same step")
        .font(.system(size: 12))
        .foregroundStyle(muted)
    }
  }
}

private struct StagesLine: View {
  let decision: AppModel.Decision

  var body: some View {
    if let total = decision.totalSeconds {
      let head = String(format: "%d inference%@ · %.1f s in all", decision.runs.count, decision.runs.count == 1 ? "" : "s", total)
      let lines = decision.runs.enumerated().map { index, run -> String in
        var parts = [String(format: "%.1f s", run.totalSeconds)]
        if let vision = run.visionSeconds { parts.append(String(format: "vision %.1f s", vision)) }
        if let prefill = run.prefillSeconds { parts.append(String(format: "prefill %.1f s", prefill)) }
        if let decode = run.decodeSeconds { parts.append(String(format: "decode %.1f s (%d tokens)", decode, run.decodeTokens ?? 0)) }
        return "\(index + 1): " + parts.joined(separator: " · ")
      }
      Text(([head] + lines).joined(separator: "\n"))
        .font(.system(size: 12).monospacedDigit())
        .foregroundStyle(muted)
        .textSelection(.enabled)
    }
  }
}

/// The model's reasoning, folded.
private struct Thinking: View {
  let decision: AppModel.Decision
  @State private var open = false

  var body: some View {
    let reasoning = decision.runs.map(\.thinking).filter { !$0.isEmpty }.joined(separator: "\n\n")
    if !reasoning.isEmpty {
      Button(open ? "Hide the reasoning" : "Show the reasoning") { open.toggle() }
        .buttonStyle(.link)
        .font(.system(size: 13))
      if open {
        Text(reasoning)
          .font(.system(size: 13))
          .textSelection(.enabled)
          .fixedSize(horizontal: false, vertical: true)
      }
    }
  }
}
