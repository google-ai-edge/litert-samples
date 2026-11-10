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

import PhotosUI
import SwiftUI

struct ChatView: View {
  @ObservedObject var chat: ChatModel
  @State private var photo: PhotosPickerItem?

  var body: some View {
    NavigationStack {
      VStack(spacing: 0) {
        modelBar
        if let notice = chat.notice { noticeBox(notice) }
        messages
        inputBar
      }
      .navigationTitle("LiteRT-LM Chat")
      .navigationBarTitleDisplayMode(.inline)
      .toolbar {
        Button("New chat") { Task { await chat.newChat() } }
          .disabled(chat.phase != .ready)
      }
    }
    .onChange(of: photo) { _, item in
      guard let item else { return }
      Task {
        if let data = try? await item.loadTransferable(type: Data.self) {
          chat.attach(data)
        } else {
          chat.notice = ChatModel.Notice(text: "The picture could not be read from the library.")
        }
        photo = nil
      }
    }
  }

  private var modelBar: some View {
    VStack(alignment: .leading, spacing: 8) {
      HStack {
        Picker("Model", selection: $chat.choice) {
          ForEach(Models.all) { Text($0.name).tag($0) }
        }
        .pickerStyle(.segmented)
        Picker("Backend", selection: $chat.useGPU) {
          Text("GPU").tag(true)
          Text("CPU").tag(false)
        }
        .pickerStyle(.segmented)
        .frame(width: 110)
      }
      HStack {
        Text(status).font(.subheadline).foregroundStyle(.secondary)
        Spacer()
        Button("Load") { Task { await chat.load() } }
          .buttonStyle(.borderedProminent)
          .disabled(chat.loaded == ChatModel.Loaded(model: chat.choice, gpu: chat.useGPU))
      }
      Text(chat.memoryLine).font(.caption2.monospacedDigit()).foregroundStyle(.secondary)
    }
    .disabled(chat.busy)
    .padding(.horizontal)
    .padding(.vertical, 8)
    .background(.bar)
  }

  private var status: String {
    switch chat.phase {
    case .loading:
      if chat.loaded != nil { return "Starting a new conversation…" }
      return "Loading \(chat.choice.name) on the \(chat.useGPU ? "GPU" : "CPU")…"
    case .empty: return "No model loaded"
    default: return chat.loaded?.label ?? ""
    }
  }

  private func noticeBox(_ notice: ChatModel.Notice) -> some View {
    VStack(alignment: .leading, spacing: 8) {
      Text(notice.text).font(.callout).textSelection(.enabled)
      HStack {
        if let action = notice.action {
          Button(action.rawValue) { Task { await chat.perform(action) } }
            .buttonStyle(.bordered)
        }
        Spacer()
        Button("Dismiss") { chat.notice = nil }
      }
    }
    .padding(12)
    .frame(maxWidth: .infinity, alignment: .leading)
    .background(Color.orange.opacity(0.15))
  }

  private var messages: some View {
    ScrollViewReader { scroller in
      ScrollView {
        VStack(alignment: .leading, spacing: 12) {
          ForEach(chat.lines) { line in
            lineView(line).id(line.id)
          }
        }
        .padding()
      }
      .onChange(of: chat.lines.last?.text) { _, _ in
        if let last = chat.lines.last { scroller.scrollTo(last.id, anchor: .bottom) }
      }
      .onChange(of: chat.lines.count) { _, _ in
        if let last = chat.lines.last { scroller.scrollTo(last.id, anchor: .bottom) }
      }
    }
  }

  @ViewBuilder private func lineView(_ line: ChatModel.Line) -> some View {
    switch line.role {
    case .user:
      VStack(alignment: .trailing, spacing: 6) {
        if let image = line.image {
          Image(uiImage: image).resizable().scaledToFit().frame(maxHeight: 160)
            .clipShape(RoundedRectangle(cornerRadius: 8))
        }
        Text(line.text)
          .padding(10)
          .background(Color.accentColor.opacity(0.15), in: RoundedRectangle(cornerRadius: 12))
      }
      .frame(maxWidth: .infinity, alignment: .trailing)
    case .model:
      Text(Self.reply(line.text))
        .textSelection(.enabled)
        .frame(maxWidth: .infinity, alignment: .leading)
    case .note:
      Text(line.text).font(.footnote).foregroundStyle(.secondary)
        .frame(maxWidth: .infinity, alignment: .center)
    }
  }

  /// A reply with its inline Markdown applied (Gemma often writes **bold**); line breaks stay as
  /// they are. Text that does not parse, such as a reply cut off mid-way, is shown as it is.
  private static func reply(_ text: String) -> AttributedString {
    if text.isEmpty { return AttributedString("…") }
    let options = AttributedString.MarkdownParsingOptions(
      interpretedSyntax: .inlineOnlyPreservingWhitespace)
    return (try? AttributedString(markdown: text, options: options)) ?? AttributedString(text)
  }

  private var inputBar: some View {
    VStack(alignment: .leading, spacing: 6) {
      if let picture = chat.picture {
        HStack {
          Image(uiImage: picture.image).resizable().scaledToFit().frame(height: 56)
            .clipShape(RoundedRectangle(cornerRadius: 6))
          Button("Remove") { chat.removePicture() }.font(.footnote)
        }
      }
      HStack(alignment: .bottom) {
        Menu {
          PhotosPicker(selection: $photo, matching: .images) {
            Label("Photo library", systemImage: "photo")
          }
          Button {
            chat.attachExample()
          } label: {
            Label("Example picture", systemImage: "cup.and.saucer")
          }
        } label: {
          Image(systemName: "photo.badge.plus").font(.title2)
        }
        .disabled(chat.phase != .ready)

        TextField("Message", text: $chat.draft, axis: .vertical)
          .lineLimit(1...5)
          .textFieldStyle(.roundedBorder)

        if chat.phase == .replying || chat.phase == .stopping {
          Button {
            Task { await chat.stop() }
          } label: {
            Image(systemName: "stop.circle.fill").font(.title)
          }
          .disabled(chat.phase == .stopping)
        } else {
          Button {
            Task { await chat.send() }
          } label: {
            Image(systemName: "arrow.up.circle.fill").font(.title)
          }
          .disabled(
            chat.phase != .ready
              || chat.draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        }
      }
    }
    .padding(.horizontal)
    .padding(.vertical, 8)
    .background(.bar)
  }
}
