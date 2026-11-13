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

import Foundation
import LiteRTLM

/// One model on LiteRT-LM: an `Engine` and its conversation (an engine runs one conversation at a
/// time). Every LiteRT-LM call and every release happens on this actor, in this order: the reply
/// ends, then the conversation goes, then the engine.
actor ModelRunner {
  /// The context of one conversation: prompt, pictures and replies together.
  static let contextTokens = 4000
  /// The longest reply.
  static let replyTokens = 1024

  private var engine: Engine?
  private var conversation: Conversation?

  /// Loads a model file. Throws LiteRT-LM's error when the file cannot be read or the backend
  /// cannot start.
  func load(file: URL, gpu: Bool) async throws {
    await unload()
    if SelfCheck.failGPU && gpu { throw SimulatedGPUFailure() }
    let config = try EngineConfig(
      modelPath: file.path,
      // The text decoder: the GPU (Metal) or the CPU.
      backend: gpu ? .gpu : .cpu(),
      // The image encoder runs on the CPU, as in the LiteRT-LM Swift guide. With nil the engine
      // does not load it and the model cannot see pictures.
      visionBackend: .cpu(),
      maxNumTokens: Self.contextTokens,
      // LiteRT-LM writes its weight and GPU program caches here; later loads read them.
      cacheDir: Self.cacheDirectory().path)
    let engine = Engine(engineConfig: config)
    try await engine.initialize()
    let conversation = try await engine.createConversation()
    self.engine = engine
    self.conversation = conversation
  }

  /// Sends a message, with the picture first as in the LiteRT-LM Swift guide. The reply arrives in
  /// pieces on the returned stream.
  func send(_ text: String, picture: URL?) throws -> AsyncThrowingStream<Message, Error> {
    guard let conversation else { throw NoModel() }
    var contents: [Content] = []
    if let picture { contents.append(.imageFile(picture.path)) }
    contents.append(.text(text))
    return conversation.sendMessageStream(
      Message(contents: contents), maxOutputTokens: Self.replyTokens)
  }

  /// Asks LiteRT-LM to stop the reply in progress; its stream then ends with an error.
  func stop() {
    try? conversation?.cancel()
  }

  /// Replaces the conversation with a new one on the same engine: for New chat, and after Stop or
  /// an error.
  func newConversation() async throws {
    guard let engine else { throw NoModel() }
    await releaseConversation()
    conversation = try await engine.createConversation()
  }

  /// LiteRT-LM's own timing of the last reply; nil unless the self-check turned it on.
  func benchmark() -> BenchmarkInfo? {
    try? conversation?.getBenchmarkInfo()
  }

  /// Frees the model's memory.
  func unload() async {
    await releaseConversation()
    // The conversation held the engine too; `Engine.deinit` deletes it here, on this actor.
    engine = nil
  }

  /// A reply's stream holds the conversation until LiteRT-LM delivers its last piece, and lets go
  /// of it on LiteRT-LM's callback thread. If that were the last reference, `Conversation.deinit`
  /// would run on the callback thread and wait for the callback's own task to end: in LiteRT-LM
  /// 0.19.1 it waits out a 10-minute timeout. So the app keeps its reference until it is the only
  /// one, and the conversation is deleted here.
  private func releaseConversation() async {
    guard var last = conversation else { return }
    conversation = nil
    var waited = 0
    while !isKnownUniquelyReferenced(&last), waited < 500 {
      try? await Task.sleep(for: .milliseconds(10))
      waited += 1
    }
  }

  private static func cacheDirectory() -> URL {
    let url = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
      .appendingPathComponent("litert-lm", isDirectory: true)
    try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
  }

  struct NoModel: LocalizedError {
    var errorDescription: String? { "No model is loaded." }
  }
}
