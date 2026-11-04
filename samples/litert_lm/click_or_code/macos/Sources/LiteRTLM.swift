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

import CLiteRTLM
import Foundation

// A small Swift layer over the LiteRT-LM 0.19.0 C API (the headers inside CLiteRTLM_mac.xcframework). It makes the
// same calls, in the same order and with the same values, as LiteRT-LM's own Python API 0.19.0
// (python/litert_lm/engine.py and conversation.py at the v0.19.0 tag), so a reply here can be compared word for
// word with one from `litert_lm.Engine(...).create_conversation(...).send_message_async(...)`.

struct LiteRTLMError: LocalizedError {
  let call: String
  let status: UInt32
  let message: String

  var errorDescription: String? {
    message.isEmpty ? "\(call) failed (LiteRT-LM status \(status))." : message
  }
}

private func check(_ status: LiteRtLmStatusCode, _ call: String) throws {
  guard status != kLiteRtLmStatusOk else { return }
  let message = litert_lm_get_last_error_message().map { String(cString: $0) } ?? ""
  litert_lm_clear_last_error()
  throw LiteRTLMError(call: call, status: status.rawValue, message: message)
}

enum RuntimeLog {
  /// verbose, debug, info, warning, error or fatal; nil keeps the runtime's default.
  static func setMinimumLevel(_ name: String?) {
    let levels: [String: LiteRtLmLogSeverity] = [
      "verbose": kLiteRtLmLogSeverityVerbose, "debug": kLiteRtLmLogSeverityDebug, "info": kLiteRtLmLogSeverityInfo,
      "warning": kLiteRtLmLogSeverityWarning, "error": kLiteRtLmLogSeverityError, "fatal": kLiteRtLmLogSeverityFatal,
    ]
    if let name, let level = levels[name] {
      _ = litert_lm_set_min_log_level(level)
    }
  }
}

/// What a `.litertlm` file's header says, read without building an engine.
struct FileInfo: Codable, Equatable {
  var modelType: String
  var readsPictures: Bool
  var maxContextTokens: Int?
  var minRuntimeVersion: String?

  static func read(path: String) throws -> FileInfo {
    var loaded: OpaquePointer?
    try check(litert_lm_loaded_file_create(path, &loaded), "litert_lm_loaded_file_create")
    defer { litert_lm_loaded_file_delete(loaded) }
    var type = kLiteRtLmModelTypeUnknown
    try check(litert_lm_loaded_file_model_type(loaded, &type), "litert_lm_loaded_file_model_type")
    var vision = false
    try check(
      litert_lm_loaded_file_supports_input_modality(loaded, kLiteRtLmModalityVision, &vision),
      "litert_lm_loaded_file_supports_input_modality")
    var context: UInt32 = 0
    let contextKnown = litert_lm_loaded_file_max_context_tokens(loaded, &context) == kLiteRtLmStatusOk
    var version: UnsafePointer<CChar>?
    let versionKnown = litert_lm_loaded_file_min_runtime_version(loaded, &version) == kLiteRtLmStatusOk
    litert_lm_clear_last_error()
    let typeName: String
    switch type {
    case kLiteRtLmModelTypeLlm: typeName = "llm"
    case kLiteRtLmModelTypeEmbedding: typeName = "embedding"
    default: typeName = "unknown"
    }
    return FileInfo(
      modelType: typeName, readsPictures: vision, maxContextTokens: contextKnown ? Int(context) : nil,
      minRuntimeVersion: versionKnown ? version.map { String(cString: $0) } : nil)
  }
}

/// One LiteRT-LM engine: the text decoder and the vision encoder, both on the Mac's GPU.
final class Engine {
  let handle: OpaquePointer

  /// The calls of the Python API's `Engine(path, backend=GPU, vision_backend=GPU, max_num_images=1,
  /// cache_dir=..., enable_benchmark=True)`, in its order.
  init(modelPath: String, cacheDir: String) throws {
    var settings: OpaquePointer?
    try check(
      litert_lm_engine_settings_create(modelPath, "gpu", "gpu", nil, &settings), "litert_lm_engine_settings_create")
    defer { litert_lm_engine_settings_delete(settings) }
    try check(litert_lm_engine_settings_enable_benchmark(settings), "litert_lm_engine_settings_enable_benchmark")
    try check(litert_lm_engine_settings_set_max_num_images(settings, 1), "litert_lm_engine_settings_set_max_num_images")
    try check(litert_lm_engine_settings_set_cache_dir(settings, cacheDir), "litert_lm_engine_settings_set_cache_dir")
    var engine: OpaquePointer?
    try check(litert_lm_engine_create(settings, &engine), "litert_lm_engine_create")
    guard let engine else { throw LiteRTLMError(call: "litert_lm_engine_create", status: 0, message: "No engine.") }
    handle = engine
  }

  deinit {
    litert_lm_engine_delete(handle)
  }
}

/// The runtime's counters for a conversation's last turn (the Python API's BenchmarkInfo).
struct TurnCounters: Codable, Equatable {
  var prefillTokens: Int?
  var prefillTokensPerSecond: Double?
  var decodeTokens: Int?
  var decodeTokensPerSecond: Double?
  /// BenchmarkInfo's time to first token.
  var timeToFirstToken: Double?
  var contextTokens: Int?
}

/// One conversation: the messages it starts with, then one reply. The file's own sampler, as the model card's script
/// runs it.
final class Conversation {
  enum Event {
    case text(String)
    case finished
    case failed(String)
  }

  let handle: OpaquePointer
  /// Every stream's callback state, kept until the conversation is deleted: a chunk that arrives after the stream's
  /// last event (after a cancel) must still find it.
  private var streams: [StreamBox] = []
  private let streamsLock = NSLock()

  /// The calls of the Python API's `engine.create_conversation(messages=...)`.
  init(engine: Engine, messagesJSON: String?) throws {
    var session: OpaquePointer?
    try check(litert_lm_session_config_create(&session), "litert_lm_session_config_create")
    defer { litert_lm_session_config_delete(session) }
    var config: OpaquePointer?
    try check(litert_lm_conversation_config_create(&config), "litert_lm_conversation_config_create")
    defer { litert_lm_conversation_config_delete(config) }
    try check(
      litert_lm_conversation_config_set_session_config(config, session), "litert_lm_conversation_config_set_session_config")
    if let messagesJSON {
      try check(litert_lm_conversation_config_set_messages(config, messagesJSON), "litert_lm_conversation_config_set_messages")
    }
    var conversation: OpaquePointer?
    try check(litert_lm_conversation_create(engine.handle, config, &conversation), "litert_lm_conversation_create")
    guard let conversation else {
      throw LiteRTLMError(call: "litert_lm_conversation_create", status: 0, message: "No conversation.")
    }
    handle = conversation
  }

  deinit {
    // Deleting waits for the runtime's work on this conversation, so no callback runs after it; then the stream
    // states go with this object.
    litert_lm_conversation_delete(handle)
  }

  /// Streams one reply (the Python API's `send_message_async(message, max_output_tokens=...)`): `onEvent` runs on a
  /// runtime thread, once per piece of text, then once with `.finished` or `.failed`. A reply can fail with
  /// DEADLINE_EXCEEDED when a wait inside LiteRT-LM passes its time limit: in 0.19.0 the WebGPU sampler waits at most
  /// 120 s for the GPU's result (runtime/components/top_k_webgpu_sampler.cc), counted in wall-clock time, also while
  /// the process is stopped. The engine cannot answer again after that, so the engine process then exits (EngineHost).
  func send(messageJSON: String, maxOutputTokens: Int, onEvent: @escaping (Event) -> Void) throws {
    let box = StreamBox(onEvent)
    streamsLock.lock()
    streams.append(box)
    streamsLock.unlock()
    var optional: OpaquePointer?
    try check(litert_lm_conversation_optional_args_create(&optional), "litert_lm_conversation_optional_args_create")
    defer { litert_lm_conversation_optional_args_delete(optional) }
    try check(
      litert_lm_conversation_optional_args_set_max_output_tokens(optional, Int32(maxOutputTokens)),
      "litert_lm_conversation_optional_args_set_max_output_tokens")
    // "{}" = no extra context, as the Python API sends it.
    let status = litert_lm_conversation_send_message_stream(
      handle, messageJSON, "{}", optional,
      { context, chunk in
        guard let context else { return }
        Unmanaged<StreamBox>.fromOpaque(context).takeUnretainedValue().handle(chunk)
      }, Unmanaged.passUnretained(box).toOpaque())
    try check(status, "litert_lm_conversation_send_message_stream")
  }

  /// Asks the runtime to stop the reply. The conversation must not be used afterwards, only deleted (C API header).
  func cancel() {
    _ = litert_lm_conversation_cancel_process(handle)
  }

  func counters() -> TurnCounters {
    var result = TurnCounters()
    var count = 0
    if litert_lm_conversation_get_token_count(handle, &count) == kLiteRtLmStatusOk { result.contextTokens = count }
    var info: OpaquePointer?
    guard litert_lm_conversation_get_benchmark_info(handle, &info) == kLiteRtLmStatusOk, let info else {
      litert_lm_clear_last_error()
      return result
    }
    defer { litert_lm_benchmark_info_delete(info) }
    var turns = 0
    if litert_lm_benchmark_info_get_num_prefill_turns(info, &turns) == kLiteRtLmStatusOk, turns > 0 {
      var tokens = 0
      var rate = 0.0
      if litert_lm_benchmark_info_get_prefill_token_count_at(info, turns - 1, &tokens) == kLiteRtLmStatusOk {
        result.prefillTokens = tokens
      }
      if litert_lm_benchmark_info_get_prefill_tokens_per_sec_at(info, turns - 1, &rate) == kLiteRtLmStatusOk {
        result.prefillTokensPerSecond = rate
      }
    }
    turns = 0
    if litert_lm_benchmark_info_get_num_decode_turns(info, &turns) == kLiteRtLmStatusOk, turns > 0 {
      var tokens = 0
      var rate = 0.0
      if litert_lm_benchmark_info_get_decode_token_count_at(info, turns - 1, &tokens) == kLiteRtLmStatusOk {
        result.decodeTokens = tokens
      }
      if litert_lm_benchmark_info_get_decode_tokens_per_sec_at(info, turns - 1, &rate) == kLiteRtLmStatusOk {
        result.decodeTokensPerSecond = rate
      }
    }
    var seconds = 0.0
    if litert_lm_benchmark_info_get_time_to_first_token(info, &seconds) == kLiteRtLmStatusOk {
      result.timeToFirstToken = seconds
    }
    litert_lm_clear_last_error()
    return result
  }
}

/// One stream's callback. It reports one last event (an error, or the final chunk) and ignores anything after it.
private final class StreamBox {
  /// Dropped after the reply's last event: the closure usually holds the conversation that holds this box, and that
  /// cycle would keep every conversation (and its runtime state) alive.
  private var onEvent: ((Conversation.Event) -> Void)?
  private let lock = NSLock()

  init(_ onEvent: @escaping (Conversation.Event) -> Void) {
    self.onEvent = onEvent
  }

  func handle(_ chunk: OpaquePointer?) {
    lock.lock()
    defer { lock.unlock() }
    guard let onEvent else { return }
    var error: UnsafePointer<CChar>?
    if litert_lm_stream_chunk_get_error(chunk, &error) == kLiteRtLmStatusOk, let error, error.pointee != 0 {
      self.onEvent = nil
      onEvent(.failed(String(cString: error)))
      return
    }
    var text: UnsafePointer<CChar>?
    if litert_lm_stream_chunk_get_text(chunk, &text) == kLiteRtLmStatusOk, let text {
      let piece = Self.text(ofMessageJSON: String(cString: text))
      if !piece.isEmpty { onEvent(.text(piece)) }
    }
    var final = false
    if litert_lm_stream_chunk_is_final(chunk, &final) == kLiteRtLmStatusOk, final {
      self.onEvent = nil
      onEvent(.finished)
    }
  }

  /// A chunk is a message in JSON ({"role": "assistant", "content": [{"type": "text", "text": ...}]}); text that is
  /// not JSON is taken as it is, as the Python API does.
  static func text(ofMessageJSON json: String) -> String {
    guard !json.isEmpty else { return "" }
    guard let data = json.data(using: .utf8),
      let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    else { return json }
    if let content = object["content"] as? String { return content }
    var text = ""
    for part in object["content"] as? [[String: Any]] ?? [] where (part["type"] as? String) == "text" {
      text += part["text"] as? String ?? ""
    }
    return text
  }
}

enum MessageJSON {
  static func encode(_ object: Any) -> String {
    let data = (try? JSONSerialization.data(withJSONObject: object, options: [.withoutEscapingSlashes])) ?? Data()
    return String(decoding: data, as: UTF8.self)
  }

  /// What a conversation starts with: the system prompt and, for the agent's second step, its first step as history
  /// (the agent's `_build_messages` with one picture kept: the first screenshot collapsed).
  static func preface(system: String, request: String, firstReply: String?) -> String {
    var messages: [[String: Any]] = [["role": "system", "content": system]]
    if let firstReply {
      messages.append(["role": "user", "content": [["type": "text", "text": Step.requestTurn(request)]]])
      messages.append(["role": "assistant", "content": [["type": "text", "text": firstReply]]])
    }
    return encode(messages)
  }

  /// The agent's first step: the screenshot, then its step text.
  static func firstStep(picturePath: String, request: String) -> String {
    encode(["role": "user", "content": [["type": "image", "path": picturePath], ["type": "text", "text": Step.stepText(request)]]])
  }

  /// The agent's second step: the screenshot as the answer to the call for one.
  static func secondStep(picturePath: String) -> String {
    encode([
      "role": "user",
      "content": [
        ["type": "text", "text": Step.resultBefore], ["type": "image", "path": picturePath],
        ["type": "text", "text": Step.resultAfter],
      ],
    ])
  }

  static func warmUp(picturePath: String) -> String {
    encode(["role": "user", "content": [["type": "image", "path": picturePath], ["type": "text", "text": "Hi"]]])
  }
}
