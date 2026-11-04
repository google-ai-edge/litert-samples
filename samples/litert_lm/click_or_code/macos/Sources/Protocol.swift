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

// The window and the engine process talk in JSON lines: commands on the engine's stdin, events on its stdout. The
// engine's stderr is its log (LiteRT-LM's own messages plus `[engine]` lines).

struct EngineCommand: Codable {
  enum Kind: String, Codable {
    /// One inference: a new conversation holding the system prompt and, for the second step, the first reply.
    case step
    case cancel
    case quit
  }

  var cmd: Kind
  var id: Int?
  /// The 1920 x 1088 PNG the window made from the screenshot.
  var picture: String?
  var request: String?
  /// The first step's reply (a call for a screenshot): this inference is the agent's second step.
  var firstReply: String?
  var system: String?
  var maxOutputTokens: Int?
}

struct TurnStats: Codable, Equatable {
  /// Creating the conversation (its system prompt and history are prefilled when the message goes out).
  var conversationSeconds: Double?
  /// Send to the first piece of text, measured in the engine process.
  var firstTextSeconds: Double?
  /// Send to the end of the reply.
  var replySeconds: Double?
  var chunks = 0
  /// LiteRT-LM's own counters for this inference.
  var counters: TurnCounters?
  /// The engine process's memory footprint (Activity Monitor's Memory column): now, and the most it has used.
  var footprintBytes: UInt64?
  var peakFootprintBytes: UInt64?
}

struct EngineEvent: Codable {
  enum Kind: String, Codable {
    case fileInfo = "file_info"
    case building
    case warming
    case ready
    case chunk
    case turnDone = "turn_done"
    case error
  }

  enum Stage: String, Codable {
    case file, engine, generate
  }

  var event: Kind
  var id: Int?
  var text: String?
  var stage: Stage?
  var message: String?
  var file: FileInfo?
  var buildSeconds: Double?
  var warmUpSeconds: Double?
  var peakFootprintBytes: UInt64?
  var stats: TurnStats?
  var cancelled: Bool?
  /// The reply stopped at the output-token limit or at the end of the context, not at the model's own stop token.
  var limitReached: Bool?
  /// For a chunk: seconds since the engine got the command, on the engine's clock.
  var t: Double?
}

/// Exit codes of the engine process, so the window can say what went wrong after it has gone.
enum EngineExit {
  static let badFile: Int32 = 3
  static let engineFailed: Int32 = 4
  /// A reply failed with DEADLINE_EXCEEDED (a wait in LiteRT-LM passed its time limit); the engine cannot answer
  /// again after that.
  static let deadline: Int32 = 5

  static func isDeadline(_ message: String) -> Bool { message.contains("DEADLINE_EXCEEDED") }
}
