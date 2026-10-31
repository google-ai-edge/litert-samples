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
    /// A new conversation about a picture: the picture and the first question.
    case ask
    /// A follow-up question in the current conversation.
    case followUp = "follow_up"
    case cancel
    case quit
  }

  var cmd: Kind
  var id: Int?
  var text: String?
  /// The 448 x 448 PNG the window made from the picture (`ask` only).
  var picture: String?
  /// The system message (`ask` only), e.g. "Reasoning strength: low." for Muse-Glimmer-30B.
  var system: String?
  var maxOutputTokens: Int?
}

/// How a model writes its replies, read from its chat template when the engine starts.
enum ReplyFormat: String, Codable {
  /// `<|start|>assistant to=self<|message|>reasoning<|eom|><|start|>assistant to=user<|message|>answer`
  /// (Muse-Glimmer-30B): a reasoning message, then the answer.
  case reasoningMessages = "reasoning_messages"
  /// ChatML (Qwen2.5-VL): the reply is the answer.
  case plain
}

struct TurnStats: Codable, Equatable {
  /// Send to the first piece of text, measured in the engine process.
  var firstTextSeconds: Double?
  /// Send to the first piece of the answer (after a reasoning message, for Muse-Glimmer-30B).
  var firstAnswerSeconds: Double?
  /// Send to the end of the reply.
  var replySeconds: Double?
  var chunks = 0
  /// LiteRT-LM's own counters for this turn.
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
  var format: ReplyFormat?
  var buildSeconds: Double?
  var warmUpSeconds: Double?
  var peakFootprintBytes: UInt64?
  var stats: TurnStats?
  var cancelled: Bool?
  /// The reply stopped at the output-token limit or at the end of the context, not at the model's own stop token.
  var limitReached: Bool?
  /// For a chunk: seconds since the engine received the question, on the engine's clock.
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
