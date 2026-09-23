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
import LiteRT

/// An unexpected signature in the neural pronunciation graph.
public enum MatchaG2PError: Error {
  case unexpectedBufferCount(inputs: Int, outputs: Int)
}

// The caller serializes access because the model reuses its tensor buffers.
/// Runs dictionary-missing pronunciation requests through a compiled CPU graph.
public final class MatchaG2PRunner: WordPhonemizer {
  public let codec: G2PCodec
  private let graph: LiteRTGraph
  public var backendStatus: GraphBackendStatus { graph.backendStatus }
  public var lastTiming: GraphTiming? { graph.lastTiming }
  public var totalRunAndReadSeconds: Double { graph.totalRunAndReadSeconds }

  /// Loads and checks the pronunciation graph signature.
  public init(
    modelPath: String, metadata: G2PMetadata, environment: Environment,
    configuration: MatchaGraphConfiguration = .cpu
  ) throws {
    codec = G2PCodec(metadata: metadata)
    graph = try LiteRTGraph(
      modelPath: modelPath, environment: environment, configuration: configuration)
    guard
      graph.inputShapes == [[1, metadata.maxTokens]]
        && graph.outputShapes == [[1, metadata.maxTokens, metadata.phonemeCount]]
    else {
      throw MatchaG2PError.unexpectedBufferCount(
        inputs: graph.inputShapes.count, outputs: graph.outputShapes.count)
    }
  }

  /// Executes the fixed-size pronunciation graph and returns token-major logits.
  public func logits(for input: G2PInput) throws -> [Float] {
    guard input.values.count == codec.metadata.maxTokens else {
      throw G2PCodecError.invalidLength(input.values.count)
    }
    let logits = try graph.run([input.values])[0]
    guard logits.count == codec.metadata.maxTokens * codec.metadata.phonemeCount else {
      throw G2PCodecError.invalidLogitCount(
        expected: codec.metadata.maxTokens * codec.metadata.phonemeCount, actual: logits.count)
    }
    return logits
  }

  /// Returns the pronunciation of a single dictionary-missing word.
  public func phonemizeWord(_ word: String) throws -> String {
    let input = codec.input(for: word)
    return try codec.decode(logits(for: input), length: input.length)
  }

}
