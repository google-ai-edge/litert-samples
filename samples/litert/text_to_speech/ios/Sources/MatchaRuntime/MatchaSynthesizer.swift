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

// The package exposes MatchaCore; the app compiles these sources into one module.
#if canImport(MatchaCore)
  import MatchaCore
#endif

/// Invalid synthesis configuration or nonfinite intermediate tensors.
public enum MatchaSynthesizerError: Error, LocalizedError {
  case unsupportedSignature(String)
  case invalidStepCount(Int)
  case nonfiniteTensor(stage: String, index: Int)

  public var errorDescription: String? {
    switch self {
    case .unsupportedSignature(let reason): return reason
    case .invalidStepCount(let count): return "The step count must be positive (received \(count))."
    case .nonfiniteTensor(let stage, let index):
      return "Synthesis stopped: \(stage) contains a non-finite value at index \(index)."
    }
  }
}

/// Independent accelerator configurations for the three acoustic graphs.
public struct MatchaPlacement: Codable {
  public var textEncoder: MatchaGraphConfiguration
  public var decoder: MatchaGraphConfiguration
  public var vocoder: MatchaGraphConfiguration

  /// Selects the accelerator and precision independently for each acoustic graph.
  public init(
    textEncoder: MatchaGraphConfiguration = .cpu, decoder: MatchaGraphConfiguration = .cpu,
    vocoder: MatchaGraphConfiguration = .gpu
  ) {
    self.textEncoder = textEncoder
    self.decoder = decoder
    self.vocoder = vocoder
  }

  public static let cpuOnly = MatchaPlacement(vocoder: .cpu)
}

/// Generated audio, acoustic controls, optional tensors, and graph measurements.
public struct MatchaSynthesis {
  public let audio: [Float]
  public let textLength: Int
  public let paddedIDs: [Int32]
  public let durations: [Float]
  public let melLength: Int
  public let steps: Int
  public let dt: Float
  public let times: [Float]
  public let tensors: [String: [Float]]
  public let graphTimings: [String: GraphTiming]
  public let graphStatuses: [String: GraphBackendStatus]
}

/// Owns reused graph buffers. Calls to synthesize are serialized for their full duration.
public final class MatchaSynthesizer {
  private let textEncoder: LiteRTGraph
  private let decoder: LiteRTGraph
  private let vocoder: LiteRTGraph
  private let embedding: [Float]
  private let muIndex: Int
  private let logwIndex: Int
  private let lock = NSLock()
  public let textEncoderOutputNames: [String]
  public let textEncoderOutputShapes: [[Int]]
  public let muOutputName: String
  public let logwOutputName: String

  /// Pass the same environment that the G2P runner uses.
  public init(assets: URL, environment: Environment, placement: MatchaPlacement = MatchaPlacement())
    throws
  {
    embedding = try MatchaMath.readLittleEndianFloats(
      Data(contentsOf: assets.appendingPathComponent("emb.bin")))
    let encoderGraph = try LiteRTGraph(
      modelPath: assets.appendingPathComponent("matcha_textenc_fp16.tflite").path,
      environment: environment, configuration: placement.textEncoder)
    let decoderGraph = try LiteRTGraph(
      modelPath: assets.appendingPathComponent("matcha_decoder_fp16.tflite").path,
      environment: environment, configuration: placement.decoder)
    let vocoderGraph = try LiteRTGraph(
      modelPath: assets.appendingPathComponent("matcha_vocoder_fp16.tflite").path,
      environment: environment, configuration: placement.vocoder)
    let muIndices = encoderGraph.outputShapes.indices.filter {
      encoderGraph.outputShapes[$0] == [1, 80, 256]
    }
    let logwIndices = encoderGraph.outputShapes.indices.filter {
      encoderGraph.outputShapes[$0] == [1, 1, 256]
    }
    guard muIndices.count == 1 && logwIndices.count == 1 && encoderGraph.outputShapes.count == 2,
      encoderGraph.inputShapes == [[1, 256, 192], [1, 1, 256]],
      decoderGraph.inputShapes == [[1, 80, 512], [1, 80, 512], [1, 160], [1, 1, 512]],
      decoderGraph.outputShapes == [[1, 80, 512]],
      vocoderGraph.inputShapes == [[1, 80, 512]],
      vocoderGraph.outputShapes == [[1, 1, 131072]]
    else {
      throw MatchaSynthesizerError.unsupportedSignature("Unexpected acoustic graph shapes")
    }
    textEncoder = encoderGraph
    decoder = decoderGraph
    vocoder = vocoderGraph
    muIndex = muIndices[0]
    logwIndex = logwIndices[0]
    textEncoderOutputNames = encoderGraph.outputNames
    textEncoderOutputShapes = encoderGraph.outputShapes
    muOutputName = encoderGraph.outputNames[muIndices[0]]
    logwOutputName = encoderGraph.outputNames[logwIndices[0]]
  }

  /// Executes text encoding, length regulation, Euler integration, and vocoding.
  public func synthesize(
    ids: [Int32], steps: Int = MatchaMath.defaultSteps, seed: UInt64 = 0,
    noise: [Float]? = nil, captureTensors: Bool = false
  ) throws -> MatchaSynthesis {
    lock.lock()
    defer { lock.unlock() }
    guard steps > 0 else {
      throw MatchaSynthesizerError.invalidStepCount(steps)
    }
    var tensors = [String: [Float]]()
    var timings = [String: GraphTiming]()
    let text = try MatchaMath.prepareText(ids: ids, embedding: embedding)
    let encoded = try textEncoder.run([text.embedded, text.mask])
    timings["text_encoder"] = textEncoder.lastTiming
    let mu = encoded[muIndex]
    let logw = encoded[logwIndex]
    try Self.requireFinite(mu, stage: "text encoder mu")
    try Self.requireFinite(logw, stage: "text encoder logw")
    let regulated = try MatchaMath.regulate(mu: mu, logw: logw, textMask: text.mask)
    try Self.requireFinite(regulated.durations, stage: "durations")
    try Self.requireFinite(regulated.cumulative, stage: "cumulative durations")
    var random = GaussianNoise(seed: seed)
    var x =
      try noise.map { try MatchaMath.maskNoise($0, melLength: regulated.melLength) }
      ?? random.tensor(melLength: regulated.melLength)
    try Self.requireFinite(x, stage: "initial noise")
    if captureTensors {
      tensors["embedded"] = text.embedded
      tensors["text_mask"] = text.mask
      tensors["mu"] = mu
      tensors["logw"] = logw
      tensors["durations"] = regulated.durations
      tensors["cumulative"] = regulated.cumulative
      tensors["mu_y"] = regulated.muY
      tensors["mel_mask"] = regulated.mask
      tensors["x0"] = x
    }
    let dt: Float = 1 / Float(steps)
    var time: Float = 0
    var times = [Float]()
    for step in 0..<steps {
      let prefix = String(format: "step_%02d", step)
      let timeEmbedding = MatchaMath.timeEmbedding(time)
      let velocity = try decoder.run([x, regulated.muY, timeEmbedding, regulated.mask])[0]
      timings[prefix] = decoder.lastTiming
      try Self.requireFinite(velocity, stage: "decoder velocity at step \(step)")
      try MatchaMath.eulerUpdate(x: &x, velocity: velocity, dt: dt)
      try Self.requireFinite(x, stage: "Euler output at step \(step)")
      if captureTensors {
        tensors[prefix + "_time"] = timeEmbedding
        tensors[prefix + "_velocity"] = velocity
        tensors[prefix + "_x"] = x
      }
      times.append(time)
      time = time + dt
    }
    let mel = try MatchaMath.denormalize(x, melLength: regulated.melLength)
    try Self.requireFinite(mel, stage: "mel")
    let fullWaveform = try vocoder.run([mel])[0]
    try Self.requireFinite(fullWaveform, stage: "vocoder waveform")
    timings["vocoder"] = vocoder.lastTiming
    let waveform = try MatchaMath.crop(fullWaveform, melLength: regulated.melLength)
    let audio = MatchaMath.clip(waveform)
    if captureTensors {
      tensors["mel"] = mel
      tensors["vocoder_full"] = fullWaveform
      tensors["waveform"] = waveform
      tensors["audio"] = audio
    }
    return MatchaSynthesis(
      audio: audio, textLength: text.textLength, paddedIDs: text.paddedIDs,
      durations: regulated.durations, melLength: regulated.melLength,
      steps: steps, dt: dt, times: times, tensors: tensors, graphTimings: timings,
      graphStatuses: [
        "text_encoder": textEncoder.backendStatus,
        "decoder": decoder.backendStatus, "vocoder": vocoder.backendStatus,
      ])
  }

  /// Rejects invalid values before they can affect another synthesis stage.
  private static func requireFinite(_ values: [Float], stage: String) throws {
    if let index = values.firstIndex(where: { !$0.isFinite }) {
      throw MatchaSynthesizerError.nonfiniteTensor(stage: stage, index: index)
    }
  }
}
