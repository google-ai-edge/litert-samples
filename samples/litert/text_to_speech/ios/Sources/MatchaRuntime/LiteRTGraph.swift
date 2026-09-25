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

/// The accelerator requested for a compiled graph.
public enum MatchaBackend: String, Codable, CaseIterable {
  case cpu
  case gpu
}

/// The optional Metal compute precision override.
public enum MatchaGPUPrecision: String, Codable, CaseIterable {
  case defaultPrecision = "default"
  case fp32
}

/// Independent accelerator and precision choices for one graph.
public struct MatchaGraphConfiguration: Codable, Equatable {
  public var backend: MatchaBackend
  public var precision: MatchaGPUPrecision

  /// Selects the requested accelerator and optional Metal precision override.
  public init(backend: MatchaBackend = .cpu, precision: MatchaGPUPrecision = .defaultPrecision) {
    self.backend = backend
    self.precision = precision
  }

  public static let cpu = MatchaGraphConfiguration(backend: .cpu)
  public static let gpu = MatchaGraphConfiguration(backend: .gpu)
}

/// Requested and effective execution information reported by the runtime.
public struct GraphBackendStatus: Codable {
  public let requested: MatchaBackend
  public let precision: MatchaGPUPrecision
  public let effective: String
  public let fullyAccelerated: Bool
  public let gpuEnvironmentAvailable: Bool
  public let inputBufferTypes: [Int32]
  public let outputBufferTypes: [Int32]
}

/// Host write, synchronous execution, and output readback durations.
public struct GraphTiming: Codable {
  public let writeSeconds: Double
  public let runSeconds: Double
  public let readSeconds: Double
  public var runAndReadSeconds: Double { runSeconds + readSeconds }
}

/// Invalid graph metadata, buffers, or runtime option payloads.
public enum LiteRTGraphError: Error {
  case missingOptionPayload
  case unsupportedTensor(String)
  case inputCount(expected: Int, actual: Int)
  case tensorSize(String, expected: Int, actual: Int)
}

/// Owns one compiled model and serializes access to reusable tensor buffers.
public final class LiteRTGraph {
  public let inputNames: [String]
  public let outputNames: [String]
  public let inputShapes: [[Int]]
  public let outputShapes: [[Int]]
  public let backendStatus: GraphBackendStatus
  public private(set) var lastTiming: GraphTiming?
  public private(set) var totalRunAndReadSeconds: Double = 0
  private let environment: Environment
  private let model: CompiledModel
  private var inputs: [TensorBuffer]
  private var outputs: [TensorBuffer]
  private let lock = NSLock()

  /// Compiles the graph and allocates reusable buffers in the shared environment.
  public init(
    modelPath: String, environment: Environment, configuration: MatchaGraphConfiguration = .cpu
  ) throws {
    self.environment = environment
    let options = try Options()
    // Request GPU without CPU fallback: unsupported graphs must fail instead of running partly on CPU.
    try options.setHardwareAccelerators(configuration.backend == .gpu ? [.gpu] : [.cpu])
    if configuration.backend == .gpu && configuration.precision == .fp32 {
      // precision = 2 is kLiteRtDelegatePrecisionFp32 in litert/c/litert_common.h,
      // parsed from gpu_options TOML in litert/c/options/litert_gpu_options.cc.
      guard let payload = strdup("precision = 2\n") else {
        throw LiteRTGraphError.missingOptionPayload
      }
      let opaque: OpaqueOptions
      do {
        opaque = try OpaqueOptions(
          identifier: "gpu_options", payload: payload,
          destructor: { free($0) })
      } catch {
        free(payload)
        throw error
      }
      // On success Options takes ownership. On failure opaque owns the payload.
      try options.addOpaqueOptions(opaque)
    }
    let compiled = try CompiledModel(
      filePath: modelPath, environment: environment, options: options)
    model = compiled
    let inputCount = try compiled.inputCount()
    let outputCount = try compiled.outputCount()
    inputNames = try (0..<inputCount).map { try compiled.inputName(inputIndex: $0) }
    outputNames = try (0..<outputCount).map { try compiled.outputName(outputIndex: $0) }
    let inputTypes = try (0..<inputCount).map { try compiled.inputTensorType(inputIndex: $0) }
    let outputTypes = try (0..<outputCount).map { try compiled.outputTensorType(outputIndex: $0) }
    for (name, type) in zip(inputNames + outputNames, inputTypes + outputTypes) {
      guard type.elementType == .float32 && type.layout.elementCount != nil else {
        throw LiteRTGraphError.unsupportedTensor(name)
      }
    }
    inputShapes = inputTypes.map { $0.layout.dimensions }
    outputShapes = outputTypes.map { $0.layout.dimensions }
    inputs = try model.createInputBuffers()
    outputs = try model.createOutputBuffers()
    let fullyAccelerated = try compiled.isFullyAccelerated()
    let gpuAvailable = environment.hasGpuEnvironment()
    let effective: String
    if configuration.backend == .cpu {
      effective = "CPU"
    } else if !gpuAvailable {
      effective = "CPU (fallback detected)"
    } else if fullyAccelerated {
      effective = "Metal"
    } else {
      effective = "CPU or partial Metal (fallback detected)"
    }
    backendStatus = GraphBackendStatus(
      requested: configuration.backend, precision: configuration.precision,
      effective: effective, fullyAccelerated: fullyAccelerated,
      gpuEnvironmentAvailable: gpuAvailable,
      inputBufferTypes: inputs.map { $0.type.rawValue },
      outputBufferTypes: outputs.map { $0.type.rawValue })
  }

  deinit {
    // TensorBuffer does not retain the environment that allocated it.
    withExtendedLifetime((model, environment)) {
      inputs.removeAll()
      outputs.removeAll()
    }
  }

  /// Writes inputs, executes the compiled graph, and reads every output.
  public func run(_ values: [[Float]]) throws -> [[Float]] {
    lock.lock()
    defer { lock.unlock() }
    guard values.count == inputs.count else {
      throw LiteRTGraphError.inputCount(expected: inputs.count, actual: values.count)
    }
    let start = DispatchTime.now().uptimeNanoseconds
    for index in values.indices {
      let expected = inputShapes[index].reduce(1, *)
      guard values[index].count == expected else {
        throw LiteRTGraphError.tensorSize(
          inputNames[index], expected: expected, actual: values[index].count)
      }
      try inputs[index].write(values[index])
    }
    let wrote = DispatchTime.now().uptimeNanoseconds
    try model.run(inputs: inputs, outputs: outputs)
    let ran = DispatchTime.now().uptimeNanoseconds
    var results = [[Float]]()
    for index in outputs.indices {
      let output: [Float] = try outputs[index].read()
      let expected = outputShapes[index].reduce(1, *)
      guard output.count == expected else {
        throw LiteRTGraphError.tensorSize(
          outputNames[index], expected: expected, actual: output.count)
      }
      results.append(output)
    }
    let read = DispatchTime.now().uptimeNanoseconds
    lastTiming = GraphTiming(
      writeSeconds: Double(wrote - start) / 1e9,
      runSeconds: Double(ran - wrote) / 1e9,
      readSeconds: Double(read - ran) / 1e9)
    totalRunAndReadSeconds += Double(read - wrote) / 1e9
    return results
  }
}
