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

/// One serial owner for the shared environment, frontend, and reused model buffers.
final class SpeechWorker {
  let assets: URL
  let queue = DispatchQueue(label: "speech.synthesis", qos: .userInitiated)
  private let runtimeLibraryDirectory: String?
  private var environment: Environment?
  private var activeConfiguration: AppRunConfiguration?
  private var frontend: MatchaFrontend?
  private var g2p: MatchaG2PRunner?
  private var synthesizer: MatchaSynthesizer?

  /// Selects assets and optionally an explicit plugin directory for command-line hosts.
  init(assets: URL, runtimeLibraryDirectory: String? = nil) {
    self.assets = assets
    self.runtimeLibraryDirectory = runtimeLibraryDirectory
  }

  static let requiredFiles = [
    "dp_g2p_matcha_fp16.tflite", "matcha_textenc_fp16.tflite", "matcha_decoder_fp16.tflite",
    "matcha_vocoder_fp16.tflite", "emb.bin", "config.json", "g2p_meta.json", "g2p_dict.txt",
  ]
  var missingResources: [String] {
    Self.requiredFiles.filter {
      !FileManager.default.fileExists(atPath: assets.appendingPathComponent($0).path)
    }
  }

  /// Symbol metadata stored alongside the embedding table.
  private struct Configuration: Decodable { let symbols: [String] }

  /// Loads or reuses the requested graph set on the worker queue.
  func prepare(_ configuration: AppRunConfiguration) throws {
    dispatchPrecondition(condition: .onQueue(queue))
    if activeConfiguration == configuration { return }
    let missing = missingResources
    guard missing.isEmpty else { throw AppSpeechError.missingResources(missing) }
    if environment == nil {
      let options: [Environment.Option] =
        runtimeLibraryDirectory.map { [.runtimeLibraryDir($0)] } ?? []
      environment = try Environment(options: options)
    }
    guard let environment else { throw AppSpeechError.notPrepared }
    if frontend == nil {
      let decoder = JSONDecoder()
      let meta = try decoder.decode(
        G2PMetadata.self, from: Data(contentsOf: assets.appendingPathComponent("g2p_meta.json")))
      let config = try decoder.decode(
        Configuration.self, from: Data(contentsOf: assets.appendingPathComponent("config.json")))
      let dictionary = try MatchaFrontend.readDictionary(
        at: assets.appendingPathComponent("g2p_dict.txt"))
      let newG2P = try MatchaG2PRunner(
        modelPath: assets.appendingPathComponent("dp_g2p_matcha_fp16.tflite").path, metadata: meta,
        environment: environment)
      frontend = try MatchaFrontend(
        dictionary: dictionary, symbols: config.symbols, phonemizer: newG2P)
      g2p = newG2P
    }
    synthesizer = try MatchaSynthesizer(
      assets: assets, environment: environment, placement: configuration.placement)
    activeConfiguration = configuration
  }

  /// Synchronous result used by both interactive and command-line synthesis.
  struct Measurement {
    let synthesis: MatchaSynthesis
    let phonemeCount: Int
    let statuses: [String: GraphBackendStatus]
    let milliseconds: [String: Double]
    let totalMilliseconds: Double
  }

  /// Measures text processing, model execution and readback after model preparation.
  func measure(text: String, seed: UInt64, steps: Int) throws -> Measurement {
    dispatchPrecondition(condition: .onQueue(queue))
    guard let frontend, let g2p, let synthesizer else { throw AppSpeechError.notPrepared }
    let start = DispatchTime.now().uptimeNanoseconds
    let beforeG2P = g2p.totalRunAndReadSeconds
    let phonemes = try frontend.phonemize(text)
    let g2pSeconds = g2p.totalRunAndReadSeconds - beforeG2P
    let synthesis = try synthesizer.synthesize(ids: phonemes.ids, steps: steps, seed: seed)
    let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
    var statuses = synthesis.graphStatuses
    statuses["g2p"] = g2p.backendStatus
    let times: [String: Double] = [
      "g2p": g2pSeconds * 1000,
      "text_encoder": (synthesis.graphTimings["text_encoder"]?.runAndReadSeconds ?? 0) * 1000,
      "decoder": synthesis.graphTimings.filter { $0.key.hasPrefix("step_") }.values.reduce(0) {
        $0 + $1.runAndReadSeconds
      } * 1000,
      "vocoder": (synthesis.graphTimings["vocoder"]?.runAndReadSeconds ?? 0) * 1000,
    ]
    return Measurement(
      synthesis: synthesis, phonemeCount: phonemes.ids.count, statuses: statuses,
      milliseconds: times, totalMilliseconds: elapsed)
  }

  /// Enqueues interactive synthesis and returns a displayable result.
  func speak(
    text: String, seed: UInt64, steps: Int, configuration: AppRunConfiguration,
    completion: @escaping (Result<AppSpeechResult, Error>) -> Void
  ) {
    queue.async {
      do {
        try self.prepare(configuration)
        let measurement = try self.measure(text: text, seed: seed, steps: steps)
        let rows = [
          ("g2p", "Pronunciation"), ("text_encoder", "Text encoder"), ("decoder", "Decoder"),
          ("vocoder", "Vocoder"),
        ].compactMap { key, name -> AppGraphRow? in
          guard let status = measurement.statuses[key] else { return nil }
          let requested = status.requested == .cpu ? "CPU" : "Metal · " + status.precision.rawValue
          return AppGraphRow(
            id: key, name: name, requested: requested, effective: status.effective,
            milliseconds: measurement.milliseconds[key] ?? 0)
        }
        completion(
          .success(
            AppSpeechResult(
              audio: measurement.synthesis.audio, rows: rows,
              totalMilliseconds: measurement.totalMilliseconds,
              audioSeconds: Double(measurement.synthesis.audio.count)
                / Double(MatchaMath.sampleRate))))
      } catch { completion(.failure(error)) }
    }
  }
}
