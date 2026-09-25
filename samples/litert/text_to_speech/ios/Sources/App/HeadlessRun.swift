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

import Darwin
import Foundation

/// Validated launch arguments for repeatable synthesis without interactive controls.
struct HeadlessRequest {
  let text: String
  let seed: UInt64
  let steps: Int
  let runs: Int
  let configuration: AppRunConfiguration

  /// Returns nil for an ordinary interactive launch.
  static func parse(_ arguments: [String]) throws -> HeadlessRequest? {
    guard let index = arguments.firstIndex(of: "-speak") else { return nil }
    guard arguments.indices.contains(index + 1) else {
      throw AppSpeechError.invalidArgument("-speak")
    }
    var values = [String: String]()
    var offset = index
    while offset < arguments.count {
      let key = arguments[offset]
      guard ["-speak", "-seed", "-steps", "-runs", "-placement"].contains(key),
        arguments.indices.contains(offset + 1), values[key] == nil
      else {
        throw AppSpeechError.invalidArgument(key)
      }
      values[key] = arguments[offset + 1]
      offset += 2
    }
    guard let seed = UInt64(values["-seed"] ?? "0"), let steps = Int(values["-steps"] ?? "10"),
      steps > 0, let runs = Int(values["-runs"] ?? "1"), runs > 0
    else {
      throw AppSpeechError.invalidArgument("seed, steps or runs")
    }
    let configuration = try AppRunConfiguration.parse(
      values["-placement"] ?? "te=cpu,dec=cpu,voc=gpu")
    return HeadlessRequest(
      text: arguments[index + 1], seed: seed, steps: steps, runs: runs, configuration: configuration
    )
  }
}

/// Timing statistics in milliseconds for a set of synchronized measurements.
struct TimingSummary: Codable {
  let median: Double
  let min: Double
  let max: Double

  /// Accepts a nonempty collection of elapsed times.
  init(_ values: [Double]) throws {
    guard !values.isEmpty else { throw AppSpeechError.invalidArgument("runs") }
    let sorted = values.sorted()
    let middle = sorted.count / 2
    median =
      sorted.count.isMultiple(of: 2) ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle]
    min = sorted[0]
    max = sorted[sorted.count - 1]
  }
}

/// Serializes audio and system information shared by command-line reports.
enum SpeechReport {
  /// Returns the hardware identifier, including the selected simulated model.
  static func modelIdentifier() -> String {
    if let simulated = ProcessInfo.processInfo.environment["SIMULATOR_MODEL_IDENTIFIER"] {
      return simulated
    }
    var system = utsname()
    uname(&system)
    return withUnsafeBytes(of: &system.machine) {
      String(decoding: $0.prefix(while: { $0 != 0 }), as: UTF8.self)
    }
  }

  /// Gives a stable label to the platform's current thermal state.
  static func thermalState() -> String {
    switch ProcessInfo.processInfo.thermalState {
    case .nominal: return "nominal"
    case .fair: return "fair"
    case .serious: return "serious"
    case .critical: return "critical"
    @unknown default: return "unknown"
    }
  }

  /// Writes mono IEEE Float32 WAV samples with a fixed synthesis sample rate.
  static func writeWave(_ samples: [Float], to url: URL) throws {
    guard samples.allSatisfy(\.isFinite), samples.count <= Int(UInt32.max - 36) / 4 else {
      throw AppSpeechError.nonfiniteAudio
    }
    var data = Data()
    func text(_ value: String) { data.append(contentsOf: value.utf8) }
    func integer<T: FixedWidthInteger>(_ value: T) {
      var little = value.littleEndian
      withUnsafeBytes(of: &little) { data.append(contentsOf: $0) }
    }
    text("RIFF")
    integer(UInt32(36 + samples.count * 4))
    text("WAVEfmt ")
    integer(UInt32(16))
    integer(UInt16(3))
    integer(UInt16(1))
    integer(UInt32(MatchaMath.sampleRate))
    integer(UInt32(MatchaMath.sampleRate * 4))
    integer(UInt16(4))
    integer(UInt16(32))
    text("data")
    integer(UInt32(samples.count * 4))
    for sample in samples { integer(sample.bitPattern) }
    try data.write(to: url, options: .atomic)
  }

  /// Converts a typed report fragment into JSON-compatible values.
  static func json<T: Encodable>(_ value: T) throws -> Any {
    try JSONSerialization.jsonObject(with: JSONEncoder().encode(value))
  }
}

extension SpeechWorker {
  /// Performs two warm-ups, measured runs, and writes the final audio and report.
  func runHeadless(
    _ request: HeadlessRequest, documents: URL, completion: @escaping (Result<URL, Error>) -> Void
  ) {
    queue.async {
      do {
        try FileManager.default.createDirectory(at: documents, withIntermediateDirectories: true)
        let thermalBefore = SpeechReport.thermalState()
        try self.prepare(request.configuration)
        for _ in 0..<2 {
          _ = try self.measure(text: request.text, seed: request.seed, steps: request.steps)
        }
        var measurements = [Measurement]()
        for _ in 0..<request.runs {
          measurements.append(
            try self.measure(text: request.text, seed: request.seed, steps: request.steps))
        }
        guard let last = measurements.last else { throw AppSpeechError.invalidArgument("runs") }
        var graphs = [String: Any]()
        for (name, status) in last.statuses {
          graphs[name] = [
            "backend": try SpeechReport.json(status),
            "milliseconds": try SpeechReport.json(
              TimingSummary(measurements.map { $0.milliseconds[name] ?? 0 })),
          ]
        }
        let total = try TimingSummary(measurements.map(\.totalMilliseconds))
        let audioSeconds = Double(last.synthesis.audio.count) / Double(MatchaMath.sampleRate)
        let report: [String: Any] = [
          "graphs": graphs, "total_ms": try SpeechReport.json(total), "audio_seconds": audioSeconds,
          "rtf": total.median / 1000 / audioSeconds, "ylen": last.synthesis.melLength,
          "phoneme_count": last.phonemeCount, "sample_count": last.synthesis.audio.count,
          "sample_rate_hz": MatchaMath.sampleRate,
          "seed": String(request.seed), "steps": request.steps, "runs": request.runs, "warmups": 2,
          "thermal_before": thermalBefore, "thermal_after": SpeechReport.thermalState(),
          "device_model": SpeechReport.modelIdentifier(),
          "os_version": ProcessInfo.processInfo.operatingSystemVersionString,
          "build_configuration": Bundle.main.object(
            forInfoDictionaryKey: "MatchaBuildConfiguration") as? String ?? "unknown",
          "litert_commit": Bundle.main.object(forInfoDictionaryKey: "MatchaLiteRTCommit") as? String
            ?? "unknown",
        ]
        try SpeechReport.writeWave(
          last.synthesis.audio, to: documents.appendingPathComponent("out.wav"))
        let result = documents.appendingPathComponent("result.json")
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
          .write(to: result, options: .atomic)
        completion(.success(result))
      } catch { completion(.failure(error)) }
    }
  }
}
