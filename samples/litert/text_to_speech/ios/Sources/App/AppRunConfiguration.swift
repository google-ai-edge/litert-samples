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

/// Acoustic graph placement; pronunciation and the flow decoder remain on CPU.
struct AppRunConfiguration: Codable, Equatable {
  var textEncoder: MatchaGraphConfiguration = .cpu
  var vocoder: MatchaGraphConfiguration = .gpu

  var placement: MatchaPlacement {
    MatchaPlacement(textEncoder: textEncoder, decoder: .cpu, vocoder: vocoder)
  }

  static var cpuOnly: AppRunConfiguration {
    var value = AppRunConfiguration()
    value.vocoder = .cpu
    return value
  }

  static var highPrecision: AppRunConfiguration {
    var value = AppRunConfiguration()
    value.textEncoder = MatchaGraphConfiguration(backend: .gpu, precision: .fp32)
    value.vocoder = MatchaGraphConfiguration(backend: .gpu, precision: .fp32)
    return value
  }

  /// Parses an explicit, complete te/dec/voc placement without implicit fallbacks.
  static func parse(_ specification: String) throws -> AppRunConfiguration {
    var fields = [String: MatchaGraphConfiguration]()
    for field in specification.split(separator: ",", omittingEmptySubsequences: false) {
      let pair = field.split(separator: "=", omittingEmptySubsequences: false)
      guard pair.count == 2, ["te", "dec", "voc"].contains(String(pair[0])),
        fields[String(pair[0])] == nil
      else {
        throw AppSpeechError.invalidPlacement(specification)
      }
      let value: MatchaGraphConfiguration
      switch pair[1] {
      case "cpu": value = .cpu
      case "gpu", "gpu:default": value = .gpu
      case "gpu:fp32": value = MatchaGraphConfiguration(backend: .gpu, precision: .fp32)
      default: throw AppSpeechError.invalidPlacement(specification)
      }
      fields[String(pair[0])] = value
    }
    guard fields.count == 3, let encoder = fields["te"], let decoder = fields["dec"],
      let vocoder = fields["voc"], decoder == .cpu
    else {
      throw AppSpeechError.invalidPlacement(specification)
    }
    return AppRunConfiguration(textEncoder: encoder, vocoder: vocoder)
  }
}

/// Recoverable application setup, argument, and synthesis failures.
enum AppSpeechError: LocalizedError {
  case missingResources([String])
  case invalidSeed
  case invalidPlacement(String)
  case invalidArgument(String)
  case notPrepared
  case unavailableAudioBuffer
  case nonfiniteAudio

  var errorDescription: String? {
    switch self {
    case .missingResources(let files):
      return "Models missing — run prep_resources.sh, then rebuild. Missing: "
        + files.joined(separator: ", ")
    case .invalidSeed: return "Enter a whole seed from 0 through 18446744073709551615."
    case .invalidPlacement(let value):
      return "Invalid placement: \(value). Specify te, dec and voc; dec must be cpu."
    case .invalidArgument(let name): return "Missing or invalid argument: " + name
    case .notPrepared: return "The speech engine is not ready."
    case .unavailableAudioBuffer: return "The audio buffer could not be created."
    case .nonfiniteAudio: return "Synthesis produced invalid audio."
    }
  }
}

/// A graph's requested and observed backend with synchronous readback timing.
struct AppGraphRow: Identifiable {
  let id: String
  let name: String
  let requested: String
  let effective: String
  let milliseconds: Double
}

/// One completed synthesis and the information presented by the interface.
struct AppSpeechResult {
  let audio: [Float]
  let rows: [AppGraphRow]
  let totalMilliseconds: Double
  let audioSeconds: Double
  var realTimeFactor: Double { totalMilliseconds / 1000 / audioSeconds }
}
