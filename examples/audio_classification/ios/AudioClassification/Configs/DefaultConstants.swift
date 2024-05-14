// Copyright 2022 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import Foundation

enum Model: String, CaseIterable {
  case Yamnet = "YAMNet"
  case speechCommand = "Speech Command"

  var modelPath: String? {
      switch self {
      case .Yamnet:
          return Bundle.main.path(
              forResource: "yamnet", ofType: "tflite")
      case .speechCommand:
          return Bundle.main.path(
              forResource: "speech_commands", ofType: "tflite")
      }
  }
}


struct DefaultConstants {
  static var model: Model = .Yamnet
  static var overLap: Double = 0.5
  static var maxResults: Int = 3
  static var threshold: Float = 0.3
  static var threadCount: Int = 2
}
