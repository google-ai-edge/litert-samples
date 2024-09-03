// Copyright 2024 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

import Foundation
import UIKit

// MARK: Define default constants
struct DefaultConstants {
  static let model: Model = .deeplabV3
}

// MARK: Model
enum Model: Int, CaseIterable {
  case deeplabV3

  var name: String {
    switch self {
    case .deeplabV3:
      return "Deeplab V3"
    }
  }

  var modelPath: String? {
    switch self {
    case .deeplabV3:
      return Bundle.main.path(
        forResource: "deeplab_v3", ofType: "tflite")
    }
  }

  init?(name: String) {
    switch name {
    case "Deeplab V3":
      self.init(rawValue: 1)
    default:
      return nil
    }
  }
}
