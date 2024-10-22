// Copyright 2024 The Google AI Edge Authors. All Rights Reserved.
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
  static let maxResults: Int = 3
  static let scoreThreshold: Float = 0.2
  static let model: Model = .efficientnetLite0
}


// MARK: Tflite Model
enum Model: String, CaseIterable {
  case efficientnetLite0 = "EfficientNet-Lite0"
  case efficientnetLite2 = "EfficientNet-Lite2"
  
  var modelPath: String? {
    switch self {
    case .efficientnetLite0:
      return Bundle.main.path(
        forResource: "efficientnet_lite0", ofType: "tflite")
    case .efficientnetLite2:
      return Bundle.main.path(
        forResource: "efficientnet_lite2", ofType: "tflite")
    }
  }
  
  var labelPath: String? {
    switch self {
    case .efficientnetLite0:
      return Bundle.main.path(
        forResource: "labels_efficientnet_lite0", ofType: "txt")
    case .efficientnetLite2:
      return Bundle.main.path(
        forResource: "labels_efficientnet_lite2", ofType: "txt")
    }
  }
}
