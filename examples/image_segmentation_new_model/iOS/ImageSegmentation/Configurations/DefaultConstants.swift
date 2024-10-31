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
  static let model: Model = .isnetPt2eDrq
}

// MARK: Model
enum Model: Int, CaseIterable {
  case isnetPt2eDrq
  case isnetTflDrq
  case isnet


  var name: String {
    switch self {
    case .isnetPt2eDrq:
      return "Isnet pt2e drq"
    case .isnetTflDrq:
      return "Isnet tfl drq"
    case .isnet:
      return "Isnet"
    }
  }

  var modelPath: String? {
    switch self {
    case .isnetPt2eDrq:
      return Bundle.main.path(
        forResource: "isnet_pt2e_drq", ofType: "tflite")
    case .isnetTflDrq:
      return Bundle.main.path(
        forResource: "isnet_tfl_drq", ofType: "tflite")
    case .isnet:
      return Bundle.main.path(
        forResource: "isnet", ofType: "tflite")
    }
  }

  init?(name: String) {
    switch name {
    case "Isnet pt2e drq":
      self = .isnetPt2eDrq
    case "Isnet tfl drq":
      self = .isnetTflDrq
    case "Isnet":
      self = .isnet
    default:
      return nil
    }
  }
}
