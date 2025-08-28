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

/**
 * Singleton storing the configs needed to initialize an tflite Tasks object and run inference.
 * Controllers can observe the `InferenceConfigManager.notificationName` for any changes made by the user.
 */
class InferenceConfigManager: NSObject {

  var model: Model = DefaultConstants.model {
    didSet { postConfigChangedNotification() }
  }

  var scoreThreshold: Float = DefaultConstants.scoreThreshold {
    didSet { postConfigChangedNotification() }
  }
  
  var maxResults: Int = DefaultConstants.maxResults {
    didSet {
      postConfigChangedNotification()
      postMaxResultChangedNotification()
    }
  }

  static let sharedInstance = InferenceConfigManager()
  
  static let notificationName = Notification.Name.init(rawValue: "com.google.tflite.inferenceConfigChanged")
  static let maxResultChangeNotificationName = Notification.Name.init(rawValue: "com.google.tflite.inferenceMaxResultsChanged")
  
  private func postConfigChangedNotification() {
    NotificationCenter.default
      .post(name: InferenceConfigManager.notificationName, object: nil)
  }

  private func postMaxResultChangedNotification() {
    NotificationCenter.default
      .post(name: InferenceConfigManager.maxResultChangeNotificationName, object: nil)
  }

}
