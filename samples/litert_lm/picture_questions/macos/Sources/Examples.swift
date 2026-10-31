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

/// A bundled example: a 448 x 448 picture drawn for this sample and a question about it (Resources/examples).
struct Example: Codable, Identifiable, Equatable {
  var id: String
  var title: String
  var picture: String
  var question: String
  /// Muse-Glimmer-30B's reasoning strength for this example (nil = leave the window's choice).
  var reasoning: String?

  var pictureURL: URL? {
    let name = (picture as NSString).deletingPathExtension
    let ext = (picture as NSString).pathExtension
    return Bundle.main.url(forResource: name, withExtension: ext, subdirectory: "examples")
  }

  static let all: [Example] = {
    guard let url = Bundle.main.url(forResource: "examples", withExtension: "json", subdirectory: "examples"),
      let data = try? Data(contentsOf: url),
      let list = try? JSONDecoder().decode([Example].self, from: data)
    else { return [] }
    return list
  }()
}
