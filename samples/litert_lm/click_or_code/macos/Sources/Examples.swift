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

/// A bundled screenshot and the requests to try on it (`examples/examples.json`, shared with the Android app).
struct Example: Identifiable, Equatable {
  var id: String
  var name: String
  var file: String
  var requests: [String]

  var pictureURL: URL? {
    Bundle.main.url(forResource: (file as NSString).deletingPathExtension, withExtension: (file as NSString).pathExtension, subdirectory: "examples")
  }

  static let all: [Example] = {
    struct List: Decodable {
      struct Item: Decodable {
        var id: String
        var name: String
        var file: String
        var requests: [String]
      }
      var examples: [Item]
    }
    guard let url = Bundle.main.url(forResource: "examples", withExtension: "json", subdirectory: "examples"),
      let data = try? Data(contentsOf: url), let list = try? JSONDecoder().decode(List.self, from: data)
    else { return [] }
    return list.examples.map { Example(id: $0.id, name: $0.name, file: $0.file, requests: $0.requests) }
  }()
}
