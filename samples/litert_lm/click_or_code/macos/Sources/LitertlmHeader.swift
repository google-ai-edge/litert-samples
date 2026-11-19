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

/// What a .litertlm file says about itself. The file starts with "LITERTLM", a version and the offset where its header
/// ends; the header, a flatbuffer from byte 32 on, lists the file's sections. A model's weights are either inside its
/// graphs or in TFLiteWeights sections of their own, and the list shows which.
struct LitertlmHeader {
  /// Each section's AnySectionDataType in the schema; 7 is TFLiteWeights.
  var sectionTypes: [Int]

  /// Whether the file's weights are in TFLiteWeights sections, not inside its graphs.
  var weightsInSections: Bool { sectionTypes.contains(7) }

  /// The header of the file at `url`; nil when it has none: not a .litertlm file, or cut inside it.
  static func read(_ url: URL) -> LitertlmHeader? {
    guard let handle = try? FileHandle(forReadingFrom: url.resolvingSymlinksInPath()) else { return nil }
    defer { try? handle.close() }
    guard let start = try? handle.read(upToCount: 32), start.count == 32, start.prefix(8) == Data("LITERTLM".utf8),
      let end = Flatbuffer(bytes: [UInt8](start)).uint(at: 24, size: 8), end >= 32, end <= 1 << 24,
      let rest = try? handle.read(upToCount: end - 32), rest.count == end - 32
    else { return nil }
    let buffer = Flatbuffer(bytes: [UInt8](rest))
    guard let metadata = buffer.target(of: 0), let sections = buffer.field(6, of: metadata).flatMap(buffer.target),
      let objects = buffer.field(4, of: sections).flatMap(buffer.target),
      let count = buffer.uint(at: objects, size: 4), count > 0, count <= (buffer.bytes.count - objects) / 4
    else { return nil }
    var types: [Int] = []
    for index in 0..<count {
      guard let section = buffer.target(of: objects + 4 + 4 * index) else { return nil }
      types.append(buffer.field(10, of: section).flatMap { buffer.uint(at: $0, size: 1) } ?? 0)
    }
    return LitertlmHeader(sectionTypes: types)
  }
}

/// Reads a flatbuffer. A read outside its bytes is nil.
private struct Flatbuffer {
  let bytes: [UInt8]

  func uint(at position: Int, size: Int) -> Int? {
    guard position >= 0, position + size <= bytes.count else { return nil }
    return (0..<size).reversed().reduce(0) { $0 << 8 | Int(bytes[position + $1]) }
  }

  /// Where the offset stored at `position` points.
  func target(of position: Int) -> Int? { uint(at: position, size: 4).map { position + $0 } }

  /// Where a field of the table at `table` is; a field is named by its place in the vtable (4, 6, 8...). Nil when the
  /// table leaves it out.
  func field(_ field: Int, of table: Int) -> Int? {
    guard let back = uint(at: table, size: 4) else { return nil }
    let vtable = table - Int(Int32(truncatingIfNeeded: back))
    guard let size = uint(at: vtable, size: 2), field < size, let offset = uint(at: vtable + field, size: 2), offset != 0
    else { return nil }
    return table + offset
  }
}
