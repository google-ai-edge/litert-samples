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

import ImageIO
import UIKit

/// A picture attached to the next message: the PNG file the model reads, and what the chat shows.
struct Picture {
  let file: URL
  let image: UIImage

  /// The longest side of the file the model reads.
  static let maxSide = 1024

  /// LiteRT-LM decodes the file itself: it reads PNG and JPEG but not HEIC, and it ignores a
  /// photo's EXIF orientation and color profile. So the app saves an upright sRGB PNG, at most
  /// 1024 pixels on its longer side.
  static func make(from data: Data) throws -> Picture {
    // ImageIO decodes a reduced copy (a 48-megapixel photo is never decoded at full size, which
    // would take about 200 MB more while a model is loaded) and turns it upright.
    let options: [CFString: Any] = [
      kCGImageSourceCreateThumbnailFromImageAlways: true,
      kCGImageSourceCreateThumbnailWithTransform: true,
      kCGImageSourceThumbnailMaxPixelSize: maxSide,
    ]
    guard let source = CGImageSourceCreateWithData(data as CFData, nil),
      let reduced = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary)
    else { throw NotAPicture() }
    // Drawn again in 8-bit sRGB: iPhone photos are in Display P3.
    let size = CGSize(width: reduced.width, height: reduced.height)
    let format = UIGraphicsImageRendererFormat()
    format.scale = 1
    format.preferredRange = .standard
    let upright = UIGraphicsImageRenderer(size: size, format: format).image { _ in
      UIImage(cgImage: reduced).draw(in: CGRect(origin: .zero, size: size))
    }
    guard let png = upright.pngData() else { throw NotAPicture() }
    let file = FileManager.default.temporaryDirectory
      .appendingPathComponent(UUID().uuidString).appendingPathExtension("png")
    try png.write(to: file)
    return Picture(file: file, image: upright)
  }

  /// The example picture in the app bundle: a drawn café menu.
  static func example() throws -> Picture {
    guard let url = Bundle.main.url(forResource: "example", withExtension: "png") else {
      throw NotAPicture()
    }
    return try make(from: Data(contentsOf: url))
  }

  struct NotAPicture: LocalizedError {
    var errorDescription: String? { "This file could not be read as a picture." }
  }
}
