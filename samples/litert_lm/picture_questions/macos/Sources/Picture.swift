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

import AppKit
import CryptoKit
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// A picture as the model gets it: 448 x 448 pixels, upright, the whole picture kept.
///
/// The two models' files read one fixed 448 x 448 picture. LiteRT-LM would stretch any other size to 448 x 448 and
/// does not apply a photo's EXIF orientation, so the app makes the 448 x 448 picture itself: it turns the photo
/// upright, scales its longer side to 448 and centers it on a white square (nothing is cut off and nothing is
/// squeezed). That PNG is the file the model reads and the picture the window shows. A PNG that is already
/// 448 x 448 and upright is used as it is, byte for byte.
struct PreparedPicture: Equatable {
  /// Where the picture came from: a file name, or "Pasted picture".
  var sourceName: String
  var sourcePixelWidth: Int
  var sourcePixelHeight: Int
  /// EXIF orientation 1-8 (1 = upright as stored).
  var orientation: Int
  /// The 448 x 448 PNG given to the model.
  var modelPNG: URL
  var sha256: String
  /// How the 448 x 448 picture was made, in words.
  var note: String

  static let side = 448
  /// Larger pictures are refused before they are decoded (a 200-megapixel picture is a 2.6 GB bitmap).
  static let maxSourcePixels = 200_000_000

  enum Failure: LocalizedError {
    case unreadable(String)
    case tooLarge(String, Int, Int)
    case notWritten(String)

    var errorDescription: String? {
      switch self {
      case .unreadable(let name):
        return "macOS could not read “\(name)” as a picture. It may be damaged, cut short, or not a picture at all."
      case .tooLarge(let name, let width, let height):
        return "“\(name)” is \(width) × \(height) pixels, more than this app reads "
          + "(\(PreparedPicture.maxSourcePixels / 1_000_000) megapixels). The model sees 448 × 448 pixels anyway: "
          + "save a smaller copy, or crop to the part you are asking about."
      case .notWritten(let why):
        return "The app could not write the 448 × 448 picture: \(why)"
      }
    }
  }

  static func make(from url: URL, directory: URL) throws -> PreparedPicture {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else {
      throw Failure.unreadable(url.lastPathComponent)
    }
    let original = (try? Data(contentsOf: url, options: .mappedIfSafe)) ?? Data()
    return try make(source: source, name: url.lastPathComponent, originalBytes: original, directory: directory)
  }

  static func make(from data: Data, name: String, directory: URL) throws -> PreparedPicture {
    guard let source = CGImageSourceCreateWithData(data as CFData, nil) else { throw Failure.unreadable(name) }
    return try make(source: source, name: name, originalBytes: data, directory: directory)
  }

  private static func make(source: CGImageSource, name: String, originalBytes: Data, directory: URL) throws
    -> PreparedPicture
  {
    guard CGImageSourceGetCount(source) > 0,
      let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
      let width = properties[kCGImagePropertyPixelWidth] as? Int,
      let height = properties[kCGImagePropertyPixelHeight] as? Int, width > 0, height > 0
    else { throw Failure.unreadable(name) }
    if width * height > maxSourcePixels { throw Failure.tooLarge(name, width, height) }
    let orientation = properties[kCGImagePropertyOrientation] as? Int ?? 1
    let type = CGImageSourceGetType(source).map { $0 as String } ?? ""

    let png: Data
    let note: String
    if width == side, height == side, orientation == 1, type == UTType.png.identifier {
      // Already the model's size and upright: the same bytes, so the same pixels.
      guard CGImageSourceCreateImageAtIndex(source, 0, nil) != nil else { throw Failure.unreadable(name) }
      png = originalBytes
      note = "448 × 448 PNG, used as it is"
    } else {
      // Upright and with its longer side at most 448 (ImageIO's own downsampling; large and HEIC photos are never
      // decoded at full size), then centered on a white 448 x 448 square in sRGB.
      let options: [CFString: Any] = [
        kCGImageSourceCreateThumbnailFromImageAlways: true,
        kCGImageSourceCreateThumbnailWithTransform: true,
        kCGImageSourceThumbnailMaxPixelSize: side,
        kCGImageSourceShouldCacheImmediately: true,
      ]
      guard let image = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else {
        throw Failure.unreadable(name)
      }
      png = try squarePNG(image)
      let upright = orientation == 1 ? "" : ", turned upright (EXIF orientation \(orientation))"
      note = "\(width) × \(height)\(upright), scaled into 448 × 448 on white"
    }

    let sha = SHA256.hash(data: png).map { String(format: "%02x", $0) }.joined()
    let file = directory.appendingPathComponent("\(sha).png")
    do {
      try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
      if !FileManager.default.fileExists(atPath: file.path) { try png.write(to: file, options: .atomic) }
    } catch {
      throw Failure.notWritten(error.localizedDescription)
    }
    return PreparedPicture(
      sourceName: name, sourcePixelWidth: width, sourcePixelHeight: height, orientation: orientation,
      modelPNG: file, sha256: sha, note: note)
  }

  /// `image` (at most 448 on its longer side, already upright) scaled to fit and centered on a white 448 x 448
  /// square, as an 8-bit sRGB PNG without alpha.
  private static func squarePNG(_ image: CGImage) throws -> Data {
    guard let space = CGColorSpace(name: CGColorSpace.sRGB),
      let context = CGContext(
        data: nil, width: side, height: side, bitsPerComponent: 8, bytesPerRow: 0, space: space,
        bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
    else { throw Failure.notWritten("no drawing context") }
    context.setFillColor(CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 1))
    context.fill(CGRect(x: 0, y: 0, width: side, height: side))
    context.interpolationQuality = .high
    let scale = Double(side) / Double(max(image.width, image.height))
    let w = (Double(image.width) * scale).rounded()
    let h = (Double(image.height) * scale).rounded()
    context.draw(image, in: CGRect(x: (Double(side) - w) / 2, y: (Double(side) - h) / 2, width: w, height: h))
    guard let square = context.makeImage() else { throw Failure.notWritten("no image") }
    let data = NSMutableData()
    guard let destination = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil) else {
      throw Failure.notWritten("no PNG encoder")
    }
    CGImageDestinationAddImage(destination, square, nil)
    guard CGImageDestinationFinalize(destination) else { throw Failure.notWritten("PNG encoding failed") }
    return data as Data
  }
}
