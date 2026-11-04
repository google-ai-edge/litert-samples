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
import CoreGraphics
import CryptoKit
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// A screenshot as the app holds it: its size and a copy for the window (upright), and the 1920 x 1088 PNG the runtime
/// reads.
struct Picture {
  /// Where it came from: a file name, or what was captured.
  var name: String
  var width: Int
  var height: Int
  var display: NSImage
  var modelPNG: URL
  var sourceSha256: String
  /// SHA-256 of the 1920 x 1088 RGB bytes, comparable with Pillow's on the same picture.
  var pictureRgbSha256: String

  /// Larger pictures are refused before they are decoded (a 200-megapixel picture is a 2.6 GB bitmap).
  static let maxSourcePixels = 200_000_000
  static let displayMaxSide = 2048

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
          + "(\(Picture.maxSourcePixels / 1_000_000) megapixels). The model reads 1920 × 1088 pixels anyway: save a smaller copy."
      case .notWritten(let why):
        return "The app could not write the 1920 × 1088 picture: \(why)"
      }
    }
  }

  /// Decodes `data`, turns it upright by its EXIF orientation, stretches it and writes the PNG. Runs off the main thread.
  static func make(data: Data, name: String, directory: URL) throws -> Picture {
    guard let source = CGImageSourceCreateWithData(data as CFData, nil), CGImageSourceGetCount(source) > 0,
      let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
      let width = properties[kCGImagePropertyPixelWidth] as? Int,
      let height = properties[kCGImagePropertyPixelHeight] as? Int, width > 0, height > 0
    else { throw Failure.unreadable(name) }
    if width * height > maxSourcePixels { throw Failure.tooLarge(name, width, height) }
    let orientation = properties[kCGImagePropertyOrientation] as? Int ?? 1
    guard let image = CGImageSourceCreateImageAtIndex(source, 0, nil), let stored = storedRGB(image) else {
      throw Failure.unreadable(name)
    }
    let upright = stored.oriented(orientation)
    let frame = Pictures.stretch(upright)
    let rgbSHA = frame.sha256
    guard let png = pngData(frame) else { throw Failure.notWritten("no PNG encoder") }
    let file = directory.appendingPathComponent("\(rgbSHA).png")
    do {
      try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
      if !FileManager.default.fileExists(atPath: file.path) { try png.write(to: file, options: .atomic) }
    } catch {
      throw Failure.notWritten(error.localizedDescription)
    }
    let options: [CFString: Any] = [
      kCGImageSourceCreateThumbnailFromImageAlways: true,
      kCGImageSourceCreateThumbnailWithTransform: true,
      kCGImageSourceThumbnailMaxPixelSize: displayMaxSide,
    ]
    let thumbnail = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) ?? image
    let display = NSImage(cgImage: thumbnail, size: NSSize(width: thumbnail.width, height: thumbnail.height))
    return Picture(
      name: name, width: upright.width, height: upright.height, display: display, modelPNG: file,
      sourceSha256: SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined(), pictureRgbSha256: rgbSHA)
  }

  /// The pixels as they are stored, alpha dropped and no color conversion, as Pillow reads them: the context has the
  /// picture's own color space, so drawing into it changes no value of an opaque pixel. A partly transparent pixel (the
  /// rounded corners of a window capture, 340 of 1,085,056 pixels in a Finder window) comes out multiplied by its alpha.
  private static func storedRGB(_ image: CGImage) -> Rgb? {
    let w = image.width
    let h = image.height
    let space = image.colorSpace.flatMap { $0.model == .rgb ? $0 : nil } ?? CGColorSpace(name: CGColorSpace.sRGB)
    guard let space,
      let context = CGContext(
        data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4, space: space,
        bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
    else { return nil }
    context.interpolationQuality = .none
    context.setBlendMode(.copy)
    context.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
    guard let data = context.data else { return nil }
    let rgba = data.assumingMemoryBound(to: UInt8.self)
    var out = [UInt8](repeating: 0, count: w * h * 3)
    out.withUnsafeMutableBufferPointer { o in
      for i in 0..<(w * h) {
        o[i * 3] = rgba[i * 4]
        o[i * 3 + 1] = rgba[i * 4 + 1]
        o[i * 3 + 2] = rgba[i * 4 + 2]
      }
    }
    return Rgb(width: w, height: h, bytes: out)
  }

  /// `image` as PNG bytes (lossless, no color profile: the runtime's decoder reads the same values).
  static func pngData(_ image: Rgb) -> Data? {
    let stride = image.width * 3
    guard let provider = CGDataProvider(data: Data(image.bytes) as CFData),
      let space = CGColorSpace(name: CGColorSpace.sRGB),
      let cg = CGImage(
        width: image.width, height: image.height, bitsPerComponent: 8, bitsPerPixel: 24, bytesPerRow: stride, space: space,
        bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue), provider: provider, decode: nil,
        shouldInterpolate: false, intent: .defaultIntent)
    else { return nil }
    let data = NSMutableData()
    guard let destination = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil) else { return nil }
    CGImageDestinationAddImage(destination, cg, nil)
    return CGImageDestinationFinalize(destination) ? data as Data : nil
  }

  /// A plain mid-gray 1920 x 1088 PNG for the engine's warm-up.
  static func writeGray(to url: URL) throws {
    let frame = Rgb(
      width: Pictures.width, height: Pictures.height,
      bytes: [UInt8](repeating: 128, count: Pictures.width * Pictures.height * 3))
    guard let png = pngData(frame) else { throw HostError("could not make the warm-up picture") }
    try png.write(to: url, options: .atomic)
  }
}

/// Taking a screenshot with macOS's own `screencapture`. The Screen Recording permission is the app's (or, started from
/// a terminal, the terminal's).
enum Capture {
  enum Failure: LocalizedError {
    case noPermission
    case failed(String)

    var errorDescription: String? {
      switch self {
      case .noPermission:
        return "macOS has not allowed this app to take screenshots. Turn on Click or Code in System Settings > "
          + "Privacy & Security > Screen & System Audio Recording, then quit and open the app again. Or open a saved "
          + "screenshot with Open picture…"
      case .failed(let why):
        return "The screenshot failed: \(why)"
      }
    }
  }

  /// Asks macOS once (the system's own dialog, the first time) when the permission is missing.
  static func ensureAllowed() throws {
    if CGPreflightScreenCaptureAccess() { return }
    _ = CGRequestScreenCaptureAccess()
    throw Failure.noPermission
  }

  private static func run(_ arguments: [String]) async throws -> Int32 {
    try await withCheckedThrowingContinuation { (done: CheckedContinuation<Int32, Error>) in
      let process = Process()
      process.executableURL = URL(fileURLWithPath: "/usr/sbin/screencapture")
      process.arguments = arguments
      process.terminationHandler = { done.resume(returning: $0.terminationStatus) }
      do {
        try process.run()
      } catch {
        done.resume(throwing: error)
      }
    }
  }

  /// The main display, without a sound. Returns the PNG's bytes.
  static func screen(to url: URL) async throws -> Data {
    let status = try await run(["-x", "-m", url.path])
    return try read(url, status: status)
  }

  /// macOS's window picker: the user clicks a window. Nil when the user pressed Esc.
  static func pickWindow(to url: URL) async throws -> Data? {
    try? FileManager.default.removeItem(at: url)
    _ = try await run(["-i", "-W", "-x", "-o", url.path])
    guard FileManager.default.fileExists(atPath: url.path) else { return nil }
    return try read(url, status: 0)
  }

  /// One window, by its number, without its shadow (the recorder's way to the picker's result).
  static func window(id: CGWindowID, to url: URL) async throws -> Data {
    let status = try await run(["-x", "-o", "-l", String(id), url.path])
    return try read(url, status: status)
  }

  private static func read(_ url: URL, status: Int32) throws -> Data {
    guard status == 0, let data = try? Data(contentsOf: url), !data.isEmpty else {
      throw Failure.failed("screencapture ended with status \(status).")
    }
    return data
  }

  /// The frontmost normal window of an app, for the recorder.
  static func windowID(owner: String, title: String?) -> CGWindowID? {
    let list = CGWindowListCopyWindowInfo([.optionOnScreenOnly, .excludeDesktopElements], kCGNullWindowID) as? [[String: Any]] ?? []
    for window in list {
      guard window[kCGWindowOwnerName as String] as? String == owner, window[kCGWindowLayer as String] as? Int == 0
      else { continue }
      if let title, window[kCGWindowName as String] as? String != title { continue }
      if let number = window[kCGWindowNumber as String] as? UInt32 { return number }
    }
    return nil
  }
}
