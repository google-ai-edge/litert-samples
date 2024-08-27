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

import UIKit

extension UIImage {

  convenience init?(data: Data, size: CGSize) {
    let width = Int(size.width)
    let height = Int(size.height)

    let floats = [Float32](unsafeData: data) ?? []

    let bufferCapacity = width * height * 4
    let unsafePointer = UnsafeMutablePointer<UInt8>.allocate(capacity: bufferCapacity)
    let unsafeBuffer = UnsafeMutableBufferPointer<UInt8>(start: unsafePointer,
                                                         count: bufferCapacity)
    defer {
      unsafePointer.deallocate()
    }

    for x in 0 ..< width {
      for y in 0 ..< height {
        let floatIndex = (y * width + x) * 3
        let index = (y * width + x) * 4
        var fred = floats[floatIndex]
        var fgreen = floats[floatIndex + 1]
        var fblue = floats[floatIndex + 2]
        if fred < Float(UInt8.min) { fred = Float(UInt8.min) }
        if fgreen < Float(UInt8.min) { fgreen = Float(UInt8.min) }
        if fblue < Float(UInt8.min) { fblue = Float(UInt8.min) }

        if fred > 255 { fred = 255 }
        if fgreen > 255 { fgreen = 255 }
        if fblue > 255 { fblue = 255 }

        let red = UInt8(fred)
        let green = UInt8(fgreen)
        let blue = UInt8(fblue)

        unsafeBuffer[index] = UInt8(red)
        unsafeBuffer[index + 1] = UInt8(green)
        unsafeBuffer[index + 2] = UInt8(blue)
        unsafeBuffer[index + 3] = 0
      }
    }

    let outData = Data(buffer: unsafeBuffer)

    // Construct image from output tensor data
    let alphaInfo = CGImageAlphaInfo.noneSkipLast
    let bitmapInfo = CGBitmapInfo(rawValue: alphaInfo.rawValue)
      .union(.byteOrder32Big)
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    guard
      let imageDataProvider = CGDataProvider(data: outData as CFData),
      let cgImage = CGImage(
        width: width,
        height: height,
        bitsPerComponent: 8,
        bitsPerPixel: 32,
        bytesPerRow: MemoryLayout<UInt8>.size * 4 * Int(size.width),
        space: colorSpace,
        bitmapInfo: bitmapInfo,
        provider: imageDataProvider,
        decode: nil,
        shouldInterpolate: false,
        intent: .defaultIntent
      )
    else {
      return nil
    }
    self.init(cgImage: cgImage)
  }

  func croppedFixedOrientation(boundingBox: CGRect) -> UIImage? {
    guard let cgImage = self.cgImage else {
      //CGImage is not available
      return nil
    }
    guard imageOrientation != UIImage.Orientation.up else {
      //This is default orientation, don't need to do anything
      guard let cgImage = self.cgImage?.cropping(to: boundingBox) else {
        return nil
      }
      return UIImage(cgImage: cgImage)
    }
    guard let colorSpace = cgImage.colorSpace, let ctx = CGContext(data: nil, width: Int(size.width), height: Int(size.height), bitsPerComponent: cgImage.bitsPerComponent, bytesPerRow: 0, space: colorSpace, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
      return nil //Not able to create CGContext
    }

    var transform: CGAffineTransform = CGAffineTransform.identity

    switch imageOrientation {
    case .down, .downMirrored:
      transform = transform.translatedBy(x: size.width, y: size.height)
      transform = transform.rotated(by: CGFloat.pi)
      break
    case .left, .leftMirrored:
      transform = transform.translatedBy(x: size.width, y: 0)
      transform = transform.rotated(by: CGFloat.pi / 2.0)
      break
    case .right, .rightMirrored:
      transform = transform.translatedBy(x: 0, y: size.height)
      transform = transform.rotated(by: CGFloat.pi / -2.0)
      break
    default:
      break
    }

    //Flip image one more time if needed to, this is to prevent flipped image
    switch imageOrientation {
    case .upMirrored, .downMirrored:
      transform = transform.translatedBy(x: size.width, y: 0)
      transform = transform.scaledBy(x: -1, y: 1)
      break
    case .leftMirrored, .rightMirrored:
      transform = transform.translatedBy(x: size.height, y: 0)
      transform = transform.scaledBy(x: -1, y: 1)
    default:
      break
    }

    ctx.concatenate(transform)

    switch imageOrientation {
    case .left, .leftMirrored, .right, .rightMirrored:
      ctx.draw(self.cgImage!, in: CGRect(x: 0, y: 0, width: size.height, height: size.width))
    default:
      ctx.draw(self.cgImage!, in: CGRect(x: 0, y: 0, width: size.width, height: size.height))
      break
    }

    guard let newCGImage = ctx.makeImage() else { return nil }
    guard let cgImage = newCGImage.cropping(to: boundingBox) else {
      return nil
    }
    return UIImage(cgImage: cgImage)
  }
}
