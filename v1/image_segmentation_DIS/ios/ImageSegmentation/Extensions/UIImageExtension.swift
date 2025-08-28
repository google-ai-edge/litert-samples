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

        if fred > Float32(UInt8.max) { fred = Float32(UInt8.max) }
        if fgreen > Float32(UInt8.max) { fgreen = Float32(UInt8.max) }
        if fblue > Float32(UInt8.max) { fblue = Float32(UInt8.max) }

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
}
