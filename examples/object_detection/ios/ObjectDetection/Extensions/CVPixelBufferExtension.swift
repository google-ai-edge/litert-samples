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
import Accelerate

extension CVPixelBuffer {

  /**
   Returns thumbnail by cropping pixel buffer to biggest square and scaling the cropped image to
   model dimensions.
   */
  func centerThumbnail(ofSize size: CGSize ) -> CVPixelBuffer? {

    let imageWidth = CVPixelBufferGetWidth(self)
    let imageHeight = CVPixelBufferGetHeight(self)
    let pixelBufferType = CVPixelBufferGetPixelFormatType(self)

    assert(pixelBufferType == kCVPixelFormatType_32BGRA)

    let inputImageRowBytes = CVPixelBufferGetBytesPerRow(self)
    let imageChannels = 4

    let thumbnailSize = min(imageWidth, imageHeight)
    CVPixelBufferLockBaseAddress(self, CVPixelBufferLockFlags(rawValue: 0))

    var originX = 0
    var originY = 0

    if imageWidth > imageHeight {
      originX = (imageWidth - imageHeight) / 2
    }
    else {
      originY = (imageHeight - imageWidth) / 2
    }

    // Finds the biggest square in the pixel buffer and advances rows based on it.
    guard let inputBaseAddress = CVPixelBufferGetBaseAddress(self)?.advanced(
        by: originY * inputImageRowBytes + originX * imageChannels) else {
      return nil
    }

    // Gets vImage Buffer from input image
    var inputVImageBuffer = vImage_Buffer(
        data: inputBaseAddress, height: UInt(thumbnailSize), width: UInt(thumbnailSize),
        rowBytes: inputImageRowBytes)

    let thumbnailRowBytes = Int(size.width) * imageChannels
    guard  let thumbnailBytes = malloc(Int(size.height) * thumbnailRowBytes) else {
      return nil
    }

    // Allocates a vImage buffer for thumbnail image.
    var thumbnailVImageBuffer = vImage_Buffer(data: thumbnailBytes, height: UInt(size.height), width: UInt(size.width), rowBytes: thumbnailRowBytes)

    // Performs the scale operation on input image buffer and stores it in thumbnail image buffer.
    let scaleError = vImageScale_ARGB8888(&inputVImageBuffer, &thumbnailVImageBuffer, nil, vImage_Flags(0))

    CVPixelBufferUnlockBaseAddress(self, CVPixelBufferLockFlags(rawValue: 0))

    guard scaleError == kvImageNoError else {
      return nil
    }

    let releaseCallBack: CVPixelBufferReleaseBytesCallback = {mutablePointer, pointer in

      if let pointer = pointer {
        free(UnsafeMutableRawPointer(mutating: pointer))
      }
    }

    var thumbnailPixelBuffer: CVPixelBuffer?

    // Converts the thumbnail vImage buffer to CVPixelBuffer
    let conversionStatus = CVPixelBufferCreateWithBytes(
        nil, Int(size.width), Int(size.height), pixelBufferType, thumbnailBytes,
        thumbnailRowBytes, releaseCallBack, nil, nil, &thumbnailPixelBuffer)

    guard conversionStatus == kCVReturnSuccess else {

      free(thumbnailBytes)
      return nil
    }

    return thumbnailPixelBuffer
  }

/// This function takes an UIImage and converts it into a CVPixelBuffer
///
/// - Parameters:
///   - image: The UIImage to convert into a CVPixelBuffer
/// - Returns: The CVPixelBuffer created from the UIImage, or nil if unsuccessful
  static func buffer(from image: UIImage) -> CVPixelBuffer? {
    let attrs = [
      kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
      kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue
    ] as CFDictionary
    var pixelBuffer: CVPixelBuffer?
    let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                     Int(image.size.width),
                                     Int(image.size.height),
                                     kCVPixelFormatType_32BGRA,
                                     attrs,
                                     &pixelBuffer)

    guard let buffer = pixelBuffer, status == kCVReturnSuccess else {
      return nil
    }

    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    let pixelData = CVPixelBufferGetBaseAddress(buffer)

    let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
    guard let context = CGContext(data: pixelData,
                                  width: Int(image.size.width),
                                  height: Int(image.size.height),
                                  bitsPerComponent: 8,
                                  bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                                  space: rgbColorSpace,
                                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else {
      return nil
    }

    context.translateBy(x: 0, y: image.size.height)
    context.scaleBy(x: 1.0, y: -1.0)

    UIGraphicsPushContext(context)
    image.draw(in: CGRect(x: 0, y: 0, width: image.size.width, height: image.size.height))
    UIGraphicsPopContext()

    return pixelBuffer
  }

/// This function is used to create a dictionary of attributes for Metal compatibility with pixel buffers.
///
/// - Returns: A dictionary containing Metal compatibility attributes for pixel buffers.
  fileprivate func metalCompatiblityAttributes() -> [String: Any] {
    let attributes: [String: Any] = [
      String(kCVPixelBufferMetalCompatibilityKey): true,
      String(kCVPixelBufferOpenGLCompatibilityKey): true,
      String(kCVPixelBufferIOSurfacePropertiesKey): [
        String(kCVPixelBufferIOSurfaceOpenGLESTextureCompatibilityKey): true,
        String(kCVPixelBufferIOSurfaceOpenGLESFBOCompatibilityKey): true,
        String(kCVPixelBufferIOSurfaceCoreAnimationCompatibilityKey): true
      ]
    ]
    return attributes
  }

  /**
    Creates a pixel buffer of the specified width, height, and pixel format.

    - Note: This pixel buffer is backed by an IOSurface and therefore can be
      turned into a Metal texture.
  */
  public func createPixelBuffer(width: Int, height: Int, pixelFormat: OSType) -> CVPixelBuffer? {
    let attributes = metalCompatiblityAttributes() as CFDictionary
    var pixelBuffer: CVPixelBuffer?
    let status = CVPixelBufferCreate(nil, width, height, pixelFormat, attributes, &pixelBuffer)
    if status != kCVReturnSuccess {
      print("Error: could not create pixel buffer", status)
      return nil
    }
    return pixelBuffer
  }

  /**
    First crops the pixel buffer, then resizes it.

    This function requires the caller to pass in both the source and destination
    pixel buffers. The dimensions of destination pixel buffer should be at least
    `scaleWidth` x `scaleHeight` pixels.
  */
  public func resizePixelBuffer(from srcPixelBuffer: CVPixelBuffer,
                                to dstPixelBuffer: CVPixelBuffer,
                                cropX: Int,
                                cropY: Int,
                                cropWidth: Int,
                                cropHeight: Int,
                                scaleWidth: Int,
                                scaleHeight: Int) {

    assert(CVPixelBufferGetWidth(dstPixelBuffer) >= scaleWidth)
    assert(CVPixelBufferGetHeight(dstPixelBuffer) >= scaleHeight)

    let srcFlags = CVPixelBufferLockFlags.readOnly
    let dstFlags = CVPixelBufferLockFlags(rawValue: 0)

    guard kCVReturnSuccess == CVPixelBufferLockBaseAddress(srcPixelBuffer, srcFlags) else {
      print("Error: could not lock source pixel buffer")
      return
    }
    defer { CVPixelBufferUnlockBaseAddress(srcPixelBuffer, srcFlags) }

    guard kCVReturnSuccess == CVPixelBufferLockBaseAddress(dstPixelBuffer, dstFlags) else {
      print("Error: could not lock destination pixel buffer")
      return
    }
    defer { CVPixelBufferUnlockBaseAddress(dstPixelBuffer, dstFlags) }

    guard let srcData = CVPixelBufferGetBaseAddress(srcPixelBuffer),
          let dstData = CVPixelBufferGetBaseAddress(dstPixelBuffer) else {
      print("Error: could not get pixel buffer base address")
      return
    }

    let srcBytesPerRow = CVPixelBufferGetBytesPerRow(srcPixelBuffer)
    let offset = cropY*srcBytesPerRow + cropX*4
    var srcBuffer = vImage_Buffer(data: srcData.advanced(by: offset),
                                  height: vImagePixelCount(cropHeight),
                                  width: vImagePixelCount(cropWidth),
                                  rowBytes: srcBytesPerRow)

    let dstBytesPerRow = CVPixelBufferGetBytesPerRow(dstPixelBuffer)
    var dstBuffer = vImage_Buffer(data: dstData,
                                  height: vImagePixelCount(scaleHeight),
                                  width: vImagePixelCount(scaleWidth),
                                  rowBytes: dstBytesPerRow)

    let error = vImageScale_ARGB8888(&srcBuffer, &dstBuffer, nil, vImage_Flags(0))
    if error != kvImageNoError {
      print("Error:", error)
    }
  }

  /**
    First crops the pixel buffer, then resizes it.

    This allocates a new destination pixel buffer that is Metal-compatible.
  */
  public func resizePixelBuffer(cropX: Int,
                                cropY: Int,
                                cropWidth: Int,
                                cropHeight: Int,
                                scaleWidth: Int,
                                scaleHeight: Int) -> CVPixelBuffer? {

    let pixelFormat = CVPixelBufferGetPixelFormatType(self)
    let dstPixelBuffer = createPixelBuffer(width: scaleWidth, height: scaleHeight,
                                           pixelFormat: pixelFormat)

    if let dstPixelBuffer = dstPixelBuffer {
      CVBufferPropagateAttachments(self, dstPixelBuffer)

      resizePixelBuffer(from: self, to: dstPixelBuffer,
                        cropX: cropX, cropY: cropY,
                        cropWidth: cropWidth, cropHeight: cropHeight,
                        scaleWidth: scaleWidth, scaleHeight: scaleHeight)
    }

    return dstPixelBuffer
  }

}
