import AVFoundation
import UIKit
import Accelerate

extension CVPixelBuffer {
  public func convertToSquarePixelBuffer(outputSize: Int) -> CVPixelBuffer? {
    let srcWidth = CVPixelBufferGetWidth(self)
    let srcHeight = CVPixelBufferGetHeight(self)
    
    var dstPixelBuffer: CVPixelBuffer?
    
    let pixelBufferAttributes: [String: Any] = [
      kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
      kCVPixelBufferWidthKey as String: outputSize,
      kCVPixelBufferHeightKey as String: outputSize,
      kCVPixelBufferIOSurfacePropertiesKey as String: [:]
    ]
    
    let status = CVPixelBufferCreate(kCFAllocatorDefault, outputSize, outputSize, kCVPixelFormatType_32BGRA, pixelBufferAttributes as CFDictionary, &dstPixelBuffer)
    
    guard status == kCVReturnSuccess, let dstPixelBuffer = dstPixelBuffer else {
      print("Error: could not create destination pixel buffer")
      return nil
    }
    
    let srcFlags = CVPixelBufferLockFlags.readOnly
    let dstFlags = CVPixelBufferLockFlags(rawValue: 0)
    
    guard kCVReturnSuccess == CVPixelBufferLockBaseAddress(self, srcFlags) else {
      print("Error: could not lock source pixel buffer")
      return nil
    }
    defer { CVPixelBufferUnlockBaseAddress(self, srcFlags) }
    
    guard kCVReturnSuccess == CVPixelBufferLockBaseAddress(dstPixelBuffer, dstFlags) else {
      print("Error: could not lock destination pixel buffer")
      return nil
    }
    defer { CVPixelBufferUnlockBaseAddress(dstPixelBuffer, dstFlags) }
    
    guard let srcData = CVPixelBufferGetBaseAddress(self),
          let dstData = CVPixelBufferGetBaseAddress(dstPixelBuffer) else {
      print("Error: could not get pixel buffer base address")
      return nil
    }
    
    let srcBytesPerRow = CVPixelBufferGetBytesPerRow(self)
    let dstBytesPerRow = CVPixelBufferGetBytesPerRow(dstPixelBuffer)
    
    // Initialize destination buffer with black color
    memset(dstData, 0, dstBytesPerRow * outputSize)
    
    // Calculate the scaling factor to fit the source image within the destination square
    let maxDimension = Float(max(srcWidth, srcHeight))
    let scaleFactor = Float(outputSize) / maxDimension
    
    // Calculate the size of the scaled image
    let scaledWidth = Int(Float(srcWidth) * scaleFactor)
    let scaledHeight = Int(Float(srcHeight) * scaleFactor)
    
    // Calculate the position to place the scaled image in the destination buffer
    let dstX = (outputSize - scaledWidth) / 2
    let dstY = (outputSize - scaledHeight) / 2
    
    var srcCropBuffer = vImage_Buffer(data: srcData,
                                      height: vImagePixelCount(srcHeight),
                                      width: vImagePixelCount(srcWidth),
                                      rowBytes: srcBytesPerRow)
    
    var dstCropBuffer = vImage_Buffer(data: dstData.advanced(by: dstY * dstBytesPerRow + dstX * 4),
                                      height: vImagePixelCount(scaledHeight),
                                      width: vImagePixelCount(scaledWidth),
                                      rowBytes: dstBytesPerRow)
    
    let error = vImageScale_ARGB8888(&srcCropBuffer, &dstCropBuffer, nil, vImage_Flags(0))
    if error != kvImageNoError {
      print("Error:", error)
      return nil
    }
    
    return dstPixelBuffer
  }
  
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
  
}
