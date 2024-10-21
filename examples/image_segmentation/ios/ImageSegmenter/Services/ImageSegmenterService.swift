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
import TensorFlowLite
import AVFoundation

// Initializes and calls the Tflite APIs for segmention.
class ImageSegmenterService: NSObject {
  
  private var batchSize = 1
  private var inputWidth: Int = 257
  private var inputHeight: Int = 257
  private var inputChannels: Int = 3

  var interpreter: Interpreter!
  var modelPath: String
  
  // MARK: - Custom Initializer
  init?(model: Model) {
    guard let modelPath = model.modelPath else { return nil }
    self.modelPath = modelPath
    super.init()
    
    createInterpreter()
  }
  
  private func createInterpreter() {
    do {
      interpreter = try Interpreter(modelPath: modelPath)
      try interpreter.allocateTensors()
      
      let input = try interpreter.input(at: 0)
      batchSize = input.shape.dimensions[0]
      inputWidth = input.shape.dimensions[1]
      inputHeight = input.shape.dimensions[2]
      inputChannels = input.shape.dimensions[3]
    } catch {
      print("Failed to create Interpreter: \(error.localizedDescription)")
    }
  }
  
  // MARK: - Segmention Methods for Different Modes
  /**
   This method return ImageSegmenterResult and infrenceTime when receive an image
   **/
  func segment(image: UIImage) -> ResultBundle? {
    guard let pixelBuffer = CVPixelBuffer.buffer(from: image) else {
      return nil
    }
    
    return runModel(onFrame: pixelBuffer)
  }
  
  func segmentAsync(
    sampleBuffer: CMSampleBuffer, completion: (ResultBundle?) -> Void) {
      guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
        completion(nil)
        return
      }
      completion(runModel(onFrame: pixelBuffer))
    }
  
  func segment(
    videoFrame: CGImage)
  -> ResultBundle?  {
    let image = UIImage(cgImage: videoFrame)
    return segment(image: image)
  }
  
  private func runModel(onFrame pixelBuffer: CVPixelBuffer) -> ResultBundle? {
    guard let resizedPixelBuffer = pixelBuffer.convertToSquare(squareSize: inputWidth) else {
      return nil
    }
    let sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
    assert(sourcePixelFormat == kCVPixelFormatType_32ARGB ||
           sourcePixelFormat == kCVPixelFormatType_32BGRA ||
           sourcePixelFormat == kCVPixelFormatType_32RGBA)
    
    
    let imageChannels = 4
    assert(imageChannels >= inputChannels)
    
    let interval: TimeInterval
    let outputTensor: Tensor
    
    do {
      let input = try interpreter.input(at: 0)
      guard let rgbData = rgbDataFromBuffer(
        resizedPixelBuffer,
        byteCount: batchSize * inputWidth * inputHeight * inputChannels,
        isModelQuantized: input.dataType == .uInt8
      ) else {
        print("Failed to convert the image buffer to RGB data.")
        return nil
      }
      
      try interpreter.copy(rgbData, toInputAt: 0)
      // Run inference by invoking the `Interpreter`.
      let startDate = Date()
      try interpreter.invoke()
      interval = Date().timeIntervalSince(startDate) * 1000
      
      // Get the output `Tensor` to process the inference results.
      outputTensor = try interpreter.output(at: 0)
      
    } catch {
      print("Failed to invoke the interpreter with error: \(error.localizedDescription)")
      return nil
    }
    let outputData = outputTensor.data
        
    let segmentMask = createSegmentationMask(from: outputData, width: outputTensor.shape.dimensions[1], height: outputTensor.shape.dimensions[2], numClasses: outputTensor.shape.dimensions[3])
    
    let categoryMask = CategoryMask(width: outputTensor.shape.dimensions[1], height: outputTensor.shape.dimensions[1], data: segmentMask)
    
    return ResultBundle(inferenceTime: interval, categoryMasks: [categoryMask])
  }
  
  // Function to convert TensorFlow Lite Data (float32) to [Float] array
  func convertTensorDataToFloatArray(tensorData: Data) -> [Float] {
    return tensorData.withUnsafeBytes {
      Array(UnsafeBufferPointer<Float32>(start: $0.bindMemory(to: Float32.self).baseAddress, count: tensorData.count / MemoryLayout<Float32>.stride))
    }
  }
  
  // Function to create a segmentation mask by picking the class with the maximum probability
  func createSegmentationMask(from data: Data, width: Int, height: Int, numClasses: Int) -> [UInt8] {
    let expectedDataSize = width * height * numClasses
    guard data.count == expectedDataSize * MemoryLayout<Float>.size else {
        print("Data size mismatch")
        return []
    }
    let floatArray = data.withUnsafeBytes {
        Array(UnsafeBufferPointer<Float>(start: $0.bindMemory(to: Float.self).baseAddress!, count: expectedDataSize))
    }
        
    var maskData = [UInt8](repeating: 0, count: width * height)
    
    floatArray.withUnsafeBufferPointer { bufferPointer in
      for i in 0..<width * height {
        let startIdx = i * numClasses
        let endIdx = startIdx + numClasses
        let classValues = Array(bufferPointer[startIdx..<endIdx])
        
        // Use vDSP to find the index of the maximum value efficiently
        var maxClassIndex: Int = 0
        var maxClassValue: Float = -Float.greatestFiniteMagnitude
        vDSP_maxvi(classValues, 1, &maxClassValue, &maxClassIndex, vDSP_Length(numClasses))
        
        // Store the index of the class with the maximum probability (as grayscale)
        maskData[i] = UInt8(maxClassIndex)
      }
    }
    
    return maskData
  }
  
  /// Returns the float datas from output tensor
  private func getFloatsData(tensor: Tensor) -> [Float]? {
    let results: [Float]
    switch tensor.dataType {
    case .uInt8:
      guard let quantization = tensor.quantizationParameters else {
        print("No results returned because the quantization values for the output tensor are nil.")
        return nil
      }
      let quantizedResults = [UInt8](tensor.data)
      results = quantizedResults.map {
        quantization.scale * Float(Int($0) - quantization.zeroPoint)
      }
    case .float32:
      results = [Float32](unsafeData: tensor.data) ?? []
    default:
      print("Output tensor data type \(tensor.dataType) is unsupported for this example app.")
      return nil
    }
    return results
  }
  
  /// Returns the RGB data representation of the given image buffer with the specified `byteCount`.
  ///
  /// - Parameters
  ///   - buffer: The pixel buffer to convert to RGB data.
  ///   - byteCount: The expected byte count for the RGB data calculated using the values that the
  ///       model was trained on: `batchSize * imageWidth * imageHeight * componentsCount`.
  ///   - isModelQuantized: Whether the model is quantized (i.e. fixed point values rather than
  ///       floating point values).
  /// - Returns: The RGB data representation of the image buffer or `nil` if the buffer could not be
  ///     converted.
  private func rgbDataFromBuffer(
    _ buffer: CVPixelBuffer,
    byteCount: Int,
    isModelQuantized: Bool
  ) -> Data? {
    CVPixelBufferLockBaseAddress(buffer, .readOnly)
    defer {
      CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
    }
    guard let sourceData = CVPixelBufferGetBaseAddress(buffer) else {
      return nil
    }
    
    let width = CVPixelBufferGetWidth(buffer)
    let height = CVPixelBufferGetHeight(buffer)
    let sourceBytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
    let destinationChannelCount = inputChannels
    let destinationBytesPerRow = destinationChannelCount * width
    
    var sourceBuffer = vImage_Buffer(data: sourceData,
                                     height: vImagePixelCount(height),
                                     width: vImagePixelCount(width),
                                     rowBytes: sourceBytesPerRow)
    
    guard let destinationData = malloc(height * destinationBytesPerRow) else {
      print("Error: out of memory")
      return nil
    }
    
    defer {
      free(destinationData)
    }
    
    var destinationBuffer = vImage_Buffer(data: destinationData,
                                          height: vImagePixelCount(height),
                                          width: vImagePixelCount(width),
                                          rowBytes: destinationBytesPerRow)
    
    let pixelBufferFormat = CVPixelBufferGetPixelFormatType(buffer)
    
    switch (pixelBufferFormat) {
    case kCVPixelFormatType_32BGRA:
      vImageConvert_BGRA8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
    case kCVPixelFormatType_32ARGB:
      vImageConvert_ARGB8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
    case kCVPixelFormatType_32RGBA:
      vImageConvert_RGBA8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
    default:
      // Unknown pixel format.
      return nil
    }
    
    let byteData = Data(bytes: destinationBuffer.data, count: destinationBuffer.rowBytes * height)
    
    // Quantized model: keep it as uint8
    if isModelQuantized {
      return byteData
    }
    
    // Non-quantized: normalize to float values
    guard let bytes = Array<UInt8>(unsafeData: byteData) else { return nil }
    return Data(copyingBufferOf: normalizeBytesToFloats(bytes: bytes))
  }
  
  /// Normalizes an array of UInt8 bytes to an array of Float values, reducing CPU usage.
  ///
  /// This function uses vectorized operations from the Accelerate framework, which are
  /// optimized for performance, allowing for efficient conversion and normalization of
  /// the byte array. Instead of using a loop and appending values one by one, this
  /// method minimizes overhead by processing the entire array in a single operation,
  /// thus significantly reducing CPU load during the conversion.
  ///
  /// - Parameter bytes: An array of UInt8 values representing pixel data.
  /// - Returns: An array of normalized Float values, where each value is in the range [0.0, 1.0].
  private func normalizeBytesToFloats(bytes: [UInt8]) -> [Float] {
    let count = bytes.count
    var floats = [Float](repeating: 0, count: count)
    vDSP_vfltu8(bytes, 1, &floats, 1, vDSP_Length(count))
    
    var divisor: Float = 255.0
    vDSP_vsdiv(floats, 1, &divisor, &floats, 1, vDSP_Length(count))
    
    return floats
  }
}

/// A result from the `ImageSegmenterService`.
struct ResultBundle {
  let inferenceTime: Double
  let categoryMasks: [CategoryMask]
  var size: CGSize = .zero
}

struct CategoryMask {
  let width: Int
  let height: Int
  let data: [UInt8]
  var mask: UnsafeRawPointer {
    return data.withUnsafeBytes { rawBufferPointer in
      return UnsafeRawPointer(rawBufferPointer.baseAddress!)
    }
  }
}

struct VideoFrame {
  let pixelBuffer: CVPixelBuffer
  let formatDescription: CMFormatDescription
}


import Accelerate

extension CVPixelBuffer {
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

extension Array {
  /// Creates a new array from the bytes of the given unsafe data.
  ///
  /// - Warning: The array's `Element` type must be trivial in that it can be copied bit for bit
  ///     with no indirection or reference-counting operations; otherwise, copying the raw bytes in
  ///     the `unsafeData`'s buffer to a new array returns an unsafe copy.
  /// - Note: Returns `nil` if `unsafeData.count` is not a multiple of
  ///     `MemoryLayout<Element>.stride`.
  /// - Parameter unsafeData: The data containing the bytes to turn into an array.
  init?(unsafeData: Data) {
    guard unsafeData.count % MemoryLayout<Element>.stride == 0 else { return nil }
    self = unsafeData.withUnsafeBytes { .init($0.bindMemory(to: Element.self)) }
  }
}

extension Data {
  /// Creates a new buffer by copying the buffer pointer of the given array.
  ///
  /// - Warning: The given array's element type `T` must be trivial in that it can be copied bit
  ///     for bit with no indirection or reference-counting operations; otherwise, reinterpreting
  ///     data from the resulting buffer has undefined behavior.
  /// - Parameter array: An array with elements of type `T`.
  init<T>(copyingBufferOf array: [T]) {
    self = array.withUnsafeBufferPointer(Data.init)
  }
}
