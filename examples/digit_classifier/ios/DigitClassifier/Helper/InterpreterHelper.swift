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

import Foundation
import TensorFlowLite
import CoreImage
import Accelerate

class InterpreterHelper: NSObject {
  private var modelPath: String
  
  // MARK: - Model Parameters
  var batchSize = 1
  var inputChannels = 3
  var inputWidth = 50
  var inputHeight = 50
  
  private var labels: [String] = []
  
  /// TensorFlow Lite `Interpreter` object for performing inference on a given model.
  private var interpreter: Interpreter!
  
  // MARK: - Custom Initializer
  init?(modelPath: String?) {
    guard let modelPath = modelPath else { return nil }
    self.modelPath = modelPath
    super.init()
    
    createInterpreter()
  }
  
  private func createInterpreter() {
    do {
      // Create the `Interpreter`.
      interpreter = try Interpreter(modelPath: modelPath)
      // Allocate memory for the model's input `Tensor`s.
      try interpreter.allocateTensors()
      
      let input = try interpreter.input(at: 0)
      batchSize = input.shape.dimensions[0]
      inputWidth = input.shape.dimensions[1]
      inputHeight = input.shape.dimensions[2]
      inputChannels = input.shape.dimensions[3]
      print(input)

    } catch let error {
      print("Failed to create the interpreter with error: \(error.localizedDescription)")
    }
  }
  
  // MARK: - Recovering Methods
  /**
   This method return new image and infrenceTime when receive an image
   **/
  func proccess(image: UIImage) -> Result? {
    let interval: TimeInterval
    let outputTensor: Tensor
    do {
      guard let pixelBuffer = CVPixelBuffer.buffer(from: image) else { return nil }
      let sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
      assert(sourcePixelFormat == kCVPixelFormatType_32ARGB ||
               sourcePixelFormat == kCVPixelFormatType_32BGRA ||
                 sourcePixelFormat == kCVPixelFormatType_32RGBA)
      let scaledSize = CGSize(width: inputWidth, height: inputHeight)
      guard let thumbnailPixelBuffer = pixelBuffer.centerThumbnail(ofSize: scaledSize) else {
        return nil
      }
      guard let rgbData = rgbDataFromBuffer(
        thumbnailPixelBuffer,
        byteCount: batchSize * inputWidth * inputHeight * inputChannels
      ) else {
        print("Failed to convert the image buffer to RGB data.")
        return nil
      }
      // Copy the RGB data to the input `Tensor`.
      try interpreter.copy(rgbData, toInputAt: 0)
      
      // Run inference by invoking the `Interpreter`.
      let startDate = Date()
      try interpreter.invoke()
      interval = Date().timeIntervalSince(startDate) * 1000
      
      // Get the output `Tensor` to process the inference results.
      outputTensor = try interpreter.output(at: 0)
      guard let floatData = getFloatsData(tensor: outputTensor),
            let digit = floatData.firstIndex(where: { $0 == 1.0 }) else { return nil }
      return Result(inferenceTime: interval,
                    digit: digit)
    } catch let error {
      print("Failed to invoke the interpreter with error: \(error.localizedDescription)")
      return nil
    }
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
    byteCount: Int
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
    let destinationChannelCount = 3
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

    vImageConvert_RGBA8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))

    let byteData = Data(bytes: destinationBuffer.data, count: destinationBuffer.rowBytes * height)

    //Convert to floats
    let bytes = Array<UInt8>(unsafeData: byteData)!
    var floats = [Float]()
    for i in 0..<bytes.count {
        floats.append(Float(bytes[i]))
    }
    return Data(copyingBufferOf: floats)
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

/// A result from invoking the `Interpreter`.
struct Result {
  let inferenceTime: Double
  let digit: Int
}

