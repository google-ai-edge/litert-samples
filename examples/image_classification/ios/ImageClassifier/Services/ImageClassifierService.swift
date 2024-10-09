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
import TensorFlowLite
import AVFoundation
import Accelerate
/**
 This protocol must be adopted by any class that wants to get the classification results of the image classifier in live stream mode.
 */
protocol ImageClassifierServiceLiveStreamDelegate: AnyObject {
  func imageClassifierService(_ imageClassifierService: ImageClassifierService,
                              didFinishClassification result: ResultBundle?,
                              error: Error?)
}

/**
 This protocol must be adopted by any class that wants to take appropriate actions during  different stages of image classification on videos.
 */
protocol ImageClassifierServiceVideoDelegate: AnyObject {
  func imageClassifierService(_ imageClassifierService: ImageClassifierService,
                              didFinishClassificationOnVideoFrame index: Int)
  func imageClassifierService(_ imageClassifierService: ImageClassifierService,
                              willBeginClassification totalframeCount: Int)
}


// Initializes and calls the tflite APIs for classification.
class ImageClassifierService: NSObject {
  
  weak var liveStreamDelegate: ImageClassifierServiceLiveStreamDelegate?
  weak var videoDelegate: ImageClassifierServiceVideoDelegate?
  
  private var interpreter: Interpreter!
  private var scoreThreshold: Float
  private var maxResult: Int
  private var model: Model
  
  private var batchSize = 1
  private var inputWidth: Int = 224
  private var inputHeight: Int = 224
  private var inputChannels: Int = 3
  /// List of labels from the given labels file.
  private var labels: [String] = []
  
  // MARK: - Custom Initializer
  init?(model: Model, scoreThreshold: Float, maxResult: Int) {
    self.model = model
    self.scoreThreshold = scoreThreshold
    self.maxResult = maxResult
    super.init()
    
    createInterpreter()
  }
  
  private func createInterpreter() {
    guard let modelPath = model.modelPath else {
      fatalError("Failed to load model \(model.rawValue)")
    }
    do {
      interpreter = try Interpreter(modelPath: modelPath)
      try interpreter.allocateTensors()
      
      let input = try interpreter.input(at: 0)
      batchSize = input.shape.dimensions[0]
      inputWidth = input.shape.dimensions[1]
      inputHeight = input.shape.dimensions[2]
      inputChannels = input.shape.dimensions[3]
    } catch {
      print("Failed to create Interpreter with error: \(error)")
    }
    
    loadLabels()
  }
  
  /// Loads the labels from the labels file and stores them in the `labels` property.
  private func loadLabels() {
    guard let labelPath = model.labelPath else { return }
    do {
      let content = try String(contentsOfFile: labelPath, encoding: .utf8)
      labels = content.components(separatedBy: .newlines)
    } catch {
      fatalError("Labels file named of \(model.rawValue) cannot be read. Please add a " +
                 "valid labels file and try again.")
    }
  }
  
  // MARK: - Classification Methods
  /**
   This method return ImageClassifierResult and infrenceTime when receive an image
   **/
  func classify(image: UIImage) -> ResultBundle? {
    guard let buffer = CVPixelBuffer.buffer(from: image) else { return nil }
    return runModel(onFrame: buffer)
  }
  
  func classify(
    sampleBuffer: CMSampleBuffer, completion: (ResultBundle?) -> Void
  ) {
    guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
      completion(nil)
      return
    }
    if let result = runModel(onFrame: pixelBuffer) {
      completion(result)
    } else {
      completion(nil)
    }
  }
  
  func classify(
    videoAsset: AVAsset,
    durationInMilliseconds: Double,
    inferenceIntervalInMilliseconds: Double
  ) async -> ResultBundle? {
    let startDate = Date()
    let assetGenerator = imageGenerator(with: videoAsset)
    
    let frameCount = Int(durationInMilliseconds / inferenceIntervalInMilliseconds)
    Task { @MainActor in
      videoDelegate?.imageClassifierService(self, willBeginClassification: frameCount)
    }
    
    let imageClassifierResultTuple = classifyObjectsInFramesGenerated(
      by: assetGenerator,
      totalFrameCount: frameCount,
      atIntervalsOf: inferenceIntervalInMilliseconds)
    
    return ResultBundle(inferenceTime: Date().timeIntervalSince(startDate) / Double(frameCount) * 1000, categories: imageClassifierResultTuple.imageClassificationCategories, size: imageClassifierResultTuple.videoSize)
  }
  
  // MARK: - Private methods
  private func runModel(onFrame pixelBuffer: CVPixelBuffer) -> ResultBundle? {
    guard let resizedPixelBuffer = pixelBuffer.convertToSquarePixelBuffer(outputSize: inputWidth) else {
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
    
    guard let floatData = getFloatsData(tensor: outputTensor) else { return nil }
    
    let categories = getClassificationCategories(floatData)
    
    return ResultBundle(inferenceTime: interval, categories: categories)
  }
  
  /// Returns the top classification categories inference results sorted in descending order.
  private func getClassificationCategories(_ floatData: [Float]) -> [ClassificationCategory] {
    var classificationCategories: [ClassificationCategory] = []
    let maxIndex = floatData.indices.max { floatData[$0] < floatData[$1] } ?? -1
    if maxIndex >= 0 {
      let confidence = floatData[maxIndex]
      if confidence > scoreThreshold, maxIndex < labels.count {
        let predictedLabel = labels[maxIndex]
        let category = ClassificationCategory(label: predictedLabel, score: confidence)
        classificationCategories.append(category)
      }
    }
    let sortedClassificationCategories = classificationCategories
      .sorted { $0.score > $1.score }.prefix(maxResult)
    return Array(sortedClassificationCategories)
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
    if isModelQuantized {
      return byteData
    }
    
    // Not quantized, convert to floats
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
  
  private func imageGenerator(with videoAsset: AVAsset) -> AVAssetImageGenerator {
    let generator = AVAssetImageGenerator(asset: videoAsset)
    generator.requestedTimeToleranceBefore = CMTimeMake(value: 1, timescale: 25)
    generator.requestedTimeToleranceAfter = CMTimeMake(value: 1, timescale: 25)
    generator.appliesPreferredTrackTransform = true
    
    return generator
  }
  
  private func classifyObjectsInFramesGenerated(
    by assetGenerator: AVAssetImageGenerator,
    totalFrameCount frameCount: Int,
    atIntervalsOf inferenceIntervalMs: Double)
  -> (imageClassificationCategories: [ClassificationCategory], videoSize: CGSize)  {
    var imageClassificationCategories: [ClassificationCategory] = []
    var videoSize = CGSize.zero
    
    for i in 0..<frameCount {
      let timestampMs = Int(inferenceIntervalMs) * i // ms
      let image: CGImage
      do {
        let time = CMTime(value: Int64(timestampMs), timescale: 1000)
        //        CMTime(seconds: Double(timestampMs) / 1000, preferredTimescale: 1000)
        image = try assetGenerator.copyCGImage(at: time, actualTime: nil)
      } catch {
        print(error)
        return (imageClassificationCategories, videoSize)
      }
      
      let uiImage = UIImage(cgImage:image)
      videoSize = uiImage.size
      if let buffer = CVPixelBuffer.buffer(from: uiImage) {
        if let result = runModel(onFrame: buffer) {
          imageClassificationCategories.append(contentsOf: result.categories)
          Task { @MainActor in
            videoDelegate?.imageClassifierService(self, didFinishClassificationOnVideoFrame: i)
          }
        }
      }
    }
    
    return (imageClassificationCategories, videoSize)
  }
}

/// A result from inference, the time it takes for inference to be
/// performed.
struct ResultBundle {
  let inferenceTime: Double
  let categories: [ClassificationCategory]
  var size: CGSize = .zero
}

struct ClassificationCategory {
  let label: String
  let score: Float
}
