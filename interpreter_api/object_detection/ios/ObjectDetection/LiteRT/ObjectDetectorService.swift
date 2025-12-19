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
import CoreImage
import Accelerate

/**
 This protocol must be adopted by any class that wants to take appropriate actions during  different stages of image classification on videos.
 */
protocol ObjectDetectorServiceVideoDelegate: AnyObject {
 func objectDetectorService(_ objectDetectorService: ObjectDetectorService,
                                  didFinishClassificationOnVideoFrame index: Int)
 func objectDetectorService(_ objectDetectorService: ObjectDetectorService,
                             willBeginClassification totalframeCount: Int)
}


// Initializes and calls the tflite APIs for classification.
class ObjectDetectorService: NSObject {

  weak var videoDelegate: ObjectDetectorServiceVideoDelegate?

  private let colors = [
    UIColor.black,  // 0.0 white
    UIColor.darkGray,  // 0.333 white
    UIColor.lightGray,  // 0.667 white
    UIColor.white,  // 1.0 white
    UIColor.gray,  // 0.5 white
    UIColor.red,  // 1.0, 0.0, 0.0 RGB
    UIColor.green,  // 0.0, 1.0, 0.0 RGB
    UIColor.blue,  // 0.0, 0.0, 1.0 RGB
    UIColor.cyan,  // 0.0, 1.0, 1.0 RGB
    UIColor.yellow,  // 1.0, 1.0, 0.0 RGB
    UIColor.magenta,  // 1.0, 0.0, 1.0 RGB
    UIColor.orange,  // 1.0, 0.5, 0.0 RGB
    UIColor.purple,  // 0.5, 0.0, 0.5 RGB
    UIColor.brown,  // 0.6, 0.4, 0.2 RGB
  ]

  private var scoreThreshold: Float
  private var maxResult: Int
  private var model: Model

  // MARK: - Model Parameters

  var batchSize = 1
  var inputChannels = 3
  var inputWidth = 224
  var inputHeight = 224

  /// List of labels from the given labels file.
  private var labels: [String] = []

  /// TensorFlow Lite `Interpreter` object for performing inference on a given model.
  private var interpreter: Interpreter!

  // MARK: - Custom Initializer
  init?(model: Model, scoreThreshold: Float, maxResult: Int) {
    self.model = model
    self.scoreThreshold = scoreThreshold
    self.maxResult = maxResult
    super.init()

    createObjectDetector()
  }

  private func createObjectDetector() {
    guard let modelPath = model.modelPath else {
      fatalError("Failed to load model \(model.rawValue)")
    }
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


    } catch let error {
      fatalError("Failed to create the interpreter with error: \(error.localizedDescription)")
    }
    // Load the classes listed in the labels file.
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
   This method return ObjectDetectorResult and infrenceTime when receive an image
   **/
  func detect(image: UIImage) -> Result? {

    guard let buffer = CVPixelBuffer.buffer(from: image) else { return nil }
    let result = runModel(onFrame: buffer)
    return result
  }

  func detect(
    pixelBuffer: CVPixelBuffer, completion: (Result?) -> Void) {
      let result = runModel(onFrame: pixelBuffer)
      completion(result)
  }

  func detect(
    videoAsset: AVAsset,
    durationInMilliseconds: Double,
    inferenceIntervalInMilliseconds: Double) async -> [Result?] {
    let assetGenerator = imageGenerator(with: videoAsset)

    let frameCount = Int(durationInMilliseconds / inferenceIntervalInMilliseconds)
    Task { @MainActor in
      videoDelegate?.objectDetectorService(self, willBeginClassification: frameCount)
    }

    let results = detectObjectsInFramesGenerated(
      by: assetGenerator,
      totalFrameCount: frameCount,
      atIntervalsOf: inferenceIntervalInMilliseconds)

    return results
  }

  private func imageGenerator(with videoAsset: AVAsset) -> AVAssetImageGenerator {
    let generator = AVAssetImageGenerator(asset: videoAsset)
    generator.requestedTimeToleranceBefore = CMTimeMake(value: 1, timescale: 25)
    generator.requestedTimeToleranceAfter = CMTimeMake(value: 1, timescale: 25)
    generator.appliesPreferredTrackTransform = true

    return generator
  }

  private func detectObjectsInFramesGenerated(
    by assetGenerator: AVAssetImageGenerator,
    totalFrameCount frameCount: Int,
    atIntervalsOf inferenceIntervalMs: Double)
  -> [Result?] {
    var results: [Result?] = []

    for i in 0..<frameCount {
      let timestampMs = Int(inferenceIntervalMs) * i // ms
      let image: CGImage
      do {
        let time = CMTime(value: Int64(timestampMs), timescale: 1000)
        image = try assetGenerator.copyCGImage(at: time, actualTime: nil)
      } catch {
        print(error)
        return results
      }

      let uiImage = UIImage(cgImage:image)
      if let buffer = CVPixelBuffer.buffer(from: uiImage) {
        results.append(runModel(onFrame: buffer))
        videoDelegate?.objectDetectorService(self, didFinishClassificationOnVideoFrame: i)
      }
    }
    return results
  }

  // MARK: - Internal Methods

  /// Performs image preprocessing, invokes the `Interpreter`, and processes the inference results.
  func runModel(onFrame pixelBuffer: CVPixelBuffer) -> Result? {

    let sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
    assert(sourcePixelFormat == kCVPixelFormatType_32ARGB ||
             sourcePixelFormat == kCVPixelFormatType_32BGRA ||
               sourcePixelFormat == kCVPixelFormatType_32RGBA)


    let imageChannels = 4
    assert(imageChannels >= inputChannels)

    guard let thumbnailPixelBuffer = pixelBuffer.convertToSquarePixelBuffer(outputSize: inputWidth) else { return nil }

    let interval: TimeInterval
    let boxsOutputTensor: Tensor
    let scoreOutputTensor: Tensor
    let categoriesOutputTensor: Tensor
    do {
      let inputTensor = try interpreter.input(at: 0)

      // Remove the alpha component from the image buffer to get the RGB data.
      guard let rgbData = rgbDataFromBuffer(
        thumbnailPixelBuffer,
        byteCount: batchSize * inputWidth * inputHeight * inputChannels,
        isModelQuantized: inputTensor.dataType == .uInt8
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
      boxsOutputTensor = try interpreter.output(at: 0)
      categoriesOutputTensor = try interpreter.output(at: 1)
      scoreOutputTensor = try interpreter.output(at: 2)
    } catch let error {
      print("Failed to invoke the interpreter with error: \(error.localizedDescription)")
      return nil
    }

    guard let boxs = getFloatsData(tensor: boxsOutputTensor),
          let categories = getFloatsData(tensor: categoriesOutputTensor),
          let scores = getFloatsData(tensor: scoreOutputTensor) else { return nil }

    // Process the results.
    let topNObject = getTopN(boxs: boxs, categories: categories, scores: scores)

    // Return the inference time and inference results.
    return Result(inferenceTime: interval, objects: topNObject)
  }

  // MARK: - Private Methods

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

  /// Returns the top N inference results sorted in descending order.
  private func getTopN(boxs: [Float], categories: [Float], scores: [Float]) -> [DetectionObject] {
    var objects: [DetectionObject] = []
    for i in 0 ..< categories.count {
      if scores[i] > scoreThreshold {
        let box = CGRect(origin: CGPoint(x: CGFloat(boxs[i * 4 + 1]), y: CGFloat(boxs[i * 4])),
                       size: CGSize(width: CGFloat(boxs[i * 4 + 3] - boxs[i * 4 + 1]), height: CGFloat(boxs[i * 4 + 2] - boxs[i * 4])))
        let displayColor = colors[Int(categories[i]) % colors.count]
        let object = DetectionObject(color: displayColor,
                                     boundingBox: box,
                                     categoryLabel: labels[Int(categories[i])],
                                     score: scores[i])
        objects.append(object)
      }
    }
    let sortedObjects = objects
      .sorted { $0.score > $1.score }.prefix(maxResult)
    return Array(sortedObjects)
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
    let bytes = Array<UInt8>(unsafeData: byteData)!
    var floats = [Float]()
    for i in 0..<bytes.count {
        floats.append(Float(bytes[i]) / 255.0)
    }
    return Data(copyingBufferOf: floats)
  }
}

// MARK: - Extensions

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
  let objects: [DetectionObject]
}

struct DetectionObject {
  let color: UIColor
  let boundingBox: CGRect
  let categoryLabel: String
  let score: Float

}

/// An inference from invoking the `Interpreter`.
struct Inference {
  let confidence: Float
  let label: String
}

extension UIImage {
  func resized(to size: CGSize) -> UIImage {
    return UIGraphicsImageRenderer(size: size).image { _ in
      draw(in: CGRect(origin: .zero, size: size))
    }
  }
}
