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

import TensorFlowLite

fileprivate let errorDomain = "org.liteRT.examples"

protocol AudioClassificationHelperDelegate: AnyObject {
  func audioClassificationHelper(
    _ audioClassificationHelper: AudioClassificationHelper,
    didfinishClassfification result: Result)
}

/// Stores results for a particular audio snipprt that was successfully classified.
struct Result {
  let inferenceTime: Double
  let categories: [ClassificationCategory]
}

struct ClassificationCategory {
  let label: String
  let score: Float
}

/// This class handles all data preprocessing and makes calls to run inference on a audio snippet
/// by invoking the Task Library's `AudioClassifier`.
class AudioClassificationHelper {

  weak var delegate: AudioClassificationHelperDelegate?

  // MARK: Private properties
  private var model: Model

  /// TensorFlow Lite `Interpreter` object for performing inference on a given model.
  private var interpreter: Interpreter!

  /// An object to continously record audio using the device's microphone.
  private let audioBufferInputTensorIndex: Int = 0

  /// Sample rate for input sound buffer. Caution: generally this value is a bit less than 1 second's audio sample.
  private(set) var sampleRate = 0
  /// Lable names described in the lable file
  private(set) var labelNames: [String] = []

  private var scoreThreshold: Float = 0
  private var maxResults: Int = 0

  // MARK: - Initialization

  /// A failable initializer for `AudioClassificationHelper`. A new instance is created if the model
  /// is successfully loaded from the app's main bundle.
  init?(model: Model, scoreThreshold: Float, maxResults: Int) {
    self.model = model
    // Construct the path to the model file.
    guard let modelPath = model.modelPath else {
      print("Failed to load the model file \(model.rawValue)")
      return nil
    }
    self.scoreThreshold = scoreThreshold
    self.maxResults = maxResults
    createInterpreter(modelPath: modelPath)
    labelNames = loadLabels(labelPath: model.labelPath)
  }

  private func createInterpreter(modelPath: String) {
    do {
      interpreter = try Interpreter(modelPath: modelPath)
      try interpreter.allocateTensors()
      let inputShape = try interpreter.input(at: 0).shape
      switch model {
      case .Yamnet:
        sampleRate = inputShape.dimensions[0]
      case .speechCommand:
        sampleRate = inputShape.dimensions[1]
      }
      try interpreter.invoke()
    } catch {
      fatalError("Can not create interpreter")
    }
  }

  private func loadLabels(labelPath: String?) -> [String] {
    guard let labelPath = labelPath else { return [] }

    var content = ""
    do {
      content = try String(contentsOfFile: labelPath, encoding: .utf8)
      let labels = content.components(separatedBy: "\n")
        .filter { !$0.isEmpty }
      return labels
    } catch {
      print("Failed to load label content: '\(content)' with error: \(error.localizedDescription)")
      return []
    }
  }

  /// Invokes the `Interpreter` and processes and returns the inference results.
  public func start(inputBuffer: [Int16]) {
    do {
      let startTime = Date().timeIntervalSince1970
      let audioBufferData = int16ArrayToData(inputBuffer)
      try interpreter.copy(audioBufferData, toInputAt: audioBufferInputTensorIndex)
      try interpreter.invoke()
      let inferenceTime = Date().timeIntervalSince1970 - startTime
      let outputTensor = try interpreter.output(at: 0)
      // Gets the formatted and averaged results.
      let probabilities = dataToFloatArray(outputTensor.data) ?? []
      var classificationCategories: [ClassificationCategory] = []
      for (index, probability) in probabilities.enumerated() {
        if probability > scoreThreshold {
          classificationCategories.append(ClassificationCategory(label: labelNames[index], score: probability))
        }
      }
      classificationCategories = Array(classificationCategories.sorted(by: {$0.score > $1.score}).prefix(self.maxResults))
      let result = Result(inferenceTime: inferenceTime, categories: classificationCategories)
      delegate?.audioClassificationHelper(self, didfinishClassfification: result)
    } catch let error {
      print(">>> Failed to invoke the interpreter with error: \(error.localizedDescription)")
      return
    }
  }

  /// Creates a new buffer by copying the buffer pointer of the given `Int16` array.
  private func int16ArrayToData(_ buffer: [Int16]) -> Data {
    let floatData = buffer.map { Float($0) / Float(Int16.max) }
    return floatData.withUnsafeBufferPointer(Data.init)
  }

  /// Creates a new array from the bytes of the given unsafe data.
  /// - Returns: `nil` if `unsafeData.count` is not a multiple of `MemoryLayout<Float>.stride`.
  private func dataToFloatArray(_ data: Data) -> [Float]? {
    guard data.count % MemoryLayout<Float>.stride == 0 else { return nil }

    #if swift(>=5.0)
    return data.withUnsafeBytes { .init($0.bindMemory(to: Float.self)) }
    #else
    return data.withUnsafeBytes {
      .init(UnsafeBufferPointer<Float>(
        start: $0,
        count: unsafeData.count / MemoryLayout<Element>.stride
      ))
    }
    #endif // swift(>=5.0)
  }

}
