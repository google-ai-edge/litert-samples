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

import TensorFlowLite
import Foundation

class TextClassificationService {

  private var interpreter: Interpreter!
  private var tokenizer: WordpieceTokenizer!

  init(model: Model) {
    guard
      let modelPath = model.modelPath
    else {
      fatalError("Failed to load the model file: \(model.rawValue)")
    }
    let options = Interpreter.Options()
    do {
      // Create the `Interpreter`.
      interpreter = try Interpreter(modelPath: modelPath, options: options)

      // Initialize input and output `Tensor`s.
      try interpreter.allocateTensors()

    } catch {
      print(error)
    }
    tokenizer = WordpieceTokenizer(with: model)
  }

  func classify(text: String) -> ClassificationResult? {
    let inputIds = getIds(input: text)

    // MARK: - Inferencing
    let inferenceStartTime = Date()

    do {

      let inputShape = try interpreter.input(at: 0).shape
      var inputBuffer = [Int32](repeating: 0, count: inputShape.dimensions[1])
      inputBuffer.replaceSubrange(0..<inputIds.count, with: inputIds)
      let inputIdsData = Data(copyingBufferOf: inputBuffer)
      // Assign input `Data` to the `interpreter`.
      try interpreter.copy(inputIdsData, toInputAt: 0)

      try interpreter.invoke()

      // Get the output `Tensor` to process the inference results
      let outputTensor = try interpreter.output(at: 0)
      let output = outputTensor.data.toArray(type: Float32.self)
      let inferenceTime = Date().timeIntervalSince(inferenceStartTime) * 1000
//    labels: negative(0) & positive(1)
      let result = ClassificationResult(inferenceTime: inferenceTime, categories:
                                          ["negative" : output[0], "positive": output[1]])
      return result
    } catch let error {
      print(error)
      return nil
    }
  }

  private func getIds(input: String) -> [Int32] {
    let tokens = tokenizer.tokenize(input)
    return tokenizer.convertToIDs(tokens: tokens)
  }
}

enum Model: String, CaseIterable {
  case mobileBert = "Mobile Bert"
  case avgWordClassifier = "Avg Word Classifier"

  var modelPath: String? {
    switch self {
    case .mobileBert:
      return Bundle.main.path(
        forResource: "bert_classifier", ofType: "tflite")
    case .avgWordClassifier:
      return Bundle.main.path(
        forResource: "average_word_classifier", ofType: "tflite")
    }
  }

  var vocabPath: String? {
    switch self {
    case .mobileBert:
      return Bundle.main.path(
        forResource: "bert_vocab", ofType: "txt")
    case .avgWordClassifier:
      return Bundle.main.path(
        forResource: "average_vocab", ofType: "txt")
    }
  }
}

struct ClassificationResult {
  let inferenceTime: Double
  let categories: [String: Float32]
}

// MARK: - Data extension
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

  /// Convert a Data instance to Array representation.
  func toArray<T>(type: T.Type) -> [T] where T: AdditiveArithmetic {
    var array = [T](repeating: T.zero, count: self.count / MemoryLayout<T>.stride)
    _ = array.withUnsafeMutableBytes { self.copyBytes(to: $0) }
    return array
  }
}
