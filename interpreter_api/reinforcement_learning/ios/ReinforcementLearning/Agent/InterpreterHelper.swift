// 2024 The Google AI Edge Authors. All Rights Reserved.
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
import Accelerate

class InterpreterHelper: NSObject {

  private var modelPath: String

  /// `Interpreter` object for performing inference on a given model.
  private var interpreter: Interpreter!

  var sizeWidth = 8
  var sizeHeight = 8

  // MARK: - Custom Initializer
  init(modelPath: String) {
    self.modelPath = modelPath
    super.init()

    createInterpreter()
  }

/// This function is used to create an interpreter for a given model path.
///
/// - Throws: An error if there is an issue creating the interpreter or allocating memory for the model's input `Tensor`s.
/// - Returns: None.
  private func createInterpreter() {
    do {
      // Create the `Interpreter`.
      interpreter = try Interpreter(modelPath: modelPath)
      // Allocate memory for the model's input `Tensor`s.
      try interpreter.allocateTensors()

      let input = try interpreter.input(at: 0)
      sizeWidth = input.shape.dimensions[1]
      sizeHeight = input.shape.dimensions[2]
      let output = try interpreter.output(at: 0)
      guard sizeWidth * sizeHeight == output.shape.dimensions[1] else {
        fatalError("interpreter input/output are not correct size")
      }
    } catch let error {
      fatalError("Failed to create the interpreter with error: \(error.localizedDescription)")
    }
  }

/// This function runs a machine learning model using the given input states and returns the index of the maximum output value.
///
/// - Parameters:
///   - states: An array of integers representing the input states for the model
/// - Returns: The index of the maximum output value from the model
  func runModel(states: [Int]) -> Int {
    let float32Data = states.compactMap({Float32($0)})
    let inputData = float32Data.withUnsafeBytes {Data($0)}
    do {
      try interpreter.copy(inputData, toInputAt: 0)
      try interpreter.invoke()
      let outputRensor = try interpreter.output(at: 0)
      let dataOutput = [Float32](unsafeData: outputRensor.data) ?? []
      guard dataOutput.count > 0 else { fatalError("Output wrong")}
      var max: Float = 0.0
      var index = 0
      for i in 0..<dataOutput.count {
        if states[i] != 0 { continue }
        if dataOutput[i] > max {
          max = dataOutput[i]
          index = i
        }
      }
      return index
    } catch {
      fatalError(error.localizedDescription)
    }
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
