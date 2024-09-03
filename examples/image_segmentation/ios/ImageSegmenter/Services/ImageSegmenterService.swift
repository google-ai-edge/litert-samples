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
import TensorFlowLiteTaskVision
import AVFoundation

// Initializes and calls the Tflite APIs for segmention.
class ImageSegmenterService: NSObject {

  var imageSegmenter: ImageSegmenter?
  var modelPath: String

  // MARK: - Custom Initializer
  init?(model: Model) {
    guard let modelPath = model.modelPath else { return nil }
    self.modelPath = modelPath
    super.init()

    createImageSegmenter()
  }

  private func createImageSegmenter() {
    let imageSegmenterOptions = ImageSegmenterOptions(modelPath: modelPath)

    imageSegmenterOptions.outputType = .categoryMask
    do {
      imageSegmenter = try ImageSegmenter.segmenter(options: imageSegmenterOptions)
    }
    catch {
      print(error)
    }
  }

  // MARK: - Segmention Methods for Different Modes
  /**
   This method return ImageSegmenterResult and infrenceTime when receive an image
   **/
  func segment(image: UIImage) -> ResultBundle? {
    guard let cgImage = image.fixedOrientation() else { return nil }
    let fixImage = UIImage(cgImage: cgImage, scale: 1.0, orientation: .up)
    guard let mlImage = MLImage(image: fixImage) else {
      return nil
    }
    do {
      let startDate = Date()
      let result = try imageSegmenter?.segment(mlImage: mlImage)
      let inferenceTime = Date().timeIntervalSince(startDate) * 1000
      return ResultBundle(inferenceTime: inferenceTime, imageSegmenterResults: [result])
    } catch {
      print(error)
      return nil
    }
  }

  func segmentAsync(
    sampleBuffer: CMSampleBuffer, completion: (ResultBundle?) -> Void) {
      guard let mlImage = MLImage(sampleBuffer: sampleBuffer) else {
        completion(nil)
        return
      }
      do {
        let startDate = Date()
        let result = try imageSegmenter?.segment(mlImage: mlImage)
        let inferenceTime = Date().timeIntervalSince(startDate) * 1000
        let resultBundle = ResultBundle(inferenceTime: inferenceTime, imageSegmenterResults: [result])
        completion(resultBundle)
      } catch {
        print(error)
        completion(nil)
      }
    }

  func segment(
    videoFrame: CGImage)
  -> ResultBundle?  {
    let image = UIImage(cgImage: videoFrame)
    return segment(image: image)
  }
}

/// A result from the `ImageSegmenterService`.
struct ResultBundle {
  let inferenceTime: Double
  let imageSegmenterResults: [SegmentationResult?]
  var size: CGSize = .zero
}

struct VideoFrame {
  let pixelBuffer: CVPixelBuffer
  let formatDescription: CMFormatDescription
}
