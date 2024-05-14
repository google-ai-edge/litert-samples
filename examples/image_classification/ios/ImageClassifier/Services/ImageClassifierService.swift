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

  var imageClassifier: ImageClassifier?
  private var scoreThreshold: Float
  private var maxResult: Int
  private var modelPath: String

  // MARK: - Custom Initializer
  init?(model: Model, scoreThreshold: Float, maxResult: Int) {
    // Construct the path to the model file.
    guard let modelPath = model.modelPath else {
      print("Failed to load the model : \(model.rawValue).")
      return nil
    }
    self.modelPath = modelPath
    self.scoreThreshold = scoreThreshold
    self.maxResult = maxResult
    super.init()

    createImageClassifier()
  }

  private func createImageClassifier() {
    let options = ImageClassifierOptions(modelPath: modelPath)
    options.classificationOptions.maxResults = maxResult
    options.classificationOptions.scoreThreshold = scoreThreshold
    do {
      imageClassifier = try ImageClassifier.classifier(options: options)
    } catch {
      print(error)
    }
  }

  // MARK: - Classification Methods
  /**
   This method return ImageClassifierResult and infrenceTime when receive an image
   **/
  func classify(image: UIImage) -> ResultBundle? {
    var newImage = image
    if image.size.width > 2000 {
      let newSize = CGSize(width: 2000, height: 2000 / image.size.width * image.size.height)
      newImage = image.resized(to: newSize)
    }
    guard let mlImage = MLImage(image: newImage) else {
      return nil
    }
    do {
      let startDate = Date()
      let result = try imageClassifier?.classify(mlImage: mlImage)
      let inferenceTime = Date().timeIntervalSince(startDate) * 1000
      return ResultBundle(inferenceTime: inferenceTime, imageClassifierResults: [result])
    } catch {
        print(error)
        return nil
    }
  }

  func classify(
    sampleBuffer: CMSampleBuffer, completion: (ResultBundle?) -> Void) {
    guard let image = MLImage(sampleBuffer: sampleBuffer) else {
      completion(nil)
      return
    }
    do {
      let startDate = Date()
      let result = try imageClassifier?.classify(mlImage: image)
      let inferenceTime = Date().timeIntervalSince(startDate) * 1000
      let resultBundle = ResultBundle(inferenceTime: inferenceTime, imageClassifierResults: [result])
      completion(resultBundle)
    } catch {
      print(error)
      completion(nil)
    }
  }

  func classify(
    videoAsset: AVAsset,
    durationInMilliseconds: Double,
    inferenceIntervalInMilliseconds: Double) async -> ResultBundle? {
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

    return ResultBundle(
      inferenceTime: Date().timeIntervalSince(startDate) / Double(frameCount) * 1000,
      imageClassifierResults: imageClassifierResultTuple.imageClassifierResults,
      size: imageClassifierResultTuple.videoSize)
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
  -> (imageClassifierResults: [ClassificationResult?], videoSize: CGSize)  {
    var imageClassifierResults: [ClassificationResult?] = []
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
        return (imageClassifierResults, videoSize)
      }

      let uiImage = UIImage(cgImage:image)
      videoSize = uiImage.size
      guard let mlImage = MLImage(image: uiImage) else { return (imageClassifierResults, videoSize) }
      do {
        let result = try imageClassifier?.classify(mlImage: mlImage)
          imageClassifierResults.append(result)
        Task { @MainActor in
          videoDelegate?.imageClassifierService(self, didFinishClassificationOnVideoFrame: i)
        }
        } catch {
          print(error)
        }
      }

    return (imageClassifierResults, videoSize)
  }
}

/// A result from inference, the time it takes for inference to be
/// performed.
struct ResultBundle {
  let inferenceTime: Double
  let imageClassifierResults: [ClassificationResult?]
  var size: CGSize = .zero
}

extension UIImage {
  func resized(to size: CGSize) -> UIImage {
    return UIGraphicsImageRenderer(size: size).image { _ in
      draw(in: CGRect(origin: .zero, size: size))
    }
  }
}
