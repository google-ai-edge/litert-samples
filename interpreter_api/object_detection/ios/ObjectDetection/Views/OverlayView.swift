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

/**
 This structure holds the display parameters for the overlay to be drawon on a detected object.
 */
struct ObjectOverlay {
  let name: String
  let borderRect: CGRect
  let nameStringSize: CGSize
  let color: UIColor
  let font: UIFont
}

/**
 This UIView draws overlay on a detected object.
 */
class OverlayView: UIView {

  var objectOverlays: [ObjectOverlay] = []
  private let cornerRadius: CGFloat = 10.0
  private let stringBgAlpha: CGFloat
    = 0.7
  private let lineWidth: CGFloat = 3
  private let stringFontColor = UIColor.white
  private let stringHorizontalSpacing: CGFloat = 13.0
  private let stringVerticalSpacing: CGFloat = 7.0

  private let displayFont = UIFont.systemFont(ofSize: 14.0, weight: .medium)

  override func draw(_ rect: CGRect) {

    // Drawing code
    for objectOverlay in objectOverlays {

      drawBorders(of: objectOverlay)
      drawBackground(of: objectOverlay)
      drawName(of: objectOverlay)
    }
  }

  /**
   This method draws the borders of the detected objects.
   */
  func drawBorders(of objectOverlay: ObjectOverlay) {

    let path = UIBezierPath(rect: objectOverlay.borderRect)
    path.lineWidth = lineWidth
    objectOverlay.color.setStroke()

    path.stroke()
  }

  /**
   This method draws the background of the string.
   */
  func drawBackground(of objectOverlay: ObjectOverlay) {

    let stringBgRect = CGRect(x: objectOverlay.borderRect.origin.x, y: objectOverlay.borderRect.origin.y , width: 2 * stringHorizontalSpacing + objectOverlay.nameStringSize.width, height: 2 * stringVerticalSpacing + objectOverlay.nameStringSize.height
    )

    let stringBgPath = UIBezierPath(rect: stringBgRect)
    objectOverlay.color.withAlphaComponent(stringBgAlpha).setFill()
    stringBgPath.fill()
  }

  /**
   This method draws the name of object overlay.
   */
  func drawName(of objectOverlay: ObjectOverlay) {

    // Draws the string.
    let stringRect = CGRect(x: objectOverlay.borderRect.origin.x + stringHorizontalSpacing, y: objectOverlay.borderRect.origin.y + stringVerticalSpacing, width: objectOverlay.nameStringSize.width, height: objectOverlay.nameStringSize.height)

    let attributedString = NSAttributedString(string: objectOverlay.name, attributes: [NSAttributedString.Key.foregroundColor : stringFontColor, NSAttributedString.Key.font : objectOverlay.font])
    attributedString.draw(in: stringRect)
  }

  /**
   This method takes the results, translates the bounding box rects to the current view, draws the bounding boxes, classNames and confidence scores of inferences.
   */
  func drawAfterPerformingCalculations(
    onDetections detections: [DetectionObject], withImageSize imageSize: CGSize, scale: UIView.ContentMode
  ) {

    self.objectOverlays = []
    self.setNeedsDisplay()

    guard !detections.isEmpty else {
      return
    }

    // Calculate the offsets and scale factor for the bounding boxes to correctly scale and
    // translate them into the overlay view based on its content mode.
    let offsetsAndScaleFactor = OverlayView.offsetsAndScaleFactor(
      forImageOfSize: imageSize,
      tobeDrawnInViewOfSize: self.bounds.size,
      withContentMode: scale)

    // All models used in this example expect a square image (width = height) of certain dimension
    // as the input. To satisfy this requirement, we scaled the input image down, preserving its
    // aspect ratio to fit the model dimensions and centred its pixels within the pixels sent to
    // the model for inference.
    // The following code calculates the offsets by which the image was translated w.r.t the
    // original image size. These values will be used to reverse this translation in the transforms
    // applied later.
    //
    // The calculations are considered within the largest square that can fit the entire input image
    // i.e, side of square = `max(imageSize.width, imageSize.Height)` since the bounding box
    // coordinates output by the model are normalized. The offsets are later scaled to the bounds
    // of the overlay view.
    let maxImageDimension = max(imageSize.width, imageSize.height)

    let offsetToReverseModelInputTranslationX =
      (maxImageDimension - imageSize.width) * offsetsAndScaleFactor.scaleFactor / 2
    let offsetToReverseModelInputTranslationY =
      (maxImageDimension - imageSize.height) * offsetsAndScaleFactor.scaleFactor / 2

    var objectOverlays: [ObjectOverlay] = []
    for detection in detections {
      // The bounding boxes are first scaled to the smallest square that can completely fit the
      // image. Bounding boxes need not be scaled down to the original image size since the valid
      // pixels are of size `imageSize` and centered within the smallest square that can completely
      // fit the imag.
      // They are then scaled to the overlay view's bounds.
      //
      // The scaled bounding boxes are then translated to the correct origin within the bounds of
      // the overlay view. Note that, the translation done while processing the image for the model
      // is also reversed.
      var convertedRect = detection.boundingBox
        .applying(
          CGAffineTransform(
            scaleX: maxImageDimension * offsetsAndScaleFactor.scaleFactor,
            y: maxImageDimension * offsetsAndScaleFactor.scaleFactor)
        )
        .applying(
          CGAffineTransform(
            translationX: offsetsAndScaleFactor.xOffset - offsetToReverseModelInputTranslationX,
            y: offsetsAndScaleFactor.yOffset - offsetToReverseModelInputTranslationY)
        )

      // Adjust the bouding boxes that extend beyond the bounds of the overlay view to fall within
      // the overlay view.
      if convertedRect.origin.x < 0 {
        convertedRect.size.width = max(0, convertedRect.size.width + convertedRect.origin.x)
        convertedRect.origin.x = 0
      }

      if convertedRect.origin.y < 0 {
        convertedRect.size.height = max(0, convertedRect.size.height + convertedRect.origin.y)
        convertedRect.origin.y = 0
      }

      if convertedRect.maxY > bounds.maxY {
        convertedRect.size.height =
          bounds.maxY - convertedRect.origin.y
      }

      if convertedRect.maxX > bounds.maxX {
        convertedRect.size.width =
          bounds.maxX - convertedRect.origin.x
      }

      let objectDescription = String(
        format: "\(detection.categoryLabel) (%.2f)",
        detection.score)

      let size = objectDescription.size(withAttributes: [.font: self.displayFont])

      let objectOverlay = ObjectOverlay(
        name: objectDescription, borderRect: convertedRect, nameStringSize: size,
        color: detection.color,
        font: self.displayFont)

      objectOverlays.append(objectOverlay)
    }

    // Hands off drawing to the OverlayView
    self.draw(objectOverlays: objectOverlays)
  }

  // MARK: Helper Functions
    static func offsetsAndScaleFactor(
      forImageOfSize imageSize: CGSize,
      tobeDrawnInViewOfSize viewSize: CGSize,
      withContentMode contentMode: UIView.ContentMode
    )
      -> (xOffset: CGFloat, yOffset: CGFloat, scaleFactor: Double)
    {

      let widthScale = viewSize.width / imageSize.width
      let heightScale = viewSize.height / imageSize.height

      var scaleFactor = 0.0

      switch contentMode {
      case .scaleAspectFill:
        scaleFactor = max(widthScale, heightScale)
      case .scaleAspectFit:
        scaleFactor = min(widthScale, heightScale)
      default:
        scaleFactor = 1.0
      }

      let scaledSize = CGSize(
        width: imageSize.width * scaleFactor,
        height: imageSize.height * scaleFactor)
      let xOffset = (viewSize.width - scaledSize.width) / 2
      let yOffset = (viewSize.height - scaledSize.height) / 2

      return (xOffset, yOffset, scaleFactor)
    }

  /** Calls methods to update overlay view with detected bounding boxes and class names.
   */
  func draw(objectOverlays: [ObjectOverlay]) {

    self.objectOverlays = objectOverlays
    setNeedsDisplay()
  }

}
