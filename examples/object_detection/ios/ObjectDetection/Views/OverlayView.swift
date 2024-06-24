// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

    var offSetx: CGFloat = 0
    var offSety: CGFloat = 0
    var transformWidth = bounds.size.width
    var transformHeight = bounds.size.height
    switch scale {
    case .scaleAspectFill:
      if imageSize.width / imageSize.height > bounds.width / bounds.height {
        transformWidth = bounds.height * imageSize.width / imageSize.height
        offSetx = (transformWidth - bounds.width) / 2
      } else {
        transformHeight = bounds.width * imageSize.height / imageSize.width
        offSety = (transformHeight - bounds.height) / 2
      }
    case .scaleAspectFit:
      if imageSize.width / imageSize.height > bounds.width / bounds.height {
        transformHeight = bounds.width * imageSize.height / imageSize.width
        offSety = (transformHeight - bounds.height) / 2
      } else {
        transformWidth = bounds.height * imageSize.width / imageSize.height
        offSetx = (transformWidth - bounds.width) / 2
      }
    default:
      // Do not use now
      break
    }

    var objectOverlays: [ObjectOverlay] = []
    for detection in detections {
      // Translates bounding box rect to current view.
      var convertedRect = detection.boundingBox.applying(
        CGAffineTransform(
          scaleX: transformWidth,
          y: transformHeight))

      convertedRect.origin.x -= offSetx
      convertedRect.origin.y -= offSety

      if convertedRect.origin.x < 0 {
        convertedRect.size.width = max(1, convertedRect.size.width - convertedRect.origin.x)
        convertedRect.origin.x = 0
      }

      if convertedRect.origin.y < 0 {
        convertedRect.size.height = max(1, convertedRect.size.height - convertedRect.origin.y)
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

  /** Calls methods to update overlay view with detected bounding boxes and class names.
   */
  func draw(objectOverlays: [ObjectOverlay]) {

    self.objectOverlays = objectOverlays
    setNeedsDisplay()
  }

}
