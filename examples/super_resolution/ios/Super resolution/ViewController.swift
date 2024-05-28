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
import AVKit

class ViewController: UIViewController {

  // MARK: Constants
  private struct Constants {
    static let milliSeconds = 1000.0
    static let savedPhotosNotAvailableText = "Saved photos album is not available."
    static let mediaEmptyText = "Click + to add an image to begin running recovering a high resolution"
  }

  @IBOutlet weak var regionView: UIView!
  @IBOutlet weak var imageEmptyLabel: UILabel!
  @IBOutlet weak var inferenceTimeLabel: UILabel!
  @IBOutlet weak var pickedImageView: UIImageView!
  @IBOutlet weak var cropedImageView: UIImageView!
  @IBOutlet weak var pickFromGalleryButton: UIButton!
  @IBOutlet weak var switchShowImageButton: UIButton!

  private var interprenterHelper: InterpreterHelper?

  private var pickedImage: UIImage?
  private var cropedImage: UIImage?
  private var highResolutionImage: UIImage?
  private lazy var pickerController = UIImagePickerController()



  override func viewDidLoad() {
    super.viewDidLoad()

    interprenterHelper = InterpreterHelper(modelPath: DefaultConstants.modelPath)

    guard UIImagePickerController.isSourceTypeAvailable(.savedPhotosAlbum) else {
      pickFromGalleryButton.isEnabled = false
      imageEmptyLabel.text = Constants.savedPhotosNotAvailableText
      return
    }
    pickFromGalleryButton.isEnabled = true
    imageEmptyLabel.text = Constants.mediaEmptyText
    setupUI()
    addTouch()
  }

  private func setupUI() {
    cropedImageView.layer.borderWidth = 1.0
    cropedImageView.layer.borderColor = DefaultConstants.orangeColor.cgColor
    regionView.layer.borderWidth = 1.0
    regionView.layer.borderColor = DefaultConstants.orangeColor.cgColor
    regionView.isHidden = true
    switchShowImageButton.isHidden = true
    cropedImageView.isHidden = true
  }

  private func addTouch() {
    let tapRecognizer = UITapGestureRecognizer(target: self, action: #selector(onTapScreen))
    pickedImageView.addGestureRecognizer(tapRecognizer)
  }

  private func clear() {
    regionView.isHidden = true
    switchShowImageButton.isHidden = true
    cropedImageView.isHidden = true
    cropedImage = nil
    highResolutionImage = nil
    pickedImage = nil
    pickedImageView.image = nil
  }

  @objc func onTapScreen(_ sender: UITapGestureRecognizer) {
    let point = sender.location(in: pickedImageView)
    cropImageAndProcessing(point: point)
  }

  private func configurePickerController() {
    pickerController.delegate = self
    pickerController.sourceType = .savedPhotosAlbum
    pickerController.mediaTypes = [UTType.image.identifier]
    pickerController.allowsEditing = false
  }

  private func getCropedImage(image: UIImage, centerInView: CGPoint) -> UIImage? {
    var origin: CGPoint!
    let imageSize = image.size
    if imageSize.width / imageSize.height > pickedImageView.bounds.width / pickedImageView.bounds.height {
      let newW = imageSize.width
      let newH = pickedImageView.bounds.height / pickedImageView.bounds.width * imageSize.width
      origin = CGPoint(x: centerInView.x * (newW / pickedImageView.bounds.width) - 25, y: centerInView.y * (newH / pickedImageView.bounds.height) - (newH - imageSize.height)/2 - 25)
    } else {
      let newH = imageSize.height
      let newW = pickedImageView.bounds.width / pickedImageView.bounds.height * imageSize.height
      origin = CGPoint(x: centerInView.x * (newW / pickedImageView.bounds.width) - (newW - imageSize.width)/2 - 25, y: centerInView.y * (newH / pickedImageView.bounds.height) - 25)
    }
    if origin.x < 0 {
      origin.x = 0
    } else if origin.x + 50 > imageSize.width {
      origin.x = imageSize.width - 50
    }

    if origin.y < 0 {
      origin.y = 0
    } else if origin.y + 50 > imageSize.height {
      origin.y = imageSize.height - 50
    }
    return image.croppedFixedOrientation(boundingBox: CGRect(x: origin.x, y: origin.y, width: 50, height: 50))
  }

  private func focusToCroppedImage(point: CGPoint) {
    guard let image = pickedImage else { return }
    regionView.isHidden = false
    let imageSize = image.size
    var imageOrigin: CGPoint!
    var size: CGSize!
    var newWidth = pickedImageView.bounds.width
    var newHeight = pickedImageView.bounds.height
    if imageSize.width / imageSize.height > pickedImageView.bounds.width / pickedImageView.bounds.height {
      size = CGSize(width: 50.0 * pickedImageView.bounds.width / imageSize.width,
                            height: 50.0 * pickedImageView.bounds.width / imageSize.width)
      newHeight = imageSize.height * (pickedImageView.bounds.width / imageSize.width)
      imageOrigin = CGPoint(x: 0,
                       y: (pickedImageView.bounds.height - newHeight)/2)
    } else {
      size = CGSize(width: 50.0 * pickedImageView.bounds.height / imageSize.height,
                            height: 50.0 * pickedImageView.bounds.height / imageSize.height)
      newWidth = imageSize.width * (pickedImageView.bounds.height / imageSize.height)
      imageOrigin = CGPoint(x: (pickedImageView.bounds.width - newWidth)/2,
                       y: 0)
    }

    var regionOrigin: CGPoint = CGPoint(x: point.x - size.width/2, y: point.y - size.height/2)

    if regionOrigin.x < imageOrigin.x {
      regionOrigin.x = imageOrigin.x
    } else if regionOrigin.x + size.width > imageOrigin.x + newWidth {
      regionOrigin.x = imageOrigin.x + newWidth - size.width
    }

    if regionOrigin.y < imageOrigin.y {
      regionOrigin.y = imageOrigin.y
    } else if regionOrigin.y + size.height > imageOrigin.y + newHeight {
      regionOrigin.y = imageOrigin.y + newHeight  - size.height
    }

    regionOrigin.x = regionOrigin.x + pickedImageView.frame.origin.x
    regionOrigin.y = regionOrigin.y + pickedImageView.frame.origin.y
    regionView.frame = CGRect(origin: regionOrigin, size: size)
  }

  private func cropImageAndProcessing(point: CGPoint) {

    guard let pickedImage = pickedImage,
          let cropedImage = getCropedImage(image: pickedImage, centerInView: point) else { return }
    focusToCroppedImage(point: point)
    self.cropedImage = cropedImage
    cropedImageView.image = cropedImage
    switchShowImageButton.isSelected = false
    switchShowImageButton.isHidden = false
    cropedImageView.isHidden = false
    DispatchQueue.global(qos: .userInteractive).async { [weak self] in
      guard let self = self,
            let result = self
        .interprenterHelper?
        .proccess(image: cropedImage) else {
        return
      }
      highResolutionImage = result.image
      DispatchQueue.main.async {
        self.cropedImageView.image = self.highResolutionImage
        self.switchShowImageButton.isSelected = true
      }
    }
  }

  // Mark: IBAction
  @IBAction func pickFromGalleryButtonTouchupInside(_ sender: UIButton) {
    configurePickerController()
    present(pickerController, animated: true)
  }

  @IBAction func switchButtonTouchUpInside(_ sender: Any) {
    let image = switchShowImageButton.isSelected ? cropedImage : highResolutionImage
    if let image = image {
      cropedImageView.image = image
      switchShowImageButton.isSelected.toggle()
    }
  }
}

// MARK: UIImagePickerControllerDelegate, UINavigationControllerDelegate
extension ViewController: UIImagePickerControllerDelegate, UINavigationControllerDelegate {

  func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
    picker.dismiss(animated: true)
  }

  func imagePickerController(
    _ picker: UIImagePickerController,
    didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
    clear()
    picker.dismiss(animated: true)
    guard let mediaType = info[.mediaType] as? String else {
      return
    }

    switch mediaType {
    case UTType.image.identifier:
      guard let image = info[.originalImage] as? UIImage else {
        imageEmptyLabel.isHidden = false
        break
      }
      pickedImageView.image = image
      pickedImage = image
      imageEmptyLabel.isHidden = true
      let center = CGPoint(x: pickedImageView.bounds.width / 2, y: pickedImageView.bounds.height / 2)
      cropImageAndProcessing(point: center)
    default:
      break
    }
  }

}

extension UIImage {
  func croppedFixedOrientation(boundingBox: CGRect) -> UIImage? {
    guard let cgImage = self.cgImage else {
      //CGImage is not available
      return nil
    }
    guard imageOrientation != UIImage.Orientation.up else {
      //This is default orientation, don't need to do anything
      guard let cgImage = self.cgImage?.cropping(to: boundingBox) else {
        return nil
      }
      return UIImage(cgImage: cgImage)
    }
    guard let colorSpace = cgImage.colorSpace, let ctx = CGContext(data: nil, width: Int(size.width), height: Int(size.height), bitsPerComponent: cgImage.bitsPerComponent, bytesPerRow: 0, space: colorSpace, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
      return nil //Not able to create CGContext
    }

    var transform: CGAffineTransform = CGAffineTransform.identity

    switch imageOrientation {
    case .down, .downMirrored:
      transform = transform.translatedBy(x: size.width, y: size.height)
      transform = transform.rotated(by: CGFloat.pi)
      break
    case .left, .leftMirrored:
      transform = transform.translatedBy(x: size.width, y: 0)
      transform = transform.rotated(by: CGFloat.pi / 2.0)
      break
    case .right, .rightMirrored:
      transform = transform.translatedBy(x: 0, y: size.height)
      transform = transform.rotated(by: CGFloat.pi / -2.0)
      break
    default:
      break
    }

    //Flip image one more time if needed to, this is to prevent flipped image
    switch imageOrientation {
    case .upMirrored, .downMirrored:
      transform = transform.translatedBy(x: size.width, y: 0)
      transform = transform.scaledBy(x: -1, y: 1)
      break
    case .leftMirrored, .rightMirrored:
      transform = transform.translatedBy(x: size.height, y: 0)
      transform = transform.scaledBy(x: -1, y: 1)
    default:
      break
    }

    ctx.concatenate(transform)

    switch imageOrientation {
    case .left, .leftMirrored, .right, .rightMirrored:
      ctx.draw(self.cgImage!, in: CGRect(x: 0, y: 0, width: size.height, height: size.width))
    default:
      ctx.draw(self.cgImage!, in: CGRect(x: 0, y: 0, width: size.width, height: size.height))
      break
    }

    guard let newCGImage = ctx.makeImage() else { return nil }
    guard let cgImage = newCGImage.cropping(to: boundingBox) else {
      return nil
    }
    return UIImage(cgImage: cgImage)
  }
}
