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

import AVKit
import UIKit
import Metal
import MetalKit
import MetalPerformanceShaders

/**
 * The view controller is responsible for performing segmention on videos or images selected by the user from the device media library and
 * presenting them with the new backgrourd of the image to the user.
 */
class MediaLibraryViewController: UIViewController {

  // MARK: Constants
  private struct Constants {
    static let edgeOffset: CGFloat = 2.0
    static let inferenceTimeIntervalInMilliseconds = 200.0
    static let milliSeconds = 1000.0
    static let savedPhotosNotAvailableText = "Saved photos album is not available."
    static let mediaEmptyText =
    "Click + to add an image to begin running the image sengmentation."
    static let captureImageEmptyText =
    "Click + to take a photo to begin running the image sengmentation."
    static let pickFromGalleryButtonInset: CGFloat = 10.0
  }
  // MARK: Face Segmenter Service
  weak var inferenceResultDeliveryDelegate: InferenceResultDeliveryDelegate?

  // MARK: Controllers that manage functionality
  private lazy var pickerController = UIImagePickerController()
  private var playerViewController: AVPlayerViewController?

  private let backgroundQueue = DispatchQueue(label: "com.google.cameraController.backgroundQueue")
  private let imageSegmenterServiceQueue = DispatchQueue(
    label: "com.google.cameraController.imageSegmenterServiceQueue",
    attributes: .concurrent)
  private var selectImage: UIImage?

  // MARK: Image Segmenter Service
  private var imageSegmenterService: ImageSegmenterService?

  private let render = SegmentedImageRenderer()
  private var isRunning = false
  var imageSourceType: UIImagePickerController.SourceType = .camera

  // MARK: Storyboards Connections
  @IBOutlet weak var pickFromGalleryButton: UIButton!
  @IBOutlet weak var progressView: UIProgressView!
  @IBOutlet weak var imageEmptyLabel: UILabel!
  @IBOutlet weak var pickedImageView: UIImageView!
  @IBOutlet weak var pickFromGalleryButtonBottomSpace: NSLayoutConstraint!

  override func viewDidLoad() {
    super.viewDidLoad()
    startObserveConfigChanges()
  }

  override func viewWillAppear(_ animated: Bool) {
    super.viewWillAppear(animated)

    guard UIImagePickerController.isSourceTypeAvailable(.savedPhotosAlbum) else {
      pickFromGalleryButton.isEnabled = false
      self.imageEmptyLabel.text = Constants.savedPhotosNotAvailableText
      return
    }
    pickFromGalleryButton.isEnabled = true
    if imageSourceType == .camera {
      self.imageEmptyLabel.text = Constants.captureImageEmptyText
    } else {
      self.imageEmptyLabel.text = Constants.mediaEmptyText
    }
  }

  override func viewWillDisappear(_ animated: Bool) {
    super.viewWillDisappear(animated)
    imageSegmenterService = nil
  }

  deinit {
    playerViewController?.player?.removeTimeObserver(self)
  }

  @IBAction func onClickPickFromGallery(_ sender: Any) {
    configurePickerController()
    present(pickerController, animated: true)
  }

  private func startObserveConfigChanges() {
    NotificationCenter.default
      .addObserver(self,
                   selector: #selector(initializeImageSegmenterServiceAndSegment),
                   name: InferenceConfigurationManager.notificationName,
                   object: nil)
  }

  private func stopObserveConfigChanges() {
    NotificationCenter.default.removeObserver(self)
  }

  private func configurePickerController() {
    pickerController.delegate = self
    pickerController.sourceType = imageSourceType
    pickerController.mediaTypes = [UTType.image.identifier]
    pickerController.allowsEditing = false
  }

  private func addPlayerViewControllerAsChild() {
    guard let playerViewController = playerViewController else {
      return
    }
    playerViewController.view.translatesAutoresizingMaskIntoConstraints = false

    self.addChild(playerViewController)
    self.view.addSubview(playerViewController.view)
    self.view.bringSubviewToFront(self.pickFromGalleryButton)
    NSLayoutConstraint.activate([
      playerViewController.view.leadingAnchor.constraint(
        equalTo: view.leadingAnchor, constant: 0.0),
      playerViewController.view.trailingAnchor.constraint(
        equalTo: view.trailingAnchor, constant: 0.0),
      playerViewController.view.topAnchor.constraint(
        equalTo: view.topAnchor, constant: 0.0),
      playerViewController.view.bottomAnchor.constraint(
        equalTo: view.bottomAnchor, constant: 0.0)
    ])
    playerViewController.didMove(toParent: self)
  }

  private func removePlayerViewController() {
    defer {
      playerViewController?.view.removeFromSuperview()
      playerViewController?.willMove(toParent: nil)
      playerViewController?.removeFromParent()
    }

    playerViewController?.player?.pause()
    playerViewController?.player = nil
  }

  private func openMediaLibrary() {
    configurePickerController()
    present(pickerController, animated: true)
  }

  private func showProgressView() {
    guard let progressSuperview = progressView.superview?.superview else {
      return
    }
    progressSuperview.isHidden = false
    progressView.progress = 0.0
    progressView.observedProgress = nil
    self.view.bringSubviewToFront(progressSuperview)
  }

  private func hideProgressView() {
    guard let progressSuperview = progressView.superview?.superview else {
      return
    }
    self.view.sendSubviewToBack(progressSuperview)
    self.progressView.superview?.superview?.isHidden = true
  }

  func layoutUIElements(withInferenceViewHeight height: CGFloat) {
    pickFromGalleryButtonBottomSpace.constant =
    height + Constants.pickFromGalleryButtonInset
    view.layoutSubviews()
  }
}

extension MediaLibraryViewController: UIImagePickerControllerDelegate, UINavigationControllerDelegate {

  func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
    picker.dismiss(animated: true)
  }

  func imagePickerController(
    _ picker: UIImagePickerController,
    didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
      pickedImageView.image = nil

      picker.dismiss(animated: true)

      guard let mediaType = info[.mediaType] as? String else {
        return
      }
      render.reset()

      switch mediaType {
      case UTType.image.identifier:
        guard let image = info[.originalImage] as? UIImage else {
          imageEmptyLabel.isHidden = false
          break
        }
        imageEmptyLabel.isHidden = true
        selectImage = image
        initializeImageSegmenterServiceAndSegment()
      default:
        break
      }
    }

  private func imageGenerator(with videoAsset: AVAsset) -> AVAssetImageGenerator {
    let generator = AVAssetImageGenerator(asset: videoAsset)
    generator.requestedTimeToleranceBefore = CMTimeMake(value: 1, timescale: 25)
    generator.requestedTimeToleranceAfter = CMTimeMake(value: 1, timescale: 25)
    generator.appliesPreferredTrackTransform = true

    return generator
  }

  @objc func initializeImageSegmenterServiceAndSegment() {
    guard let image = selectImage else { return }
    showProgressView()
    imageSegmenterService = ImageSegmenterService(model: InferenceConfigurationManager.sharedInstance.model)
    DispatchQueue.global(qos: .userInteractive).async { [weak self] in
      guard let self = self,
            let resultBundle = self.imageSegmenterService?.segment(image: image) else {
        DispatchQueue.main.async {
          self?.hideProgressView()
        }
        return
      }

      DispatchQueue.main.async {
        self.hideProgressView()
        self.render.prepare(with: image.size, outputRetainedBufferCountHint: 3)
        self.inferenceResultDeliveryDelegate?.didPerformInference(result: resultBundle)
        let newImage = self.render.render(image: image, segmenterData: resultBundle.outputData, maskSize: resultBundle.size)
        self.pickedImageView.image = newImage
      }
    }
  }
}
