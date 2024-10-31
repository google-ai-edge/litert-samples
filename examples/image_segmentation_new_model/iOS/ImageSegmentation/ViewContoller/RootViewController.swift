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

protocol InferenceResultDeliveryDelegate: AnyObject {
  func didPerformInference(result: ResultBundle?)
}

/** The view controller is responsible for presenting and handling the tabbed controls for switching between the live camera feed and
  * media library view controllers. It also handles the presentation of the inferenceVC
  */
class RootViewController: UIViewController {

  // MARK: Storyboards Connections
  @IBOutlet weak var tabBarContainerView: UIView!
  @IBOutlet weak var runningModeTabbar: UITabBar!
  @IBOutlet weak var bottomViewHeightConstraint: NSLayoutConstraint!

  // MARK: Constants
  private struct Constants {
    static let mediaLibraryViewControllerStoryBoardId = "MEDIA_LIBRARY_VIEW_CONTROLLER"
    static let inferenceVCEmbedSegueName = "EMBED"
    static let storyBoardName = "Main"
    static let tabBarItemsCount = 2
    static let bottomSheetMaxHeight = 110
    static let bottomSheetMinHeight = 42
  }
  
  // MARK: Controllers that manage functionality
  private var mediaLibraryViewController: MediaLibraryViewController?
  private var cameraPickerViewController: MediaLibraryViewController?
  private var bottomSheetViewController: BottomSheetViewController?

  // MARK: View Handling Methods
  override func viewDidLoad() {
    super.viewDidLoad()
    
    runningModeTabbar.selectedItem = runningModeTabbar.items?.first
    runningModeTabbar.delegate = self
    instantiateCameraPickerViewController()
    switchTo(childViewController: cameraPickerViewController, fromViewController: nil)
  }

  override var preferredStatusBarStyle: UIStatusBarStyle {
    return .lightContent
  }
  
  private func instantiateMediaLibraryViewController() {
    guard mediaLibraryViewController == nil else {
      return
    }
    guard let viewController = UIStoryboard(name: Constants.storyBoardName, bundle: .main)
      .instantiateViewController(
        withIdentifier: Constants.mediaLibraryViewControllerStoryBoardId)
            as? MediaLibraryViewController else {
      return
    }
    viewController.imageSourceType = .savedPhotosAlbum

    viewController.inferenceResultDeliveryDelegate = self
    mediaLibraryViewController = viewController
  }

  private func instantiateCameraPickerViewController() {
    guard cameraPickerViewController == nil else {
      return
    }
    guard let viewController = UIStoryboard(name: Constants.storyBoardName, bundle: .main)
      .instantiateViewController(
        withIdentifier: Constants.mediaLibraryViewControllerStoryBoardId)
            as? MediaLibraryViewController else {
      return
    }

    viewController.inferenceResultDeliveryDelegate = self
    cameraPickerViewController = viewController
  }

  // MARK: Storyboard Segue Handlers
  override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
    super.prepare(for: segue, sender: sender)
    if segue.identifier == Constants.inferenceVCEmbedSegueName {
      bottomSheetViewController = segue.destination as? BottomSheetViewController
      bottomSheetViewController?.delegate = self
    }
  }

  private func updateMediaLibraryControllerUI() {
    if let mediaLibraryViewController = mediaLibraryViewController {
      mediaLibraryViewController.layoutUIElements(
        withInferenceViewHeight: bottomViewHeightConstraint.constant)
    }

    if let cameraPickerViewController = cameraPickerViewController {
      cameraPickerViewController.layoutUIElements(
        withInferenceViewHeight: bottomViewHeightConstraint.constant)
    }
  }
}

// MARK: UITabBarDelegate
extension RootViewController: UITabBarDelegate {
  func switchTo(
    childViewController: UIViewController?,
    fromViewController: UIViewController?) {
    fromViewController?.willMove(toParent: nil)
    fromViewController?.view.removeFromSuperview()
    fromViewController?.removeFromParent()
    
    guard let childViewController = childViewController else {
      return
    }
      
    addChild(childViewController)
    childViewController.view.translatesAutoresizingMaskIntoConstraints = false
    tabBarContainerView.addSubview(childViewController.view)
    NSLayoutConstraint.activate(
      [
        childViewController.view.leadingAnchor.constraint(
          equalTo: tabBarContainerView.leadingAnchor,
          constant: 0.0),
        childViewController.view.trailingAnchor.constraint(
          equalTo: tabBarContainerView.trailingAnchor,
          constant: 0.0),
        childViewController.view.topAnchor.constraint(
          equalTo: tabBarContainerView.topAnchor,
          constant: 0.0),
        childViewController.view.bottomAnchor.constraint(
          equalTo: tabBarContainerView.bottomAnchor,
          constant: 0.0)
      ]
    )
    childViewController.didMove(toParent: self)
  }
  
  func tabBar(_ tabBar: UITabBar, didSelect item: UITabBarItem) {
    guard let tabBarItems = tabBar.items, tabBarItems.count == Constants.tabBarItemsCount else {
      return
    }

    var fromViewController: UIViewController?
    var toViewController: UIViewController?
    
    switch item {
    case tabBarItems[0]:
        fromViewController = mediaLibraryViewController
        toViewController = cameraPickerViewController
    case tabBarItems[1]:
        instantiateMediaLibraryViewController()
        fromViewController = cameraPickerViewController
        toViewController = mediaLibraryViewController
    default:
      break
    }
    
    switchTo(
      childViewController: toViewController,
      fromViewController: fromViewController)
    self.updateMediaLibraryControllerUI()
  }
}

// MARK: InferenceResultDeliveryDelegate Methods
extension RootViewController: InferenceResultDeliveryDelegate {
  func didPerformInference(result: ResultBundle?) {
    DispatchQueue.main.async {
      self.bottomSheetViewController?.update(result: result)
    }
  }
}

// MARK: BottomSheetViewControllerDelegate Methods
extension RootViewController: BottomSheetViewControllerDelegate {
  func viewController(
    _ viewController: BottomSheetViewController,
    didSwitchBottomSheetViewState isOpen: Bool) {
      bottomViewHeightConstraint.constant = CGFloat(isOpen ? Constants.bottomSheetMaxHeight : Constants.bottomSheetMinHeight)
      UIView.animate(withDuration: 0.3) {[weak self] in
        guard let weakSelf = self else {
          return
        }
        weakSelf.view.layoutSubviews()
        weakSelf.updateMediaLibraryControllerUI()
      }
    }
}
