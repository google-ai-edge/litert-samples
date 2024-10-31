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

protocol BottomSheetViewControllerDelegate: AnyObject {
  /**
   This method is called when the user opens or closes the bottom sheet.
  **/
  func viewController(
    _ viewController: BottomSheetViewController,
    didSwitchBottomSheetViewState isOpen: Bool)
}

/** The view controller is responsible for presenting the controls to change the meta data for the image classifier (model, max results,
 * score threshold) and updating the singleton`` ClassifierMetadata`` on user input.
 */
class BottomSheetViewController: UIViewController {

  // MARK: Delegates
  weak var delegate: BottomSheetViewControllerDelegate?

  // MARK: Storyboards Connections
  @IBOutlet weak var inferenceTimeLabel: UILabel!
  @IBOutlet weak var inferenceTimeNameLabel: UILabel!

  @IBOutlet weak var choseModelButton: UIButton!

  @IBOutlet weak var toggleBottomSheetButton: UIButton!


  // MARK: Computed properties


  override func viewDidLoad() {
    super.viewDidLoad()
    setupUI()
  }

  // MARK: - Public Functions
  func update(result: ResultBundle?) {
    if let inferenceTime = result?.inferenceTime {
      inferenceTimeLabel.text = String(format: "%.2fms", inferenceTime)
    } else {
      inferenceTimeLabel.text = ""
    }
  }

  // MARK: - Private function
  private func setupUI() {

    // Chose model option
    let choseModel = {(action: UIAction) in
      self.updateModel(modelTitle: action.title)
    }
    let actions: [UIAction] = Model.allCases.compactMap { model in
      let action = UIAction(title: model.name, handler: choseModel)
      if model == InferenceConfigurationManager.sharedInstance.model {
        action.state = .on
      }
      return action
    }
    choseModelButton.menu = UIMenu(children: actions)
    choseModelButton.showsMenuAsPrimaryAction = true
    choseModelButton.changesSelectionAsPrimaryAction = true
  }

  private func updateModel(modelTitle: String) {
    guard let model = Model(name: modelTitle) else { return }
    InferenceConfigurationManager.sharedInstance.model = model
  }

  // MARK: IBAction
  @IBAction func expandButtonTouchUpInside(_ sender: UIButton) {
    sender.isSelected.toggle()
    delegate?.viewController(self, didSwitchBottomSheetViewState: sender.isSelected)
  }
}
