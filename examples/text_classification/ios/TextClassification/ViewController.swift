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

class ViewController: UIViewController {

  @IBOutlet weak var inputTextView: UITextView!
  @IBOutlet weak var outputLabel: UILabel!
  @IBOutlet weak var choseModelButton: UIButton!
  @IBOutlet weak var inferenceTimeLabel: UILabel!
  @IBOutlet weak var inferenceViewLayoutConstraint: NSLayoutConstraint!

  private var model: Model = Constants.defaultModel {
    didSet {
      textClassificationService = TextClassificationService(model: model)
    }
  }
  private var textClassificationService: TextClassificationService?

  override func viewDidLoad() {
    super.viewDidLoad()
    setupUI()

    view.addGestureRecognizer(UITapGestureRecognizer(target: self, action: #selector(screenTap)))
    // Initialize a TextClassification instance
    textClassificationService = TextClassificationService(model: model)
  }

  private func setupUI() {
    inputTextView.text = Constants.defaultText
    inputTextView.layer.cornerRadius = 5
    inputTextView.layer.borderColor = UIColor(displayP3Red: 1, green: 137/255.0, blue: 0, alpha: 1).cgColor
    inputTextView.layer.borderWidth = 1


    // Chose model option
    let choseModel = {(action: UIAction) in
      guard let model = Model(rawValue: action.title) else { return }
      self.model = model
    }
    let actions: [UIAction] = Model.allCases.compactMap { model in
      let action = UIAction(title: model.rawValue, handler: choseModel)
      if model == self.model {
        action.state = .on
      }
      return action
    }
    choseModelButton.menu = UIMenu(children: actions)
    choseModelButton.showsMenuAsPrimaryAction = true
    choseModelButton.changesSelectionAsPrimaryAction = true
  }

  @objc private func screenTap() {
    inputTextView.resignFirstResponder()
  }

  /// Action when user tap the "Classify" button.
  @IBAction func classifyButtonTouchupInside(_ sender: Any) {
    inputTextView.resignFirstResponder()
    guard let text = inputTextView.text,
          text.isEmpty == false else { return }
    classify(text: text)
  }

  @IBAction func showHidenButtonTouchUpInside(_ sender: UIButton) {
    sender.isSelected.toggle()
    inferenceViewLayoutConstraint.constant = sender.isSelected ? 140 : 84
    UIView.animate(withDuration: 0.3, animations: {
      self.view.layoutIfNeeded()
    }, completion: nil)
  }

  /// Classify the text and display the result.
  private func classify(text: String) {
    // Run TF Lite inference in a background thread to avoid blocking app UI
    DispatchQueue.global(qos: .userInitiated).async {
      guard let result = self.textClassificationService?.classify(text: text) else { return }
      DispatchQueue.main.async {
        self.inferenceTimeLabel.text = String(format: "%.3f ms", result.inferenceTime)
        self.outputLabel.text = String(format: "Positive: (%.3f)\nNegative: (%.3f)",
                                       result.categories["positive"]!,
                                       result.categories["negative"]!)
      }
    }
  }
}

struct Constants {
  static let defaultText = "Google has released 24 versions of the Android operating system since 2008 and continues to make substantial investments to develop, grow, and improve the OS."
  static let defaultModel = Model.mobileBert
}
