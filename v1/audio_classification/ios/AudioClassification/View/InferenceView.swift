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

protocol InferenceViewDelegate {
  /// This method is called when the user changes the value to update model used for inference.
  func view(_ view: InferenceView, needPerformActions action: InferenceView.Action)
}

/// View to allows users changing the inference configs.
class InferenceView: UIView {

  enum Action {
    case changeModel(Model)
    case changeOverlap(Double)
    case changeMaxResults(Int)
    case changeScoreThreshold(Float)
    case changeThreadCount(Int)
  }

  var delegate: InferenceViewDelegate?

  @IBOutlet weak var inferenceTimeLabel: UILabel!
  @IBOutlet weak var overlabLabel: UILabel!
  @IBOutlet weak var maxResulteLabel: UILabel!
  @IBOutlet weak var thresholdLabel: UILabel!
  @IBOutlet weak var threadsLabel: UILabel!
  @IBOutlet weak var overLapStepper: UIStepper!
  @IBOutlet weak var maxResultsStepper: UIStepper!
  @IBOutlet weak var thresholdStepper: UIStepper!
  @IBOutlet weak var threadsStepper: UIStepper!
  @IBOutlet weak var choseModelButton: UIButton!
  @IBOutlet weak var showHidenButtonLayoutConstraint: NSLayoutConstraint!
  @IBOutlet weak var showHidenButton: UIButton!

  /// Set the default settings.
  func setDefault() {
    // Chose model option
    let choseModel = { [weak self] (action: UIAction) in
      guard let self else { return }
      guard let model = Model(rawValue: action.title) else { return }
      delegate?.view(self, needPerformActions: .changeModel(model))
    }
    let actions: [UIAction] = Model.allCases.compactMap { model in
      let action = UIAction(title: model.rawValue, handler: choseModel)
      if model == DefaultConstants.model {
        action.state = .on
      }
      return action
    }
    choseModelButton.menu = UIMenu(children: actions)
    choseModelButton.showsMenuAsPrimaryAction = true
    choseModelButton.changesSelectionAsPrimaryAction = true

    overlabLabel.text = "\(Int(DefaultConstants.overLap * 100))%"
    overLapStepper.value = DefaultConstants.overLap
    maxResulteLabel.text = "\(DefaultConstants.maxResults)"
    maxResultsStepper.value = Double(DefaultConstants.maxResults)
    thresholdLabel.text = String(format: "%.1f", DefaultConstants.threshold)
    thresholdStepper.value = Double(DefaultConstants.threshold)
    threadsLabel.text = "\(DefaultConstants.threadCount)"
    threadsStepper.value = Double(DefaultConstants.threadCount)
  }

  @IBAction func overlapStepperValueChanged(_ sender: UIStepper) {
    overlabLabel.text = String(format: "%.0f", sender.value * 100) + "%"
    delegate?.view(self, needPerformActions: .changeOverlap(sender.value))
  }

  @IBAction func maxResultsStepperValueChanged(_ sender: UIStepper) {
    maxResulteLabel.text = "\(Int(sender.value))"
    delegate?.view(self, needPerformActions: .changeMaxResults(Int(sender.value)))
  }

  @IBAction func thresholdStepperValueChanged(_ sender: UIStepper) {
    thresholdLabel.text = String(format: "%.1f", sender.value)
    delegate?.view(self, needPerformActions: .changeScoreThreshold(Float(sender.value)))
  }

  @IBAction func threadsStepperValueChanged(_ sender: UIStepper) {
    threadsLabel.text = "\(Int(sender.value))"
    delegate?.view(self, needPerformActions: .changeThreadCount(Int(sender.value)))
  }

  @IBAction func showHidenButtonTouchUpInside(_ sender: UIButton) {
    sender.isSelected.toggle()
    showHidenButtonLayoutConstraint.constant = sender.isSelected ? 296 : 74
    UIView.animate(withDuration: 0.3, animations: {
      self.superview?.layoutIfNeeded()
    }, completion: nil)
  }
}
