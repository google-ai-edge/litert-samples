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
import AVFoundation

class ViewController: UIViewController {

  // MARK: - Variables
  @IBOutlet weak var tableView: UITableView!
  @IBOutlet weak var inferenceView: InferenceView!

  private var audioInputManager: AudioInputManager?
  private var audioClassificationHelper: AudioClassificationHelper!

  private var bufferSize: Int = 0

  private var model: Model = DefaultConstants.model
  private var overLap: Double = DefaultConstants.overLap
  private var maxResults: Int = DefaultConstants.maxResults
  private var threshold: Float = DefaultConstants.threshold
  private var threadCount: Int = DefaultConstants.threadCount

  private var datas: [ClassificationCategory] = []

  override func viewDidLoad() {
    super.viewDidLoad()
    inferenceView.delegate = self
    inferenceView.setDefault()
    restartClassifier()
  }

  // MARK: - Private Methods

  /// Initializes the AudioInputManager and starts recognizing on the output buffers.
  private func startAudioRecognition() {
    audioInputManager?.stop()
    audioInputManager = AudioInputManager(bufferSize: audioClassificationHelper.sampleRate, overlap: Float(overLap))
    audioInputManager?.delegate = self

    bufferSize = audioInputManager?.bufferSize ?? 0

    audioInputManager?.checkPermissionsAndStartTappingMicrophone()
  }

  private func runModel(inputBuffer: [Int16]) {
    audioClassificationHelper.start(inputBuffer: inputBuffer)
  }

  /// Start a new audio classification routine.
  private func restartClassifier() {
    // Create a new classifier instance.
    guard let audioClassificationHelper = AudioClassificationHelper(
      model: model,
      scoreThreshold: threshold,
      maxResults: maxResults) else { fatalError("can not init AudioClassificationHelper") }
    audioClassificationHelper.delegate = self
    self.audioClassificationHelper = audioClassificationHelper

    startAudioRecognition()
  }
}

// MARK: extension implement show permission error
extension ViewController {
  private func showPermissionsErrorAlert() {
    let alertController = UIAlertController(
      title: "Microphone Permissions Denied",
      message: "Microphone permissions have been denied for this app. You can change this by going to Settings",
      preferredStyle: .alert
    )

    let cancelAction = UIAlertAction(title: "Cancel", style: .cancel, handler: nil)
    let settingsAction = UIAlertAction(title: "Settings", style: .default) { _ in
      UIApplication.shared.open(
        URL(string: UIApplication.openSettingsURLString)!,
        options: [:],
        completionHandler: nil
      )
    }
    alertController.addAction(cancelAction)
    alertController.addAction(settingsAction)

    present(alertController, animated: true, completion: nil)
  }
}

// MARK: UITableViewDataSource, UITableViewDelegate
extension ViewController: UITableViewDataSource, UITableViewDelegate {

  func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
    return datas.count
  }

  func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
    guard let cell = tableView.dequeueReusableCell(withIdentifier: "ResultCell") as? ResultTableViewCell else { fatalError() }
    cell.setData(datas[indexPath.row])
    return cell
  }
}

// MARK: AudioClassificationHelperDelegate
extension ViewController: AudioClassificationHelperDelegate {
  func audioClassificationHelper(_ audioClassificationHelper: AudioClassificationHelper, didfinishClassfification result: Result) {
    datas = result.categories
    DispatchQueue.main.async {
      self.tableView.reloadData()
      self.inferenceView.inferenceTimeLabel.text = "\(Int(result.inferenceTime * 1000)) ms"
    }
  }
}

// MARK: InferenceViewDelegate
extension ViewController: InferenceViewDelegate {
  func view(_ view: InferenceView, needPerformActions action: InferenceView.Action) {
    switch action {
    case .changeModel(let model):
      self.model = model
    case .changeOverlap(let overLap):
      self.overLap = overLap
    case .changeMaxResults(let maxResults):
      self.maxResults = maxResults
    case .changeScoreThreshold(let threshold):
      self.threshold = threshold
    case .changeThreadCount(let threadCount):
      self.threadCount = threadCount
    }

    // Restart the audio classifier as the config as changed.
    restartClassifier()
  }
}

// MARK: - AudioInputManagerDelegate
extension ViewController: AudioInputManagerDelegate {
  func audioInputManagerDidFailToAchievePermission(_ audioInputManager: AudioInputManager) {
    let alertController = UIAlertController(
      title: "Microphone Permissions Denied",
      message: "Microphone permissions have been denied for this app. You can change this by going to Settings",
      preferredStyle: .alert
    )

    let cancelAction = UIAlertAction(title: "Cancel", style: .cancel, handler: nil)
    let settingsAction = UIAlertAction(title: "Settings", style: .default) { _ in
      UIApplication.shared.open(
        URL(string: UIApplication.openSettingsURLString)!,
        options: [:],
        completionHandler: nil
      )
    }
    alertController.addAction(cancelAction)
    alertController.addAction(settingsAction)

    present(alertController, animated: true, completion: nil)
  }

  func audioInputManager(
    _ audioInputManager: AudioInputManager,
    didCaptureChannelData channelData: [Int16]
  ) {
    let sampleRate = audioClassificationHelper.sampleRate
    guard channelData.count >= sampleRate else { return }
    self.runModel(inputBuffer: Array(channelData[0..<sampleRate]))
  }
}
