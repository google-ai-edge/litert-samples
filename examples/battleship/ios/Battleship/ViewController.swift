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

  private var agentBoardState: [Int] = []
  private var agentHiddenBoardState: [Int] = []
  private var playerBoardState: [Int] = []
  private var playerHiddenBoardState: [Int] = []
  private var playerBoadHits = 0 {
    didSet {
      playerHitsLabel.text = "Your board (hits: \(playerBoadHits))"
    }
  }
  private var agentBoadHits = 0 {
    didSet {
      agentHitsLabel.text = "Agent's board (hits: \(agentBoadHits))"
    }
  }

  private var agent: Agent!

  @IBOutlet weak var agentBoardView: PlayView!
  @IBOutlet weak var playerBoardView: PlayView!
  @IBOutlet weak var playerHitsLabel: UILabel!
  @IBOutlet weak var agentHitsLabel: UILabel!

  override func viewDidLoad() {
    super.viewDidLoad()

    guard let modelPath = Bundle.main.path(forResource: "planestrike", ofType: "tflite") else {
      fatalError("can not load model")
    }
    agent = Agent(modelPath: modelPath)
    agentBoardView.boardSize = agent.boardSize
    playerBoardView.boardSize = agent.boardSize

    agentBoardView.layer.borderColor = UIColor.gray.cgColor
    agentBoardView.layer.borderWidth = 2
    playerBoardView.layer.borderColor = UIColor.gray.cgColor
    playerBoardView.layer.borderWidth = 2

    agentBoardView.delegate = self

    restartGame()
  }
  
  @IBAction func resetButtonTouchUpInside(_ sender: Any) {
    restartGame()
  }
  
  private func restartGame() {

    playerBoadHits = 0
    agentBoadHits = 0

    agentBoardView.reset()
    playerBoardView.reset()

    let defaultBoardState = [Int](repeating: 0, count: agent.boardSize * agent.boardSize)
    agentBoardState = [Int](repeating: 0, count: agent.boardSize * agent.boardSize)
    playerBoardState = [Int](repeating: 0, count: agent.boardSize * agent.boardSize)
    agentHiddenBoardState = setRandomBoardState(state: defaultBoardState)
    playerHiddenBoardState = setRandomBoardState(state: defaultBoardState)

    for i in 0..<playerHiddenBoardState.count {
      if playerHiddenBoardState[i] == 1 {
        playerBoardView.setLocation(i, state: 2)
      }
    }
  }

  private func setRandomBoardState(state: [Int]) -> [Int] {
    let planeOrientation = Int.random(in: 0..<4)

        // Figrue out the location of plane core as the '*' below
        //     0      1      2       3
        //   | |      |      | |    ---
        //   |-*-    -*-    -*-|     |
        //   | |      |      | |    -*-
        //           ---             |
    var newState = state
    var planeCoreX = 0
    var planeCoreY = 0

    switch planeOrientation {
    case 0:
      planeCoreX = Int.random(in: 0..<(agent.boardSize - 2)) + 1
      planeCoreY = Int.random(in: 0..<(agent.boardSize - 3)) + 2

      newState[planeCoreX * agent.boardSize + planeCoreY - 2] = 1
      newState[(planeCoreX - 1) * agent.boardSize + planeCoreY - 2] = 1
      newState[(planeCoreX + 1) * agent.boardSize + planeCoreY - 2] = 1
    case 1:
      planeCoreX = Int.random(in: 0..<(agent.boardSize - 3)) + 1
      planeCoreY = Int.random(in: 0..<(agent.boardSize - 2)) + 1

      newState[(planeCoreX + 2) * agent.boardSize + planeCoreY] = 1
      newState[(planeCoreX + 2) * agent.boardSize + planeCoreY + 1] = 1
      newState[(planeCoreX + 2) * agent.boardSize + planeCoreY - 1] = 1
    case 2:
      planeCoreX = Int.random(in: 0..<(agent.boardSize - 2)) + 1
      planeCoreY = Int.random(in: 0..<(agent.boardSize - 3)) + 1

      newState[planeCoreX * agent.boardSize + planeCoreY + 2] = 1
      newState[(planeCoreX - 1)  * agent.boardSize + planeCoreY + 2] = 1
      newState[(planeCoreX + 1) * agent.boardSize + planeCoreY + 2] = 1
    default:
      planeCoreX = Int.random(in: 0..<(agent.boardSize - 3)) + 2
      planeCoreY = Int.random(in: 0..<(agent.boardSize - 2)) + 1

      newState[(planeCoreX - 2) * agent.boardSize + planeCoreY] = 1
      newState[(planeCoreX - 2) * agent.boardSize + planeCoreY + 1] = 1
      newState[(planeCoreX - 2) * agent.boardSize + planeCoreY - 1] = 1
    }
    newState[planeCoreX  * agent.boardSize  + planeCoreY] = 1
    newState[(planeCoreX + 1) * agent.boardSize + planeCoreY] = 1
    newState[(planeCoreX - 1) * agent.boardSize + planeCoreY] = 1
    newState[planeCoreX * agent.boardSize + planeCoreY + 1] = 1
    newState[planeCoreX * agent.boardSize + planeCoreY - 1] = 1

    return newState
  }

  private func stopGame(isPlayerWin: Bool) {
    let title = isPlayerWin ? "You win!" : "Agent win!"
    let alertController = UIAlertController(title: title, message: nil, preferredStyle: .alert)
    alertController.addAction(UIAlertAction(title: "OK", style: .default, handler: { _ in
      self.restartGame()
    }))
    present(alertController, animated: true)
  }
}

// MARK: PlayViewDelegate
extension ViewController: PlayViewDelegate {
  func playViewChosseLocation(_ location: Int) {

    guard agentBoardState[location] == 0 else { return }
    switch agentHiddenBoardState[location] {
    case 1:
      agentBoardState[location] = 1
      agentBoadHits += 1
    default:
      agentBoardState[location] = -1
    }
    agentBoardView.setLocation(location, state: agentBoardState[location])
    if agentBoadHits == 8 {
      stopGame(isPlayerWin: true)
      return
    }

    let agentChooseLocation = agent.play(states: playerBoardState)
    guard playerBoardState[agentChooseLocation] == 0 else { fatalError("location is choosed") }
    switch playerHiddenBoardState[agentChooseLocation] {
    case 1:
      playerBoardState[agentChooseLocation] = 1
      playerBoadHits += 1
    default:
      playerBoardState[agentChooseLocation] = -1
    }
    playerBoardView.setLocation(agentChooseLocation, state: playerBoardState[agentChooseLocation])
    if playerBoadHits == 8 {
      stopGame(isPlayerWin: false)
      return
    }
  }
}
