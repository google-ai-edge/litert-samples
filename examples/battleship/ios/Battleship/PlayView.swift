//
//  PlayView.swift
//  Battleship
//
//  Created by MBA0077 on 6/24/24.
//

import UIKit

protocol PlayViewDelegate: AnyObject {
  func playViewChosseLocation(_ location: Int)
}

class PlayView: UIView {

  weak var delegate: PlayViewDelegate?

  var boardSize: Int = 8 {
    didSet {
      createPlayView()
    }
  }
  
  private var buttons: [UIButton] = []

/// This function is used to create a grid of buttons for a play view.
///
/// - Parameters: None
/// - Returns: None

/// Loop through the boardSize to create buttons and add them to the view.
/// Set button properties such as border, tag, and target action.
/// Add constraints to position the buttons in a grid layout.
/// Activate the constraints to apply them to the buttons.
  private func createPlayView() {
    subviews.forEach({ $0.removeFromSuperview() })
    buttons = []
    for i in 0..<boardSize{
      for j in 0..<boardSize {
        let button = UIButton()
        button.layer.borderWidth = 1
        button.layer.borderColor = UIColor.gray.cgColor
        button.tag = i * boardSize + j
        button.addTarget(self, action: #selector(buttonTouchupInside), for: .touchUpInside)
        addSubview(button)
        buttons.append(button)
        button.translatesAutoresizingMaskIntoConstraints = false
        var constraints = [
          button.widthAnchor.constraint(equalTo: widthAnchor, multiplier: 1/CGFloat(boardSize)),
          button.heightAnchor.constraint(equalTo: heightAnchor, multiplier: 1/CGFloat(boardSize))
        ]
        if i == 0 {
          constraints.append(button.topAnchor.constraint(equalTo: topAnchor))
        } else {
          let topButton = buttons[(i - 1) * boardSize]
          constraints.append(button.topAnchor.constraint(equalTo: topButton.bottomAnchor))
        }

        if j == 0 {
          constraints.append(button.leftAnchor.constraint(equalTo: leftAnchor))
        } else {
          let leftButton = buttons[j - 1]
          constraints.append(button.leftAnchor.constraint(equalTo: leftButton.rightAnchor))
        }
        NSLayoutConstraint.activate(constraints)
      }
    }
  }

/// Resets the background color of all buttons in the array to white.
  func reset() {
    for button in buttons {
      button.backgroundColor = .white
    }
  }

/// Sets the background color of a button at a specific location based on the state provided.

/// - Parameters:
///   - location: The index of the button whose background color needs to be set.
///   - state: The state value that determines the color to be set ( -1 for yellow, 1 for red, 2 for green, any other value for white).
  func setLocation(_ location: Int, state: Int) {
    switch state {
    case -1:
      buttons[location].backgroundColor = .yellow
    case 1:
      buttons[location].backgroundColor = .red
    case 2:
      buttons[location].backgroundColor = .green
    default:
      buttons[location].backgroundColor = .white
    }
  }

/// Handles the touch up inside event of a button.
///
/// - Parameters:
///   - sender: The button that triggered the event
/// - Returns: None
  @objc private func buttonTouchupInside(_ sender: UIButton) {
    delegate?.playViewChosseLocation(sender.tag)
  }
}
