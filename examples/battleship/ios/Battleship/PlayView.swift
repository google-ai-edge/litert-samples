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

  func reset() {
    for button in buttons {
      button.backgroundColor = .white
    }
  }

  // -1: miss, 1: hit, 2: hidenBoard, default(0): none
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

  @objc private func buttonTouchupInside(_ sender: UIButton) {
    delegate?.playViewChosseLocation(sender.tag)
  }
}
