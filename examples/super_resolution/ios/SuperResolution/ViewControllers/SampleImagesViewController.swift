// 2024 The Google AI Edge Authors. All Rights Reserved.
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

class SampleImagesViewController: UIViewController {

  private var interprenterHelper: InterpreterHelper?
  private var scaledImage: UIImage?

  @IBOutlet weak var superResolutionImageView: UIImageView!
  @IBOutlet weak var scaledImageView: UIImageView!

  weak var inferenceResultDeliveryDelegate: InferenceResultDeliveryDelegate?

  override func viewDidLoad() {
      super.viewDidLoad()

      interprenterHelper = InterpreterHelper(modelPath: DefaultConstants.modelPath)
        // Do any additional setup after loading the view.
    }

  @IBAction func chooseImageButtonTouchupInside(_ sender: UIButton) {
    guard scaledImage != sender.imageView?.image else { return }
    scaledImage = sender.imageView?.image
    scaledImageView.image = sender.imageView?.image
    superResolutionImageView.image = nil
  }

  @IBAction func upsampleButtonTouchupInside(_ sender: Any) {
    guard let image = scaledImage,
    image.size == CGSize(width: 50, height: 50) else { return }

    DispatchQueue.global(qos: .userInteractive).async { [weak self] in
      guard let self = self,
            let result = self
        .interprenterHelper?
        .proccess(image: image) else {
        return
      }
      DispatchQueue.main.async {
        self.superResolutionImageView.image = result.image
        self.inferenceResultDeliveryDelegate?.didPerformInference(result: result)
      }
    }
  }
  


}
