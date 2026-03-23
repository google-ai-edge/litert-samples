/*
 * Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import SwiftUI
import Combine

enum AcceleratorOption: String, CaseIterable {
    case cpu = "CPU"

    var liteRTAccelerator: LiteRTAccelerator {
        switch self {
        case .cpu: return .CPU
        }
    }
}

enum AppTab: String, CaseIterable {
    case camera = "Camera"
    case gallery = "Gallery"
}

/// Mirrors Kotlin's MainViewModel — owns all segmentation state and model lifecycle.
final class SegmentationViewModel: ObservableObject {
    @Published var currentMask: UIImage?
    @Published var inferenceTimeMs: Double = 0
    @Published var preProcessTimeMs: Double = 0
    @Published var postProcessTimeMs: Double = 0
    @Published var selectedAccelerator: AcceleratorOption = .cpu
    @Published var selectedTab: AppTab = .camera
    @Published var errorMessage: String?
    @Published var isProcessing: Bool = false

    private var segmenter: LiteRTSegmenter?
    private let modelQueue = DispatchQueue(label: "com.litert.segmentation.model", qos: .userInitiated)
    private let modelPath: String?

    init() {
        modelPath = Bundle.main.path(forResource: "selfie_multiclass_256x256", ofType: "tflite")
        initSegmenter()
    }

    func initSegmenter() {
        guard let path = modelPath else {
            errorMessage = "Model file not found in bundle"
            return
        }

        let accelerator = selectedAccelerator.liteRTAccelerator
        modelQueue.async { [weak self] in
            guard let self = self else { return }
            do {
                let newSegmenter = try LiteRTSegmenter(modelPath: path, accelerator: accelerator)
                self.segmenter = newSegmenter
                DispatchQueue.main.async {
                    self.errorMessage = nil
                }
            } catch {
                DispatchQueue.main.async {
                    self.errorMessage = "Failed to initialize segmenter: \(error.localizedDescription)"
                }
            }
        }
    }

    func setAccelerator(_ accelerator: AcceleratorOption) {
        selectedAccelerator = accelerator
        currentMask = nil
        initSegmenter()
    }

    func segment(image: CGImage) {
        guard !isProcessing else { return }

        modelQueue.async { [weak self] in
            guard let self = self, let segmenter = self.segmenter else { return }

            DispatchQueue.main.async { self.isProcessing = true }

            do {
                let result = try segmenter.segmentImage(image)
                DispatchQueue.main.async {
                    self.currentMask = result.maskImage
                    self.inferenceTimeMs = result.inferenceTimeMs
                    self.preProcessTimeMs = result.preProcessTimeMs
                    self.postProcessTimeMs = result.postProcessTimeMs
                    self.isProcessing = false
                }
            } catch {
                DispatchQueue.main.async {
                    self.errorMessage = "Segmentation error: \(error.localizedDescription)"
                    self.isProcessing = false
                }
            }
        }
    }

    func stopSegmentation() {
        currentMask = nil
        inferenceTimeMs = 0
        preProcessTimeMs = 0
        postProcessTimeMs = 0
    }

    func clearError() {
        errorMessage = nil
    }
}
