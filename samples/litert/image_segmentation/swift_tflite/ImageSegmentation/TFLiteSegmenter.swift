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

import Foundation
import CoreGraphics
import UIKit
import TensorFlowLite

public final class TFLiteSegmentationResult {
    public let maskImage: UIImage
    public let preProcessTimeMs: Double
    public let inferenceTimeMs: Double
    public let postProcessTimeMs: Double

    public init(
        maskImage: UIImage,
        preProcessTimeMs: Double,
        inferenceTimeMs: Double,
        postProcessTimeMs: Double
    ) {
        self.maskImage = maskImage
        self.preProcessTimeMs = preProcessTimeMs
        self.inferenceTimeMs = inferenceTimeMs
        self.postProcessTimeMs = postProcessTimeMs
    }
}

public final class TFLiteSegmenter {
    private var interpreter: Interpreter!

    private struct ColoredLabel {
        let r: UInt8
        let g: UInt8
        let b: UInt8
    }

    private static let colors: [ColoredLabel] = [
        ColoredLabel(r: 0, g: 0, b: 0),       // Background
        ColoredLabel(r: 255, g: 0, b: 0),     // Class 1 (Hair)
        ColoredLabel(r: 0, g: 255, b: 0),     // Class 2 (Body skin)
        ColoredLabel(r: 0, g: 0, b: 255),     // Class 3 (Face skin)
        ColoredLabel(r: 255, g: 255, b: 0),   // Class 4 (Clothes)
        ColoredLabel(r: 0, g: 255, b: 255)    // Class 5 (Accessories)
    ]

    deinit {
        interpreter = nil
    }

    public init(modelPath: String, threadCount: Int = 4) throws {
        var options = Interpreter.Options()
        options.threadCount = threadCount
        options.isXNNPackEnabled = true

        self.interpreter = try Interpreter(
            modelPath: modelPath,
            options: options
        )
        print("[TFLiteSegmenter] Initialized TFLite Interpreter (v\(TensorFlowLite.Runtime.version)) on CPU (XNNPACK, \(threadCount) threads)")

        // Allocate tensors for inference
        try self.interpreter.allocateTensors()
    }

    public func segmentImage(_ image: CGImage) throws -> TFLiteSegmentationResult {
        let modelWidth = 256
        let modelHeight = 256
        let channels = 3
        let pixelCount = modelWidth * modelHeight

        let startPreprocess = DispatchTime.now()

        // 1. Draw image into 256x256 32bpp RGBA buffer
        var imageData = [UInt8](repeating: 0, count: pixelCount * 4)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue

        guard let context = CGContext(
            data: &imageData,
            width: modelWidth,
            height: modelHeight,
            bitsPerComponent: 8,
            bytesPerRow: modelWidth * 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo
        ) else {
            throw NSError(
                domain: "com.google.tflite.segmentation",
                code: 11,
                userInfo: [NSLocalizedDescriptionKey: "Failed to create CGContext for preprocessing"]
            )
        }

        context.draw(image, in: CGRect(x: 0, y: 0, width: modelWidth, height: modelHeight))

        // 2. Normalize pixel values to [-1, 1] (NHWC float32) and copy to input Tensor at index 0
        var inputFloat = [Float](repeating: 0, count: pixelCount * channels)
        imageData.withUnsafeBufferPointer { src in
            inputFloat.withUnsafeMutableBufferPointer { dst in
                for i in 0..<pixelCount {
                    let r = src[i * 4 + 0]
                    let g = src[i * 4 + 1]
                    let b = src[i * 4 + 2]
                    dst[i * 3 + 0] = (Float(r) - 127.5) / 127.5
                    dst[i * 3 + 1] = (Float(g) - 127.5) / 127.5
                    dst[i * 3 + 2] = (Float(b) - 127.5) / 127.5
                }
            }
        }

        let inputData = inputFloat.withUnsafeBufferPointer { Data(buffer: $0) }
        try interpreter.copy(inputData, toInputAt: 0)

        let endPreprocess = DispatchTime.now()
        let startInference = DispatchTime.now()

        // 3. Run TFLite Interpreter inference
        try interpreter.invoke()

        let endInference = DispatchTime.now()
        let startPostprocess = DispatchTime.now()

        // 4. Read output tensor (1 x 256 x 256 x 6) and generate RGBA mask bitmap
        let outputTensor = try interpreter.output(at: 0)
        let expectedFloatCount = pixelCount * 6
        let actualFloatCount = outputTensor.data.count / MemoryLayout<Float>.stride

        guard actualFloatCount >= expectedFloatCount else {
            throw NSError(
                domain: "com.google.tflite.segmentation",
                code: 14,
                userInfo: [NSLocalizedDescriptionKey: "Output tensor size \(actualFloatCount) is smaller than expected \(expectedFloatCount)"]
            )
        }

        var maskBytes = [UInt8](repeating: 0, count: pixelCount * 4)

        outputTensor.data.withUnsafeBytes { rawBuffer in
            let outBuf = rawBuffer.bindMemory(to: Float.self)
            maskBytes.withUnsafeMutableBufferPointer { maskBuf in
                for pixelOffset in 0..<pixelCount {
                    let classOffset = pixelOffset * 6
                    var maxClass = 0
                    var maxVal = outBuf[classOffset]
                    for c in 1..<6 {
                        let val = outBuf[classOffset + c]
                        if val > maxVal {
                            maxVal = val
                            maxClass = c
                        }
                    }

                    let col = Self.colors[maxClass]
                    maskBuf[pixelOffset * 4 + 0] = col.r
                    maskBuf[pixelOffset * 4 + 1] = col.g
                    maskBuf[pixelOffset * 4 + 2] = col.b
                    maskBuf[pixelOffset * 4 + 3] = (maxClass == 0) ? 0 : 180
                }
            }
        }

        guard let maskContext = CGContext(
            data: &maskBytes,
            width: modelWidth,
            height: modelHeight,
            bitsPerComponent: 8,
            bytesPerRow: modelWidth * 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo
        ), let maskCGImage = maskContext.makeImage() else {
            throw NSError(
                domain: "com.google.tflite.segmentation",
                code: 13,
                userInfo: [NSLocalizedDescriptionKey: "Failed to create output mask image"]
            )
        }

        let maskImage = UIImage(cgImage: maskCGImage)
        let endPostprocess = DispatchTime.now()

        let preProcessTimeMs = Double(endPreprocess.uptimeNanoseconds - startPreprocess.uptimeNanoseconds) / 1_000_000.0
        let inferenceTimeMs = Double(endInference.uptimeNanoseconds - startInference.uptimeNanoseconds) / 1_000_000.0
        let postProcessTimeMs = Double(endPostprocess.uptimeNanoseconds - startPostprocess.uptimeNanoseconds) / 1_000_000.0

        return TFLiteSegmentationResult(
            maskImage: maskImage,
            preProcessTimeMs: preProcessTimeMs,
            inferenceTimeMs: inferenceTimeMs,
            postProcessTimeMs: postProcessTimeMs
        )
    }
}
