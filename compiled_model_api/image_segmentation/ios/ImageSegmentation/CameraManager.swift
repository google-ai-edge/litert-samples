/*
 * Copyright 2025 The Google AI Edge Authors. All Rights Reserved.
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

import AVFoundation
import CoreImage
import SwiftUI

enum CameraPermissionStatus {
    case notDetermined
    case authorized
    case denied
}

enum CameraPosition {
    case front
    case back

    var avPosition: AVCaptureDevice.Position {
        switch self {
        case .front: return .front
        case .back: return .back
        }
    }

    var toggled: CameraPosition {
        switch self {
        case .front: return .back
        case .back: return .front
        }
    }
}

final class CameraManager: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var currentFrame: CGImage?
    @Published var permissionStatus: CameraPermissionStatus = .notDetermined
    @Published var cameraPosition: CameraPosition = .front

    private let captureSession = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "com.litert.camera.session")
    private let context = CIContext()

    override init() {
        super.init()
    }

    func requestPermissionAndStart() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            DispatchQueue.main.async { self.permissionStatus = .authorized }
            setupAndStart()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                guard let self = self else { return }
                DispatchQueue.main.async {
                    self.permissionStatus = granted ? .authorized : .denied
                }
                if granted {
                    self.setupAndStart()
                }
            }
        default:
            DispatchQueue.main.async { self.permissionStatus = .denied }
        }
    }

    func flipCamera() {
        cameraPosition = cameraPosition.toggled
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            self.captureSession.stopRunning()
            self.reconfigureSession()
            self.captureSession.startRunning()
        }
    }

    func stop() {
        sessionQueue.async { [weak self] in
            self?.captureSession.stopRunning()
        }
    }

    private func setupAndStart() {
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            self.configureCaptureSession()
            self.captureSession.startRunning()
        }
    }

    private func configureCaptureSession() {
        captureSession.beginConfiguration()
        captureSession.sessionPreset = .vga640x480

        addCameraInput()

        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "com.litert.camera.video"))
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)]

        if captureSession.canAddOutput(videoOutput) {
            captureSession.addOutput(videoOutput)
            configureVideoConnection()
        }

        captureSession.commitConfiguration()
    }

    private func reconfigureSession() {
        captureSession.beginConfiguration()

        // Remove existing inputs
        for input in captureSession.inputs {
            captureSession.removeInput(input)
        }

        addCameraInput()
        configureVideoConnection()

        captureSession.commitConfiguration()
    }

    private func addCameraInput() {
        guard let device = AVCaptureDevice.default(
            .builtInWideAngleCamera,
            for: .video,
            position: cameraPosition.avPosition
        ),
        let input = try? AVCaptureDeviceInput(device: device) else {
            return
        }

        if captureSession.canAddInput(input) {
            captureSession.addInput(input)
        }
    }

    private func configureVideoConnection() {
        if let connection = videoOutput.connection(with: .video) {
            connection.videoOrientation = .portrait
            if cameraPosition == .front {
                connection.isVideoMirrored = true
            }
        }
    }

    // MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        if let cgImage = context.createCGImage(ciImage, from: ciImage.extent) {
            DispatchQueue.main.async {
                self.currentFrame = cgImage
            }
        }
    }
}
