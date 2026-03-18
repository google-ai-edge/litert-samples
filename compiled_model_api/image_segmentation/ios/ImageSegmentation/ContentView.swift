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

import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = SegmentationViewModel()
    @StateObject private var cameraManager = CameraManager()

    var body: some View {
        VStack(spacing: 0) {
            HeaderView()

            TabSelector(selectedTab: $viewModel.selectedTab) {
                viewModel.stopSegmentation()
            }

            switch viewModel.selectedTab {
            case .camera:
                CameraContentView(
                    viewModel: viewModel,
                    cameraManager: cameraManager
                )
            case .gallery:
                GalleryView(viewModel: viewModel)
            }

            Spacer(minLength: 0)

            BottomControlsView(
                viewModel: viewModel,
                onFlipCamera: { cameraManager.flipCamera() }
            )
        }
        .onAppear {
            cameraManager.requestPermissionAndStart()
        }
        .onDisappear {
            cameraManager.stop()
        }
        .alert(isPresented: Binding(
            get: { viewModel.errorMessage != nil },
            set: { if !$0 { viewModel.clearError() } }
        )) {
            Alert(
                title: Text("Error"),
                message: Text(viewModel.errorMessage ?? ""),
                dismissButton: .default(Text("OK")) { viewModel.clearError() }
            )
        }
    }
}

// MARK: - Header

struct HeaderView: View {
    var body: some View {
        HStack {
            Text("LiteRT")
                .font(.headline)
                .fontWeight(.bold)
            Text("Image Segmentation")
                .font(.headline)
            Spacer()
        }
        .padding(.horizontal)
        .padding(.vertical, 10)
        .background(Color(.systemBackground))
    }
}

// MARK: - Tab Selector

struct TabSelector: View {
    @Binding var selectedTab: AppTab
    let onTabChanged: () -> Void

    var body: some View {
        HStack(spacing: 0) {
            ForEach(AppTab.allCases, id: \.self) { tab in
                Button(action: {
                    if selectedTab != tab {
                        onTabChanged()
                        selectedTab = tab
                    }
                }) {
                    Text(tab.rawValue)
                        .font(.subheadline)
                        .fontWeight(selectedTab == tab ? .semibold : .regular)
                        .foregroundColor(selectedTab == tab ? .white : .gray)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 10)
                        .background(selectedTab == tab ? Color.blue : Color(.systemGray5))
                }
            }
        }
    }
}

// MARK: - Camera Content

struct CameraContentView: View {
    @ObservedObject var viewModel: SegmentationViewModel
    @ObservedObject var cameraManager: CameraManager

    var body: some View {
        ZStack {
            switch cameraManager.permissionStatus {
            case .authorized:
                cameraPreview
            case .denied:
                permissionDeniedView
            case .notDetermined:
                ProgressView("Requesting camera access...")
                    .frame(height: 400)
            }
        }
        .onChange(of: cameraManager.currentFrame) { frame in
            guard let frame = frame else { return }
            viewModel.segment(image: frame)
        }
    }

    private var cameraPreview: some View {
        ZStack {
            if let frame = cameraManager.currentFrame {
                Image(decorative: frame, scale: 1.0, orientation: .up)
                    .resizable()
                    .scaledToFit()
                    .frame(maxWidth: .infinity, maxHeight: 400)
            } else {
                Rectangle()
                    .fill(Color.black)
                    .frame(height: 400)
                    .overlay(ProgressView())
            }

            if let mask = viewModel.currentMask {
                Image(uiImage: mask)
                    .resizable()
                    .scaledToFit()
                    .frame(maxWidth: .infinity, maxHeight: 400)
                    .opacity(0.6)
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 15))
        .padding()
    }

    private var permissionDeniedView: some View {
        VStack(spacing: 12) {
            Image(systemName: "camera.fill")
                .font(.largeTitle)
                .foregroundColor(.gray)
            Text("Camera access denied")
                .font(.headline)
            Text("Go to Settings > Privacy > Camera to enable access.")
                .font(.caption)
                .foregroundColor(.gray)
                .multilineTextAlignment(.center)
        }
        .frame(height: 400)
        .padding()
    }
}

// MARK: - Bottom Controls

struct BottomControlsView: View {
    @ObservedObject var viewModel: SegmentationViewModel
    let onFlipCamera: () -> Void

    var body: some View {
        VStack(spacing: 12) {
            Divider()

            // Timing display
            HStack {
                Text("Inference:")
                Spacer()
                Text(String(format: "%.1f ms", viewModel.inferenceTimeMs))
                    .fontWeight(.medium)
            }
            .font(.subheadline)
            .padding(.horizontal)

            HStack {
                Text("Pre-process:")
                Spacer()
                Text(String(format: "%.1f ms", viewModel.preProcessTimeMs))
            }
            .font(.caption)
            .foregroundColor(.secondary)
            .padding(.horizontal)

            HStack {
                Text("Post-process:")
                Spacer()
                Text(String(format: "%.1f ms", viewModel.postProcessTimeMs))
            }
            .font(.caption)
            .foregroundColor(.secondary)
            .padding(.horizontal)

            // Controls row
            HStack {
                Text("Accelerator: CPU")
                    .font(.caption)
                    .foregroundColor(.secondary)

                Spacer()

                // Camera flip button
                if viewModel.selectedTab == .camera {
                    Button(action: onFlipCamera) {
                        Image(systemName: "camera.rotate")
                            .font(.title2)
                            .padding(10)
                            .background(Color(.systemGray5))
                            .clipShape(Circle())
                    }
                }
            }
            .padding(.horizontal)
            .padding(.bottom, 8)
        }
        .background(Color(.systemBackground))
    }
}
