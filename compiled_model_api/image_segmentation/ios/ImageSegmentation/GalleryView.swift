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
import UIKit

struct GalleryView: View {
    @ObservedObject var viewModel: SegmentationViewModel
    @State private var selectedImage: UIImage?
    @State private var showingImagePicker = false

    var body: some View {
        VStack {
            ZStack {
                if let image = selectedImage {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFit()
                        .frame(maxWidth: .infinity, maxHeight: 400)

                    if let mask = viewModel.currentMask {
                        Image(uiImage: mask)
                            .resizable()
                            .scaledToFit()
                            .frame(maxWidth: .infinity, maxHeight: 400)
                            .opacity(0.6)
                    }
                } else {
                    Rectangle()
                        .fill(Color.black)
                        .frame(height: 400)
                        .overlay(
                            Text("Select an image from gallery")
                                .foregroundColor(.gray)
                        )
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 15))
            .padding()

            Button(action: { showingImagePicker = true }) {
                Label("Pick from Gallery", systemImage: "photo.on.rectangle")
                    .font(.headline)
                    .foregroundColor(.white)
                    .padding()
                    .background(Color.blue)
                    .cornerRadius(10)
            }
            .sheet(isPresented: $showingImagePicker) {
                ImagePicker(selectedImage: $selectedImage)
            }
            .onChange(of: selectedImage) { newImage in
                guard let image = newImage, let cgImage = image.cgImage else { return }
                viewModel.segment(image: cgImage)
            }

            Spacer()
        }
    }
}

/// UIImagePickerController wrapper for iOS 14+ compatibility.
struct ImagePicker: UIViewControllerRepresentable {
    @Binding var selectedImage: UIImage?
    @Environment(\.presentationMode) private var presentationMode

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.sourceType = .photoLibrary
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    final class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let parent: ImagePicker

        init(_ parent: ImagePicker) {
            self.parent = parent
        }

        func imagePickerController(
            _ picker: UIImagePickerController,
            didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]
        ) {
            if let image = info[.originalImage] as? UIImage {
                parent.selectedImage = image
            }
            parent.presentationMode.wrappedValue.dismiss()
        }

        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            parent.presentationMode.wrappedValue.dismiss()
        }
    }
}
