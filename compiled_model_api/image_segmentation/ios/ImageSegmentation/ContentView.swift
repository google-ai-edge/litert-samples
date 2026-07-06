import SwiftUI

enum AcceleratorOption: String, CaseIterable {
    case cpu = "CPU"
    case gpu = "GPU (Metal)"

    var liteRTAccelerator: LiteRTAccelerator {
        switch self {
        case .cpu: return .CPU
        case .gpu: return .metal
        }
    }
}

struct ContentView: View {
    @State private var selectedAccelerator: AcceleratorOption = .cpu
    @State private var originalImage: UIImage? = nil
    @State private var segmentedMask: UIImage? = nil
    @State private var errorMessage: String? = nil
    @State private var isProcessing = false
    
    // Performance metrics
    @State private var preProcessTime: Double = 0
    @State private var inferenceTime: Double = 0
    @State private var postProcessTime: Double = 0
    
    private let modelQueue = DispatchQueue(label: "com.litert.segmentation.queue", qos: .userInitiated)
    
    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Text("LiteRT")
                    .font(.headline)
                    .fontWeight(.bold)
                Text("Image Segmentation Clean")
                    .font(.headline)
                Spacer()
            }
            .padding()
            .background(Color(.systemBackground))
            
            // Error Banner
            if let error = errorMessage {
                Text(error)
                    .font(.caption)
                    .bold()
                    .foregroundColor(.white)
                    .padding(.vertical, 8)
                    .padding(.horizontal, 16)
                    .frame(maxWidth: .infinity)
                    .background(Color.red.opacity(0.9))
            }
            
            Spacer()
            
            // Images Display
            if isProcessing {
                ProgressView("Running segmentation...")
                    .padding()
            } else {
                VStack(spacing: 16) {
                    if let image = originalImage {
                        HStack(spacing: 12) {
                            VStack {
                                Text("Original")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                Image(uiImage: image)
                                    .resizable()
                                    .scaledToFit()
                                    .frame(maxHeight: 250)
                                    .cornerRadius(8)
                            }
                            
                            VStack {
                                Text("Mask Overlay")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                ZStack {
                                    Image(uiImage: image)
                                        .resizable()
                                        .scaledToFit()
                                    if let mask = segmentedMask {
                                        Image(uiImage: mask)
                                            .resizable()
                                            .scaledToFit()
                                            .blendMode(.multiply) // Blend mask over original image
                                    }
                                }
                                .frame(maxHeight: 250)
                                .cornerRadius(8)
                            }
                        }
                        .padding(.horizontal)
                    } else {
                        Text("No test image loaded.")
                            .foregroundColor(.secondary)
                    }
                    
                    // Metrics
                    if segmentedMask != nil {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Performance Metrics:")
                                .font(.subheadline)
                                .fontWeight(.semibold)
                            Text("Preprocess: \(String(format: "%.1f", preProcessTime)) ms")
                            Text("Inference: \(String(format: "%.1f", inferenceTime)) ms")
                            Text("Postprocess: \(String(format: "%.1f", postProcessTime)) ms")
                            Text("Total: \(String(format: "%.1f", preProcessTime + inferenceTime + postProcessTime)) ms")
                        }
                        .font(.footnote)
                        .foregroundColor(.secondary)
                        .padding()
                        .background(Color(.systemGray6))
                        .cornerRadius(8)
                    }
                }
            }
            
            Spacer()
            
            // Controls
            VStack(spacing: 12) {
                HStack {
                    Text("Backend:")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                    
                    Picker("Accelerator", selection: $selectedAccelerator) {
                        ForEach(AcceleratorOption.allCases, id: \.self) { option in
                            Text(option.rawValue).tag(option)
                        }
                    }
                    .pickerStyle(.segmented)
                    .onChange(of: selectedAccelerator) { _ in
                        runSegmentationOnBundledImage()
                    }
                }
                .padding(.horizontal)
                
                Button(action: {
                    runSegmentationOnBundledImage()
                }) {
                    Text("Re-Run Segmentation")
                        .bold()
                        .foregroundColor(.white)
                        .padding()
                        .frame(maxWidth: .infinity)
                        .background(Color.blue)
                        .cornerRadius(8)
                }
                .padding(.horizontal)
            }
            .padding(.vertical)
            .background(Color(.systemBackground))
            .shadow(radius: 2)
        }
        .onAppear {
            runSegmentationOnBundledImage()
        }
    }
    
    private func runSegmentationOnBundledImage() {
        errorMessage = nil
        isProcessing = true
        
        guard let modelPath = Bundle.main.path(forResource: "selfie_multiclass_256x256", ofType: "tflite") else {
            errorMessage = "Model file selfie_multiclass_256x256.tflite not found"
            isProcessing = false
            return
        }
        
        guard let imagePath = Bundle.main.path(forResource: "image", ofType: "jpeg"),
              let uiImage = UIImage(contentsOfFile: imagePath),
              let cgImage = uiImage.cgImage else {
            errorMessage = "Sample image.jpeg not found or failed to load"
            isProcessing = false
            return
        }
        
        self.originalImage = uiImage
        let accelerator = selectedAccelerator.liteRTAccelerator
        
        modelQueue.async {
            do {
                // Initialize segmenter
                let segmenter = try LiteRTSegmenter(modelPath: modelPath, accelerator: accelerator)
                
                // Run segmentation
                let result = try segmenter.segmentImage(cgImage)
                
                DispatchQueue.main.async {
                    self.segmentedMask = result.maskImage
                    self.preProcessTime = result.preProcessTimeMs
                    self.inferenceTime = result.inferenceTimeMs
                    self.postProcessTime = result.postProcessTimeMs
                    self.isProcessing = false
                }
            } catch {
                DispatchQueue.main.async {
                    self.errorMessage = "Error: \(error.localizedDescription)"
                    self.isProcessing = false
                }
            }
        }
    }
}
