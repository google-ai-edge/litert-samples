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

enum AppMode: String, CaseIterable {
    case gallery = "Gallery Image"
    case liveCamera = "Live Camera"
}

struct ContentView: View {
    @State private var appMode: AppMode = .gallery
    @State private var selectedAccelerator: AcceleratorOption = .cpu
    
    // Gallery Mode State
    @State private var originalImage: UIImage? = nil
    @State private var showingImagePicker = false
    
    // Common State
    @State private var segmentedMask: UIImage? = nil
    @State private var errorMessage: String? = nil
    @State private var isProcessingStatic = false
    
    // Live Mode State
    @StateObject private var cameraManager = CameraManager()
    @State private var isProcessingLiveFrame = false
    
    // Cached segmenter instance to avoid re-compilation on every frame
    @State private var activeSegmenter: LiteRTSegmenter? = nil
    @State private var activeAccelerator: LiteRTAccelerator? = nil
    @State private var isInitializingSegmenter = false
    
    // Performance metrics
    @State private var preProcessTime: Double = 0
    @State private var inferenceTime: Double = 0
    @State private var postProcessTime: Double = 0
    
    private let modelQueue = DispatchQueue(label: "com.litert.segmentation.queue", qos: .userInitiated)
    
    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("LiteRT")
                        .font(.headline)
                        .fontWeight(.bold)
                    Text("Image Segmentation Clean")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                Spacer()
                
                if appMode == .liveCamera && cameraManager.permissionStatus == .authorized {
                    Button(action: {
                        cameraManager.flipCamera()
                    }) {
                        Image(systemName: "camera.rotate")
                            .font(.title2)
                            .foregroundColor(.blue)
                            .padding(8)
                            .background(Color(.systemGray6))
                            .clipShape(Circle())
                    }
                }
            }
            .padding()
            .background(Color(.systemBackground))
            
            // Mode Picker
            Picker("Mode", selection: $appMode) {
                ForEach(AppMode.allCases, id: \.self) { mode in
                    Text(mode.rawValue).tag(mode)
                }
            }
            .pickerStyle(.segmented)
            .padding(.horizontal)
            .padding(.bottom, 8)
            .onChange(of: appMode) { newMode in
                segmentedMask = nil
                errorMessage = nil
                if newMode == .liveCamera {
                    cameraManager.requestPermissionAndStart()
                } else {
                    cameraManager.stop()
                    runSegmentationOnSelectedImage()
                }
            }
            
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
            
            // Display Area
            if appMode == .gallery {
                // Static image view
                if isProcessingStatic {
                    ProgressView("Running segmentation...")
                        .padding()
                } else if let image = originalImage {
                    HStack(spacing: 12) {
                        VStack {
                            Text("Selected")
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
                                        .blendMode(.multiply)
                                }
                            }
                            .frame(maxHeight: 250)
                            .cornerRadius(8)
                        }
                    }
                    .padding(.horizontal)
                } else {
                    Text("No image loaded.")
                        .foregroundColor(.secondary)
                }
            } else {
                // Live camera view
                switch cameraManager.permissionStatus {
                case .notDetermined:
                    ProgressView("Requesting camera access...")
                case .denied:
                    VStack(spacing: 8) {
                        Image(systemName: "camera.meter.lost.fill")
                            .font(.largeTitle)
                            .foregroundColor(.secondary)
                        Text("Camera access denied.")
                            .font(.headline)
                        Text("Please allow camera access in Settings to use live mode.")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal)
                    }
                case .authorized:
                    if let cameraFrame = cameraManager.currentFrame {
                        VStack {
                            Text("Realtime Stream Overlay")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            ZStack {
                                Image(decorative: cameraFrame, scale: 1.0, orientation: .up)
                                    .resizable()
                                    .scaledToFit()
                                if let mask = segmentedMask {
                                    Image(uiImage: mask)
                                        .resizable()
                                        .scaledToFit()
                                        .blendMode(.multiply)
                                }
                            }
                            .frame(maxHeight: 350)
                            .cornerRadius(8)
                            .onChange(of: cameraManager.currentFrame) { newFrame in
                                if let frame = newFrame {
                                    processLiveFrame(frame)
                                }
                            }
                        }
                        .padding(.horizontal)
                    } else {
                        ProgressView("Waiting for camera feed...")
                    }
                }
            }
            
            // Metrics Display
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
                .padding(.top, 16)
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
                        if appMode == .gallery {
                            runSegmentationOnSelectedImage()
                        } else {
                            // Re-initialize segmenter on background thread
                            isInitializingSegmenter = true
                            segmentedMask = nil
                            let accelerator = selectedAccelerator.liteRTAccelerator
                            getOrInitializeSegmenter(for: accelerator) { _ in
                                self.isInitializingSegmenter = false
                            }
                        }
                    }
                }
                .padding(.horizontal)
                
                if appMode == .gallery {
                    Button(action: {
                        showingImagePicker = true
                    }) {
                        Text(originalImage == nil ? "Select Image from Gallery" : "Change Image")
                            .bold()
                            .foregroundColor(.white)
                            .padding()
                            .frame(maxWidth: .infinity)
                            .background(Color.blue)
                            .cornerRadius(8)
                    }
                    .padding(.horizontal)
                }
            }
            .padding(.vertical)
            .background(Color(.systemBackground))
            .shadow(radius: 2)
        }
        .onAppear {
            if appMode == .gallery {
                runSegmentationOnSelectedImage()
            }
        }
        .onDisappear {
            cameraManager.stop()
        }
        .sheet(isPresented: $showingImagePicker) {
            ImagePicker(image: $originalImage)
        }
        .onChange(of: originalImage) { _ in
            runSegmentationOnSelectedImage()
        }
    }
    
    // Core logic: Caches and returns the active segmenter instance
    private func getOrInitializeSegmenter(for accelerator: LiteRTAccelerator, completion: @escaping (LiteRTSegmenter?) -> Void) {
        if let segmenter = activeSegmenter, activeAccelerator == accelerator {
            completion(segmenter)
            return
        }
        
        guard let modelPath = Bundle.main.path(forResource: "selfie_multiclass_256x256", ofType: "tflite") else {
            errorMessage = "Model file selfie_multiclass_256x256.tflite not found"
            completion(nil)
            return
        }
        
        modelQueue.async {
            do {
                let segmenter = try LiteRTSegmenter(modelPath: modelPath, accelerator: accelerator)
                DispatchQueue.main.async {
                    self.activeSegmenter = segmenter
                    self.activeAccelerator = accelerator
                    completion(segmenter)
                }
            } catch {
                DispatchQueue.main.async {
                    self.errorMessage = "Failed to initialize segmenter: \(error.localizedDescription)"
                    completion(nil)
                }
            }
        }
    }
    
    private func runSegmentationOnSelectedImage() {
        errorMessage = nil
        isProcessingStatic = true
        
        let cgImage: CGImage
        if let uiImage = originalImage {
            guard let cg = uiImage.cgImage else {
                errorMessage = "Selected image has no CGImage backing"
                isProcessingStatic = false
                return
            }
            cgImage = cg
        } else {
            // Default portrait image on startup
            guard let imagePath = Bundle.main.path(forResource: "image", ofType: "jpeg"),
                  let uiImage = UIImage(contentsOfFile: imagePath),
                  let cg = uiImage.cgImage else {
                errorMessage = "Default image.jpeg not found"
                isProcessingStatic = false
                return
            }
            self.originalImage = uiImage
            cgImage = cg
        }
        
        let accelerator = selectedAccelerator.liteRTAccelerator
        
        getOrInitializeSegmenter(for: accelerator) { segmenter in
            guard let segmenter = segmenter else {
                self.isProcessingStatic = false
                return
            }
            
            modelQueue.async {
                do {
                    let result = try segmenter.segmentImage(cgImage)
                    DispatchQueue.main.async {
                        self.segmentedMask = result.maskImage
                        self.preProcessTime = result.preProcessTimeMs
                        self.inferenceTime = result.inferenceTimeMs
                        self.postProcessTime = result.postProcessTimeMs
                        self.isProcessingStatic = false
                    }
                } catch {
                    DispatchQueue.main.async {
                        self.errorMessage = "Error: \(error.localizedDescription)"
                        self.isProcessingStatic = false
                    }
                }
            }
        }
    }
    
    private func processLiveFrame(_ cgImage: CGImage) {
        // Skip frame if already running inference or if setup is loading
        guard appMode == .liveCamera, !isProcessingLiveFrame, !isInitializingSegmenter else { return }
        
        let accelerator = selectedAccelerator.liteRTAccelerator
        getOrInitializeSegmenter(for: accelerator) { segmenter in
            guard let segmenter = segmenter else { return }
            
            self.isProcessingLiveFrame = true
            
            modelQueue.async {
                do {
                    let result = try segmenter.segmentImage(cgImage)
                    DispatchQueue.main.async {
                        // Only update UI if we are still in live mode
                        if self.appMode == .liveCamera {
                            self.segmentedMask = result.maskImage
                            self.preProcessTime = result.preProcessTimeMs
                            self.inferenceTime = result.inferenceTimeMs
                            self.postProcessTime = result.postProcessTimeMs
                        }
                        self.isProcessingLiveFrame = false
                    }
                } catch {
                    DispatchQueue.main.async {
                        self.isProcessingLiveFrame = false
                    }
                }
            }
        }
    }
}
