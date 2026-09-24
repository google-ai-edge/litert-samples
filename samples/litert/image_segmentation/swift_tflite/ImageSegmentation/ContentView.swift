import SwiftUI
import TensorFlowLite

enum AcceleratorOption: String, CaseIterable {
    case cpu = "CPU"
    case gpu = "Metal"
    case coreML = "CoreML"

    var title: String {
        switch self {
        case .cpu: return "CPU (XNNPACK)"
        case .gpu: return "GPU (MetalDelegate)"
        case .coreML: return "NPU (CoreMLDelegate)"
        }
    }

    var tfliteAccelerator: TFLiteAccelerator {
        switch self {
        case .cpu: return .cpu
        case .gpu: return .metal
        case .coreML: return .coreML
        }
    }
}

enum AppTab {
    case camera
    case gallery
}

struct ContentView: View {
    // Theme colors matching the Kotlin Android sample (darkBlue and teal)
    private let primaryColor = Color(red: 2/255.0, green: 15/255.0, blue: 89/255.0)  // #020F59
    private let accentColor = Color(red: 0/255.0, green: 201/255.0, blue: 158/255.0)  // #00C99E

    #if targetEnvironment(simulator)
    @State private var activeTab: AppTab = .gallery
    #else
    @State private var activeTab: AppTab = .camera
    #endif
    @State private var selectedAccelerator: AcceleratorOption = .gpu

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

    // Cached segmenter instance
    @State private var activeSegmenter: TFLiteSegmenter? = nil
    @State private var activeAccelerator: TFLiteAccelerator? = nil
    @State private var isInitializingSegmenter = false

    // Performance metrics
    @State private var preProcessTime: Double = 0
    @State private var inferenceTime: Double = 0
    @State private var postProcessTime: Double = 0

    // Bottom Sheet Drag State
    @State private var sheetOffset: CGFloat = 0
    @State private var isSheetExpanded = false

    private let modelQueue = DispatchQueue(label: "com.tflite.segmentation.queue", qos: .userInitiated)

    var body: some View {
        ZStack(alignment: .bottom) {
            VStack(spacing: 0) {
                HStack(spacing: 8) {
                    if let logoPath = Bundle.main.path(forResource: "logo", ofType: "png"),
                       let uiImage = UIImage(contentsOfFile: logoPath) {
                        Image(uiImage: uiImage)
                            .resizable()
                            .scaledToFit()
                            .frame(height: 28)
                    }
                    Text("Image Segmentation (TFLite)")
                        .font(.title3)
                        .fontWeight(.bold)
                        .foregroundColor(.white)

                    Spacer()
                }
                .padding(.horizontal)
                .padding(.vertical, 12)
                .background(accentColor)

                // Tab Selection bar
                HStack(spacing: 0) {
                    Button(action: {
                        activeTab = .camera
                        cameraManager.requestPermissionAndStart()
                        segmentedMask = nil
                        errorMessage = nil
                    }) {
                        VStack(spacing: 8) {
                            Text("Camera")
                                .fontWeight(activeTab == .camera ? .bold : .regular)
                                .foregroundColor(activeTab == .camera ? accentColor : .secondary)
                            Rectangle()
                                .fill(activeTab == .camera ? accentColor : Color.clear)
                                .frame(height: 3)
                        }
                    }
                    .frame(maxWidth: .infinity)

                    Button(action: {
                        activeTab = .gallery
                        cameraManager.stop()
                        segmentedMask = nil
                        errorMessage = nil
                        runSegmentationOnSelectedImage()
                    }) {
                        VStack(spacing: 8) {
                            Text("Gallery")
                                .fontWeight(activeTab == .gallery ? .bold : .regular)
                                .foregroundColor(activeTab == .gallery ? accentColor : .secondary)
                            Rectangle()
                                .fill(activeTab == .gallery ? accentColor : Color.clear)
                                .frame(height: 3)
                        }
                    }
                    .frame(maxWidth: .infinity)
                }
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

                // Main Content Viewport
                ZStack {
                    Color(.systemGray6)
                        .edgesIgnoringSafeArea(.all)

                    if activeTab == .camera {
                        CameraViewport()
                    } else {
                        GalleryViewport()
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            .edgesIgnoringSafeArea(.bottom)

            // Bottom Sheet
            BottomSheetView()
        }
        .onAppear {
            if activeTab == .camera {
                cameraManager.requestPermissionAndStart()
            } else {
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

    // MARK: - Viewports

    @ViewBuilder
    private func CameraViewport() -> some View {
        switch cameraManager.permissionStatus {
        case .notDetermined:
            ProgressView("Requesting camera access...")
        case .denied:
            VStack(spacing: 12) {
                Image(systemName: "camera.meter.lost.fill")
                    .font(.system(size: 48))
                    .foregroundColor(.secondary)
                Text("Camera Access Denied")
                    .font(.headline)
                Text("Please allow camera access in Settings.")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
            }
            .padding()
        case .authorized:
            if let cameraFrame = cameraManager.currentFrame {
                ZStack(alignment: .topTrailing) {
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
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .clipped()

                    // Floating Camera Flip FAB
                    Button(action: {
                        cameraManager.flipCamera()
                    }) {
                        Image(systemName: "camera.rotate.fill")
                            .font(.title3)
                            .foregroundColor(.white)
                            .padding(12)
                            .background(Color.black.opacity(0.6))
                            .clipShape(Circle())
                            .shadow(radius: 4)
                    }
                    .padding()
                }
                .onChange(of: cameraManager.currentFrame) { newFrame in
                    if let frame = newFrame {
                        processLiveFrame(frame)
                    }
                }
            } else {
                ProgressView("Waiting for camera feed...")
            }
        }
    }

    @ViewBuilder
    private func GalleryViewport() -> some View {
        ZStack(alignment: .bottomTrailing) {
            if isProcessingStatic {
                ProgressView("Running segmentation...")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if let image = originalImage {
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
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .padding(.bottom, 120)
            } else {
                Text("No image loaded.")
                    .foregroundColor(.secondary)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }

            // Add Photo FAB
            Button(action: {
                showingImagePicker = true
            }) {
                Image(systemName: "plus")
                    .font(.title2.bold())
                    .foregroundColor(.white)
                    .padding(18)
                    .background(accentColor)
                    .clipShape(Circle())
                    .shadow(radius: 5)
            }
            .padding(.trailing, 20)
            .padding(.bottom, 120)
        }
    }

    // MARK: - Bottom Sheet Component

    @ViewBuilder
    private func BottomSheetView() -> some View {
        let peekHeight: CGFloat = 85
        let expandedHeight: CGFloat = 275

        VStack(spacing: 0) {
            // Drag Indicator & Peek Content
            VStack(spacing: 8) {
                RoundedRectangle(cornerRadius: 3)
                    .fill(accentColor)
                    .frame(width: 40, height: 5)
                    .padding(.top, 8)

                HStack {
                    Text("Inference Time (\(selectedAccelerator.title))")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                    Spacer()
                    Text(segmentedMask != nil ? "\(String(format: "%.1f", inferenceTime)) ms" : "-- ms")
                        .font(.headline)
                        .fontWeight(.bold)
                        .foregroundColor(primaryColor)
                }
                .padding(.horizontal, 24)
                .padding(.bottom, 10)
            }
            .contentShape(Rectangle())
            .onTapGesture {
                withAnimation(.spring()) {
                    isSheetExpanded.toggle()
                }
            }

            Divider()

            // Expanded content (Metrics & Backend Controls)
            VStack(spacing: 14) {
                // Secondary Metrics Rows
                VStack(spacing: 6) {
                    HStack {
                        Text("Preprocess Time:")
                            .font(.footnote)
                            .foregroundColor(.secondary)
                        Spacer()
                        Text(segmentedMask != nil ? "\(String(format: "%.1f", preProcessTime)) ms" : "-- ms")
                            .font(.footnote)
                            .bold()
                    }
                    HStack {
                        Text("Postprocess Time:")
                            .font(.footnote)
                            .foregroundColor(.secondary)
                        Spacer()
                        Text(segmentedMask != nil ? "\(String(format: "%.1f", postProcessTime)) ms" : "-- ms")
                            .font(.footnote)
                            .bold()
                    }
                    HStack {
                        Text("TFLite Runtime Version:")
                            .font(.footnote)
                            .foregroundColor(.secondary)
                        Spacer()
                        Text(TensorFlowLite.Runtime.version)
                            .font(.footnote)
                            .bold()
                    }
                }

                Divider()

                // Backend Selection Picker (CPU XNNPACK / GPU MetalDelegate / NPU CoreMLDelegate)
                VStack(alignment: .leading, spacing: 8) {
                    Text("Delegate / Accelerator")
                        .font(.subheadline)
                        .fontWeight(.medium)

                    Picker("Accelerator", selection: $selectedAccelerator) {
                        ForEach(AcceleratorOption.allCases, id: \.self) { option in
                            Text(option.rawValue).tag(option)
                        }
                    }
                    .pickerStyle(.segmented)
                    .onChange(of: selectedAccelerator) { _ in
                        if activeTab == .gallery {
                            runSegmentationOnSelectedImage()
                        } else {
                            segmentedMask = nil
                            let accelerator = selectedAccelerator.tfliteAccelerator
                            getOrInitializeSegmenter(for: accelerator) { _ in }
                        }
                    }
                }
            }
            .padding(.top, 14)
            .padding(.horizontal, 24)

            Spacer()
        }
        .frame(height: expandedHeight)
        .background(Color(.systemBackground))
        .cornerRadius(18)
        .shadow(radius: 6)
        .offset(y: isSheetExpanded ? 0 : expandedHeight - peekHeight)
        .gesture(
            DragGesture()
                .onEnded { value in
                    withAnimation(.spring()) {
                        if value.translation.height < -50 {
                            isSheetExpanded = true
                        } else if value.translation.height > 50 {
                            isSheetExpanded = false
                        }
                    }
                }
        )
    }

    // MARK: - Core Processing Logic

    private func getOrInitializeSegmenter(for accelerator: TFLiteAccelerator, completion: @escaping (TFLiteSegmenter?) -> Void) {
        if let segmenter = activeSegmenter, activeAccelerator == accelerator {
            completion(segmenter)
            return
        }

        if isInitializingSegmenter {
            completion(nil)
            return
        }

        isInitializingSegmenter = true

        guard let modelPath = Bundle.main.path(forResource: "selfie_multiclass_256x256", ofType: "tflite") else {
            errorMessage = "Model file selfie_multiclass_256x256.tflite not found"
            isInitializingSegmenter = false
            completion(nil)
            return
        }

        modelQueue.async {
            do {
                let segmenter = try TFLiteSegmenter(modelPath: modelPath, accelerator: accelerator)
                DispatchQueue.main.async {
                    self.activeSegmenter = segmenter
                    self.activeAccelerator = accelerator
                    self.isInitializingSegmenter = false
                    completion(segmenter)
                }
            } catch {
                DispatchQueue.main.async {
                    self.errorMessage = "Failed to initialize segmenter: \(error.localizedDescription)"
                    self.isInitializingSegmenter = false
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

        let accelerator = selectedAccelerator.tfliteAccelerator

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
        guard activeTab == .camera, !isProcessingLiveFrame, !isInitializingSegmenter else { return }

        let accelerator = selectedAccelerator.tfliteAccelerator
        getOrInitializeSegmenter(for: accelerator) { segmenter in
            guard let segmenter = segmenter else { return }

            self.isProcessingLiveFrame = true

            modelQueue.async {
                do {
                    let result = try segmenter.segmentImage(cgImage)
                    DispatchQueue.main.async {
                        if self.activeTab == .camera {
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
