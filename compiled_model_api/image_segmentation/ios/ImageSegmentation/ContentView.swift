import SwiftUI
import CoreGraphics

struct ContentView: View {
    @StateObject private var cameraManager = CameraManager()
    @State private var useMetal = false
    @State private var segmenter: LiteRTSegmenter?
    @State private var currentMask: UIImage?
    @State private var inferenceTimeMs: Double = 0.0
    
    let modelPath = Bundle.main.path(forResource: "selfie_multiclass_256x256", ofType: "tflite")
    
    var body: some View {
        VStack {
            Text("LiteRT iOS Segmentation")
                .font(.headline)
                .padding()
            
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
                        .overlay(Text("No Camera Feed").foregroundColor(.white))
                }
                
                if let mask = currentMask {
                    Image(uiImage: mask)
                        .resizable()
                        .scaledToFill()
                        .frame(maxWidth: .infinity, maxHeight: 400)
                        .opacity(0.8)
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 15))
            .padding()
            
            HStack {
                Text("Backend: ")
                Picker("Backend", selection: $useMetal) {
                    Text("CPU").tag(false)
                    Text("Metal").tag(true)
                }
                .pickerStyle(SegmentedPickerStyle())
                .onChange(of: useMetal) { _ in
                    initSegmenter()
                }
            }
            .padding()
            
            Text(String(format: "Inference time: %.1f ms", inferenceTimeMs))
                .font(.subheadline)
                .padding()
            
            Spacer()
        }
        .onAppear {
            initSegmenter()
            cameraManager.start()
        }
        .onDisappear {
            cameraManager.stop()
        }
        .onChange(of: cameraManager.currentFrame) { frame in
            guard let frame = frame, let segmenter = segmenter else { return }
            
            // Run segmentation asynchronously
            DispatchQueue.global(qos: .userInitiated).async {
                var error: NSError?
                if let result = segmenter.segmentImage(frame, error: &error) {
                    DispatchQueue.main.async {
                        self.currentMask = result.maskImage
                        self.inferenceTimeMs = result.inferenceTimeMs
                    }
                } else if let error = error {
                    print("Segmentation error: \(error.localizedDescription)")
                }
            }
        }
    }
    
    private func initSegmenter() {
        guard let path = modelPath else {
            print("Model path not found")
            return
        }
        var error: NSError?
        let accelerator = useMetal ? LiteRTAcceleratorMetal : LiteRTAcceleratorCPU
        segmenter = LiteRTSegmenter(modelPath: path, accelerator: accelerator, error: &error)
        if let error = error {
            print("Failed to initialize segmenter: \(error)")
        }
    }
}
