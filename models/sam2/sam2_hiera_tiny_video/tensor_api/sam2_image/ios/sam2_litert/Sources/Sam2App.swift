import SwiftUI

@main
struct Sam2App: App {
    var body: some Scene {
        WindowGroup { ContentView() }
    }
}

struct ContentView: View {
    @State private var status = "Compiling SAM2 (512x512) on GPU…"
    @State private var running = true

    var body: some View {
        VStack(spacing: 20) {
            Text("SAM2 · LiteRT Tensor API (512x512)")
                .font(.headline)
            Text(status)
                .font(.system(.footnote, design: .monospaced))
                .multilineTextAlignment(.leading)
                .padding()
            Button(running ? "Running…" : "Re-run benchmark") { run() }
                .disabled(running)
        }
        .padding()
        .onAppear { run() }
    }

    private func run() {
        running = true
        status = "Compiling SAM2 (512x512) on GPU…"
        DispatchQueue.global(qos: .userInitiated).async {
            let result: String
            do {
                result = try Benchmark.run()
            } catch {
                result = "FAIL \(error)"
            }
            print("SAM2BENCH\n\(result)")
            fflush(stdout)
            NSLog("SAM2BENCH\n%@", result)
            DispatchQueue.main.async {
                status = result
                running = false
            }
        }
    }
}

enum Benchmark {
    static let SIZE = 512
    static let MG = 128
    static let MEAN: [Float] = [0.485, 0.456, 0.406]
    static let STD: [Float] = [0.229, 0.224, 0.225]
    static let WARMUP = 5
    static let RUNS = 20

    static func run() throws -> String {
        guard let path = Bundle.main.path(forResource: "sam2_image", ofType: "tflite") else {
            throw NSError(domain: "SAM2", code: 1, userInfo: [NSLocalizedDescriptionKey: "sam2_image.tflite not in bundle"])
        }
        let model = LiteRTModel()
        let (fully, compileSec) = try model.compileOnGPU(path: path)

        let bPixels = try model.createBuffer(sig: 0, isInput: true, index: 0, dims: [1, 512, 512, 3])
        let bImgEmb = try model.createBuffer(sig: 0, isInput: false, index: 0, dims: [1, 32, 32, 256])
        let bFeatS1 = try model.createBuffer(sig: 0, isInput: false, index: 1, dims: [1, 64, 64, 64])
        let bFeatS0 = try model.createBuffer(sig: 0, isInput: false, index: 2, dims: [1, 128, 128, 32])

        let bCoords = try model.createBuffer(sig: 1, isInput: true, index: 3, dims: [1, 1, 2])
        let bMasks  = try model.createBuffer(sig: 1, isInput: false, index: 0, dims: [1, 3, 128, 128])
        let bIou    = try model.createBuffer(sig: 1, isInput: false, index: 1, dims: [1, 3])
        let bObj    = try model.createBuffer(sig: 1, isInput: false, index: 2, dims: [1, 1])

        defer {
            for b in [bPixels, bImgEmb, bFeatS1, bFeatS0, bCoords, bMasks, bIou, bObj] {
                if let b = b { LiteRtDestroyTensorBuffer(b) }
            }
        }

        try model.writeFloats(bPixels, circleInputNHWC())
        try model.writeFloats(bCoords, [Float(SIZE / 2), Float(SIZE / 2)])

        var encIn: [LiteRtTensorBuffer?] = [bPixels]
        var encOut: [LiteRtTensorBuffer?] = [bImgEmb, bFeatS1, bFeatS0]
        var decIn: [LiteRtTensorBuffer?] = [bImgEmb, bFeatS1, bFeatS0, bCoords]
        var decOut: [LiteRtTensorBuffer?] = [bMasks, bIou, bObj]

        for _ in 0..<WARMUP {
            try model.runSignature(0, inputs: &encIn, outputs: &encOut)
            try model.runSignature(1, inputs: &decIn, outputs: &decOut)
        }

        var encTimes: [Double] = []
        for _ in 0..<RUNS {
            let t = Date()
            try model.runSignature(0, inputs: &encIn, outputs: &encOut)
            encTimes.append(-t.timeIntervalSinceNow * 1000.0)
        }

        try model.runSignature(0, inputs: &encIn, outputs: &encOut)
        var decTimes: [Double] = []
        for _ in 0..<RUNS {
            let t = Date()
            try model.runSignature(1, inputs: &decIn, outputs: &decOut)
            decTimes.append(-t.timeIntervalSinceNow * 1000.0)
        }

        let masks = try model.readFloats(bMasks, count: 3 * MG * MG)
        let iou = try model.readFloats(bIou, count: 3)
        let obj = try model.readFloats(bObj, count: 1)

        var best = 0
        for m in 1..<3 where iou[m] > iou[best] { best = m }
        let plane = MG * MG
        let fg = masks[(best * plane)..<((best + 1) * plane)].reduce(0) { $0 + ($1 > 0 ? 1 : 0) }

        return String(
            format: "SAM2 hiera-tiny · LiteRT Tensor API (512x512, Metal fp16)\nfullyGPU=%@ compile=%.2fs\nenc_median=%.2fms  dec_median=%.2fms  (runs=%d)\niou_scores=[%.4f, %.4f, %.4f]  object_score=%.2f\nbest mask = [%d]: fg=%d/%d",
            fully ? "YES" : "NO", compileSec,
            median(encTimes), median(decTimes), RUNS,
            iou[0], iou[1], iou[2], obj[0],
            best, fg, plane)
    }

    static func circleInputNHWC() -> [Float] {
        var out = [Float](repeating: 0, count: SIZE * SIZE * 3)
        let cx = SIZE / 2, cy = SIZE / 2, r = SIZE / 4 - 8, r2 = r * r
        for y in 0..<SIZE {
            for x in 0..<SIZE {
                let v: Float = ((x - cx) * (x - cx) + (y - cy) * (y - cy) <= r2) ? 1.0 : 0.0
                let base = (y * SIZE + x) * 3
                out[base + 0] = (v - MEAN[0]) / STD[0]
                out[base + 1] = (v - MEAN[1]) / STD[1]
                out[base + 2] = (v - MEAN[2]) / STD[2]
            }
        }
        return out
    }

    static func median(_ v: [Double]) -> Double { v.sorted()[v.count / 2] }
}
