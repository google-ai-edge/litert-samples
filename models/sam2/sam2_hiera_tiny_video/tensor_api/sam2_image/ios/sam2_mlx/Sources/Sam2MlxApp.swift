import SwiftUI
import Foundation
import MLX
import MLXNN

@main
struct Sam2MlxApp: App {
    var body: some Scene {
        WindowGroup { ContentView() }
    }
}

/// SAM2 (hiera-tiny) MLX-swift on-device benchmark — the MLX side of the LiteRT-vs-MLX
/// comparison, measured on the same device as the LiteRT harness. Fixed to 512x512 input
/// (matching the LiteRT Tensor API benchmark's default). Runs the ported model — validated
/// at corr 0.9999 against the PyTorch reference at 512 — and reports warm encoder / decoder
/// latency plus the full decoder outputs (3 multimask logits, iou_scores, object score) to
/// the device console (NSLog "SAM2MLXBENCH") and on screen.
struct ContentView: View {
    @State private var status = "Loading MLX SAM2…"
    @State private var running = true

    var body: some View {
        VStack(spacing: 20) {
            Text("SAM2 · MLX-swift benchmark (512)").font(.headline)
            Text(status).font(.system(.footnote, design: .monospaced))
                .multilineTextAlignment(.leading).padding()
            Button(running ? "Running…" : "Re-run") { run() }.disabled(running)
        }
        .padding()
        .onAppear { run() }
    }

    private func run() {
        running = true
        status = "Loading MLX SAM2…"
        DispatchQueue.global(qos: .userInitiated).async {
            let result: String
            do { result = try Benchmark.run() } catch { result = "FAIL \(error)" }
            NSLog("SAM2MLXBENCH\n%@", result)
            DispatchQueue.main.async { status = result; running = false }
        }
    }
}

enum Benchmark {
    /// Input side, fixed to 512 to match the LiteRT Tensor API benchmark.
    /// The bundled weights must be the 512-baked set (`sam2_tiny_512.safetensors`,
    /// produced by `tools/make_512_weights.py`): the Hiera positional embedding is
    /// re-composed at the 128x128 token grid (bicubic base + tiled window embed) —
    /// slicing or naively resizing the composed 1024 grid is geometrically wrong.
    static let size = 512
    static let mean: [Float] = [0.485, 0.456, 0.406]
    static let std: [Float] = [0.229, 0.224, 0.225]

    static func circle() -> MLXArray {
        var arr = [Float](repeating: 0, count: 3 * size * size)
        let plane = size * size
        let (cx, cy, r2) = (size / 2, size / 2, (size / 4 - 8) * (size / 4 - 8))
        for y in 0 ..< size {
            for x in 0 ..< size {
                let value: Float = ((x - cx) * (x - cx) + (y - cy) * (y - cy) <= r2) ? 1.0 : 0.0
                let i = y * size + x
                arr[i] = (value - mean[0]) / std[0]
                arr[plane + i] = (value - mean[1]) / std[1]
                arr[2 * plane + i] = (value - mean[2]) / std[2]
            }
        }
        return MLXArray(arr, [1, 3, size, size]).asType(.float16)
    }

    static func run() throws -> String {
        guard let url = Bundle.main.url(forResource: "sam2_tiny_512", withExtension: "safetensors") else {
            throw NSError(domain: "SAM2MLX", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "sam2_tiny_512.safetensors not bundled — generate it with tools/make_512_weights.py"])
        }
        let prefixes = ["trunk.", "neck.", "sam_prompt_encoder.", "sam_mask_decoder.", "no_mem_embed"]
        let all = try loadArrays(url: url)
        var weights: [String: MLXArray] = [:]
        for (k, v) in all where prefixes.contains(where: { k.hasPrefix($0) }) { weights[k] = v }

        var cfg = HieraCfg()
        cfg.posEmbedHW = (size / 4, size / 4)
        let model = Sam2ImageSegmenter(config: cfg, imageSize: size)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .none)
        eval(model)

        let pixels = circle()
        let coords = MLXArray([Float(size / 2), Float(size / 2)], [1, 1, 2])
        let labels = MLXArray([Int32(1)], [1, 1])

        func median(_ v: [Double]) -> Double { v.sorted()[v.count / 2] }
        let warmup = 5, runs = 20
        for _ in 0 ..< warmup {
            let (v, hr) = model.encodeImage(pixels)
            let (m, i, o) = model.predict(v, hr, coords, labels)
            eval(m, i, o)
        }
        var encTimes: [Double] = []
        for _ in 0 ..< runs {
            let t = Date()
            let (v, _) = model.encodeImage(pixels); eval(v)
            encTimes.append(-t.timeIntervalSinceNow * 1000)
        }
        let (v0, hr0) = model.encodeImage(pixels); eval(v0)
        var decTimes: [Double] = []
        for _ in 0 ..< runs {
            let t = Date()
            let (m, i, o) = model.predict(v0, hr0, coords, labels)
            eval(m, i, o)
            decTimes.append(-t.timeIntervalSinceNow * 1000)
        }
        let (masks, iou, objScore) = model.predict(v0, hr0, coords, labels)
        eval(masks, iou, objScore)

        let maskSide = size / 4
        let iouArr = (0 ..< 3).map { iou[0, $0].item(Float.self) }
        var best = 0
        for j in 1 ..< 3 where iouArr[j] > iouArr[best] { best = j }
        let fgBest = (masks[0, best] .> 0).sum().item(Int32.self)
        let fg0 = (masks[0, 0] .> 0).sum().item(Int32.self)

        return String(format: """
            SAM2 hiera-tiny · MLX-swift (fp16, %dx%d)
            enc_median=%.1fms  dec_median=%.1fms  (runs=%d)
            iou_scores=[%.4f, %.4f, %.4f]  object_score=%.2f
            best mask = [%d]: fg=%d/%d   (mask[0] fg=%d/%d)
            """, size, size, median(encTimes), median(decTimes), runs,
            iouArr[0], iouArr[1], iouArr[2], objScore[0, 0].item(Float.self),
            best, fgBest, maskSide * maskSide, fg0, maskSide * maskSide)
    }
}
