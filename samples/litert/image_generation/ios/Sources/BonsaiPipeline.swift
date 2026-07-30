// Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

// Bonsai Image 4B on-device pipeline — the verification Runner (device/
// BonsaiDeviceTest) turned into an app engine: arbitrary prompt via the Swift
// Qwen tokenizer, seeded on-device noise, and the Mac-verified host math from
// BonsaiMath. Runtime path is unchanged from the verified run: classic TFLite
// C API + explicit XNNPACK delegate (mandatory — without it this build falls
// back to reference kernels: orders of magnitude slower AND wrong numerics
// for blockwise int4), graphs loaded sequentially and freed between stages so
// peak memory stays ~DiT-sized (~2.9 GiB on iPhone 17 Pro).

import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

final class BonsaiPipeline {
    struct Failure: Error, LocalizedError {
        let what: String
        let code: Int32
        var errorDescription: String? { "\(what) (status \(code))" }
    }
    struct Cancelled: Error {}

    struct Meta: Decodable {
        struct Files: Decodable {
            let dit: String
            let textenc: String
            let vae: String
        }
        let files: Files
        let latent_bn_scale: [Float]
        let latent_bn_shift: [Float]
    }

    struct Result {
        let rgb: [UInt8]              // 512*512*3
        let pngURL: URL
        let seconds: Double
        let stepSeconds: [Double]
    }

    static let threads: Int32 = 6

    // MARK: single fixed-shape graph, CPU

    final class Graph {
        private var model: OpaquePointer?
        private var opts: OpaquePointer?
        private var interp: OpaquePointer?
        private var delegate: OpaquePointer?
        private var order: [Int32] = []
        let loadSeconds: Double

        init(path: String) throws {
            let t = Date()
            model = TfLiteModelCreateFromFile(path)
            guard model != nil else { throw Failure(what: "ModelCreateFromFile", code: -1) }
            opts = TfLiteInterpreterOptionsCreate()
            TfLiteInterpreterOptionsSetNumThreads(opts, BonsaiPipeline.threads)
            delegate = BonsaiCreateXnnpackDelegate(BonsaiPipeline.threads)
            guard delegate != nil else { throw Failure(what: "XNNPackDelegateCreate", code: -9) }
            TfLiteInterpreterOptionsAddDelegate(opts, delegate)
            interp = TfLiteInterpreterCreate(model, opts)
            guard interp != nil else { throw Failure(what: "InterpreterCreate", code: -2) }
            guard TfLiteInterpreterAllocateTensors(interp) == 0 else {
                throw Failure(what: "AllocateTensors", code: -3)
            }
            loadSeconds = -t.timeIntervalSinceNow
            // input order by serving_default_args_<n>, NEVER by shape/index
            let n = TfLiteInterpreterGetInputTensorCount(interp)
            func argpos(_ i: Int32) -> Int {
                guard let t = TfLiteInterpreterGetInputTensor(interp, i),
                      let c = TfLiteTensorName(t) else { return Int(i) }
                let name = String(cString: c)
                guard let r = name.range(of: "args_") else { return Int(i) }
                return Int(name[r.upperBound...].prefix(while: { $0.isNumber })) ?? Int(i)
            }
            order = (0..<n).sorted { argpos($0) < argpos($1) }
        }

        func run(_ inputs: [Data], outCount: Int) throws -> [Float] {
            guard inputs.count == order.count else {
                throw Failure(what: "inputCount \(inputs.count) != \(order.count)", code: -4)
            }
            for (input, idx) in zip(inputs, order) {
                guard let t = TfLiteInterpreterGetInputTensor(interp, idx) else {
                    throw Failure(what: "GetInputTensor \(idx)", code: -5)
                }
                guard TfLiteTensorByteSize(t) == input.count else {
                    throw Failure(what: "byteSize input \(idx): graph \(TfLiteTensorByteSize(t)) "
                                  + "!= host \(input.count)", code: -6)
                }
                let st = input.withUnsafeBytes {
                    TfLiteTensorCopyFromBuffer(t, $0.baseAddress, input.count)
                }
                guard st == 0 else { throw Failure(what: "CopyFromBuffer \(idx)", code: st) }
            }
            guard TfLiteInterpreterInvoke(interp) == 0 else {
                throw Failure(what: "Invoke", code: -7)
            }
            guard let ot = TfLiteInterpreterGetOutputTensor(interp, 0) else {
                throw Failure(what: "GetOutputTensor", code: -8)
            }
            var out = [Float](repeating: 0, count: outCount)
            let st = out.withUnsafeMutableBytes {
                TfLiteTensorCopyToBuffer(ot, $0.baseAddress, outCount * MemoryLayout<Float>.stride)
            }
            guard st == 0 else { throw Failure(what: "CopyToBuffer", code: st) }
            return out
        }

        deinit {
            if interp != nil { TfLiteInterpreterDelete(interp) }
            if opts != nil { TfLiteInterpreterOptionsDelete(opts) }
            if delegate != nil { TfLiteXNNPackDelegateDelete(delegate) }
            if model != nil { TfLiteModelDelete(model) }
        }
    }

    // MARK: assets

    let dir: URL
    let meta: Meta
    let tokenizer: QwenTokenizer
    var cancelled = false

    /// pipeline_meta.json comes from the models dir if present (HF repo
    /// layout), else from the app bundle; the tokenizer tables ship in the
    /// bundle. The 4 GiB of .tflite always live in the models dir.
    init(modelsDir: URL) throws {
        dir = modelsDir
        let metaURL = Self.bundleFallback(modelsDir.appendingPathComponent("pipeline_meta.json"),
                                          resource: "pipeline_meta", ext: "json")
        guard let metaURL else { throw Failure(what: "pipeline_meta.json missing", code: -20) }
        meta = try JSONDecoder().decode(Meta.self, from: Data(contentsOf: metaURL))
        guard let vocab = Bundle.main.url(forResource: "vocab", withExtension: "json"),
              let merges = Bundle.main.url(forResource: "merges", withExtension: "txt") else {
            throw Failure(what: "tokenizer resources missing from bundle", code: -21)
        }
        tokenizer = try QwenTokenizer(vocabURL: vocab, mergesURL: merges)
    }

    private static func bundleFallback(_ url: URL, resource: String, ext: String) -> URL? {
        if FileManager.default.fileExists(atPath: url.path) { return url }
        return Bundle.main.url(forResource: resource, withExtension: ext)
    }

    /// The published file name, or its `_fixed` sibling (the device
    /// verification set used dit_int4b32_fixed.tflite).
    static func resolveModel(_ name: String, in dir: URL) -> URL? {
        for candidate in [name, name.replacingOccurrences(of: ".tflite", with: "_fixed.tflite")] {
            let u = dir.appendingPathComponent(candidate)
            if FileManager.default.fileExists(atPath: u.path) { return u }
        }
        return nil
    }

    /// Missing required model files (empty = ready to generate).
    static func missingFiles(modelsDir: URL) -> [String] {
        guard let metaURL = bundleFallback(modelsDir.appendingPathComponent("pipeline_meta.json"),
                                           resource: "pipeline_meta", ext: "json"),
              let meta = try? JSONDecoder().decode(Meta.self, from: Data(contentsOf: metaURL)) else {
            return ["pipeline_meta.json"]
        }
        return [meta.files.dit, meta.files.textenc, meta.files.vae].filter {
            resolveModel($0, in: modelsDir) == nil
        }
    }

    private func modelPath(_ name: String) throws -> String {
        guard let u = Self.resolveModel(name, in: dir) else {
            throw Failure(what: "\(name) not found in \(dir.lastPathComponent)", code: -22)
        }
        return u.path
    }

    private func checkCancel() throws {
        if cancelled { throw Cancelled() }
    }

    // MARK: generation

    func generate(prompt: String, seed: UInt64, steps: Int,
                  status: @escaping (String) -> Void,
                  progress: @escaping (String, Double) -> Void) throws -> Result {
        let t00 = Date()
        cancelled = false
        func fdata(_ a: [Float]) -> Data { a.withUnsafeBufferPointer { Data(buffer: $0) } }
        func idata(_ a: [Int32]) -> Data { a.withUnsafeBufferPointer { Data(buffer: $0) } }

        // ---- stage 1: tokenize + text encoder ----------------------------
        progress("Encoding prompt…", 0.02)
        let enc = tokenizer.encodePrompt(prompt)
        status("prompt: \(enc.promptTokenCount) tokens")
        var embeds: [Float] = []
        try autoreleasepool {
            let te = try Graph(path: try modelPath(meta.files.textenc))
            try checkCancel()
            let t = Date()
            embeds = try te.run([idata(enc.ids), idata(enc.mask)],
                                outCount: BonsaiMath.seq * 7680)
            status(String(format: "text encoder %.1fs (load %.1fs)",
                          -t.timeIntervalSinceNow, te.loadSeconds))
        }
        try checkCancel()

        // ---- stage 2: DiT Euler loop --------------------------------------
        progress("Loading DiT (2.1 GiB)…", 0.10)
        let sigmas = BonsaiMath.sigmas(steps: steps)
        let imgIds = fdata(BonsaiMath.imgIDs())
        let txtIds = fdata(BonsaiMath.txtIDs())
        let embedsData = fdata(embeds)
        var lat = BonsaiMath.noise(seed: seed)
        var stepSeconds: [Double] = []
        try autoreleasepool {
            let dit = try Graph(path: try modelPath(meta.files.dit))
            status(String(format: "DiT loaded %.1fs", dit.loadSeconds))
            for k in 0..<steps {
                try checkCancel()
                progress("Step \(k + 1) of \(steps)…",
                         0.16 + 0.72 * Double(k) / Double(steps))
                let t = Date()
                let v = try dit.run(
                    [fdata(lat), embedsData, fdata([sigmas[k]]), imgIds, txtIds],
                    outCount: BonsaiMath.tokens * BonsaiMath.packedChannels)
                let ds = sigmas[k + 1] - sigmas[k]
                for i in 0..<lat.count { lat[i] += ds * v[i] }
                stepSeconds.append(-t.timeIntervalSinceNow)
                status(String(format: "step %d/%d  sigma %.3f  %.1fs",
                              k + 1, steps, sigmas[k], stepSeconds[k]))
            }
        }
        try checkCancel()

        // ---- stage 3: unpatchify + VAE decode ------------------------------
        progress("Decoding image…", 0.90)
        let z = BonsaiMath.unpatchify(lat, scale: meta.latent_bn_scale,
                                      shift: meta.latent_bn_shift)
        var rgb = [UInt8](repeating: 0, count: 512 * 512 * 3)
        try autoreleasepool {
            let vae = try Graph(path: try modelPath(meta.files.vae))
            let t = Date()
            let y = try vae.run([fdata(z)], outCount: 3 * 512 * 512)
            for c in 0..<3 {
                for p in 0..<(512 * 512) {
                    rgb[p * 3 + c] = UInt8(max(0, min(255, ((y[c * 262144 + p] / 2 + 0.5) * 255).rounded())))
                }
            }
            status(String(format: "VAE decode %.1fs", -t.timeIntervalSinceNow))
        }

        // ---- save -----------------------------------------------------------
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let stamp = ISO8601DateFormatter().string(from: Date())
            .replacingOccurrences(of: ":", with: "-")
        let png = docs.appendingPathComponent("bonsai_\(stamp)_seed\(seed).png")
        try Self.writePNG(rgb, to: png)
        let total = -t00.timeIntervalSinceNow
        status(String(format: "TOTAL %.1fs -> %@", total, png.lastPathComponent))
        progress("Done", 1.0)
        return Result(rgb: rgb, pngURL: png, seconds: total, stepSeconds: stepSeconds)
    }

    // MARK: png

    static func writePNG(_ rgb: [UInt8], to url: URL) throws {
        var rgba = [UInt8](repeating: 255, count: 512 * 512 * 4)
        for p in 0..<(512 * 512) {
            rgba[p * 4] = rgb[p * 3]
            rgba[p * 4 + 1] = rgb[p * 3 + 1]
            rgba[p * 4 + 2] = rgb[p * 3 + 2]
        }
        guard let ctx = CGContext(data: &rgba, width: 512, height: 512, bitsPerComponent: 8,
                                  bytesPerRow: 512 * 4, space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue),
              let img = ctx.makeImage(),
              let dest = CGImageDestinationCreateWithURL(
                url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            throw Failure(what: "writePNG", code: -30)
        }
        CGImageDestinationAddImage(dest, img, nil)
        CGImageDestinationFinalize(dest)
    }
}
