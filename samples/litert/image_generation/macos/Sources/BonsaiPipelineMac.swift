// Copyright 2026 Daisuke Majima. All Rights Reserved.
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

// Bonsai Image 4B pipeline for macOS — the iOS BonsaiPipeline's math with a
// different runtime split: the DiT runs on the Apple GPU through the LiteRT
// Metal accelerator (fp32 forced; default fp16 corrupts this model), while
// text encoder and VAE stay on CPU (XNNPACK), where they are bit-exact vs the
// device fixtures. All three graphs stay resident: the one-time ~40 s Metal
// compile buys ~0.75 s/step, so a 4-step 512x512 image takes ~6 s steady-state.

import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

final class BonsaiPipelineMac {
    struct Failure: Error, LocalizedError {
        let what: String
        var errorDescription: String? { what }
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
    /// Model files may sit flat in the chosen dir or in these subdirs
    /// (the local staging layout uses gpu_work/ + hf_upload/).
    static let searchSubdirs = ["", "gpu_work", "hf_upload"]

    // MARK: assets

    let dir: URL
    let meta: Meta
    let tokenizer: QwenTokenizer
    var cancelled = false

    private let runtime: BonsaiRuntime
    private var textenc: BonsaiGraph?
    private var dit: BonsaiGraph?
    private var vae: BonsaiGraph?

    init(modelsDir: URL) throws {
        dir = modelsDir
        guard let metaURL = Self.findFile("pipeline_meta.json", in: modelsDir)
                ?? Bundle.main.url(forResource: "pipeline_meta", withExtension: "json") else {
            throw Failure(what: "pipeline_meta.json missing")
        }
        meta = try JSONDecoder().decode(Meta.self, from: Data(contentsOf: metaURL))
        guard let vocab = Bundle.main.url(forResource: "vocab", withExtension: "json"),
              let merges = Bundle.main.url(forResource: "merges", withExtension: "txt") else {
            throw Failure(what: "tokenizer resources missing from bundle")
        }
        tokenizer = try QwenTokenizer(vocabURL: vocab, mergesURL: merges)
        guard let fw = Bundle.main.privateFrameworksPath else {
            throw Failure(what: "no Frameworks path")
        }
        runtime = try BonsaiRuntime(libraryDir: fw)
    }

    /// The GPU-shaped DiT export (rope constants folded, gather-free): the
    /// published CPU-shaped dit_int4b32.tflite does NOT run on the Metal
    /// delegate — the app requires the dit_gpu_ sibling.
    static func gpuDitName(for ditFile: String) -> String {
        ditFile
            .replacingOccurrences(of: "_fixed.tflite", with: ".tflite")
            .replacingOccurrences(of: "dit_", with: "dit_gpu_")
    }
    var gpuDitName: String { Self.gpuDitName(for: meta.files.dit) }

    static func findFile(_ name: String, in dir: URL) -> URL? {
        for sub in searchSubdirs {
            for candidate in [name, name.replacingOccurrences(of: ".tflite", with: "_fixed.tflite")] {
                let u = sub.isEmpty ? dir.appendingPathComponent(candidate)
                                    : dir.appendingPathComponent(sub).appendingPathComponent(candidate)
                if FileManager.default.fileExists(atPath: u.path) { return u }
            }
        }
        return nil
    }

    /// Missing required model files (empty = ready to prepare).
    static func missingFiles(modelsDir: URL) -> [String] {
        guard let metaURL = findFile("pipeline_meta.json", in: modelsDir)
                ?? Bundle.main.url(forResource: "pipeline_meta", withExtension: "json"),
              let meta = try? JSONDecoder().decode(Meta.self, from: Data(contentsOf: metaURL)) else {
            return ["pipeline_meta.json"]
        }
        return [gpuDitName(for: meta.files.dit), meta.files.textenc, meta.files.vae].filter {
            findFile($0, in: modelsDir) == nil
        }
    }

    private func modelPath(_ name: String) throws -> String {
        guard let u = Self.findFile(name, in: dir) else {
            throw Failure(what: "\(name) not found under \(dir.path)")
        }
        return u.path
    }

    private func checkCancel() throws {
        if cancelled { throw Cancelled() }
    }

    // MARK: engine warm-up (one-time per launch)

    /// Compiles all three graphs. The DiT Metal compile takes ~40 s.
    func prepare(status: @escaping (String) -> Void) throws {
        if textenc == nil {
            status("Loading text encoder (CPU)…")
            let g = try BonsaiGraph(runtime: runtime,
                                    modelPath: modelPath(meta.files.textenc),
                                    useGpu: false, threads: Self.threads,
                                    intInputMask: 0b11)
            textenc = g
            status(String(format: "text encoder ready %.1fs", g.loadSeconds + g.compileSeconds))
        }
        if vae == nil {
            status("Loading VAE decoder (CPU)…")
            let g = try BonsaiGraph(runtime: runtime,
                                    modelPath: modelPath(meta.files.vae),
                                    useGpu: false, threads: Self.threads,
                                    intInputMask: 0)
            vae = g
            status(String(format: "VAE ready %.1fs", g.loadSeconds + g.compileSeconds))
        }
        if dit == nil {
            status("Compiling DiT for Apple GPU… (~40 s, once per launch)")
            let g = try BonsaiGraph(runtime: runtime,
                                    modelPath: modelPath(gpuDitName),
                                    useGpu: true, threads: Self.threads,
                                    intInputMask: 0)
            dit = g
            status(String(format: "DiT on GPU ready %.1fs (fully accelerated: %@)",
                          g.loadSeconds + g.compileSeconds,
                          g.fullyAccelerated ? "yes" : "NO"))
        }
    }

    var isPrepared: Bool { textenc != nil && dit != nil && vae != nil }

    // MARK: generation

    func generate(prompt: String, seed: UInt64, steps: Int,
                  status: @escaping (String) -> Void,
                  progress: @escaping (String, Double) -> Void) throws -> Result {
        let t00 = Date()
        cancelled = false
        guard let textenc, let dit, let vae else { throw Failure(what: "engine not prepared") }
        func fdata(_ a: [Float]) -> Data { a.withUnsafeBufferPointer { Data(buffer: $0) } }
        func idata(_ a: [Int32]) -> Data { a.withUnsafeBufferPointer { Data(buffer: $0) } }
        func floats(_ d: Data) -> [Float] {
            d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        }

        // ---- stage 1: tokenize + text encoder (CPU) -----------------------
        progress("Encoding prompt…", 0.05)
        let enc = tokenizer.encodePrompt(prompt)
        status("prompt: \(enc.promptTokenCount) tokens")
        var t = Date()
        let embedsData = try textenc.run(inputs: [idata(enc.ids), idata(enc.mask)])
        status(String(format: "text encoder %.2fs", -t.timeIntervalSinceNow))
        try checkCancel()

        // ---- stage 2: DiT Euler loop (GPU, fp32) --------------------------
        let sigmas = BonsaiMath.sigmas(steps: steps)
        let imgIds = fdata(BonsaiMath.imgIDs())
        let txtIds = fdata(BonsaiMath.txtIDs())
        var lat = BonsaiMath.noise(seed: seed)
        var stepSeconds: [Double] = []
        for k in 0..<steps {
            try checkCancel()
            progress("Step \(k + 1) of \(steps) (GPU)…",
                     0.15 + 0.70 * Double(k) / Double(steps))
            t = Date()
            let v = floats(try dit.run(inputs: [fdata(lat), embedsData,
                                                fdata([sigmas[k]]), imgIds, txtIds]))
            let ds = sigmas[k + 1] - sigmas[k]
            for i in 0..<lat.count { lat[i] += ds * v[i] }
            stepSeconds.append(-t.timeIntervalSinceNow)
            status(String(format: "step %d/%d  sigma %.3f  %.2fs",
                          k + 1, steps, sigmas[k], stepSeconds[k]))
        }
        try checkCancel()

        // ---- stage 3: unpatchify + VAE decode (CPU) -----------------------
        progress("Decoding image…", 0.90)
        let z = BonsaiMath.unpatchify(lat, scale: meta.latent_bn_scale,
                                      shift: meta.latent_bn_shift)
        t = Date()
        let y = floats(try vae.run(inputs: [fdata(z)]))
        var rgb = [UInt8](repeating: 0, count: 512 * 512 * 3)
        for c in 0..<3 {
            for p in 0..<(512 * 512) {
                rgb[p * 3 + c] = UInt8(max(0, min(255, ((y[c * 262144 + p] / 2 + 0.5) * 255).rounded())))
            }
        }
        status(String(format: "VAE decode %.2fs", -t.timeIntervalSinceNow))

        // ---- save ----------------------------------------------------------
        let outDir = FileManager.default.urls(for: .applicationSupportDirectory,
                                              in: .userDomainMask)[0]
            .appendingPathComponent("Bonsai", isDirectory: true)
        try? FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
        let stamp = ISO8601DateFormatter().string(from: Date())
            .replacingOccurrences(of: ":", with: "-")
        let png = outDir.appendingPathComponent("bonsai_\(stamp)_seed\(seed).png")
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
            throw Failure(what: "writePNG failed")
        }
        CGImageDestinationAddImage(dest, img, nil)
        CGImageDestinationFinalize(dest)
    }
}
