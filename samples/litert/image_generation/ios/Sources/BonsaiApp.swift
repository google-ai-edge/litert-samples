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

// Bonsai — minimal on-device text-to-image app around BonsaiPipeline.
// UI is deliberately small: prompt, steps, seed, one button, progress, image.
// Everything still prints to the console so devicectl --console runs work.

import SwiftUI

@main
struct BonsaiApp: App {
    var body: some Scene {
        WindowGroup { ContentView() }
    }
}

@MainActor
final class GenerationModel: ObservableObject {
    @Published var prompt = "a small bonsai tree in a blue ceramic pot"
    @Published var steps = 4
    @Published var seedText = ""
    @Published var running = false
    @Published var stage = ""
    @Published var fraction = 0.0
    @Published var image: UIImage?
    @Published var caption = ""
    @Published var pngURL: URL?
    @Published var errorText: String?
    @Published var missing: [String] = []
    @Published var log: [String] = []

    private var pipeline: BonsaiPipeline?

    var modelsDir: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    func checkAssets() {
        missing = BonsaiPipeline.missingFiles(modelsDir: modelsDir)
    }

    /// CLI-driven runs. Arguments (-autorun 1 [-prompt "…"] [-seed N]
    /// [-steps N]) or environment (BONSAI_AUTORUN=1, BONSAI_PROMPT,
    /// BONSAI_SEED, BONSAI_STEPS — devicectl --environment-variables,
    /// simctl SIMCTL_CHILD_ prefix).
    func maybeAutorun() {
        let d = UserDefaults.standard
        let env = ProcessInfo.processInfo.environment
        guard d.bool(forKey: "autorun") || env["BONSAI_AUTORUN"] == "1",
              missing.isEmpty, !running else { return }
        if let p = env["BONSAI_PROMPT"] ?? d.string(forKey: "prompt"), !p.isEmpty { prompt = p }
        if let s = env["BONSAI_SEED"] ?? d.string(forKey: "seed"), !s.isEmpty { seedText = s }
        let st = env["BONSAI_STEPS"].flatMap(Int.init) ?? d.integer(forKey: "steps")
        if st > 0 { steps = st }
        generate()
    }

    func generate() {
        guard !running else { return }
        errorText = nil
        image = nil
        pngURL = nil
        log = []
        running = true
        fraction = 0
        stage = "Starting…"
        let seed = UInt64(seedText) ?? UInt64.random(in: 0..<1_000_000)
        seedText = String(seed)
        let prompt = prompt
        let steps = steps
        let dir = modelsDir
        setvbuf(stdout, nil, _IONBF, 0)   // devicectl --console sees lines live

        Task.detached(priority: .userInitiated) { [weak self] in
            func post(_ f: @escaping @MainActor (GenerationModel) -> Void) {
                Task { @MainActor in if let self { f(self) } }
            }
            do {
                let pipeline: BonsaiPipeline
                if let p = await self?.pipeline {
                    pipeline = p
                } else {
                    post { $0.stage = "Loading tokenizer…" }
                    pipeline = try BonsaiPipeline(modelsDir: dir)
                    post { $0.pipeline = pipeline }
                }
                let result = try pipeline.generate(
                    prompt: prompt, seed: seed, steps: steps,
                    status: { line in
                        print("[bonsai] \(line)")
                        post { $0.log.append(line) }
                    },
                    progress: { label, f in
                        post {
                            $0.stage = label
                            $0.fraction = f
                        }
                    })
                let ui = Self.makeImage(rgb: result.rgb)
                post {
                    $0.image = ui
                    $0.pngURL = result.pngURL
                    $0.caption = String(format: "seed %llu · %d steps · %.0f s",
                                        seed, steps, result.seconds)
                    $0.running = false
                }
            } catch is BonsaiPipeline.Cancelled {
                post {
                    $0.stage = "Cancelled"
                    $0.running = false
                }
            } catch {
                print("[bonsai] FAILED: \(error.localizedDescription)")
                post {
                    $0.errorText = error.localizedDescription
                    $0.running = false
                }
            }
        }
    }

    func cancel() {
        pipeline?.cancelled = true
        stage = "Cancelling after this step…"
    }

    nonisolated static func makeImage(rgb: [UInt8]) -> UIImage? {
        var rgba = [UInt8](repeating: 255, count: 512 * 512 * 4)
        for p in 0..<(512 * 512) {
            rgba[p * 4] = rgb[p * 3]
            rgba[p * 4 + 1] = rgb[p * 3 + 1]
            rgba[p * 4 + 2] = rgb[p * 3 + 2]
        }
        guard let ctx = CGContext(data: &rgba, width: 512, height: 512, bitsPerComponent: 8,
                                  bytesPerRow: 512 * 4, space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue),
              let img = ctx.makeImage() else { return nil }
        return UIImage(cgImage: img)
    }
}

struct ContentView: View {
    @StateObject private var model = GenerationModel()
    @FocusState private var promptFocused: Bool

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    if !model.missing.isEmpty { missingBox }

                    TextField("Describe an image…", text: $model.prompt, axis: .vertical)
                        .lineLimit(2...5)
                        .textFieldStyle(.roundedBorder)
                        .focused($promptFocused)

                    HStack {
                        Text("Steps").font(.subheadline).foregroundStyle(.secondary)
                        Picker("Steps", selection: $model.steps) {
                            ForEach([2, 4, 6, 8], id: \.self) { Text("\($0)") }
                        }
                        .pickerStyle(.segmented)
                    }

                    HStack {
                        Text("Seed").font(.subheadline).foregroundStyle(.secondary)
                        TextField("random", text: $model.seedText)
                            .keyboardType(.numberPad)
                            .textFieldStyle(.roundedBorder)
                        Button {
                            model.seedText = String(UInt64.random(in: 0..<1_000_000))
                        } label: {
                            Image(systemName: "die.face.5")
                        }
                    }

                    if model.running {
                        Button(role: .destructive) { model.cancel() } label: {
                            Text("Cancel").frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered)
                        ProgressView(value: model.fraction)
                        Text(model.stage)
                            .font(.footnote.monospaced())
                            .foregroundStyle(.secondary)
                    } else {
                        Button {
                            promptFocused = false
                            model.generate()
                        } label: {
                            Text("Generate").frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(!model.missing.isEmpty
                                  || model.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    }

                    if let error = model.errorText {
                        Text(error).font(.footnote).foregroundStyle(.red)
                    }

                    if let image = model.image {
                        VStack(alignment: .leading, spacing: 6) {
                            Image(uiImage: image)
                                .resizable()
                                .scaledToFit()
                                .clipShape(RoundedRectangle(cornerRadius: 12))
                            HStack {
                                Text(model.caption)
                                    .font(.footnote)
                                    .foregroundStyle(.secondary)
                                Spacer()
                                if let url = model.pngURL {
                                    ShareLink(item: url) { Image(systemName: "square.and.arrow.up") }
                                }
                            }
                        }
                    }

                    if !model.log.isEmpty {
                        DisclosureGroup("Timing log") {
                            VStack(alignment: .leading, spacing: 2) {
                                ForEach(model.log.indices, id: \.self) {
                                    Text(model.log[$0]).font(.system(size: 11, design: .monospaced))
                                }
                            }
                            .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        .font(.footnote)
                    }
                }
                .padding()
            }
            .navigationTitle("Bonsai")
            .onAppear {
                model.checkAssets()
                model.maybeAutorun()
            }
        }
    }

    private var missingBox: some View {
        VStack(alignment: .leading, spacing: 4) {
            Label("Model files missing from Documents", systemImage: "exclamationmark.triangle")
                .font(.subheadline.bold())
            ForEach(model.missing, id: \.self) {
                Text($0).font(.footnote.monospaced())
            }
            Text("Copy them from huggingface.co/litert-community/Bonsai-Image-ternary-4B "
                 + "via Finder file sharing (USB).")
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.yellow.opacity(0.15), in: RoundedRectangle(cornerRadius: 10))
    }
}
