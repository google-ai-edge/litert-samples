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

// Bonsai for macOS — text-to-image with the DiT on the Apple GPU.
// Same minimal UI as the iOS app (prompt / steps / seed / progress / image /
// share) plus an engine panel for the one-time "Compiling for GPU…" phase.
// Headless CLI mode (no window, prints to console, exits when done):
//   BONSAI_AUTORUN=1 [BONSAI_PROMPT=…] [BONSAI_SEED=…] [BONSAI_STEPS=…]
//   [BONSAI_MODELS_DIR=…] ./Bonsai.app/Contents/MacOS/Bonsai

import AppKit
import SwiftUI
import UniformTypeIdentifiers

@main
struct BonsaiMacApp: App {
    init() {
        // Headless CLI mode: run the whole pipeline without a window (the
        // window plumbing never fires onAppear when the raw binary is
        // launched from a script) and exit.
        if ProcessInfo.processInfo.environment["BONSAI_AUTORUN"] == "1" {
            Self.runHeadless()
        }
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .defaultSize(width: 920, height: 640)
    }

    private static func runHeadless() -> Never {
        setvbuf(stdout, nil, _IONBF, 0)
        let env = ProcessInfo.processInfo.environment
        let dir = env["BONSAI_MODELS_DIR"].map { URL(fileURLWithPath: $0) }
            ?? FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("models/bonsai-image-4b-tflite")
        let prompt = env["BONSAI_PROMPT"] ?? "a small bonsai tree in a blue ceramic pot"
        let seed = env["BONSAI_SEED"].flatMap(UInt64.init) ?? 0
        let steps = env["BONSAI_STEPS"].flatMap(Int.init) ?? 4
        do {
            let missing = BonsaiPipelineMac.missingFiles(modelsDir: dir)
            guard missing.isEmpty else {
                print("[bonsai] MISSING: \(missing.joined(separator: ", "))")
                exit(2)
            }
            let pipeline = try BonsaiPipelineMac(modelsDir: dir)
            try pipeline.prepare { print("[bonsai] \($0)") }
            let result = try pipeline.generate(
                prompt: prompt, seed: seed, steps: steps,
                status: { print("[bonsai] \($0)") },
                progress: { _, _ in })
            print("[bonsai] AUTORUN_DONE \(result.pngURL.path)")
            exit(0)
        } catch {
            print("[bonsai] FAILED: \(error.localizedDescription)")
            exit(1)
        }
    }
}

@MainActor
final class GenerationModel: ObservableObject {
    enum Engine: Equatable {
        case missingModels([String])
        case compiling
        case ready
        case failed(String)
    }

    @Published var engine: Engine = .compiling
    @Published var engineStatus = ""
    @Published var prompt = "a small bonsai tree in a blue ceramic pot"
    @Published var steps = 4
    @Published var seedText = ""
    @Published var running = false
    @Published var stage = ""
    @Published var fraction = 0.0
    @Published var image: NSImage?
    @Published var caption = ""
    @Published var pngURL: URL?
    @Published var errorText: String?
    @Published var log: [String] = []

    private var pipeline: BonsaiPipelineMac?
    private var preparing = false

    var modelsDir: URL {
        if let saved = UserDefaults.standard.string(forKey: "modelsDir") {
            return URL(fileURLWithPath: saved)
        }
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("models/bonsai-image-4b-tflite")
    }

    func chooseModelsDir() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.message = "Choose the folder with the Bonsai .tflite files"
        if panel.runModal() == .OK, let url = panel.url {
            UserDefaults.standard.set(url.path, forKey: "modelsDir")
            start()
        }
    }

    /// Checks assets, then compiles the engine in the background (the DiT
    /// Metal compile is the one-time ~40 s wait).
    func start() {
        guard !preparing, !running else { return }
        let missing = BonsaiPipelineMac.missingFiles(modelsDir: modelsDir)
        guard missing.isEmpty else {
            engine = .missingModels(missing)
            return
        }
        preparing = true
        engine = .compiling
        errorText = nil
        setvbuf(stdout, nil, _IONBF, 0)
        let dir = modelsDir
        Task.detached(priority: .userInitiated) { [weak self] in
            func post(_ f: @escaping @MainActor (GenerationModel) -> Void) {
                Task { @MainActor in if let self { f(self) } }
            }
            do {
                let pipeline = try BonsaiPipelineMac(modelsDir: dir)
                try pipeline.prepare { line in
                    print("[bonsai] \(line)")
                    post {
                        $0.engineStatus = line
                        $0.log.append(line)
                    }
                }
                post {
                    $0.pipeline = pipeline
                    $0.preparing = false
                    $0.engine = .ready
                    $0.maybeGuiAutorun()
                }
            } catch {
                print("[bonsai] ENGINE FAILED: \(error.localizedDescription)")
                post {
                    $0.preparing = false
                    $0.engine = .failed(error.localizedDescription)
                }
            }
        }
    }

    /// Demo/screenshot runs with the window up:
    ///   open Bonsai.app --args -guiAutorun 1 [-prompt "…"] [-seed N] [-steps N]
    /// (open --args lands in UserDefaults via NSArgumentDomain.)
    func maybeGuiAutorun() {
        let d = UserDefaults.standard
        guard d.bool(forKey: "guiAutorun"), !running else { return }
        if let p = d.string(forKey: "prompt"), !p.isEmpty { prompt = p }
        if let s = d.string(forKey: "seed"), !s.isEmpty { seedText = s }
        let st = d.integer(forKey: "steps")
        if st > 0 { steps = st }
        generate()
    }

    func generate() {
        guard !running, let pipeline else { return }
        errorText = nil
        image = nil
        pngURL = nil
        running = true
        fraction = 0
        stage = "Starting…"
        let seed = UInt64(seedText) ?? UInt64.random(in: 0..<1_000_000)
        seedText = String(seed)
        let prompt = prompt
        let steps = steps

        Task.detached(priority: .userInitiated) { [weak self] in
            func post(_ f: @escaping @MainActor (GenerationModel) -> Void) {
                Task { @MainActor in if let self { f(self) } }
            }
            do {
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
                let ns = Self.makeImage(rgb: result.rgb)
                post {
                    $0.image = ns
                    $0.pngURL = result.pngURL
                    $0.caption = String(format: "seed %llu · %d steps · %.1f s",
                                        seed, steps, result.seconds)
                    $0.running = false
                }
            } catch is BonsaiPipelineMac.Cancelled {
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

    nonisolated static func makeImage(rgb: [UInt8]) -> NSImage? {
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
        return NSImage(cgImage: img, size: NSSize(width: 512, height: 512))
    }
}

struct ContentView: View {
    @StateObject private var model = GenerationModel()

    var body: some View {
        HSplitView {
            controls
                .frame(minWidth: 330, maxWidth: 420)
            canvas
                .frame(minWidth: 420, maxWidth: .infinity, maxHeight: .infinity)
        }
        .onAppear { model.start() }
        .navigationTitle("Bonsai — Apple GPU")
    }

    private var controls: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                engineBox

                Text("Prompt").font(.subheadline).foregroundStyle(.secondary)
                TextField("Describe an image…", text: $model.prompt, axis: .vertical)
                    .lineLimit(3...6)
                    .textFieldStyle(.roundedBorder)

                HStack {
                    Text("Steps").font(.subheadline).foregroundStyle(.secondary)
                    Picker("", selection: $model.steps) {
                        ForEach([2, 4, 6, 8], id: \.self) { Text("\($0)") }
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                }

                HStack {
                    Text("Seed").font(.subheadline).foregroundStyle(.secondary)
                    TextField("random", text: $model.seedText)
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
                    ProgressView(value: model.fraction)
                    Text(model.stage)
                        .font(.footnote.monospaced())
                        .foregroundStyle(.secondary)
                } else {
                    Button {
                        model.generate()
                    } label: {
                        Text("Generate").frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.return, modifiers: .command)
                    .disabled(model.engine != .ready
                              || model.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }

                if let error = model.errorText {
                    Text(error).font(.footnote).foregroundStyle(.red)
                }

                if !model.log.isEmpty {
                    DisclosureGroup("Timing log") {
                        VStack(alignment: .leading, spacing: 2) {
                            ForEach(model.log.indices, id: \.self) {
                                Text(model.log[$0])
                                    .font(.system(size: 11, design: .monospaced))
                                    .textSelection(.enabled)
                            }
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .font(.footnote)
                }
            }
            .padding()
        }
    }

    @ViewBuilder private var engineBox: some View {
        switch model.engine {
        case .missingModels(let names):
            VStack(alignment: .leading, spacing: 6) {
                Label("Model files not found", systemImage: "exclamationmark.triangle")
                    .font(.subheadline.bold())
                ForEach(names, id: \.self) { Text($0).font(.footnote.monospaced()) }
                Text("Looked under \(model.modelsDir.path) (and gpu_work/, hf_upload/).")
                    .font(.footnote).foregroundStyle(.secondary)
                Button("Choose Models Folder…") { model.chooseModelsDir() }
            }
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.yellow.opacity(0.15), in: RoundedRectangle(cornerRadius: 10))
        case .compiling:
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text("Preparing engine").font(.subheadline.bold())
                }
                Text(model.engineStatus.isEmpty ? "Loading…" : model.engineStatus)
                    .font(.footnote.monospaced())
                    .foregroundStyle(.secondary)
            }
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.blue.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))
        case .ready:
            Label("DiT on Apple GPU · text encoder + VAE on CPU",
                  systemImage: "bolt.fill")
                .font(.footnote)
                .foregroundStyle(.secondary)
        case .failed(let why):
            VStack(alignment: .leading, spacing: 6) {
                Label("Engine failed", systemImage: "xmark.octagon")
                    .font(.subheadline.bold()).foregroundStyle(.red)
                Text(why).font(.footnote.monospaced())
                Button("Retry") { model.start() }
            }
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.red.opacity(0.1), in: RoundedRectangle(cornerRadius: 10))
        }
    }

    private var canvas: some View {
        VStack(spacing: 10) {
            if let image = model.image {
                Image(nsImage: image)
                    .resizable()
                    .scaledToFit()
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                    .shadow(radius: 6)
                HStack {
                    Text(model.caption).font(.footnote).foregroundStyle(.secondary)
                    if let url = model.pngURL {
                        ShareLink(item: url) { Image(systemName: "square.and.arrow.up") }
                        Button {
                            NSWorkspace.shared.activateFileViewerSelecting([url])
                        } label: {
                            Image(systemName: "magnifyingglass")
                        }
                        .help("Reveal in Finder")
                    }
                }
            } else if model.running {
                ProgressView(value: model.fraction)
                    .frame(maxWidth: 320)
                Text(model.stage).font(.footnote).foregroundStyle(.secondary)
            } else {
                Image(systemName: "photo")
                    .font(.system(size: 56))
                    .foregroundStyle(.quaternary)
                Text("512 × 512 · Bonsai Image 4B (ternary DiT, int4)")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
        .padding()
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
